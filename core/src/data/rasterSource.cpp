#include "data/rasterSource.h"
#include "data/propertyItem.h"
#include "data/tileData.h"
#include "tile/tile.h"
#include "tile/tileBuilder.h"
#include "tile/tileTask.h"
#include "util/mapProjection.h"
#include "log.h"

#include <cstring>
#include <vector>

namespace Tangram {

namespace {

// Copy one WxW cell (raw bytes, `bpp` bytes/pixel) into the mosaic buffer at grid position
// (dx+1, dy+1), dx/dy in {-1, 0, 1}. See stitchElevationMosaic() for the overall layout.
void copyMosaicCell(GLubyte* mosaic, size_t mosaicRowBytes, int W, int bpp,
                     int dx, int dy, const GLubyte* srcData) {
    size_t rowOff = size_t(dy + 1) * size_t(W);
    size_t colOff = size_t(dx + 1) * size_t(W) * size_t(bpp);
    for (int r = 0; r < W; r++) {
        GLubyte* dst = mosaic + (rowOff + size_t(r)) * mosaicRowBytes + colOff;
        const GLubyte* src = srcData + size_t(r) * size_t(W) * size_t(bpp);
        std::memcpy(dst, src, size_t(W) * size_t(bpp));
    }
}

// Synthesize a missing neighbor cell at grid position (dx+1, dy+1) by mirror-extrapolating
// `ownData` (the tile's own WxW buffer): flip vertically for a N/S neighbor (dy != 0),
// horizontally for an E/W neighbor (dx != 0), both ways for a diagonal neighbor. This is a
// pure byte/index shuffle -- no pixel decode -- so it's agnostic to the source's pixel
// encoding (terrarium-RGB, float, etc.) and only ever feeds the coarsest mip levels of the
// shader-side pyramid, where exactness doesn't matter.
void fillMirroredNeighborCell(GLubyte* mosaic, size_t mosaicRowBytes, int W, int bpp,
                               int dx, int dy, const GLubyte* ownData) {
    size_t rowOff = size_t(dy + 1) * size_t(W);
    size_t colOff = size_t(dx + 1) * size_t(W) * size_t(bpp);
    for (int r = 0; r < W; r++) {
        int srcRow = (dy == 0) ? r : (W - 1 - r);
        const GLubyte* srcRowPtr = ownData + size_t(srcRow) * size_t(W) * size_t(bpp);
        GLubyte* dstRowPtr = mosaic + (rowOff + size_t(r)) * mosaicRowBytes + colOff;
        if (dx == 0) {
            std::memcpy(dstRowPtr, srcRowPtr, size_t(W) * size_t(bpp));
        } else {
            for (int c = 0; c < W; c++) {
                int srcCol = W - 1 - c;
                std::memcpy(dstRowPtr + size_t(c) * size_t(bpp), srcRowPtr + size_t(srcCol) * size_t(bpp), size_t(bpp));
            }
        }
    }
}

} // namespace

std::shared_ptr<Texture> stitchElevationMosaic(const std::shared_ptr<Texture>& own,
                                                const std::array<std::shared_ptr<Texture>, 9>& neighbors) {
    if (!own) { return nullptr; }

    int W = own->width();
    if (W <= 0 || W != own->height()) {
        LOGE("stitchElevationMosaic: source texture must be square (got %dx%d)", own->width(), own->height());
        return nullptr;
    }

    const GLubyte* ownData = own->bufferData();
    if (!ownData) {
        LOGE("stitchElevationMosaic: source texture has no CPU buffer data"
             " (RasterSource::m_keepTextureData must be set for the elevation source)");
        return nullptr;
    }

    TextureOptions options = own->getOptions();
    int bpp = options.bytesPerPixel();

    int mosaicW = 3 * W;
    size_t mosaicRowBytes = size_t(mosaicW) * size_t(bpp);
    std::vector<GLubyte> mosaic(mosaicRowBytes * size_t(mosaicW));

    // Tile's own data occupies the center third [W, 2W) x [W, 2W) -- see
    // docs/texture-shading-plan.md "Frozen Interface Contract".
    copyMosaicCell(mosaic.data(), mosaicRowBytes, W, bpp, 0, 0, ownData);

    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) { continue; }

            const auto& neighborTex = neighbors[size_t(dy + 1) * 3 + size_t(dx + 1)];

            if (neighborTex && neighborTex->bufferData() &&
                neighborTex->width() == W && neighborTex->height() == W &&
                neighborTex->getOptions().bytesPerPixel() == bpp) {
                copyMosaicCell(mosaic.data(), mosaicRowBytes, W, bpp, dx, dy, neighborTex->bufferData());
            } else {
                fillMirroredNeighborCell(mosaic.data(), mosaicRowBytes, W, bpp, dx, dy, ownData);
            }
        }
    }

    // Frozen Interface Contract: mosaic must be uploaded with hardware mipmaps and
    // LINEAR_MIPMAP_LINEAR/LINEAR filtering; wrap stays CLAMP_TO_EDGE (the mosaic already
    // contains real neighbor data -- or a mirrored approximation of it -- so it must not rely
    // on GL texture repeat/wrap). This is a per-texture override: `own`'s own TextureOptions
    // (e.g. plain "nearest" filtering used by the per-tile cache) is left untouched.
    options.generateMipmaps = true;
    options.minFilter = TextureMinFilter::LINEAR_MIPMAP_LINEAR;
    options.magFilter = TextureMagFilter::LINEAR;
    options.wrapS = TextureWrap::CLAMP_TO_EDGE;
    options.wrapT = TextureWrap::CLAMP_TO_EDGE;

    auto tex = std::make_shared<Texture>(options);
    tex->setPixelData(mosaicW, mosaicW, bpp, mosaic.data(), mosaic.size());

    return tex;
}

class RasterTileTask : public BinaryTileTask {
public:

    const bool subTask = false;

    std::unique_ptr<Texture> texture;
    std::unique_ptr<Raster> raster;

    RasterTileTask(const TileID& _tileId, TileSource* _source, bool _subTask)
        : BinaryTileTask(_tileId, _source),
          subTask(_subTask) {}


    RasterSource* rasterSource() {
        return static_cast<RasterSource*>(source());  //std::static_pointer_cast<RasterSource>(m_source.lock());
    }

    bool hasData() const override {
        // probably should be "return BinaryTileTask::hasData() || ..."
        return bool(rawTileData) || bool(texture) || bool(raster);
    }

    void process(TileBuilder& _tileBuilder) override {
        auto source = rasterSource();
        assert(!m_ready);  // shared task previously could be erroneously added to tile worker queue twice

        if (!texture && !raster) {
            // Decode texture data
            texture = source->createTexture(m_tileId, *rawTileData);
            if (!texture) {
                // cancel on decode failure to match behavior of TileTask (and behavior for download failure)
                //  empty texture will be set in addRaster() if no proxy available
                //raster = std::make_unique<Raster>(m_tileId, source->emptyTexture());
                cancel();
                return;
            }
        }

        // Create tile geometries
        if (!subTask) {
          // make raster available for tile builder; RasterSource texture cache is not thread-safe currently
          //  so we can't add raster for good until complete() on main thread; note the empty deleter since
          //  shared_ptr doesn't have a release() method
          auto ptex = raster ? raster->texture : std::shared_ptr<Texture>(texture.get(), [](auto* t){});
          m_tile = std::make_unique<Tile>(m_tileId, source->id(), source->generation());
          m_tile->rasters().emplace_back(m_tileId, ptex);
          _tileBuilder.build(*m_tile, *(source->m_tileData), *source);
          m_tile->rasters().pop_back();
        }
        m_ready = true;
    }

    void addRaster(Tile& _tile) {
        auto source = rasterSource();
        if (!raster) {
          auto tex = source->cacheTexture(m_tileId, std::move(texture));
          raster = std::make_unique<Raster>(m_tileId, tex);
        }
        // `raster->texture` is always the plain per-tile texture from RasterSource::m_textures
        // at this point (whether just decoded above, or found via the createRasterTask() reuse
        // path) -- for the elevation source, replace it with a freshly-stitched neighbor mosaic
        // for rendering, without disturbing what's cached in m_textures (other tiles still need
        // the plain per-tile texture as *their* neighbor data). See
        // docs/texture-shading-plan.md Phase 2.
        if (source->m_buildElevationMosaic) {
            if (auto mosaicTex = source->buildElevationMosaic(m_tileId, raster->texture)) {
                _tile.rasters().emplace_back(raster->tileID, mosaicTex);
                return;
            }
            // Fall through to the plain texture if the mosaic couldn't be built (e.g. no
            // CPU-side buffer data was retained).
        }
        _tile.rasters().emplace_back(raster->tileID, raster->texture);
    }

    void complete() override {

        addRaster(*m_tile);

        for (auto& subtask : m_subTasks) {
            //assert(subTask->isReady());
            subtask->complete(*this);
        }
    }

    void complete(TileTask& _mainTask) override {
        if (!isReady()) {  //isCanceled()?
            // check for alternative raster is now in TileManager
            auto source = rasterSource();
            _mainTask.tile()->rasters().emplace_back(tileId(), source->emptyTexture());
        } else {
            addRaster(*_mainTask.tile());
        }
    }
};


RasterSource::RasterSource(const std::string& _name, std::unique_ptr<DataSource> _sources,
                           TextureOptions _options, TileSource::ZoomOptions _zoomOptions)
    : TileSource(_name, std::move(_sources), _zoomOptions),
      m_texOptions(_options) {

    m_textures = std::make_shared<Cache>();
    m_emptyTexture = std::make_shared<Texture>(m_texOptions);

    GLubyte pixel[4] = { 0, 0, 0, 0 };
    auto bpp = _options.bytesPerPixel();
    m_emptyTexture->setPixelData(1, 1, bpp, pixel, bpp);
}

void RasterSource::generateGeometry(bool _generateGeometry) {
    m_generateGeometry = _generateGeometry;

    if (m_generateGeometry) {
        Feature rasterFeature;
        rasterFeature.geometryType = GeometryType::polygons;
        rasterFeature.polygons = { { {
                    {0.0f, 0.0f},
                    {1.0f, 0.0f},
                    {1.0f, 1.0f},
                    {0.0f, 1.0f},
                    {0.0f, 0.0f}
                } } };
        rasterFeature.props = Properties();

        m_tileData = std::make_shared<TileData>();
        m_tileData->layers.emplace_back("");
        m_tileData->layers.back().features.push_back(rasterFeature);
    }
}

std::unique_ptr<Texture> RasterSource::createTexture(TileID _tile, const std::vector<char>& _rawTileData) {
    if (_rawTileData.empty()) { return nullptr; }

    auto data = reinterpret_cast<const uint8_t*>(_rawTileData.data());
    auto length = _rawTileData.size();
    auto tex = std::make_unique<Texture>(m_texOptions, !m_keepTextureData);
    if (!tex->loadImageFromMemory(data, length)) { tex.reset(); }
    return tex;
}

std::shared_ptr<TileData> RasterSource::parse(const TileTask& _task) const {
    assert(false);
    return nullptr;
}

void RasterSource::addRasterTask(TileTask& _task) {

    TileID subTileID = _task.tileId();

    // apply apt downsampling for raster tiles depending on difference
    // in zoomBias (which also takes zoom offset into account)
    auto zoomDiff = m_zoomOptions.zoomBias - _task.source()->zoomBias();

    if (zoomDiff > 0) {
        subTileID = subTileID.zoomBiasAdjusted(zoomDiff).withMaxSourceZoom(m_zoomOptions.maxZoom);
    } else {
        subTileID = subTileID.withMaxSourceZoom(m_zoomOptions.maxZoom);
    }

    auto rasterTask = createRasterTask(subTileID, true);

    _task.subTasks().push_back(rasterTask);
}

std::shared_ptr<RasterTileTask> RasterSource::createRasterTask(TileID _tileId, bool subTask) {
    auto task = std::make_shared<RasterTileTask>(_tileId, this, subTask);

    // First try existing textures cache
    TileID id(_tileId.x, _tileId.y, _tileId.z);

    auto texIt = m_textures->find(id);
    if (texIt != m_textures->end()) {
        auto texture = texIt->second.lock();

        if (texture) {
            LOGV("%d - reuse %s", m_textures->size(), id.toString().c_str());

            task->raster = std::make_unique<Raster>(id, texture);
            // No more loading needed.
            task->startedLoading();
            if (subTask) { task->setReady(); }
        }
    }
    return task;
}

std::shared_ptr<TileTask> RasterSource::createTask(TileID _tileId) {
    auto task = createRasterTask(_tileId, false);

    addRasterTasks(*task);

    return task;
}

std::shared_ptr<Texture> RasterSource::cacheTexture(const TileID& _tileId, std::unique_ptr<Texture> _texture) {
    assert(_texture && _texture->bufferSize() > 0 && _texture.get() != m_emptyTexture.get());
    TileID id(_tileId.x, _tileId.y, _tileId.z);

    auto& textureEntry = (*m_textures)[id];
    auto texture = textureEntry.lock();
    if (texture) {
        LOGV("%d - drop duplicate %s", m_textures->size(), id.toString().c_str());
        // The same texture has been loaded in the meantime: Reuse it and drop _texture..
        return texture;
    }

    texture = std::shared_ptr<Texture>(_texture.release(),
                                       [c = std::weak_ptr<Cache>(m_textures), id](auto* t) {
                                           if (auto cache = c.lock()) {
                                               cache->erase(id);
                                               LOGV("%d - remove %s", cache->size(), id.toString().c_str());
                                           }
                                           delete t;
                                       });
    // Add to cache
    textureEntry = texture;
    LOGV("%d - added %s", m_textures->size(), id.toString().c_str());

    return texture;
}

std::shared_ptr<Texture> RasterSource::getTexture(TileID _tile) {
    auto texIt = m_textures->find(_tile);
    return texIt != m_textures->end() ? texIt->second.lock() : nullptr;
}

std::shared_ptr<Texture> RasterSource::buildElevationMosaic(TileID _tile, const std::shared_ptr<Texture>& _ownTexture) {
    // Look up whichever of the 8 same-zoom neighbors are already cached (getTexture() is only
    // safe to call here because addRaster()/complete() run on the main thread -- see the
    // thread-safety note on RasterTileTask::addRaster()). Missing neighbors are left null;
    // stitchElevationMosaic() synthesizes them via mirror-extrapolation.
    std::array<std::shared_ptr<Texture>, 9> neighbors;
    int32_t max = 1 << _tile.z;

    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) { continue; }

            int32_t ny = _tile.y + dy;
            if (ny < 0 || ny >= max) { continue; }  // no tile across a pole -- leave as "missing"

            int32_t nx = ((_tile.x + dx) % max + max) % max;  // wrap around the antimeridian

            neighbors[size_t(dy + 1) * 3 + size_t(dx + 1)] = getTexture(TileID(nx, ny, _tile.z, _tile.s));
        }
    }

    auto mosaic = stitchElevationMosaic(_ownTexture, neighbors);

    if (mosaic) {
        LOGD("Elevation mosaic for tile %s: %dx%d (own tile %dx%d)", _tile.toString().c_str(),
             mosaic->width(), mosaic->height(),
             _ownTexture ? _ownTexture->width() : 0, _ownTexture ? _ownTexture->height() : 0);
    }

    return mosaic;
}

Raster RasterSource::getRaster(ProjectedMeters _meters) {
    if (m_textures->empty()) { return Raster(NOT_A_TILE, nullptr); }

    TileID tileId = MapProjection::projectedMetersTile(_meters, m_textures->begin()->first.z);
    auto minz = m_textures->rbegin()->first.z;
    do {
        auto tex = getTexture(tileId);
        if(tex) { return Raster(tileId, tex); }
        tileId = tileId.getParent();
    } while(tileId.z >= minz);

    return Raster(NOT_A_TILE, nullptr);
}

}
