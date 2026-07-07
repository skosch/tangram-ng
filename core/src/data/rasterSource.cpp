#include "data/rasterSource.h"
#include "data/propertyItem.h"
#include "data/tileData.h"
#include "tile/tile.h"
#include "tile/tileBuilder.h"
#include "tile/tileTask.h"
#include "util/mapProjection.h"
#include "log.h"

#include <algorithm>
#include <cstring>
#include <vector>

namespace Tangram {

namespace {

// Mosaic layout. Cells are Wp x Wp where Wp = W - overlap:
//
// Texture rows run SOUTH -> NORTH (row 0 is the tile's south edge: rasters are sampled with
// v = 0 at MapProjection::tileSouthWestCorner - see ElevationManager::elevationLerp), while
// tile y runs north -> south. So a neighbor's ROW band in the mosaic is (1 - dy), not
// (dy + 1): the north neighbor (dy = -1) occupies the TOP rows [2Wp, 3Wp). Columns and tile
// x both run west -> east, so the column band is (dx + 1) as expected. Getting the row
// direction wrong shows up as bright/dark bands hugging every horizontal tile boundary (a
// high-pass filter across the resulting elevation step).
//
// `overlap` (o = W % 2): elevation tiles that are (2^n + 1) px wide (e.g. 257) are
// node-registered - their outermost row/col is shared with (duplicates) the adjacent tile's
// edge. Every cell (own tile included) is that tile's buffer cropped to its first Wp rows
// and columns; the dropped last line of each tile is identical to the next cell's first
// line, so the mosaic is a seamless, evenly-spaced global sample grid. Crucially this makes
// the mosaic 3 * (W - o) wide (768 for 257px tiles, instead of 771): adjacent tiles'
// mosaics are then offset by exactly Wp = 256 samples and every mip level halves evenly
// (768 -> 384 -> ... -> 3), so the hardware mip box-filter grids of neighboring mosaics
// stay ALIGNED - both tiles compute bit-identical blurred elevation for the same geography,
// which is what keeps the high-pass texture shading free of tile-boundary steps. (With the
// naive odd 3W = 771 size, neighbor grids are misaligned at every mip level and NPOT mip
// downsampling is driver-arbitrary, leaving a faint visible seam on every boundary.)
inline size_t cellRowOffset(int dy, int Wp) { return size_t(1 - dy) * size_t(Wp); }

// Copy one Wp x Wp cell (raw bytes, `bpp` bytes/pixel) from a real tile buffer (row stride
// `W` samples) into the mosaic. dx/dy in {-1, 0, 1} are TILE offsets (dx = 1: east
// neighbor, dy = 1: south neighbor).
void copyMosaicCell(GLubyte* mosaic, int Wp, int W, int bpp,
                     int dx, int dy, const GLubyte* srcData) {
    size_t mosaicRowBytes = size_t(3 * Wp) * size_t(bpp);
    size_t rowOff = cellRowOffset(dy, Wp);
    size_t colOff = size_t(dx + 1) * size_t(Wp) * size_t(bpp);
    for (int r = 0; r < Wp; r++) {
        std::memcpy(mosaic + (rowOff + size_t(r)) * mosaicRowBytes + colOff,
                    srcData + size_t(r) * size_t(W) * size_t(bpp),
                    size_t(Wp) * size_t(bpp));
    }
}

// Synthesize a missing neighbor cell by mirror-extrapolating `ownData` (the tile's own
// WxW buffer) about the shared tile edge: flipped vertically for a N/S neighbor (dy != 0),
// horizontally for an E/W neighbor (dx != 0), both ways for a diagonal neighbor. For
// node-registered tiles (overlap = 1) the reflection axis is the shared edge node itself
// (whose sample, own row/col Wp, is real data even though it's cropped out of the own
// cell). This is a pure byte/index shuffle -- no pixel decode -- so it's agnostic to the
// source's pixel encoding (terrarium-RGB, float, etc.) and only ever feeds the coarsest
// mip levels of the shader-side pyramid, where exactness doesn't matter; it is also
// transient, replaced with real data by patchNeighborMosaics() as neighbor tiles arrive.
void fillMirroredNeighborCell(GLubyte* mosaic, int Wp, int W, int bpp,
                               int dx, int dy, const GLubyte* ownData) {
    int overlap = W - Wp;
    size_t mosaicRowBytes = size_t(3 * Wp) * size_t(bpp);
    size_t rowOff = cellRowOffset(dy, Wp);
    size_t colOff = size_t(dx + 1) * size_t(Wp) * size_t(bpp);
    for (int r = 0; r < Wp; r++) {
        // reflection: srcRow = Wp - r for node-registered (about the shared node), else
        // Wp - 1 - r (about the inter-pixel edge); index Wp is valid only when overlap = 1
        int srcRow = (dy == 0) ? r : (Wp - 1 + overlap - r);
        const GLubyte* srcRowPtr = ownData + size_t(srcRow) * size_t(W) * size_t(bpp);
        GLubyte* dstRowPtr = mosaic + (rowOff + size_t(r)) * mosaicRowBytes + colOff;
        if (dx == 0) {
            std::memcpy(dstRowPtr, srcRowPtr, size_t(Wp) * size_t(bpp));
        } else {
            for (int c = 0; c < Wp; c++) {
                int srcCol = Wp - 1 + overlap - c;
                std::memcpy(dstRowPtr + size_t(c) * size_t(bpp), srcRowPtr + size_t(srcCol) * size_t(bpp), size_t(bpp));
            }
        }
    }
}

// Dimensionless RMS-slope statistic of a tile's elevation data, sampled at a stride of
// ~W/32 texels (a mid-pyramid-band scale, matching the scales texture shading emphasizes),
// normalized by true ground distance so it is comparable across zoom levels and latitudes.
// Used for texture shading auto-contrast - see RasterSource::aggregateRugosity().
float computeRugosity(const Tangram::Texture& tex, const Tangram::TileID& id) {
    int W = tex.width();
    const GLubyte* data = tex.bufferData();
    if (W < 64 || !data) { return 0.f; }
    bool isFloat = tex.getOptions().pixelFormat == Tangram::PixelFormat::FLOAT;
    auto elevAt = [&](int c, int r) -> double {  // decode matches getElevationAt in elevation.yaml
        if (isFloat) { return ((const float*)data)[size_t(r) * W + c]; }
        const GLubyte* p = data + (size_t(r) * W + c) * 4;
        return (p[0]*256 + p[1] + p[2]/256.0) - 32768;  // mapzen terrarium
    };
    int s = W / 32;
    double sum2 = 0;
    int n = 0;
    for (int r = 0; r + s < W; r += s) {
        for (int c = 0; c + s < W; c += s) {
            double h = elevAt(c, r);
            double dx = elevAt(c + s, r) - h;
            double dy = elevAt(c, r + s) - h;
            sum2 += dx*dx + dy*dy;
            n += 2;
        }
    }
    if (n == 0) { return 0.f; }
    // true meters per texel = projected meters * Mercator scale factor cos(lat)
    Tangram::ProjectedMeters center = Tangram::MapProjection::tileCenter(id);
    double unproj = std::exp(center.y / Tangram::MapProjection::EARTH_RADIUS_METERS);
    double coslat = 2.0 / (unproj + 1.0 / unproj);
    double metersPerTexel = Tangram::MapProjection::metersPerTileAtZoom(id.z) * coslat / W;
    return float(std::sqrt(sum2 / n) / (double(s) * metersPerTexel));
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

    // odd width = node-registered tiles (e.g. 257 = 2^8 + 1) whose outermost row/col is
    // shared with the adjacent tile; cells are cropped to Wp = W - overlap so mosaics stay
    // mip-grid-aligned across neighboring tiles - see the layout notes above
    int overlap = W % 2;
    int Wp = W - overlap;
    if (Wp <= 0) { return nullptr; }

    int mosaicW = 3 * Wp;
    size_t mosaicRowBytes = size_t(mosaicW) * size_t(bpp);
    std::vector<GLubyte> mosaic(mosaicRowBytes * size_t(mosaicW));

    // Tile's own data occupies the center third [Wp, 2Wp) x [Wp, 2Wp) -- see
    // docs/texture-shading-plan.md "Frozen Interface Contract".
    copyMosaicCell(mosaic.data(), Wp, W, bpp, 0, 0, ownData);

    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) { continue; }

            const auto& neighborTex = neighbors[size_t(dy + 1) * 3 + size_t(dx + 1)];

            if (neighborTex && neighborTex->bufferData() &&
                neighborTex->width() == W && neighborTex->height() == W &&
                neighborTex->getOptions().bytesPerPixel() == bpp) {
                copyMosaicCell(mosaic.data(), Wp, W, bpp, dx, dy, neighborTex->bufferData());
            } else {
                fillMirroredNeighborCell(mosaic.data(), Wp, W, bpp, dx, dy, ownData);
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

    // disposeBuffer = false: the CPU-side buffer must stay resident so
    // RasterSource::patchNeighborMosaics() can overwrite individual cells in place when
    // late-arriving neighbor tiles replace their mirror-extrapolated placeholders.
    auto tex = std::make_shared<Texture>(options, false);
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
        //
        // Phase 4 finding: skip the EXPENSIVE full stitch (8 getTexture() lookups + a fresh
        // 3Wx3W CPU buffer) for isProxy() tasks -- that covers both
        // TileManager::TileEntry::m_prefetchOnly tiles (Phase 1; set via setProxyState() in
        // tileManager.cpp, never rendered at all) and genuine lower-zoom placeholder/proxy
        // tiles (rendered only briefly while the real tile loads). Doing a full stitch for
        // every one of the ~8x as many prefetch-only tiles as real tiles was saturating the
        // tile-worker pool badly enough to stall real (non-elevation) tile loading entirely --
        // see docs/texture-shading-plan.md Frozen Interface Contract addendum.
        //
        // But a proxy tile CAN still cheaply reuse an already-registered mosaic (getExistingMosaic:
        // one map lookup, no stitching) -- and during ordinary panning one usually already exists,
        // since the visible tile(s) that this proxy is standing in for typically already triggered
        // a real stitch nearby. Without this, a proxy tile's shader falls back to its plain
        // per-tile texture while ELEVATION_MOSAIC's UV remap still assumes a 3Wx3W mosaic layout
        // (elevationMosaicUV() in elevation.yaml), reading roughly 1/9th of the correct area --
        // visible as a flat, pale, wrongly-shaded patch over any tile still shown as a proxy
        // (reported by user as "little white patches"; worse the slower tiles load, since proxies
        // then persist longer -- see docs/tile-pipeline-perf-plan.md).
        if (source->m_buildElevationMosaic) {
            auto mosaicTex = isProxy() ? source->getExistingMosaic(m_tileId)
                                        : source->buildElevationMosaic(m_tileId, raster->texture);
            if (mosaicTex) {
                _tile.rasters().emplace_back(raster->tileID, mosaicTex);
                return;
            }
            // Fall through to the plain texture if no mosaic is available yet (e.g. no other
            // tile in the area has stitched one, or the CPU-side buffer wasn't retained).
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

    // A new elevation tile has arrived (visible tile or Phase 1 neighbor prefetch): replace
    // the mirror-extrapolated placeholder cell it corresponds to in any live neighboring
    // mosaic with its real data. This is what makes mirrored fallback data transient rather
    // than permanent - see docs/texture-shading-plan.md.
    if (m_buildElevationMosaic) {
        patchNeighborMosaics(id, texture);
    }

    return texture;
}

std::shared_ptr<Texture> RasterSource::getTexture(TileID _tile) {
    auto texIt = m_textures->find(_tile);
    return texIt != m_textures->end() ? texIt->second.lock() : nullptr;
}

std::shared_ptr<Texture> RasterSource::getExistingMosaic(TileID _tile) {
    auto mosaicIt = m_mosaics.find(TileID(_tile.x, _tile.y, _tile.z));
    if (mosaicIt == m_mosaics.end()) { return nullptr; }
    return mosaicIt->second.lock();
}

std::shared_ptr<Texture> RasterSource::buildElevationMosaic(TileID _tile, const std::shared_ptr<Texture>& _ownTexture) {
    TileID id(_tile.x, _tile.y, _tile.z);

    // Reuse a live registered mosaic: every Tile referencing this elevation tile shares one
    // stitched texture (previously each addRaster() call stitched its own 3Wx3W copy - a
    // large part of the Phase 4 congestion finding), and a live mosaic's cells are already
    // kept current by patchNeighborMosaics(), so it is never staler than a fresh stitch.
    if (auto mosaic = getExistingMosaic(id)) { return mosaic; }

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

            // 3-arg TileID (s = z): cacheTexture() normalizes keys the same way, and TileID's
            // ordering compares s first - looking up with _tile.s (styling zoom) would miss
            // every cached neighbor for overzoomed tiles (s > z).
            neighbors[size_t(dy + 1) * 3 + size_t(dx + 1)] = getTexture(TileID(nx, ny, _tile.z));
        }
    }

    auto mosaic = stitchElevationMosaic(_ownTexture, neighbors);

    if (mosaic) {
        // Keep the original per-tile texture alive for as long as the mosaic lives: m_textures
        //  only holds weak refs, and Tiles reference the mosaic INSTEAD of the original, so
        //  without this the original expires as soon as the mosaic replaces it - then any
        //  neighbor stitched later can't find this (still rendered!) tile's data in the cache
        //  and permanently falls back to mirror-extrapolation for its cell, since
        //  patchNeighborMosaics() only fires when a tile is newly cached. The resulting
        //  mirrored-vs-real content mismatch between adjacent mosaics showed up as a
        //  deterministic texture-shading seam on every tile boundary.
        //  ElevationMosaicInfo also carries the tile's ruggedness statistic for auto-contrast.
        auto info = std::make_shared<ElevationMosaicInfo>();
        info->ownTexture = _ownTexture;
        info->rugosity = computeRugosity(*_ownTexture, id);
        mosaic->userData = std::move(info);
        // Register for reuse and for in-place patching by later-arriving neighbors; prune
        // dead entries while we're here (mosaics die when their last referencing Tile does).
        for (auto it = m_mosaics.begin(); it != m_mosaics.end();) {
            it = it->second.expired() ? m_mosaics.erase(it) : std::next(it);
        }
        m_mosaics[id] = mosaic;
        LOGD("Elevation mosaic for tile %s: %dx%d (own tile %dx%d, %d live mosaics)",
             _tile.toString().c_str(), mosaic->width(), mosaic->height(),
             _ownTexture ? _ownTexture->width() : 0, _ownTexture ? _ownTexture->height() : 0,
             int(m_mosaics.size()));
    }

    return mosaic;
}

float RasterSource::aggregateRugosity() {
    double sum = 0;
    int n = 0;
    for (auto& kv : m_mosaics) {
        auto mosaic = kv.second.lock();
        if (!mosaic || !mosaic->userData) { continue; }
        float rug = static_cast<ElevationMosaicInfo*>(mosaic->userData.get())->rugosity;
        // skip water/void tiles (essentially zero relief) so coastal views aren't biased
        //  toward cranked-up contrast by the flat ocean half of the window
        if (rug < 0.005f) { continue; }
        sum += rug;
        n++;
    }
    return n > 0 ? float(sum / n) : -1.f;
}

void RasterSource::patchNeighborMosaics(const TileID& _tileId, const std::shared_ptr<Texture>& _texture) {
    if (!_texture || !_texture->bufferData()) { return; }

    int W = _texture->width();
    int bpp = _texture->getOptions().bytesPerPixel();
    if (W <= 0 || W != _texture->height()) { return; }

    int32_t max = 1 << _tileId.z;

    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) { continue; }

            int32_t ny = _tileId.y + dy;
            if (ny < 0 || ny >= max) { continue; }
            int32_t nx = ((_tileId.x + dx) % max + max) % max;

            auto mosaicIt = m_mosaics.find(TileID(nx, ny, _tileId.z));
            if (mosaicIt == m_mosaics.end()) { continue; }
            auto mosaic = mosaicIt->second.lock();
            if (!mosaic) { m_mosaics.erase(mosaicIt); continue; }

            int Wp = W - W % 2;
            if (!mosaic->bufferData() || mosaic->width() != 3 * Wp ||
                mosaic->getOptions().bytesPerPixel() != bpp) { continue; }

            // The new tile sits at grid offset (-dx, -dy) relative to the neighboring tile
            // whose mosaic we're patching. resize() with unchanged dimensions just flags the
            // texture for re-upload (and thus mipmap regeneration) on next bind - the Tiles
            // rendering this mosaic share the same Texture object, so they pick up the
            // patched data without any further bookkeeping.
            copyMosaicCell(mosaic->bufferData(), Wp, W, bpp, -dx, -dy, _texture->bufferData());
            mosaic->resize(mosaic->width(), mosaic->height());

            LOGD("Patched mosaic %s cell (%d,%d) with real data from tile %s",
                 mosaicIt->first.toString().c_str(), -dx, -dy, _tileId.toString().c_str());
        }
    }
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
