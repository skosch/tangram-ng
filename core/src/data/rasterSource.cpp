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

// Copy one cell (cellW x cellH texels, raw bytes, `bpp` bytes/pixel) from a real descendant
// tile buffer (row stride `W` texels) into an N x N grid composite whose row stride is
// `compositeW` texels. `col`/`row` are TILE-grid indices (col: west -> east, row: north ->
// south -- see stitchElevationOverzoomMosaic below for why `row` is placed at composite band
// (N-1-row)). Every INTERIOR cell is cropped to Wp = W - overlap (dropping its own trailing
// row/col, which the next cell over provides instead -- same node-registration convention as
// copyMosaicCell above), i.e. cellW == cellH == Wp. The composite's own NORTH-most (row == 0)
// and EAST-most (col == N-1) cells are NOT cropped in the corresponding dimension (cellH / cellW
// == W there): no further cell exists beyond them to supply that shared edge, so cropping it
// there would silently drop the true world-1.0 sample entirely -- this was a real, shipped bug
// (docs/terrain-illumination-plan.md "missing far-edge line"): the lattice node at UV (1,*) or
// (*,1) (getTerrainMeshElevation, elevation.yaml) sampled one texel short of the tile's actual
// edge, producing streaks along the north/east border of every composited tile.
void copyOverzoomGridCell(GLubyte* composite, int compositeW, int Wp, int W, int bpp, int N,
                           int col, int row, int cellW, int cellH, const GLubyte* srcData) {
    size_t compositeRowBytes = size_t(compositeW) * size_t(bpp);
    // Band offset formula is UNCHANGED from the pre-fix (always-Wp-cell) layout: row=0's band
    // still starts at (N-1)*Wp, it just now extends cellH (== W, not Wp, only for row == 0)
    // texels upward from there instead of Wp, reaching all the way to the true top edge at
    // compositeW = (N-1)*Wp + W -- see the doc comment above for why.
    size_t rowOff = size_t(N - 1 - row) * size_t(Wp);
    size_t colOff = size_t(col) * size_t(Wp) * size_t(bpp);
    for (int r = 0; r < cellH; r++) {
        std::memcpy(composite + (rowOff + size_t(r)) * compositeRowBytes + colOff,
                    srcData + size_t(r) * size_t(W) * size_t(bpp),
                    size_t(cellW) * size_t(bpp));
    }
}

// Fill a missing/mismatched grid cell (composite cell width/height cellW/cellH -- see
// copyOverzoomGridCell's doc comment for why the north-most/east-most cell can be larger than
// Wp) by nearest-neighbor upsampling the corresponding sub-rectangle of `ownData` (the primary
// tile's own ownW x ownW buffer, covering the composite's full [0,1] footprint) -- i.e.
// approximately what that cell would show today (before this fix), just cropped to its own
// sub-region so the rest of the composite can still hold real finer data. Nearest (not
// bilinear) sampling is a deliberate simplification: this is a same-resolution-as-today
// fallback for cells where no better data is cached yet, not a quality upgrade in itself, and
// staying byte-copy-only keeps it agnostic to pixel encoding (terrarium-RGB vs float) with no
// decode/re-encode round trip. World-position mapping is done directly off the true composite
// dimensions (compositeW, itself == (N-1)*Wp + ownW, not simply N*Wp) so the north/east-edge
// fallback texels line up with the same world position as the equivalent real-data copy would.
void fillUpsampledOverzoomCell(GLubyte* composite, int compositeW, int Wp, int bpp, int N,
                                int col, int row, int cellW, int cellH,
                                const GLubyte* ownData, int ownW) {
    size_t compositeRowBytes = size_t(compositeW) * size_t(bpp);
    size_t rowOff = size_t(N - 1 - row) * size_t(Wp);
    size_t colOff = size_t(col) * size_t(Wp);
    for (int r = 0; r < cellH; r++) {
        size_t globalRow = rowOff + size_t(r);
        double v = (double(globalRow) + 0.5) / double(compositeW);
        int ownRow = std::min(ownW - 1, std::max(0, int(v * double(ownW))));
        GLubyte* dstRow = composite + globalRow * compositeRowBytes + colOff * size_t(bpp);
        const GLubyte* srcRow = ownData + size_t(ownRow) * size_t(ownW) * size_t(bpp);
        for (int c = 0; c < cellW; c++) {
            size_t globalCol = colOff + size_t(c);
            double u = (double(globalCol) + 0.5) / double(compositeW);
            int ownCol = std::min(ownW - 1, std::max(0, int(u * double(ownW))));
            std::memcpy(dstRow + size_t(c) * size_t(bpp), srcRow + size_t(ownCol) * size_t(bpp), size_t(bpp));
        }
    }
}

// Bounds how many zoom levels of overzoom the opportunistic elevation composite
// (RasterSource::overzoomTargetZoom / buildOverzoomElevationMosaic) will try to compensate for
// in one go (N x N descendants, N = 2^levels), so cost stays bounded regardless of how
// overzoomed the primary tile's display zoom (s) is -- e.g. a deeply overzoomed z14 tile
// displayed at z20 should still only try a 4x4 = 16-tile composite from z16 (this source's own
// cap), not a 64x64 = 4096-tile one. 2 covers the common case exactly (elevation.yaml caps at
// z16, the `osm` vector source at z14: a 2-level, 4x4 = 16-tile gap).
const int kOverzoomCompositeMaxLevels = 2;

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

std::shared_ptr<Texture> stitchElevationOverzoomMosaic(const std::shared_ptr<Texture>& own,
                                                        const std::vector<std::shared_ptr<Texture>>& children,
                                                        int N) {
    if (!own || N <= 1) { return nullptr; }
    if (int(children.size()) != N * N) { return nullptr; }

    int ownW = own->width();
    if (ownW <= 0 || ownW != own->height()) { return nullptr; }

    const GLubyte* ownData = own->bufferData();
    if (!ownData) { return nullptr; }

    TextureOptions options = own->getOptions();
    int bpp = options.bytesPerPixel();

    // Same node-registration crop as stitchElevationMosaic() above for every INTERIOR cell:
    // cropped to its first Wp = W - overlap rows/cols, dropping the trailing row/col shared
    // with the next cell over, so the composite is an evenly-spaced sample grid with no
    // duplicated shared-edge lines between cells. UNLIKE stitchElevationMosaic() though, this
    // composite's own outer boundary (the north-most row of cells, row == 0; the east-most
    // column, col == N-1) has no further cell beyond it to supply the shared edge it would
    // otherwise drop -- so those cells keep their FULL ownW rows/cols instead, preserving the
    // true world 0/1-line sample (docs/terrain-illumination-plan.md "missing far-edge line":
    // dropping it made the composite one texel short of the tile's real north/east edge,
    // producing streaks there even given a fully-populated N x N grid). The composite is
    // therefore (N-1)*Wp + ownW = N*Wp + overlap wide, not simply N*Wp.
    int overlap = ownW % 2;
    int Wp = ownW - overlap;
    if (Wp <= 0) { return nullptr; }

    int compositeW = (N - 1) * Wp + ownW;
    size_t rowBytes = size_t(compositeW) * size_t(bpp);
    std::vector<GLubyte> buf(rowBytes * size_t(compositeW));

    for (int row = 0; row < N; row++) {
        int cellH = (row == 0) ? ownW : Wp;
        for (int col = 0; col < N; col++) {
            int cellW = (col == N - 1) ? ownW : Wp;
            const auto& child = children[size_t(row) * size_t(N) + size_t(col)];
            if (child && child->bufferData() &&
                child->width() == ownW && child->height() == ownW &&
                child->getOptions().bytesPerPixel() == bpp) {
                copyOverzoomGridCell(buf.data(), compositeW, Wp, ownW, bpp, N, col, row,
                                      cellW, cellH, child->bufferData());
            } else {
                fillUpsampledOverzoomCell(buf.data(), compositeW, Wp, bpp, N, col, row,
                                           cellW, cellH, ownData, ownW);
            }
        }
    }

    // Unlike stitchElevationMosaic()'s 3Wp x 3Wp neighbor-ring mosaic, this composite is a
    // direct higher-resolution replacement for `own` at the SAME [0,1] UV footprint (it is not
    // read via elevationMosaicUV()'s center-third remap), so it keeps `own`'s plain texture
    // options (e.g. "nearest" filtering per elevation.yaml) rather than forcing mipmapped
    // linear filtering -- there is no neighbor-mosaic-style multi-scale shader sampling of this
    // texture to support.
    //
    // disposeBuffer left at its default (true): unlike stitchElevationMosaic()'s neighbor-ring
    // mosaic, nothing ever patches this composite's cells in place after creation (no analog of
    // patchNeighborMosaics() for this registry yet -- see the "future upgrade path" note on
    // RasterSource::buildOverzoomElevationMosaic), so there's no need to keep its CPU buffer
    // resident past the GPU upload.
    auto tex = std::make_shared<Texture>(options);
    tex->setPixelData(compositeW, compositeW, bpp, buf.data(), buf.size());

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

        // Opportunistic overzoom-compensation composite (docs/terrain-illumination-plan.md,
        // "raster-zoom mismatch" section): m_tileId carries both this raster subtask's own data
        // zoom (z, what it was actually fetched at) and the primary tile's display zoom (s). When
        // s > z and this elevation source can serve finer data than z, RasterSource::addRasterTask
        // (above) has attached this raster at the coarse z -- meaning draped polygon geometry
        // would otherwise sample genuinely coarser elevation than the co-located terrain-ground
        // mesh (which always fetches its own tiles at native resolution), a real vertical
        // mismatch between the two surfaces that grazing-angle parallax amplifies into visible
        // pale "white patch" gaps. Opportunistically upgrade the raster to a composite of
        // whatever finer descendant tiles are already cached (see
        // RasterSource::buildOverzoomElevationMosaic -- cache-only lookup, never a new fetch).
        // No-op (returns nullptr immediately) when m_tileId isn't actually overzoomed, so this
        // costs nothing for the ordinary (non-overzoomed) case, e.g. terrain-ground's own tiles.
        //
        // When m_buildElevationMosaic (texture shading) is set, do NOT bind this composite
        // directly: the ELEVATION_MOSAIC shader path (elevation.yaml) is a scene-global
        // compile-time define -- once on, EVERY texture bound to the elevation raster slot is
        // read through elevationMosaicUV(), which unconditionally remaps tile-local UV into the
        // center NINTH of whatever texture is bound, assuming the Frozen-Interface-Contract
        // 3Wp x 3Wp neighbor-ring layout. This composite is a direct same-footprint replacement
        // (see stitchElevationOverzoomMosaic's doc comment) -- NOT that layout -- so if it were
        // bound while ELEVATION_MOSAIC is compiled in, the shader would sample an essentially
        // arbitrary sub-region of it instead of the correct footprint, corrupting the
        // vertex-shader elevation read for every draped vertex (observed: land-cover polygons
        // vanish entirely, presumably displaced far outside the frustum -- confirmed by A/B
        // testing this exact scenario). Instead, buildElevationMosaic() (below) internally
        // reaches for this same composite as ITS OWN "own"/neighbor cell input -- see its doc
        // comment -- so the two mechanisms compose into one texture rather than being mutually
        // exclusive; only bind the composite directly when texture shading is off.
        if (!source->m_buildElevationMosaic) {
            auto overzoomTex = isProxy() ? source->getExistingOverzoomMosaic(m_tileId)
                                          : source->buildOverzoomElevationMosaic(m_tileId, raster->texture);
            if (overzoomTex) {
                _tile.rasters().emplace_back(raster->tileID, overzoomTex);
                return;
            }
        }

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
    // kept current by patchNeighborMosaics(), so it is never staler than a fresh stitch --
    // UNLESS it was built from the tile's plain (pre-overzoom-compensation) texture and a
    // complete opportunistic composite (see below) has since become available: composing the
    // two mechanisms means this same tile can improve over time even after its first mosaic
    // was already registered, so don't treat "a mosaic exists" alone as "nothing left to do"
    // (docs/terrain-illumination-plan.md "attachment lifecycle" finding -- see also
    // TileManager::upgradeAttachedRasters, which is what re-calls this on already-mosaic'd
    // tiles as new elevation tiles keep arriving in the background).
    auto existing = getExistingMosaic(id);
    if (existing) {
        auto* info = existing->userData ? static_cast<ElevationMosaicInfo*>(existing->userData.get()) : nullptr;
        if (info && info->ownIsOverzoomComposite) { return existing; }  // already best possible
    }

    // Prefer the higher-resolution opportunistic overzoom composite (see
    // stitchElevationOverzoomMosaic / buildOverzoomElevationMosaic's doc comments) for the
    // tile's own center cell when this tile is genuinely overzoomed and a COMPLETE composite is
    // available -- this is what lets the composite and the texture-shading neighbor-ring mosaic
    // share one texture instead of being mutually exclusive. `_ownTexture` may be null here
    // (TileManager::upgradeAttachedRasters retrying a tile that's already past the plain-texture
    // stage and may no longer have it handy) -- in that case only a cheap cache-only check is
    // possible (no new fetch/build), which is fine: a composite only ever gets registered once
    // (in buildOverzoomElevationMosaic, gated on ALL N*N descendants being cached) and is then
    // permanent, so a full build attempt was already made whenever `_ownTexture` WAS available.
    std::shared_ptr<Texture> ownForMosaic =
        _ownTexture ? buildOverzoomElevationMosaic(_tile, _ownTexture) : getExistingOverzoomMosaic(_tile);
    bool usingComposite = bool(ownForMosaic);

    if (!ownForMosaic) {
        if (existing) { return existing; }        // nothing better than what's already registered
        if (!_ownTexture) { return nullptr; }      // no fallback available to build anything at all
        ownForMosaic = _ownTexture;
    }

    // Look up whichever of the 8 same-zoom neighbors are already cached (getTexture() is only
    // safe to call here because addRaster()/complete() run on the main thread -- see the
    // thread-safety note on RasterTileTask::addRaster()). Missing neighbors are left null;
    // stitchElevationMosaic() synthesizes them via mirror-extrapolation. When this tile's own
    // cell is the overzoom composite, also try each neighbor's own composite (cache-only, same
    // gating as above) FIRST -- same reasoning as own's cell, and if a neighbor's composite
    // isn't the same size (e.g. it isn't overzoomed, or doesn't have one yet), the plain texture
    // is a safe fallback: stitchElevationMosaic()'s own dimension check rejects any mismatched
    // cell and mirror-extrapolates it instead, exactly as it already does for a missing neighbor.
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
            std::shared_ptr<Texture> neighborTex = getTexture(TileID(nx, ny, _tile.z));
            if (usingComposite) {
                // Approximate the neighbor's own display zoom with this tile's `_tile.s`: `s` is
                // continuously recomputed per visible tile from on-screen area (see
                // overzoomTargetZoom()'s doc comment) so same-zoom siblings in the same viewport
                // usually share the same or an equivalent zt; a miss here just falls back to the
                // plain neighbor texture above, so an imperfect guess costs nothing but a
                // slightly-more-often mirror-extrapolated cell, never incorrectness.
                if (auto neighborComposite = getExistingOverzoomMosaic(TileID(nx, ny, _tile.z, _tile.s))) {
                    neighborTex = neighborComposite;
                }
            }
            neighbors[size_t(dy + 1) * 3 + size_t(dx + 1)] = neighborTex;
        }
    }

    auto mosaic = stitchElevationMosaic(ownForMosaic, neighbors);

    if (mosaic) {
        // Keep the center-cell texture alive for as long as the mosaic lives: m_textures only
        //  holds weak refs, and Tiles reference the mosaic INSTEAD of the original, so without
        //  this the original expires as soon as the mosaic replaces it - then any neighbor
        //  stitched later can't find this (still rendered!) tile's data in the cache and
        //  permanently falls back to mirror-extrapolation for its cell, since
        //  patchNeighborMosaics() only fires when a tile is newly cached. The resulting
        //  mirrored-vs-real content mismatch between adjacent mosaics showed up as a
        //  deterministic texture-shading seam on every tile boundary. (When usingComposite,
        //  m_overzoomMosaics already keeps ownForMosaic alive independently -- see its doc
        //  comment -- so this is redundant but harmless in that case.)
        //  ElevationMosaicInfo also carries the tile's ruggedness statistic for auto-contrast.
        auto info = std::make_shared<ElevationMosaicInfo>();
        info->ownTexture = ownForMosaic;
        info->rugosity = computeRugosity(*ownForMosaic, id);
        info->ownIsOverzoomComposite = usingComposite;
        mosaic->userData = std::move(info);
        // Register for reuse and for in-place patching by later-arriving neighbors (prune dead
        // entries while we're here -- mosaics die when their last referencing Tile does); this
        // OVERWRITES any existing (necessarily plain-based, per the early-return above) entry
        // for `id`, which is exactly the upgrade this function exists to perform.
        for (auto it = m_mosaics.begin(); it != m_mosaics.end();) {
            it = it->second.expired() ? m_mosaics.erase(it) : std::next(it);
        }
        m_mosaics[id] = mosaic;
        LOGD("Elevation mosaic for tile %s: %dx%d (own %s, own width %d, %d live mosaics)",
             _tile.toString().c_str(), mosaic->width(), mosaic->height(),
             usingComposite ? "overzoom composite" : "plain tile",
             ownForMosaic->width(), int(m_mosaics.size()));
    }

    return mosaic;
}

int RasterSource::overzoomTargetZoom(TileID _primary) const {
    // Not actually overzoomed relative to this source's own reach: nothing to compensate for.
    // Covers both the ordinary non-overzoomed case (s == z, e.g. terrain-ground's own tiles at
    // typical zooms) and the case where this source's own maxZoom is already exhausted too
    // (_primary.z == maxZoom, e.g. deep overzoom past z16 even for terrain-ground itself) --
    // there is no finer same-source data to pull in either way. Callers check `zt <= _primary.z`.
    int zt = std::min<int>(m_zoomOptions.maxZoom, _primary.s);
    zt = std::min<int>(zt, _primary.z + kOverzoomCompositeMaxLevels);
    return zt;
}

std::shared_ptr<Texture> RasterSource::getExistingOverzoomMosaic(TileID _primary) {
    int zt = overzoomTargetZoom(_primary);
    if (zt <= _primary.z) { return nullptr; }
    // Stable (x, y, z, zt) key -- see overzoomTargetZoom()'s doc comment on why this must not be
    // the raw `_primary` TileID. m_overzoomMosaics holds strong refs (see its doc comment in
    // rasterSource.h), so a hit here is unconditional -- no weak_ptr::lock() to fail.
    auto it = m_overzoomMosaics.find(TileID(_primary.x, _primary.y, _primary.z, zt));
    if (it == m_overzoomMosaics.end()) { return nullptr; }
    return it->second;
}

void RasterSource::pruneOverzoomMosaicsIfNeeded() {
    // See the doc comment on m_overzoomMosaics (rasterSource.h) for why this is a blunt full
    // clear rather than real LRU eviction.
    const size_t kMaxOverzoomMosaicEntries = 128;
    if (m_overzoomMosaics.size() >= kMaxOverzoomMosaicEntries) {
        LOGD("Overzoom mosaic registry hit %d entries, clearing (self-heals on next rebuild of any still-visible tile)",
             int(m_overzoomMosaics.size()));
        m_overzoomMosaics.clear();
    }
}

std::shared_ptr<Texture> RasterSource::buildOverzoomElevationMosaic(TileID _primary, const std::shared_ptr<Texture>& _ownTexture) {
    if (!_ownTexture || !_ownTexture->bufferData() || !m_keepTextureData) { return nullptr; }

    int zt = overzoomTargetZoom(_primary);
    if (zt <= _primary.z) { return nullptr; }

    // Stable cache key -- see overzoomTargetZoom()'s doc comment: must be (x, y, z, zt), not the
    // raw `_primary` (whose `s` churns continuously during ordinary panning without `zt`
    // necessarily changing at all, which previously caused the same physical tile to rebuild
    // repeatedly from whatever was cached at each instant -- visible as tile-sized flicker).
    TileID key(_primary.x, _primary.y, _primary.z, zt);

    if (auto existing = getExistingOverzoomMosaic(_primary)) { return existing; }

    int levels = zt - _primary.z;
    int N = 1 << levels;

    // Enumerate the N x N descendant TileIDs covering _primary's footprint at zt and look each
    // up in the existing (weak-ref) texture cache -- deliberately cache-only for this first pass
    // (see the doc comment on this method / stitchElevationOverzoomMosaic): no new fetch is
    // triggered here even when a cell is missing. `children` is laid out row-major in the same
    // (col: west->east, row: north->south) order stitchElevationOverzoomMosaic expects.
    std::vector<std::shared_ptr<Texture>> children(size_t(N) * size_t(N));
    int32_t baseX = _primary.x << levels;
    int32_t baseY = _primary.y << levels;
    int present = 0;
    for (int row = 0; row < N; row++) {
        for (int col = 0; col < N; col++) {
            auto tex = getTexture(TileID(baseX + col, baseY + row, zt));
            if (tex && tex->bufferData()) { present++; }
            children[size_t(row) * size_t(N) + size_t(col)] = tex;
        }
    }

    // Require ALL N*N cells before ever registering a composite (changed from an earlier
    // "at least one real cell" bar -- docs/terrain-illumination-plan.md "require-all-cells for
    // determinism"). Two reasons this replaced the partial-composite design:
    //
    // 1. Determinism: m_overzoomMosaics is a STRONG, permanent-once-registered cache keyed on
    //    the stable (x, y, z, zt) tuple (see its doc comment) specifically so a physical tile's
    //    composite doesn't churn every time the primary tile itself gets rebuilt during ordinary
    //    panning. But that strength cuts both ways: once ANY composite -- even a mostly-fallback
    //    partial one built the instant a tile entered view, before its siblings finished loading
    //    -- gets registered, it is permanent for the rest of the session; nothing ever revisits
    //    or improves it, because getExistingOverzoomMosaic()/the early-return above always hands
    //    back that first snapshot. A low bar for registering therefore means WORSE determinism
    //    in practice, not better: which of many possible partial states gets frozen in depends
    //    entirely on load-order timing.
    // 2. It composes with TileManager::upgradeAttachedRasters and buildElevationMosaic()'s own
    //    upgrade-retry path (see its doc comment): both re-call this function on tiles that
    //    haven't resolved a FINAL composite yet, cheaply (see below), for as long as the tile
    //    stays visible. Requiring completeness before ever registering means those retries keep
    //    trying until the real thing is ready instead of being satisfied by (and thus never
    //    replacing) a first partial guess -- the "attachment baked at build time, never
    //    upgraded" lifecycle bug this whole mechanism exists to fix, now fixed at the composite
    //    level too, not just at the raster-attachment level.
    //
    // This does mean fewer tiles show ANY composite at a given instant (no more fallback-filled
    // partials) -- by design: nothing is displayed that couldn't be trivially recomputed
    // identically on the next call once the same terminal condition (some, but not all,
    // descendants cached) shows up again, and the plain/pre-existing-mosaic fallback the caller
    // uses in the meantime is exactly what rendered before this whole feature existed, not a
    // regression. Cost: this early-out is a cheap O(N*N) map-lookup pass (already just performed
    // above) -- safe to call every frame per visible overzoomed tile until it resolves.
    if (present < N * N) { return nullptr; }

    auto mosaic = stitchElevationOverzoomMosaic(_ownTexture, children, N);

    if (mosaic) {
        // Register for reuse -- a STRONG reference (m_overzoomMosaics.h doc comment): once
        // resolved from ALL N*N real descendants (see the require-all-cells note above), this
        // tile's composite is final and stable for the rest of the session regardless of how
        // many times the primary tile itself gets rebuilt (see the doc comment on
        // m_overzoomMosaics for the panning-flicker this fixes).
        pruneOverzoomMosaicsIfNeeded();
        m_overzoomMosaics[key] = mosaic;
        LOGD("Overzoom elevation composite for tile %s (cache key %s): %dx%d, all %d cached z%d descendants",
             _primary.toString().c_str(), key.toString().c_str(), mosaic->width(), mosaic->height(), N * N, zt);
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
