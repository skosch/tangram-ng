#include "data/rasterSource.h"
#include "debug/profiler.h"
#include "data/propertyItem.h"
#include "data/tileData.h"
#include "tile/tile.h"
#include "tile/tileBuilder.h"
#include "tile/tileTask.h"
#include "util/mapProjection.h"
#include "log.h"

#include <algorithm>
#include <chrono>
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

// Decode/encode one elevation sample in either supported pixel encoding (float, or the 3-byte
// mapzen terrarium packed in RGB) -- shared by computeRugosity() above and the triangle-PL
// upsampling helpers below (buildLatticeNodes/fillLatticeUpsampledCell). Terrarium formula
// matches getElevationAt in elevation.yaml exactly ((h.r*256+h.g+h.b/256)*255-32768 with h.r/g/b
// the shader's normalized-byte reads); multiplying that out by 255 cancels the /255
// normalization, leaving the raw-byte form used here (same as computeRugosity's decode above).
// bpp <= 1 (single raw byte, no elevation semantics) is also handled -- the value IS the byte --
// so this stays usable by unit tests that exercise the fallback-fill path with the same small
// synthetic single-channel textures the rest of this test suite uses.
double decodeElevationPixel(const GLubyte* p, int bpp, bool isFloat) {
    if (isFloat) {
        float f;
        std::memcpy(&f, p, sizeof(float));
        return double(f);
    }
    if (bpp <= 1) { return double(p[0]); }
    return (double(p[0]) * 256.0 + double(p[1]) + double(p[2]) / 256.0) - 32768.0;
}

void encodeElevationPixel(GLubyte* p, double value, int bpp, bool isFloat) {
    if (isFloat) {
        float f = float(value);
        std::memcpy(p, &f, sizeof(float));
        return;
    }
    if (bpp <= 1) {
        p[0] = GLubyte(std::min(255.0, std::max(0.0, std::round(value))));
        return;
    }
    double t = value + 32768.0;
    int b0 = std::min(255, std::max(0, int(std::floor(t / 256.0))));
    double rem = t - double(b0) * 256.0;
    int b1 = std::min(255, std::max(0, int(std::floor(rem))));
    double rem2 = (rem - double(b1)) * 256.0;
    int b2 = std::min(255, std::max(0, int(std::lround(rem2))));
    p[0] = GLubyte(b0);
    p[1] = GLubyte(b1);
    p[2] = GLubyte(b2);
    if (bpp >= 4) { p[3] = 255; }
}

// Nearest-sampled 65x65 lattice-node grid for one tile's OWN raw buffer, replicating exactly
// what the terrain mesh vertex shader reads (RasterStyle::build: resolution=64 fixed; elevation
// source `filtering: nearest`): node (i, j) at tile-local (u, v) = (i/64, j/64) reads the texel
// GL_NEAREST would pick, floor(u*W)/floor(v*W) clamped to [0, W-1]. `data`'s row axis matches
// every other raw elevation tile buffer in this file (row 0 = south, increasing = north -- see
// the mosaic layout notes above cellRowOffset()); node row index j follows the SAME convention
// purely so it composes directly with the row math in fillLatticeUpsampledCell below.
std::vector<double> buildLatticeNodes(const GLubyte* data, int W, int bpp, bool isFloat) {
    std::vector<double> nodes(65 * 65);
    for (int j = 0; j <= 64; j++) {
        int row = std::min(W - 1, std::max(0, int((double(j) / 64.0) * double(W))));
        for (int i = 0; i <= 64; i++) {
            int col = std::min(W - 1, std::max(0, int((double(i) / 64.0) * double(W))));
            const GLubyte* p = data + (size_t(row) * size_t(W) + size_t(col)) * size_t(bpp);
            nodes[size_t(j) * 65 + size_t(i)] = decodeElevationPixel(p, bpp, isFloat);
        }
    }
    return nodes;
}

// Evaluate the fixed 64-cell anti-diagonal piecewise-linear surface a buildLatticeNodes() grid
// represents at continuous tile-local (u, v) in [0,1]^2 -- the SAME formula as
// getTerrainMeshElevation() (elevation.yaml) and the terrain mesh's own triangulation
// (rasterStyle.cpp), so evaluating it at a finer nested lattice's node positions reproduces this
// tile's terrain-mesh surface exactly: a fine triangle from a same-split finer lattice always
// lies entirely within one coarse triangle (the anti-diagonal split is scale-invariant), so
// linear interpolation of the coarse corners at the fine node's exact position gives exactly what
// the coarse mesh itself would render there -- see fillLatticeUpsampledCell's doc comment.
double evalLatticePL(const std::vector<double>& nodes, double u, double v) {
    double stx = std::min(1.0, std::max(0.0, u)) * 64.0;
    double sty = std::min(1.0, std::max(0.0, v)) * 64.0;
    int cx = std::min(63, int(std::floor(stx)));
    int cy = std::min(63, int(std::floor(sty)));
    double fx = stx - double(cx);
    double fy = sty - double(cy);
    double h00 = nodes[size_t(cy) * 65 + size_t(cx)];
    double h10 = nodes[size_t(cy) * 65 + size_t(cx + 1)];
    double h01 = nodes[size_t(cy + 1) * 65 + size_t(cx)];
    double h11 = nodes[size_t(cy + 1) * 65 + size_t(cx + 1)];
    if (fx + fy <= 1.0) {
        return h00 + fx * (h10 - h00) + fy * (h01 - h00);
    }
    return h11 + (1.0 - fx) * (h01 - h11) + (1.0 - fy) * (h10 - h11);
}

// Fill one zt-grid cell of the overzoom composite (col, row grid indices; cellW x cellH texels at
// the same north/east-edge-uncropped convention as copyOverzoomGridCell below) from a REAL
// coarser ancestor tile's own 65-node lattice (`ancestorNodes`, see buildLatticeNodes), by
// evaluating that lattice's PL surface (evalLatticePL) at each destination texel's exact
// position -- see evalLatticePL's doc comment for why this reproduces the ancestor's own
// terrain-mesh surface exactly instead of the old flat nearest-neighbor approximation
// (docs/terrain-illumination-plan.md "triangle-PL upsampling").
//
// `ancestorNodes`'s tile spans a `blockSize` x `blockSize` block of zt-grid cells, anchored at
// (blockColOriginWest, blockRowOriginNorth) in the SAME zt-grid units as `col`/`row` (col: west
// -> east, row: north -> south -- see OverzoomCell's doc comment) -- used to map this cell's
// destination texels back into the ancestor's own [0,1] tile-local (u, v).
void fillLatticeUpsampledCell(GLubyte* composite, int compositeW, int Wp, int bpp, int N,
                               int col, int row, int cellW, int cellH,
                               const std::vector<double>& ancestorNodes,
                               int32_t blockColOriginWest, int32_t blockRowOriginNorth,
                               int32_t blockSize, bool isFloat) {
    size_t compositeRowBytes = size_t(compositeW) * size_t(bpp);
    size_t rowOff = size_t(N - 1 - row) * size_t(Wp);
    size_t colOff = size_t(col) * size_t(Wp);
    // South-based block origin (row 0 = south, increasing = north), matching every raw tile
    // buffer's own row convention (see buildLatticeNodes).
    int32_t blockRowOriginSouth = int32_t(N) - blockRowOriginNorth - blockSize;
    for (int r = 0; r < cellH; r++) {
        size_t globalRow = rowOff + size_t(r);
        double southFracGlobal = double(globalRow) / double(compositeW - 1);
        double vLocal = (southFracGlobal * double(N) - double(blockRowOriginSouth)) / double(blockSize);
        GLubyte* dstRow = composite + globalRow * compositeRowBytes + colOff * size_t(bpp);
        for (int c = 0; c < cellW; c++) {
            size_t globalCol = colOff + size_t(c);
            double westFracGlobal = double(globalCol) / double(compositeW - 1);
            double uLocal = (westFracGlobal * double(N) - double(blockColOriginWest)) / double(blockSize);
            double val = evalLatticePL(ancestorNodes, uLocal, vLocal);
            encodeElevationPixel(dstRow + size_t(c) * size_t(bpp), val, bpp, isFloat);
        }
    }
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

// Bounds how many zoom levels of overzoom the opportunistic elevation composite
// (RasterSource::overzoomTargetZoom / buildOverzoomElevationMosaic) will try to compensate for
// in one go (N x N descendants, N = 2^levels), so cost stays bounded regardless of how
// overzoomed the primary tile's display zoom (s) is -- e.g. a deeply overzoomed z14 tile
// displayed at z20 should still only try a 4x4 = 16-tile composite from z16 (this source's own
// cap), not a 64x64 = 4096-tile one. 2 covers the common case exactly (elevation.yaml caps at
// z16, the `osm` vector source at z14: a 2-level, 4x4 = 16-tile gap).
const int kOverzoomCompositeMaxLevels = 2;

// R2 (tile-pipeline-perf-plan.md): how long a patched mosaic may sit dirty (unflushed) before
// RasterSource::flushDirtyMosaics() forces a re-upload regardless of ring completeness -- bounds
// staleness for mosaics whose ring never completes (e.g. a permanently missing edge tile).
const double kMosaicFlushDebounceSeconds = 0.2;

double nowSeconds() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
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

    // Reuse one persistent scratch buffer across calls instead of heap-allocating a fresh
    // ~9x-tile-sized (e.g. ~7-9MB for a 512px elevation tile) vector on every stitch: this
    // function was profiled as the single biggest main-thread CPU cost (~30% of capture time),
    // and the allocation/deallocation itself -- not the per-row memcpy work below -- was the
    // waste (see docs/tile-pipeline-perf-plan.md). `static thread_local` rather than a plain
    // `static` is a defensive habit, not a requirement: stitchElevationMosaic() is only ever
    // called from RasterSource::buildElevationMosaic() on the main thread (same thread-safety
    // contract as getTexture() above -- see the note on RasterTileTask::addRaster()), so a
    // plain function-local static would already be safe here; thread_local just costs nothing
    // extra and avoids relying on that invariant never changing elsewhere. resize() never
    // shrinks capacity, so repeated calls at the same (or smaller) mosaic size reuse the
    // existing allocation with no new malloc/free at all. Every byte is unconditionally
    // overwritten below (own cell + all 8 neighbor cells, real or mirror-extrapolated) before
    // the buffer is read, so there is no dependency on prior contents or zero-initialization.
    static thread_local std::vector<GLubyte> s_scratchMosaic;
    s_scratchMosaic.resize(mosaicRowBytes * size_t(mosaicW));
    GLubyte* mosaic = s_scratchMosaic.data();

    // Tile's own data occupies the center third [Wp, 2Wp) x [Wp, 2Wp) -- see
    // docs/texture-shading-plan.md "Frozen Interface Contract".
    copyMosaicCell(mosaic, Wp, W, bpp, 0, 0, ownData);

    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) { continue; }

            const auto& neighborTex = neighbors[size_t(dy + 1) * 3 + size_t(dx + 1)];

            if (neighborTex && neighborTex->bufferData() &&
                neighborTex->width() == W && neighborTex->height() == W &&
                neighborTex->getOptions().bytesPerPixel() == bpp) {
                copyMosaicCell(mosaic, Wp, W, bpp, dx, dy, neighborTex->bufferData());
            } else {
                fillMirroredNeighborCell(mosaic, Wp, W, bpp, dx, dy, ownData);
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
    tex->setPixelData(mosaicW, mosaicW, bpp, mosaic, s_scratchMosaic.size());

    return tex;
}

std::shared_ptr<Texture> stitchElevationOverzoomMosaic(const std::shared_ptr<Texture>& own,
                                                        const std::vector<OverzoomCell>& cells,
                                                        int N) {
    if (!own || N <= 1) { return nullptr; }
    if (int(cells.size()) != N * N) { return nullptr; }

    int ownW = own->width();
    if (ownW <= 0 || ownW != own->height()) { return nullptr; }

    const GLubyte* ownData = own->bufferData();
    if (!ownData) { return nullptr; }

    TextureOptions options = own->getOptions();
    int bpp = options.bytesPerPixel();
    bool isFloat = options.pixelFormat == PixelFormat::FLOAT;

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

    // One 65x65 lattice-node grid per distinct fallback source actually used -- an ancestor tile
    // with blockSize > 1 typically supplies several zt-grid cells at once, so building its
    // lattice once and reusing it avoids redundant work.
    std::map<const Texture*, std::vector<double>> latticeCache;
    auto latticeFor = [&](const Texture* tex, const GLubyte* data, int w) -> const std::vector<double>& {
        auto it = latticeCache.find(tex);
        if (it == latticeCache.end()) {
            it = latticeCache.emplace(tex, buildLatticeNodes(data, w, bpp, isFloat)).first;
        }
        return it->second;
    };

    for (int row = 0; row < N; row++) {
        int cellH = (row == 0) ? ownW : Wp;
        for (int col = 0; col < N; col++) {
            int cellW = (col == N - 1) ? ownW : Wp;
            const OverzoomCell& cell = cells[size_t(row) * size_t(N) + size_t(col)];

            bool validReal = cell.texture && cell.texture->bufferData() &&
                cell.texture->width() == cell.texture->height() && cell.texture->width() > 0 &&
                cell.texture->getOptions().bytesPerPixel() == bpp;
            // Direct-copy path additionally requires the real cell's width to match `own`'s:
            // copyOverzoomGridCell reads it with `ownW` as the row stride (real elevation tiles
            // are always uniformly sized within one source, but stay defensive the way the
            // pre-existing dimension checks elsewhere in this file are -- a size mismatch on a
            // cell CLAIMING blockSize == 1 (exact same-zoom data) is a data-integrity red flag,
            // not just "slightly coarser than expected", so it falls all the way back to `own`
            // rather than attempting a PL fill under an assumption (this texture's footprint IS
            // exactly this one grid cell) that's already been shown to be wrong.
            bool validExact = validReal && cell.blockSize == 1 && cell.texture->width() == ownW;

            if (validExact) {
                // Exact: real data cached at the composite's own target zoom for this cell.
                copyOverzoomGridCell(buf.data(), compositeW, Wp, ownW, bpp, N, col, row,
                                      cellW, cellH, cell.texture->bufferData());
            } else if (validReal && cell.blockSize > 1) {
                // A real, but coarser, ancestor covers this cell (and typically several more) --
                // triangle-PL-upsample its own terrain-mesh surface instead of nearest-sampling
                // its raw pixels (see fillLatticeUpsampledCell's doc comment).
                const auto& nodes = latticeFor(cell.texture.get(), cell.texture->bufferData(), cell.texture->width());
                fillLatticeUpsampledCell(buf.data(), compositeW, Wp, bpp, N, col, row, cellW, cellH,
                                          nodes, cell.blockColOriginWest, cell.blockRowOriginNorth,
                                          cell.blockSize, isFloat);
            } else {
                // No usable source named for this cell (missing/malformed input) -- fall all the
                // way back to `own` itself over the whole footprint, same safety net the old
                // nearest-neighbor fallback provided, just PL-upsampled now.
                const auto& nodes = latticeFor(own.get(), ownData, ownW);
                fillLatticeUpsampledCell(buf.data(), compositeW, Wp, bpp, N, col, row, cellW, cellH,
                                          nodes, 0, 0, N, isFloat);
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
    // disposeBuffer left at its default (true): a composite is always rebuilt wholesale (a fresh
    // Texture object) when its coverage changes -- see RasterSource::buildOverzoomElevationMosaic
    // -- rather than patched in place, so there's no need to keep its CPU buffer resident past
    // the GPU upload.
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
        // R4 (tile-pipeline-perf-plan.md): a real stitch (buildOverzoomElevationMosaic/
        // buildElevationMosaic) is the expensive main-thread operation measured bursting
        // several ms at a time in clusters as tiles complete; bound how many of those this
        // task may perform per update cycle (tryConsumeStitchBudget()) and otherwise reuse
        // whatever composite/mosaic is already registered (cheap - a map lookup, same as the
        // isProxy() path already does) rather than forcing a fresh one. Falls all the way
        // through to the plain per-tile texture below if nothing is registered yet either;
        // TileManager::upgradeAttachedRasters re-checks every update and performs the real
        // stitch once the budget resets on a later, less busy frame.
        if (!source->m_buildElevationMosaic) {
            std::shared_ptr<Texture> overzoomTex;
            if (isProxy() || !source->tryConsumeStitchBudget()) {
                overzoomTex = source->getExistingOverzoomMosaic(m_tileId);
            } else {
                overzoomTex = source->buildOverzoomElevationMosaic(m_tileId, raster->texture);
            }
            if (overzoomTex) {
                _tile.rasters().emplace_back(raster->tileID, overzoomTex);
                return;
            }
        }

        if (source->m_buildElevationMosaic) {
            std::shared_ptr<Texture> mosaicTex;
            if (isProxy() || !source->tryConsumeStitchBudget()) {
                mosaicTex = source->getExistingMosaic(m_tileId);
                if (!mosaicTex) {
                    // Correctness fallback, not just a perf one: whenever m_buildElevationMosaic
                    // is set, the shader's ELEVATION_MOSAIC path (elevation.yaml) is a scene-wide
                    // compile flag - it unconditionally treats EVERY bound elevation texture as a
                    // 3Wx3W neighbor-ring mosaic and remaps tile-local UV into its center ninth
                    // (elevationMosaicUV()), with no per-texture way to know otherwise. Falling
                    // through below to attach the plain per-tile texture in that case (as this
                    // used to do) hands the shader a texture 1/3 the width it assumes: every
                    // vertex's UV collapses into a narrow, roughly-constant band near the
                    // texture's center regardless of true position, producing a flat, floating
                    // drape sitting on top of the (correctly-elevated) terrain-ground mesh
                    // underneath - confirmed live as the "landcover floating over the valley"
                    // report. A real stitch here is a bounded, one-time cost (stitchElevation
                    // Mosaic() mirror-extrapolates any missing neighbor, so it never fails given
                    // a valid own texture); skipping it for a proxy/budget-exhausted tile with
                    // nothing cached yet was trading a rare perf hit for a correctness bug -
                    // worth paying for here specifically because "wrong layout" (not just
                    // "coarser than ideal") is the failure mode.
                    mosaicTex = source->buildElevationMosaic(m_tileId, raster->texture);
                }
            } else {
                mosaicTex = source->buildElevationMosaic(m_tileId, raster->texture);
            }
            if (mosaicTex) {
                _tile.rasters().emplace_back(raster->tileID, mosaicTex);
                return;
            }
            // Only reachable if buildElevationMosaic() itself returned null (e.g. the plain
            // texture has no CPU-side buffer data to stitch from) - see its own doc comment.
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

std::shared_ptr<Texture> RasterSource::buildElevationMosaic(TileID _tile, const std::shared_ptr<Texture>& _ownTexture,
                                                            const uint8_t* _cellZoomCaps) {
    PROFILE_SCOPE(Profiler::intern("buildElevationMosaic:" + name()));
    TileID id(_tile.x, _tile.y, _tile.z);

    // Reuse a live registered mosaic: every Tile referencing this elevation tile shares one
    // stitched texture (previously each addRaster() call stitched its own 3Wx3W copy - a
    // large part of the Phase 4 congestion finding), and a live mosaic's cells are already
    // kept current by patchNeighborMosaics(), so it is never staler than a fresh stitch --
    // UNLESS the center-cell texture it was built around is no longer the CURRENT one
    // buildOverzoomElevationMosaic()/getExistingOverzoomMosaic() would hand back (see
    // ElevationMosaicInfo::ownTexture): a composite is always rebuilt as a fresh Texture object
    // whenever its per-cell coverage improves or its target zoom changes (coverage-signature
    // check, see buildOverzoomElevationMosaic), so pointer identity is exactly "is there
    // anything new to stitch" -- simpler than the old compositeZoom-equality check, which could
    // go stale whenever a composite's coverage improved without its zoom LABEL changing (see
    // docs/terrain-illumination-plan.md "triangle-PL upsampling").
    auto existing = getExistingMosaic(id);
    ElevationMosaicInfo* existingInfo = (existing && existing->userData)
        ? static_cast<ElevationMosaicInfo*>(existing->userData.get()) : nullptr;

    // Obtain center-cell data: the overzoom composite (see buildOverzoomElevationMosaic) when
    // the tile is overzoomed, else the plain texture. This is what lets the composite and the
    // texture-shading neighbor-ring mosaic share one texture instead of being mutually
    // exclusive. `_ownTexture` may be null here (TileManager::upgradeAttachedRasters retrying a
    // tile whose plain texture is no longer directly at hand) -- then only the cache-only
    // registered-composite lookup is possible, which is fine: whenever the plain texture IS
    // available (attachment build, or retry via ElevationMosaicInfo::plainTexture) a full build
    // attempt happens instead.
    std::shared_ptr<Texture> ownForMosaic =
        _ownTexture ? buildOverzoomElevationMosaic(_tile, _ownTexture, _cellZoomCaps) : getExistingOverzoomMosaic(_tile);
    bool usingComposite = bool(ownForMosaic);

    int achievedZoom = _tile.z;
    if (ownForMosaic && ownForMosaic->userData) {
        achievedZoom = static_cast<ElevationMosaicInfo*>(ownForMosaic->userData.get())->compositeZoom;
    }

    if (!ownForMosaic) {
        // No composite achievable right now. Keep an existing mosaic (even a composite-centered
        // one -- its data is closer to any overzoomed target than the plain texture) rather
        // than downgrading; only build fresh (plain-centered) when there is nothing yet.
        if (existing) {
            // Fast-path: reusing the already-live mosaic, no stitchElevationMosaic() call below.
            Profiler::counterAdd("mosaicCacheHit", 1);
            return existing;
        }
        if (!_ownTexture) {
            // Fast-path: nothing available to build anything from either -- still an early
            // return that never reaches stitchElevationMosaic() below.
            Profiler::counterAdd("mosaicCacheHit", 1);
            return nullptr;      // nothing available to build anything from
        }
        ownForMosaic = _ownTexture;
    } else if (existingInfo && existingInfo->ownTexture == ownForMosaic) {
        // Fast-path: nothing has actually changed since the live mosaic was built -- share it
        // unchanged, no stitchElevationMosaic() call below.
        Profiler::counterAdd("mosaicCacheHit", 1);
        return existing;
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
                // Look for a neighbor composite at the SAME level as our center cell (passing
                // s = achievedZoom makes overzoomTargetZoom() target exactly that level;
                // achievedZoom <= z + kOverzoomCompositeMaxLevels always holds) -- only a
                // same-size cell can be stitched anyway. A miss just falls back to the plain
                // neighbor texture above, which stitchElevationMosaic()'s dimension check then
                // mirror-extrapolates -- never incorrectness, only a coarser transition cell.
                if (auto neighborComposite = getExistingOverzoomMosaic(TileID(nx, ny, _tile.z, achievedZoom))) {
                    neighborTex = neighborComposite;
                }
            }
            neighbors[size_t(dy + 1) * 3 + size_t(dx + 1)] = neighborTex;
        }
    }

    // Every early return above (all three fast paths) counted itself as mosaicCacheHit and
    // returned before reaching here, so anything that falls through to this point is exactly
    // the expensive path: a real stitchElevationMosaic() call. Every call to buildElevationMosaic()
    // hits exactly one of these two counters.
    Profiler::counterAdd("mosaicFullRebuild", 1);
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
        // Keep the tile's plain texture reachable for future rebuilds at a different composite
        // level (styling-zoom flips) even once nothing else references it: from the direct
        // argument when available, else carried through from whichever info we already have.
        if (_ownTexture) {
            info->plainTexture = _ownTexture;
        } else if (usingComposite && ownForMosaic->userData) {
            info->plainTexture = static_cast<ElevationMosaicInfo*>(ownForMosaic->userData.get())->plainTexture;
        } else {
            info->plainTexture = ownForMosaic;  // center IS the plain texture
        }
        info->rugosity = computeRugosity(*ownForMosaic, id);
        info->compositeZoom = achievedZoom;
        // Initial neighborMask: which cells stitchElevationMosaic() above actually found real
        // data for (same dimension/bpp check it applies internally) vs. mirror-extrapolated --
        // drives the R2 patch-upload debounce (see ElevationMosaicInfo::neighborMask).
        {
            int W = ownForMosaic->width();
            int bpp = ownForMosaic->getOptions().bytesPerPixel();
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    if (dx == 0 && dy == 0) { continue; }
                    auto& nb = neighbors[size_t(dy + 1) * 3 + size_t(dx + 1)];
                    if (nb && nb->bufferData() && nb->width() == W && nb->height() == W &&
                        nb->getOptions().bytesPerPixel() == bpp) {
                        info->neighborMask |= uint16_t(1u << (size_t(dy + 1) * 3 + size_t(dx + 1)));
                        // Keep this neighbor's texture alive for as long as THIS mosaic
                        // references its data -- see ElevationMosaicInfo::neighborTextures.
                        info->neighborTextures[size_t(dy + 1) * 3 + size_t(dx + 1)] = nb;
                    }
                }
            }
        }
        // UX diagnosis (2026-07-11): a neighbor cell that read real in the PREVIOUS mosaic but
        // mirror-extrapolated in THIS rebuild means that neighbor tile's texture was evicted
        // between the two builds - nothing keeps a neighbor's Texture alive the way `own`'s is
        // deliberately kept alive via userData above (that fix was for exactly this class of bug,
        // but only for the center tile; neighbors are looked up fresh via getTexture() every
        // rebuild with no equivalent protection). This directly reintroduces a texture-shading
        // seam that wasn't there a moment ago, independent of tilt - see the "edge artifacts even
        // at zenith, fine most of the time" report.
        if (existingInfo) {
            uint16_t lost = existingInfo->neighborMask & ~info->neighborMask;
            if (lost != 0) {
                int lostCount = 0;
                for (uint16_t m = lost; m; m &= uint16_t(m - 1)) { ++lostCount; }
                Profiler::counterAdd("mosaicNeighborRegressionCells", lostCount);
                Profiler::counterAdd("mosaicNeighborRegressionEvents", 1);
            }
        }
        mosaic->userData = std::move(info);
        // Register for reuse and for in-place patching by later-arriving neighbors (prune dead
        // entries while we're here -- mosaics die when their last referencing Tile does); this
        // OVERWRITES any existing (different-center-zoom, per the early-returns above) entry
        // for `id`, which is exactly the up/downgrade this function exists to perform.
        for (auto it = m_mosaics.begin(); it != m_mosaics.end();) {
            it = it->second.expired() ? m_mosaics.erase(it) : std::next(it);
        }
        m_mosaics[id] = mosaic;
        LOGD("Elevation mosaic for tile %s: %dx%d (own %s, center data z%d, %d live mosaics)",
             _tile.toString().c_str(), mosaic->width(), mosaic->height(),
             usingComposite ? "overzoom composite" : "plain tile",
             achievedZoom, int(m_mosaics.size()));
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
    // Single stable (x, y, z, zt) key -- see overzoomTargetZoom()'s doc comment on why this must
    // not be the raw `_primary` TileID. m_overzoomMosaics holds strong refs (see its doc comment
    // in rasterSource.h), so a hit here is unconditional -- no weak_ptr::lock() to fail.
    auto it = m_overzoomMosaics.find(TileID(_primary.x, _primary.y, _primary.z, zt));
    return it != m_overzoomMosaics.end() ? it->second : nullptr;
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

std::shared_ptr<Texture> RasterSource::buildOverzoomElevationMosaic(TileID _primary, const std::shared_ptr<Texture>& _ownTexture,
                                                                    const uint8_t* _cellZoomCaps) {
    if (!_ownTexture || !_ownTexture->bufferData() || !m_keepTextureData) { return nullptr; }

    int zt = overzoomTargetZoom(_primary);
    if (zt <= _primary.z) { return nullptr; }  // not overzoomed enough to be worth compensating

    PROFILE_SCOPE(Profiler::intern("buildOverzoomElevationMosaic:" + name()));

    int levels = zt - _primary.z;
    int N = 1 << levels;
    int32_t baseX = _primary.x << levels;
    int32_t baseY = _primary.y << levels;
    TileID key(_primary.x, _primary.y, _primary.z, zt);

    // Per-cell TARGET zoom: zt, capped by the caller-provided displayed-terrain zoom for the
    // cell (see the doc comment in rasterSource.h -- matching the DISPLAYED zoom rather than
    // the finest CACHED zoom is what makes the drape sit on the rendered terrain surface).
    auto cellTargetZoom = [&](int row, int col) -> int {
        if (!_cellZoomCaps) { return zt; }
        int cap = int(_cellZoomCaps[size_t(row) * size_t(N) + size_t(col)]);
        return std::max<int>(_primary.z, std::min<int>(zt, cap));
    };

    // Cheap per-cell fingerprint of what the stitch below would actually use: the cell's
    // target zoom AND whether real data at that zoom is cached right now. Drives the
    // rebuild-vs-reuse decision without walking the full ancestor chain or re-stitching when
    // nothing has actually changed since the last call -- this is what keeps calling this
    // function every update (TileManager::upgradeAttachedRasters does) cheap in the steady
    // state. Folding the target zoom in means a displayed-zoom flip (either direction)
    // changes the signature and triggers exactly one re-stitch.
    uint64_t signature = 1469598103934665603ull;  // FNV-1a offset basis
    for (int row = 0; row < N; row++) {
        for (int col = 0; col < N; col++) {
            int zc = cellTargetZoom(row, col);
            int dl = zt - zc;
            auto tex = getTexture(TileID((baseX + col) >> dl, (baseY + row) >> dl, zc));
            uint64_t v = (uint64_t(zc) << 1) | ((tex && tex->bufferData()) ? 1u : 0u);
            signature = (signature ^ v) * 1099511628211ull;  // FNV-1a prime
        }
    }

    auto existingIt = m_overzoomMosaics.find(key);
    if (existingIt != m_overzoomMosaics.end()) {
        auto* info = existingIt->second->userData
            ? static_cast<ElevationMosaicInfo*>(existingIt->second->userData.get()) : nullptr;
        if (info && info->coverageSignature == signature) { return existingIt->second; }
        // else: more (or, if a tile was evicted, less) real data is cached now -- rebuild below.
    }

    // Per zt-grid cell (independently -- this is the fix for intra-footprint zoom variation, see
    // docs/terrain-illumination-plan.md "triangle-PL upsampling"): use real data at the cell's
    // TARGET zoom (zt, or the caller's displayed-terrain cap) if cached, else the finest real
    // ancestor down to _primary.z+1 that IS cached (triangle-PL upsampled -- see OverzoomCell/
    // fillLatticeUpsampledCell), else `_ownTexture` itself (always available). When the walk
    // starts at za == zt, dl == 0 and the block geometry degenerates to the single exact cell.
    // Deliberately cache-only (docs/terrain-illumination-plan.md): no new fetch is triggered
    // even when a cell's real data isn't cached yet.
    std::vector<OverzoomCell> cells(size_t(N) * size_t(N));
    for (int row = 0; row < N; row++) {
        for (int col = 0; col < N; col++) {
            OverzoomCell& cell = cells[size_t(row) * size_t(N) + size_t(col)];

            bool found = false;
            for (int za = cellTargetZoom(row, col); za > _primary.z; za--) {
                int dl = zt - za;
                int32_t ax = (baseX + col) >> dl;
                int32_t ay = (baseY + row) >> dl;
                auto anc = getTexture(TileID(ax, ay, za));
                if (anc && anc->bufferData()) {
                    cell.texture = anc;
                    cell.blockColOriginWest = (ax << dl) - baseX;
                    cell.blockRowOriginNorth = (ay << dl) - baseY;
                    cell.blockSize = 1 << dl;
                    found = true;
                    break;
                }
            }
            if (!found) {
                cell.texture = _ownTexture;
                cell.blockColOriginWest = 0;
                cell.blockRowOriginNorth = 0;
                cell.blockSize = N;
            }
        }
    }

    auto composite = stitchElevationOverzoomMosaic(_ownTexture, cells, N);
    if (!composite) { return nullptr; }

    // Keep the plain texture alive for future rebuilds once the styling zoom flips (see
    // ElevationMosaicInfo's doc comments; ownTexture stays null -- that is what marks this
    // texture as a composite).
    auto info = std::make_shared<ElevationMosaicInfo>();
    info->plainTexture = _ownTexture;
    info->compositeZoom = zt;
    info->coverageSignature = signature;
    composite->userData = std::move(info);

    // Register (replacing any previous entry at this key -- see m_overzoomMosaics' doc comment
    // for why this is a fresh object each time rather than an in-place patch).
    pruneOverzoomMosaicsIfNeeded();
    m_overzoomMosaics[key] = composite;
    LOGD("Overzoom elevation composite for tile %s (cache key %s): %dx%d, target z%d, %d cells, %s, signature %llx",
         _primary.toString().c_str(), key.toString().c_str(), composite->width(), composite->height(),
         zt, N * N, _cellZoomCaps ? "display-capped" : "uncapped", (unsigned long long)signature);
    return composite;
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
            // whose mosaic we're patching. The CPU-side cell copy happens immediately (data
            // must stay correct), but the GPU resize()/re-upload is deferred to
            // flushDirtyMosaics() (R2, tile-pipeline-perf-plan.md): several neighbor arrivals
            // patching the same mosaic in quick succession (up to 8 per newly-cached tile, and
            // several tiles can complete per frame) would otherwise each force their own full
            // re-upload + mip regen on next bind.
            copyMosaicCell(mosaic->bufferData(), Wp, W, bpp, -dx, -dy, _texture->bufferData());
            if (mosaic->userData) {
                auto* info = static_cast<ElevationMosaicInfo*>(mosaic->userData.get());
                info->neighborMask |= uint16_t(1u << (size_t(-dy + 1) * 3 + size_t(-dx + 1)));
                // Keep this newly-arrived neighbor's texture alive too - see
                // ElevationMosaicInfo::neighborTextures. Without this, a cell patched in here
                // (rather than found during buildElevationMosaic's own rebuild) would have no
                // strong ref protecting it, and could still be evicted before any subsequent
                // rebuild, silently re-losing exactly the data this patch just restored.
                info->neighborTextures[size_t(-dy + 1) * 3 + size_t(-dx + 1)] = _texture;
                if (info->firstDirtyTime == 0.0) { info->firstDirtyTime = nowSeconds(); }
            } else {
                // No info to track dirty state against (shouldn't happen - every registered
                // mosaic gets one in buildElevationMosaic) - fall back to an immediate resize
                // so the patch isn't silently lost.
                mosaic->resize(mosaic->width(), mosaic->height());
            }

            LOGD("Patched mosaic %s cell (%d,%d) with real data from tile %s",
                 mosaicIt->first.toString().c_str(), -dx, -dy, _tileId.toString().c_str());
        }
    }
}

void RasterSource::flushDirtyMosaics() {
    PROFILE_SCOPE(Profiler::intern("flushDirtyMosaics:" + name()));
    double now = nowSeconds();
    for (auto it = m_mosaics.begin(); it != m_mosaics.end();) {
        auto mosaic = it->second.lock();
        if (!mosaic) { it = m_mosaics.erase(it); continue; }
        ++it;

        if (!mosaic->userData) { continue; }
        auto* info = static_cast<ElevationMosaicInfo*>(mosaic->userData.get());
        bool ringComplete = (info->neighborMask & kAllRealNeighborsMask) == kAllRealNeighborsMask;

        if (info->firstDirtyTime != 0.0) {  // dirty: decide whether to flush this cycle
            bool staleTooLong = (now - info->firstDirtyTime) >= kMosaicFlushDebounceSeconds;
            if (ringComplete || staleTooLong) {
                // Perf (2026-07 profiling): gate the actual GPU re-upload on this update's
                // upload budget - a burst of mosaics going ring-complete-or-stale-dirty in the
                // same update must not all resize() in one frame (measured up to ~500MB
                // uploaded in a single frame). Leave firstDirtyTime untouched when budget-
                // limited so this mosaic is retried (with priority, since it's already
                // eligible) on the next update instead of silently dropped.
                if (tryConsumeUploadBudget()) {
                    mosaic->resize(mosaic->width(), mosaic->height());
                    info->firstDirtyTime = 0.0;
                }
            }
            continue;  // just (re)dirtied or just flushed - either way, not yet safe to free
        }

        // R6 (tile-pipeline-perf-plan.md): not dirty, so any resize() from a previous cycle
        // has had at least one full update cycle to be picked up by a bind() -- once the ring
        // is complete too (no more real neighbor data will ever patch this mosaic), free its
        // CPU buffer. shouldResize() re-check guards the (expected rare) case where it was
        // JUST bound this same frame with a pending upload not yet consumed.
        if (ringComplete && mosaic->bufferData() && !mosaic->shouldResize()) {
            mosaic->releaseBuffer();
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
