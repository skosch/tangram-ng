#include "util/textureShading.h"

#include "data/rasterSource.h"
#include "gl/texture.h"
#include "tile/tile.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <vector>

namespace Tangram {

namespace {

// Pyramid depth for this CPU-side box-filter approximation. Deliberately the ORIGINAL,
// conservatively-safe default from docs/texture-shading-plan.md's Frozen Interface Contract
// ("Keep 2^level well under W/2, not just under W") rather than the live shader's current
// TEXTURE_SHADING_MAX_LEVELS=8 (hillshade.yaml): the shader's larger depth is only safe there
// because its fractional base LOD `b` is usually > 0 (continuous view-zoom minus tile zoom) and
// it samples hardware-mipmapped, true-box-averaged GPU textures; this CPU sampler always starts
// at level 0 (native tile resolution -- see sampleTextureShading's doc comment) and hand-rolls
// the box filter over the 3Wp-wide neighbor-ring mosaic, so a deep pyramid would read well into
// the (possibly still mirror-extrapolated) neighbor cells for a peak near its own tile's edge.
constexpr int kMaxLevels = 4;

// Matches the CURRENT shipped shader defaults (hillshade.yaml's u_texture_shading_alpha /
// u_texture_shading_contrast gui_variables) as of 2026-07-12, NOT the older values still
// written in docs/texture-shading-plan.md's top "Frozen Interface Contract" section (0.75 /
// 0.05) -- that section was never updated after the auto-contrast/zoom-normalization rework
// documented further down in the same file ("Resolution of the open follow-ups", "Default
// u_texture_shading_alpha is now 0.6"). Using the live values here is what actually keeps this
// consistent with what the hillshade layer renders, which is this whole feature's stated goal
// (docs/label-placement-plan.md Phase 2). Unlike the shader, this is a single fixed value -- no
// zoom-dependent alpha_min blending, no live auto-contrast multiplier: a peak priority ranking
// wants one stable number, not a continuously-animated one (see textureShading.h).
constexpr float kAlpha = 0.6f;
constexpr float kContrast = 1.0f;

// Same decode as ElevationManager::readElevTex / RasterSource's computeRugosity /
// decodeElevationPixel (elevationManager.cpp, rasterSource.cpp) -- duplicated locally rather
// than shared, matching this codebase's existing precedent of small local re-implementations of
// this exact decode (both of the above are themselves already separate local copies).
double decodeElevation(const GLubyte* p, int bpp, bool isFloat) {
    if (isFloat) {
        float f;
        std::memcpy(&f, p, sizeof(float));
        return double(f);
    }
    if (bpp <= 1) { return double(p[0]); }
    return (double(p[0]) * 256.0 + double(p[1]) + double(p[2]) / 256.0) - 32768.0;
}

// Bilinear sample of a square, row-major float elevation grid (`size` x `size`) at continuous
// pixel coordinates (x, y). Mirrors ElevationManager::elevationLerp's pixel-center convention
// (-0.5) and edge clamping.
double bilinearSample(const std::vector<float>& grid, int size, double x, double y) {
    double px = x - 0.5, py = y - 0.5;
    int ix0 = std::max(0, int(std::floor(px)));
    int iy0 = std::max(0, int(std::floor(py)));
    int ix1 = std::min(int(std::ceil(px)), size - 1);
    int iy1 = std::min(int(std::ceil(py)), size - 1);
    double fx = px - ix0, fy = py - iy0;
    double t00 = grid[size_t(iy0) * size_t(size) + size_t(ix0)];
    double t10 = grid[size_t(iy0) * size_t(size) + size_t(ix1)];
    double t01 = grid[size_t(iy1) * size_t(size) + size_t(ix0)];
    double t11 = grid[size_t(iy1) * size_t(size) + size_t(ix1)];
    double t0 = t00 + fx * (t10 - t00);
    double t1 = t01 + fx * (t11 - t01);
    return t0 + fy * (t1 - t0);
}

// One 2x2 box-filter downsample step, standing in for a hardware mip level (Frozen Interface
// Contract, docs/texture-shading-plan.md / docs/label-placement-plan.md: "software box-filter
// downsampling ... in place of hardware mipmaps"). Close enough to GL_LINEAR_MIPMAP_LINEAR for
// this purpose -- this feeds a ranking signal, not a rendered pixel, so bit-exactness isn't
// required.
std::vector<float> boxFilterDownsample(const std::vector<float>& src, int srcSize, int& outSize) {
    outSize = std::max(1, srcSize / 2);
    std::vector<float> dst(size_t(outSize) * size_t(outSize));
    for (int y = 0; y < outSize; y++) {
        int sy0 = std::min(srcSize - 1, 2 * y), sy1 = std::min(srcSize - 1, 2 * y + 1);
        for (int x = 0; x < outSize; x++) {
            int sx0 = std::min(srcSize - 1, 2 * x), sx1 = std::min(srcSize - 1, 2 * x + 1);
            float sum = src[size_t(sy0) * size_t(srcSize) + size_t(sx0)] +
                        src[size_t(sy0) * size_t(srcSize) + size_t(sx1)] +
                        src[size_t(sy1) * size_t(srcSize) + size_t(sx0)] +
                        src[size_t(sy1) * size_t(srcSize) + size_t(sx1)];
            dst[size_t(y) * size_t(outSize) + size_t(x)] = sum * 0.25f;
        }
    }
    return dst;
}

// A mosaic's decoded level0 buffer plus its box-filter-downsampled levels -- everything
// sampleTextureShadingAtMosaic() used to rebuild from scratch on every single call, even
// though none of it depends on the specific sample point (worldPos/tileId), only on the
// mosaic's own pixel data. Factored out so it can be cached (see sampleTextureShading()
// below) -- this was the exact "known, unimplemented perf improvement" this file used to
// flag in a comment rather than fix: refineAnchor() (label.cpp) samples up to ~45 points per
// peak (5 per candidate anchor x up to 9 anchors) against the SAME mosaic, so rebuilding the
// whole pyramid per sample rather than per mosaic made anchor refinement cost tens of
// millions of redundant float ops per peak in cluttered views -- see
// docs/label-placement-plan.md round 8.
struct Pyramid {
    std::vector<std::vector<float>> levels;  // levels[0] = decoded level0, then box-filter downsamples
    std::vector<int> sizes;                  // sizes[i] = levels[i]'s width/height
};

Pyramid buildPyramid(const Texture& mosaic) {
    Pyramid p;
    int mosaicW = mosaic.width();
    const GLubyte* data = mosaic.bufferData();
    const TextureOptions& options = mosaic.getOptions();
    int bpp = options.bytesPerPixel();
    bool isFloat = options.pixelFormat == PixelFormat::FLOAT;

    std::vector<float> level0(size_t(mosaicW) * size_t(mosaicW));
    for (int y = 0; y < mosaicW; y++) {
        for (int x = 0; x < mosaicW; x++) {
            const GLubyte* px = data + (size_t(y) * size_t(mosaicW) + size_t(x)) * size_t(bpp);
            level0[size_t(y) * size_t(mosaicW) + size_t(x)] = float(decodeElevation(px, bpp, isFloat));
        }
    }
    p.levels.push_back(level0);
    p.sizes.push_back(mosaicW);

    std::vector<float> level = std::move(level0);
    int levelSize = mosaicW;
    for (int k = 0; k < kMaxLevels; k++) {
        int nextSize;
        std::vector<float> next = boxFilterDownsample(level, levelSize, nextSize);
        if (nextSize < 2) { break; }  // pyramid bottomed out
        p.levels.push_back(next);
        p.sizes.push_back(nextSize);
        level = std::move(next);
        levelSize = nextSize;
    }
    return p;
}

// The per-sample-point half of the original function: walk an already-built pyramid at
// worldPos's mosaic-space uv, difference adjacent levels weighted by Brown's power law, same
// formula/behavior as before this refactor -- just reading a pyramid instead of building one.
float sampleFromPyramid(const Pyramid& p, TileID tileId, ProjectedMeters worldPos, int mosaicW) {
    if (p.levels.empty()) { return 0.5f; }

    // Tile-local uv of worldPos within its own tile -- same formula as
    // ElevationManager::elevationLerp(tex, tileId, meters).
    double scale = MapProjection::metersPerTileAtZoom(tileId.z);
    if (scale <= 0) { return 0.5f; }
    ProjectedMeters tileOrigin = MapProjection::tileSouthWestCorner(tileId);
    ProjectedMeters offset = worldPos - tileOrigin;
    double u = offset.x / scale, v = offset.y / scale;
    if (u < 0 || u > 1 || v < 0 || v > 1) { return 0.5f; }

    // Mosaic-space uv within the center third, per the Frozen Interface Contract:
    // `muv = (uv + 1) / 3`.
    double muvU = (u + 1.0) / 3.0, muvV = (v + 1.0) / 3.0;

    // texelMeters: real ground meters spanned by one mosaic level-0 texel. The center cell is
    // an uncropped 1:1 copy of the tile's own buffer (see rasterSource.cpp's Wp/mosaic-layout
    // comment), so this equals the tile's own texel spacing -- same latitude-corrected formula
    // as RasterSource's computeRugosity(), duplicated locally for the same reason as
    // decodeElevation() above.
    ProjectedMeters center = MapProjection::tileCenter(tileId);
    double unproj = std::exp(center.y / MapProjection::EARTH_RADIUS_METERS);
    double coslat = 2.0 / (unproj + 1.0 / unproj);
    int Wp = mosaicW / 3;
    double texelMeters = scale * coslat / double(Wp);
    if (!(texelMeters > 0)) { return 0.5f; }

    // Difference adjacent levels, weighting band k by pow(2, -k*alpha) -- Brown's power-law
    // filter discretization, exactly as pinned in docs/texture-shading-plan.md and mirrored by
    // hillshade.yaml's textureShading().
    double hPrev = bilinearSample(p.levels[0], p.sizes[0], muvU * p.sizes[0], muvV * p.sizes[0]);
    double z = 0.0;
    for (size_t k = 1; k < p.levels.size(); k++) {
        double hNext = bilinearSample(p.levels[k], p.sizes[k], muvU * p.sizes[k], muvV * p.sizes[k]);
        z += (hPrev - hNext) * std::pow(2.0, -double(k - 1) * double(kAlpha));
        hPrev = hNext;
    }

    double az = double(kContrast) * z / texelMeters;
    double shade = az / (2.0 * std::sqrt(1.0 + az * az)) + 0.5;
    return float(shade);
}

} // namespace

float sampleTextureShadingAtMosaic(const Texture& mosaic, TileID tileId, ProjectedMeters worldPos) {
    int mosaicW = mosaic.width();
    if (mosaicW <= 0 || mosaicW != mosaic.height() || (mosaicW % 3) != 0) { return 0.5f; }
    if (!mosaic.bufferData()) { return 0.5f; }

    Pyramid p = buildPyramid(mosaic);
    return sampleFromPyramid(p, tileId, worldPos, mosaicW);
}

float sampleTextureShading(RasterSource& elevationSource, ProjectedMeters worldPos, bool& ok) {
    ok = false;

    // Find which (same-zoom) tile currently covers worldPos, reusing the same lookup
    // ElevationManager::getElevation() already relies on -- see rasterSource.cpp's
    // RasterSource::getRaster(). Requires m_textures (main-thread-only) to be non-empty.
    Raster raster = elevationSource.getRaster(worldPos);
    if (!raster.texture) { return 0.5f; }

    // Cheap reuse-only lookup (single map find + weak_ptr lock, no stitching) -- never
    // triggers a new mosaic build. nullptr here means no mosaic is currently resident for
    // this tile (elevation mosaic feature/toggle off, or not stitched yet) -- ok stays false,
    // caller falls back to another priority signal.
    auto mosaic = elevationSource.getExistingMosaic(raster.tileID);
    if (!mosaic || !mosaic->bufferData()) { return 0.5f; }

    int mosaicW = mosaic->width();
    if (mosaicW <= 0 || mosaicW != mosaic->height() || (mosaicW % 3) != 0) { return 0.5f; }

    // Perf (round 8): this is called up to ~45 times per peak from refineAnchor() (5 sample
    // points x up to 9 anchors) plus once more from refinePriority(), all against the SAME
    // mosaic for a given peak -- and building the pyramid is a full decode + kMaxLevels
    // box-filter downsample pass over the whole mosaic buffer, independent of the sample
    // point. Single-slot cache (weak_ptr-keyed, so a destroyed/replaced mosaic -- e.g. a fresh
    // stitch, see buildElevationMosaic()'s "always rebuilt as a fresh Texture object" comment
    // in rasterSource.cpp -- is never mistaken for a live one, unlike a raw-pointer-identity
    // check) rather than a full per-texture LRU: consecutive calls in one frame overwhelmingly
    // target the same mosaic (one label's own 46 calls, or several nearby peaks sharing one
    // tile), so this alone captures the large majority of the redundant work. A peak whose
    // mosaic differs from the immediately-preceding call just pays the normal one-time
    // rebuild cost, same as before this change.
    static std::weak_ptr<Texture> s_cachedMosaic;
    static Pyramid s_cachedPyramid;
    if (s_cachedMosaic.lock() != mosaic) {
        s_cachedPyramid = buildPyramid(*mosaic);
        s_cachedMosaic = mosaic;
    }

    ok = true;
    return sampleFromPyramid(s_cachedPyramid, raster.tileID, worldPos, mosaicW);
}

}
