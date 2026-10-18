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
    // Mosaic-space texel coordinate of levels[0]'s (0,0). Always a multiple of
    // 2^kMaxLevels so the box-filter grid of every level lines up exactly with the
    // full-mosaic pyramid's (level k texel i covers source texels [i*2^k, (i+1)*2^k)).
    int originX = 0, originY = 0;
};

Pyramid buildPyramid(const Texture& mosaic) {
    Pyramid p;
    int mosaicW = mosaic.width();
    const GLubyte* data = mosaic.bufferData();
    const TextureOptions& options = mosaic.getOptions();
    int bpp = options.bytesPerPixel();
    bool isFloat = options.pixelFormat == PixelFormat::FLOAT;

    // Perf (2026-10-10, real trace: refine calls ~100 ms each, LabelsCollect 445 ms mean
    // at 160-280 visible tiles): sampleFromPyramid() only ever reads the CENTER third of
    // the mosaic (uv in [1/3, 2/3], bilinear reaching one texel further), but this used to
    // decode and box-filter all 9 cells. A level-k texel only depends on source texels
    // within 2^k, so cropping to the center cell plus a margin of 2*2^kMaxLevels texels,
    // with the crop origin snapped to a multiple of 2^kMaxLevels (grid alignment), gives
    // BIT-IDENTICAL samples in the center third from ~1/7 of the decode work and memory.
    constexpr int kAlign = 1 << kMaxLevels;
    constexpr int kMargin = 2 * kAlign;
    int Wp = mosaicW / 3;
    int x0 = std::max(0, ((Wp - kMargin) / kAlign) * kAlign);
    int x1 = std::min(mosaicW, ((2 * Wp + kMargin + kAlign - 1) / kAlign) * kAlign);
    int cropW = x1 - x0;
    p.originX = p.originY = x0;

    std::vector<float> level0(size_t(cropW) * size_t(cropW));
    for (int y = 0; y < cropW; y++) {
        const GLubyte* row = data + (size_t(y + x0) * size_t(mosaicW) + size_t(x0)) * size_t(bpp);
        float* out = &level0[size_t(y) * size_t(cropW)];
        if (isFloat) {
            std::memcpy(out, row, size_t(cropW) * sizeof(float));
        } else {
            for (int x = 0; x < cropW; x++) {
                out[x] = float(decodeElevation(row + size_t(x) * size_t(bpp), bpp, false));
            }
        }
    }
    p.levels.push_back(std::move(level0));
    p.sizes.push_back(cropW);

    int levelSize = cropW;
    for (int k = 0; k < kMaxLevels; k++) {
        int nextSize;
        std::vector<float> next = boxFilterDownsample(p.levels.back(), levelSize, nextSize);
        if (nextSize < 2) { break; }  // pyramid bottomed out
        p.levels.push_back(std::move(next));
        p.sizes.push_back(nextSize);
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
    // Full-pyramid level-k coordinate is muv * (mosaicW >> k) (the original expression used
    //  the level's own size, which equals mosaicW >> k for the uncropped pyramid); the
    //  cropped level's origin in that space is origin >> k (origin is 2^kMaxLevels-aligned).
    auto levelSample = [&](size_t k) {
        double fullSize = double(mosaicW >> k);
        double ox = double(p.originX >> k), oy = double(p.originY >> k);
        return bilinearSample(p.levels[k], p.sizes[k], muvU * fullSize - ox, muvV * fullSize - oy);
    };
    double hPrev = levelSample(0);
    double z = 0.0;
    for (size_t k = 1; k < p.levels.size(); k++) {
        double hNext = levelSample(k);
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

    // Perf (round 8, widened 2026-10-10): called up to ~45 times per peak from refineAnchor()
    // plus once from refinePriority(), against the SAME mosaic for a given peak; the pyramid
    // is independent of the sample point. The round-8 single slot thrashed in a real
    // 160-280-visible-tile trace (labels from many tiles interleave per frame, every switch a
    // full rebuild), so this is a small LRU keyed by weak_ptr identity (a destroyed/replaced
    // mosaic is never mistaken for a live one - see buildElevationMosaic()'s "always rebuilt
    // as a fresh Texture object" comment in rasterSource.cpp). Pyramids are cropped to ~1/7
    // of the mosaic (buildPyramid), so 32 entries stay in the low tens of MB.
    struct CacheEntry {
        std::weak_ptr<Texture> mosaic;
        std::shared_ptr<Pyramid> pyramid;
        uint64_t lastUse = 0;
    };
    constexpr size_t kCacheSize = 32;
    static std::vector<CacheEntry> s_cache;
    static uint64_t s_tick = 0;
    ++s_tick;

    CacheEntry* hit = nullptr;
    CacheEntry* victim = nullptr;
    for (auto& e : s_cache) {
        auto m = e.mosaic.lock();
        if (m == mosaic) { hit = &e; break; }
        if (!m) { victim = &e; }  // expired entries are preferred victims
    }
    if (!hit) {
        if (s_cache.size() < kCacheSize && !victim) {
            s_cache.emplace_back();
            victim = &s_cache.back();
        } else if (!victim) {
            victim = &*std::min_element(s_cache.begin(), s_cache.end(),
                [](const CacheEntry& a, const CacheEntry& b) { return a.lastUse < b.lastUse; });
        }
        victim->pyramid = std::make_shared<Pyramid>(buildPyramid(*mosaic));
        victim->mosaic = mosaic;
        hit = victim;
    }
    hit->lastUse = s_tick;

    ok = true;
    return sampleFromPyramid(*hit->pyramid, raster.tileID, worldPos, mosaicW);
}

PeakDominance peakDominanceRadius(const PeakElevationFn& elevAt, ProjectedMeters pos,
                                  double groundPerProjected, float maxRadius) {
    PeakDominance out;
    const double kTwoPi = 6.283185307179586;
    double gpp = std::max(groundPerProjected, 1e-3);

    double ref = 0;
    if (!elevAt(pos, ref)) { return out; }
    out.valid = true;

    // The tagged summit is rarely exactly on the DEM's highest texel, particularly in the
    // coarse tiles used for distant terrain: take the max over a ~60 m neighbourhood.
    const double r60 = 60.0 / gpp;
    for (int k = 0; k < 8; k++) {
        double a = k * kTwoPi / 8.0, e = 0;
        if (elevAt(ProjectedMeters(pos.x + std::cos(a) * r60, pos.y + std::sin(a) * r60), e)) {
            ref = std::max(ref, e);
        }
    }
    out.peakElev = ref;

    // Polar sampling grid whose spacing is a constant fraction (~15%) of the radius, so a
    // taller neighbour's summit cap is hit at any distance with a roughly constant ~1500
    // samples for a fully dominant peak. Rings are visited outwards and the scan stops at the
    // first sample above the peak, so the many minor bumps in a view exit within a few rings
    // and cost tens of samples, not thousands.
    const double r0 = 120.0, growth = 1.25;
    const int kRingSamples = 36;
    double lastClean = r0 / growth;
    int total = 0, missing = 0;
    bool escaped = true;
    int ring = 0;
    for (double R = r0; R <= maxRadius * 1.0001; R *= growth, ring++) {
        // slack for DEM noise / resolution and for ridges that climb very gently away
        double tol = 8.0 + 0.006 * R;
        double rp = R / gpp;
        // golden-ratio phase so successive rings don't line up into radial spokes
        double phase = std::fmod(ring * 0.618034, 1.0);
        bool clean = true;
        for (int k = 0; k < kRingSamples && clean; k++) {
            double a = (k + phase) * kTwoPi / kRingSamples, e = 0;
            total++;
            if (!elevAt(ProjectedMeters(pos.x + std::cos(a) * rp, pos.y + std::sin(a) * rp), e)) {
                missing++;
                continue;
            }
            if (e > ref + tol) { clean = false; }
        }
        if (!clean) {
            out.radiusMeters = float(std::sqrt(lastClean * R));
            escaped = false;
            break;
        }
        lastClean = R;
    }
    if (escaped) { out.radiusMeters = maxRadius; }
    out.complete = missing * 10 <= total * 3;  // <= 30% of samples unavailable
    return out;
}

}
