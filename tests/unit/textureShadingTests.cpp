#include "catch.hpp"

#include "data/rasterSource.h"
#include "gl/hardware.h"
#include "gl/texture.h"
#include "tile/tile.h"
#include "util/mapProjection.h"
#include "util/textureShading.h"

#include <cmath>
#include <cstring>
#include <vector>

using namespace Tangram;

namespace {

// See rasterMosaicTests.cpp's identical helper for why this is needed outside a real GL context.
struct ForceNPOTSupport {
    ForceNPOTSupport() { Hardware::supportsTextureNPOT = true; }
} forceNPOTSupport;

TextureOptions floatOptions() {
    TextureOptions options;
    options.pixelFormat = PixelFormat::FLOAT;
    return options;
}

// A `size` x `size` FLOAT mosaic buffer built directly from a per-pixel value function --
// bypasses stitchElevationMosaic() entirely so the "known ridge pattern" is exact and doesn't
// depend on neighbor-mirroring behavior (already covered by rasterMosaicTests.cpp).
template <typename Fn>
std::shared_ptr<Texture> makeMosaic(int size, Fn valueAt) {
    auto tex = std::make_shared<Texture>(floatOptions(), false);
    std::vector<float> data(size_t(size) * size_t(size));
    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            data[size_t(y) * size_t(size) + size_t(x)] = valueAt(x, y);
        }
    }
    tex->setPixelData(size, size, sizeof(float), reinterpret_cast<const GLubyte*>(data.data()),
                       data.size() * sizeof(float));
    return tex;
}

// worldPos for tile-local (u, v) in [0,1]^2 within `tileId`'s own footprint -- same convention
// ElevationManager::elevationLerp(tex, tileId, meters) and sampleTextureShadingAtMosaic() use
// (v = 0 at the tile's south edge).
ProjectedMeters worldPosAt(TileID tileId, double u, double v) {
    double scale = MapProjection::metersPerTileAtZoom(tileId.z);
    ProjectedMeters origin = MapProjection::tileSouthWestCorner(tileId);
    return origin + ProjectedMeters(u * scale, v * scale);
}

} // namespace

TEST_CASE("Texture shading sampler - perfectly flat terrain is exactly neutral (0.5)", "[TextureShading]") {
    // A constant field box-filters to itself at every pyramid level, so every band differences
    // to exactly zero regardless of alpha/contrast -- shade = 0/(2*sqrt(1+0)) + 0.5 = 0.5
    // exactly. This is a genuine hand-computed expected value, not just a ballpark check.
    const int Wp = 16;
    const int mosaicW = 3 * Wp;
    auto mosaic = makeMosaic(mosaicW, [](int, int) { return 1000.0f; });

    TileID tileId(1, 1, 3);
    float shade = sampleTextureShadingAtMosaic(*mosaic, tileId, worldPosAt(tileId, 0.5, 0.5));
    REQUIRE(shade == Approx(0.5f).margin(1e-5f));
}

TEST_CASE("Texture shading sampler - a ridge stands out at its own crest more than nearby flat ground",
          "[TextureShading]") {
    // A north-south ridge running the length of the WHOLE mosaic (own cell + both mirrored/real
    // neighbor cells, avoiding any cell-boundary discontinuity) along mosaic column Wp*1.5 (the
    // tile's own vertical centerline) -- a "known ridge pattern" per docs/label-placement-plan.md
    // Phase 2's acceptance criteria. `ridgeHalfWidth` texels on either side of the crest are
    // raised by `ridgeAmt` meters above the flat `base` elevation everywhere else. Wp is chosen
    // large enough (128) that a point near the tile's own edge is far outside the ~16-texel
    // reach of the kMaxLevels=4 pyramid's coarsest band (textureShading.cpp) -- with a smaller
    // Wp, this high-pass filter's real "negative side lobe" just downhill of a ridge (the
    // coarser, more-blurred level is HIGHER there than the finer level, since it partially
    // averages in the nearby ridge -- exactly the Mexican-hat-like response a Laplacian pyramid
    // is expected to have) would still reach every point in a small tile, making "away from the
    // ridge" not actually flat/neutral within this buffer.
    const int Wp = 128;
    const int mosaicW = 3 * Wp;
    const double base = 1000.0;
    const double ridgeAmt = 400.0;
    const int crestX = mosaicW / 2;
    const int ridgeHalfWidth = 2;

    auto mosaic = makeMosaic(mosaicW, [&](int x, int) {
        return float(base + (std::abs(x - crestX) <= ridgeHalfWidth ? ridgeAmt : 0.0));
    });

    // z=14 (a realistic elevation-tile zoom, per docs/texture-shading-plan.md's confirmed real
    // 257px-tile serving zoom) keeps texelMeters in the tens-of-meters range, so a 400m ridge
    // registers as a large signal relative to it -- at a very coarse zoom (e.g. z=3, a
    // ~5000km-wide tile) texelMeters is enormous and even a very tall ridge normalizes away to
    // nearly nothing, which isn't a representative "known ridge pattern" for what this sampler
    // actually evaluates in production (peak label positions, always well inside a source's max
    // zoom -- see RasterSource::getRaster()'s zoom walk).
    TileID tileId(8192, 8192, 14);
    // u=0.5 is the tile's own vertical centerline, i.e. mosaic column Wp + Wp/2 = mosaicW/2 --
    // exactly the ridge crest.
    float shadeOnCrest = sampleTextureShadingAtMosaic(*mosaic, tileId, worldPosAt(tileId, 0.5, 0.5));
    // u=0.02 sits well inside the tile's own cell (u in [0,1]) but ~61 texels from the crest --
    // locally flat ground genuinely out of reach of the ridge's influence at any pyramid level.
    float shadeAwayFromRidge = sampleTextureShadingAtMosaic(*mosaic, tileId, worldPosAt(tileId, 0.02, 0.5));

    INFO("shadeOnCrest=" << shadeOnCrest << " shadeAwayFromRidge=" << shadeAwayFromRidge);

    // Brown's contrast curve is monotonic in the band-weighted sum z, and a positive local bump
    // (finer levels see more of the ridge than coarser, more-blurred levels do) makes z > 0 at
    // the crest -- so the crest should read as MORE prominent (shade > 0.5) than flat ground
    // elsewhere (shade ~= 0.5), i.e. the sampler correctly ranks "on a ridge" above "not on a
    // ridge", the entire point of using this as a peak-priority signal (Phase 2 design step 2).
    REQUIRE(shadeOnCrest > 0.5f + 0.1f);
    REQUIRE(shadeOnCrest > shadeAwayFromRidge);
    REQUIRE(shadeAwayFromRidge == Approx(0.5f).margin(0.05f));
}

TEST_CASE("Texture shading sampler - invalid/unpopulated mosaic returns neutral", "[TextureShading]") {
    TileID tileId(1, 1, 3);

    SECTION("non-square texture") {
        auto tex = std::make_shared<Texture>(floatOptions(), false);
        std::vector<float> data(32 * 16, 1000.0f);
        tex->setPixelData(32, 16, sizeof(float), reinterpret_cast<const GLubyte*>(data.data()),
                           data.size() * sizeof(float));
        REQUIRE(sampleTextureShadingAtMosaic(*tex, tileId, worldPosAt(tileId, 0.5, 0.5)) == 0.5f);
    }

    SECTION("width not a multiple of 3 (not a valid Wp*3 mosaic)") {
        auto tex = std::make_shared<Texture>(floatOptions(), false);
        std::vector<float> data(10 * 10, 1000.0f);
        tex->setPixelData(10, 10, sizeof(float), reinterpret_cast<const GLubyte*>(data.data()),
                           data.size() * sizeof(float));
        REQUIRE(sampleTextureShadingAtMosaic(*tex, tileId, worldPosAt(tileId, 0.5, 0.5)) == 0.5f);
    }

    SECTION("position outside the tile's own footprint") {
        auto mosaic = makeMosaic(48, [](int, int) { return 1000.0f; });
        ProjectedMeters farAway(1e9, 1e9);
        REQUIRE(sampleTextureShadingAtMosaic(*mosaic, tileId, farAway) == 0.5f);
    }
}

TEST_CASE("Texture shading sampler - RasterSource wrapper reports unavailable when no mosaic is resident",
          "[TextureShading]") {
    // A freshly-constructed RasterSource has never cached any tile texture, so
    // RasterSource::getRaster() (which sampleTextureShading() relies on to find the tile
    // covering worldPos) can't find anything -- ok must come back false, matching the "flat/
    // non-terrain map style, or tile not yet cached" fallback case documented in textureShading.h.
    RasterSource src("test-elevation", nullptr, floatOptions());
    TileID tileId(1, 1, 3);

    bool ok = true;
    float shade = sampleTextureShading(src, worldPosAt(tileId, 0.5, 0.5), ok);
    REQUIRE_FALSE(ok);
    REQUIRE(shade == 0.5f);
}

TEST_CASE("Texture shading sampler - integrates with RasterSource::buildElevationMosaic",
          "[TextureShading]") {
    // Exercises the actual production wiring one level down from sampleTextureShading(): build
    // a real mosaic via RasterSource::buildElevationMosaic() (same call RasterTileTask::addRaster()
    // makes -- see rasterSource.cpp), then evaluate the sampler against exactly that mosaic, the
    // same way sampleTextureShading() would once it has looked one up via getExistingMosaic().
    const int W = 32;
    TileID tileId(5, 5, 4);

    auto own = std::make_shared<Texture>(floatOptions());
    std::vector<float> data(size_t(W) * size_t(W), 1000.0f);
    own->setPixelData(W, W, sizeof(float), reinterpret_cast<const GLubyte*>(data.data()),
                       data.size() * sizeof(float));

    RasterSource src("test-elevation", nullptr, floatOptions());
    auto mosaic = src.buildElevationMosaic(tileId, own);
    REQUIRE(mosaic != nullptr);
    REQUIRE(src.getExistingMosaic(tileId) == mosaic);

    float shade = sampleTextureShadingAtMosaic(*mosaic, tileId, worldPosAt(tileId, 0.5, 0.5));
    // Flat input (no neighbors present, so mirror-extrapolated cells are flat copies too) ->
    // still exactly neutral.
    REQUIRE(shade == Approx(0.5f).margin(1e-5f));
}

TEST_CASE("Texture shading sampler - cropped pyramid matches a full-mosaic reference", "[TextureShading]") {
    // buildPyramid() crops to the center cell + an aligned margin (perf). Samples must be
    // identical to a straightforward full-mosaic box-filter pyramid, including for a Wp that
    // is not a multiple of the crop alignment.
    const int Wp = 100, W = 3 * Wp;
    auto field = [](int x, int y) {
        return 1500.0f + 80.0f * std::sin(x * 0.11f) * std::cos(y * 0.07f) +
               25.0f * std::sin(x * 0.53f + y * 0.31f) + 0.01f * float((x * 7919 + y * 104729) % 97);
    };
    auto mosaic = makeMosaic(W, field);

    // reference: full pyramid, same formula (kAlpha 0.6, kContrast 1, 4 levels)
    std::vector<std::vector<float>> lv(1, std::vector<float>(size_t(W) * W));
    for (int y = 0; y < W; y++) for (int x = 0; x < W; x++) lv[0][size_t(y) * W + x] = field(x, y);
    std::vector<int> sz(1, W);
    for (int k = 0; k < 4; k++) {
        int s = sz.back(), n = std::max(1, s / 2);
        if (n < 2) break;
        std::vector<float> d(size_t(n) * n);
        for (int y = 0; y < n; y++) for (int x = 0; x < n; x++) {
            auto at = [&](int xx, int yy) { return lv.back()[size_t(std::min(s - 1, yy)) * s + size_t(std::min(s - 1, xx))]; };
            d[size_t(y) * n + x] = 0.25f * (at(2 * x, 2 * y) + at(2 * x + 1, 2 * y) + at(2 * x, 2 * y + 1) + at(2 * x + 1, 2 * y + 1));
        }
        lv.push_back(d); sz.push_back(n);
    }
    auto bil = [&](size_t k, double x, double y) {
        int s = sz[k]; double px = x - 0.5, py = y - 0.5;
        int ix0 = std::max(0, int(std::floor(px))), iy0 = std::max(0, int(std::floor(py)));
        int ix1 = std::min(int(std::ceil(px)), s - 1), iy1 = std::min(int(std::ceil(py)), s - 1);
        double fx = px - ix0, fy = py - iy0;
        auto g = [&](int xx, int yy) { return double(lv[k][size_t(yy) * s + size_t(xx)]); };
        double a = g(ix0, iy0) + fx * (g(ix1, iy0) - g(ix0, iy0));
        double b = g(ix0, iy1) + fx * (g(ix1, iy1) - g(ix0, iy1));
        return a + fy * (b - a);
    };

    TileID tileId(5, 7, 4);
    double scale = MapProjection::metersPerTileAtZoom(tileId.z);
    double unproj = std::exp(MapProjection::tileCenter(tileId).y / MapProjection::EARTH_RADIUS_METERS);
    double coslat = 2.0 / (unproj + 1.0 / unproj);
    double texelMeters = scale * coslat / double(Wp);

    for (double u : {0.01, 0.07, 0.5, 0.93, 0.99}) {
        for (double v : {0.01, 0.31, 0.5, 0.77, 0.99}) {
            double muvU = (u + 1.0) / 3.0, muvV = (v + 1.0) / 3.0;
            double hPrev = bil(0, muvU * sz[0], muvV * sz[0]), z = 0;
            for (size_t k = 1; k < lv.size(); k++) {
                double hNext = bil(k, muvU * sz[k], muvV * sz[k]);
                z += (hPrev - hNext) * std::pow(2.0, -double(k - 1) * 0.6);
                hPrev = hNext;
            }
            double az = z / texelMeters;
            float expected = float(az / (2.0 * std::sqrt(1.0 + az * az)) + 0.5);
            float got = sampleTextureShadingAtMosaic(*mosaic, tileId, worldPosAt(tileId, u, v));
            REQUIRE(got == Approx(expected).margin(1e-6f));
        }
    }
}

// ---- peakDominanceRadius -------------------------------------------------------------------

namespace {
// Synthetic DEM in projected meters near the equator (ground == projected): a main summit and
// a lower satellite peak `dist` metres away on a connecting ridge.
double syntheticDem(double x, double y, double satX) {
    auto cone = [](double dx, double dy, double h, double r) {
        double d = std::sqrt(dx * dx + dy * dy);
        return std::max(0.0, h * (1.0 - d / r));
    };
    return std::max(cone(x, y, 3000.0, 6000.0), cone(x - satX, y, 2500.0, 2000.0));
}
}

TEST_CASE("peakDominanceRadius: summit is dominant, satellite is not", "[Core][TextureShading]") {
    const double satX = 3000.0;
    PeakElevationFn dem = [&](ProjectedMeters p, double& e) { e = syntheticDem(p.x, p.y, satX); return true; };

    PeakDominance main = peakDominanceRadius(dem, ProjectedMeters(0, 0), 1.0);
    REQUIRE(main.valid);
    REQUIRE(main.complete);
    // nothing in the scene is higher: saturates at the default max radius
    REQUIRE(main.radiusMeters == Approx(12000.f));

    // The satellite is a separate, lower summit 3 km from the main one: the first ground
    // above it is the main cone's upper slope ~2 km away, so its dominance is a couple of km.
    PeakDominance sat = peakDominanceRadius(dem, ProjectedMeters(satX, 0), 1.0);
    REQUIRE(sat.valid);
    REQUIRE(sat.radiusMeters > 500.f);
    REQUIRE(sat.radiusMeters < 4000.f);
    REQUIRE(sat.radiusMeters < main.radiusMeters);
}

TEST_CASE("peakDominanceRadius: a taller neighbour limits the radius", "[Core][TextureShading]") {
    // A modest summit with a broader, 400 m taller massif 4 km away.
    PeakElevationFn dem = [&](ProjectedMeters p, double& e) {
        auto cone = [](double dx, double dy, double h, double r) {
            return std::max(0.0, h * (1.0 - std::sqrt(dx * dx + dy * dy) / r));
        };
        e = std::max(cone(p.x, p.y, 2000.0, 1500.0), cone(p.x - 4000.0, p.y, 2400.0, 3000.0));
        return true;
    };
    PeakDominance d = peakDominanceRadius(dem, ProjectedMeters(0, 0), 1.0);
    REQUIRE(d.valid);
    // The taller ground starts ~3 km out; the estimate (geometric mean of the last clean and
    // first dirty ring) must land well short of the neighbour's distance.
    REQUIRE(d.radiusMeters > 500.f);
    REQUIRE(d.radiusMeters < 4000.f);
}

TEST_CASE("peakDominanceRadius: missing DEM at the peak is not valid, missing ring is incomplete",
          "[Core][TextureShading]") {
    PeakElevationFn none = [](ProjectedMeters, double&) { return false; };
    REQUIRE_FALSE(peakDominanceRadius(none, ProjectedMeters(0, 0), 1.0).valid);

    PeakElevationFn centerOnly = [](ProjectedMeters p, double& e) {
        if (std::hypot(p.x, p.y) > 100.0) { return false; }
        e = 1000.0;
        return true;
    };
    PeakDominance d = peakDominanceRadius(centerOnly, ProjectedMeters(0, 0), 1.0);
    REQUIRE(d.valid);
    REQUIRE_FALSE(d.complete);
}
