#include "catch.hpp"

#include "data/rasterSource.h"
#include "gl/hardware.h"
#include "gl/texture.h"

#include <array>
#include <cstring>
#include <vector>

using namespace Tangram;

namespace {

// Texture::resize() disables generateMipmaps for NPOT textures (our 3W x 3W mosaic is NPOT
// whenever W isn't a power of two divided by 3, e.g. W=256 -> 768) unless the driver is known
// to support it. Outside of a real GL context (as in this unit test) Hardware::loadCapabilities()
// never runs, so simulate what's always true in practice for this codebase: every real target
// here is GL3+/GLES3 (`#version 300 es`, confirmed in shaderSource.cpp), which always supports
// NPOT textures (see gl/hardware.cpp:59). Without this, the mipmap assertions below would fail
// for a reason that has nothing to do with the mosaic stitching logic under test.
struct ForceNPOTSupport {
    ForceNPOTSupport() { Hardware::supportsTextureNPOT = true; }
} forceNPOTSupport;

// Grid index helper matching stitchElevationMosaic()'s documented layout: neighbors[(dy+1)*3 + (dx+1)].
size_t gridIndex(int dx, int dy) { return size_t(dy + 1) * 3 + size_t(dx + 1); }

TextureOptions testOptions() {
    TextureOptions options;
    options.pixelFormat = PixelFormat::ALPHA;  // 1 byte/pixel -- simplest to reason about in tests
    return options;
}

// A WxW texture where every pixel has the same byte value.
std::shared_ptr<Texture> makeConstTexture(int W, uint8_t value) {
    auto tex = std::make_shared<Texture>(testOptions());
    std::vector<GLubyte> data(size_t(W) * size_t(W), value);
    tex->setPixelData(W, W, 1, data.data(), data.size());
    return tex;
}

// A WxW texture where pixel (r, c) = (r+1)*10 + c -- distinguishes row flips from column flips.
std::shared_ptr<Texture> makeRampTexture(int W) {
    auto tex = std::make_shared<Texture>(testOptions());
    std::vector<GLubyte> data(size_t(W) * size_t(W));
    for (int r = 0; r < W; r++) {
        for (int c = 0; c < W; c++) {
            data[size_t(r) * W + c] = GLubyte((r + 1) * 10 + c);
        }
    }
    tex->setPixelData(W, W, 1, data.data(), data.size());
    return tex;
}

uint8_t mosaicPixel(const Texture& mosaic, int row, int col) {
    return mosaic.bufferData()[size_t(row) * size_t(mosaic.width()) + size_t(col)];
}

} // namespace

TEST_CASE("Elevation mosaic - dimensions and center cell placement", "[RasterMosaic]") {
    const int W = 4;
    auto own = makeConstTexture(W, 100);

    std::array<std::shared_ptr<Texture>, 9> neighbors;
    // Fill all 8 neighbors with distinct, identifiable constant values.
    int value = 1;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) { continue; }
            neighbors[gridIndex(dx, dy)] = makeConstTexture(W, uint8_t(10 + value));
            value++;
        }
    }

    auto mosaic = stitchElevationMosaic(own, neighbors);
    REQUIRE(mosaic != nullptr);

    // Frozen Interface Contract: mosaic buffer must be exactly 3W x 3W.
    REQUIRE(mosaic->width() == 3 * W);
    REQUIRE(mosaic->height() == 3 * W);

    // Tile's own data occupies the center third [W, 2W) x [W, 2W).
    for (int r = W; r < 2 * W; r++) {
        for (int c = W; c < 2 * W; c++) {
            REQUIRE(mosaicPixel(*mosaic, r, c) == 100);
        }
    }

    // Mosaic must be uploaded with hardware mipmaps enabled, per the contract.
    REQUIRE(mosaic->getOptions().generateMipmaps == true);
    REQUIRE(mosaic->getOptions().minFilter == TextureMinFilter::LINEAR_MIPMAP_LINEAR);
    REQUIRE(mosaic->getOptions().magFilter == TextureMagFilter::LINEAR);
    REQUIRE(mosaic->getOptions().wrapS == TextureWrap::CLAMP_TO_EDGE);
    REQUIRE(mosaic->getOptions().wrapT == TextureWrap::CLAMP_TO_EDGE);
}

TEST_CASE("Elevation mosaic - all 8 neighbor cells land in the correct grid position", "[RasterMosaic]") {
    const int W = 4;
    auto own = makeConstTexture(W, 100);

    std::array<std::shared_ptr<Texture>, 9> neighbors;
    std::array<uint8_t, 9> expected{};
    int value = 1;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) { continue; }
            uint8_t v = uint8_t(10 + value);
            neighbors[gridIndex(dx, dy)] = makeConstTexture(W, v);
            expected[gridIndex(dx, dy)] = v;
            value++;
        }
    }

    auto mosaic = stitchElevationMosaic(own, neighbors);
    REQUIRE(mosaic != nullptr);

    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) { continue; }
            // buffer rows run south -> north, OPPOSITE to tile y: the north neighbor (dy=-1)
            // lands in the mosaic's top row band [2W, 3W) - see cell layout notes in
            // rasterSource.h/.cpp
            int rowOff = (1 - dy) * W;
            int colOff = (dx + 1) * W;
            uint8_t expect = expected[gridIndex(dx, dy)];
            INFO("dx=" << dx << " dy=" << dy);
            // sample every pixel in the cell, not just the center, to catch any off-by-one
            for (int r = 0; r < W; r++) {
                for (int c = 0; c < W; c++) {
                    REQUIRE(mosaicPixel(*mosaic, rowOff + r, colOff + c) == expect);
                }
            }
        }
    }
}

TEST_CASE("Elevation mosaic - missing neighbors fall back to mirrored (not garbage/zero) data", "[RasterMosaic]") {
    const int W = 4;
    auto own = makeRampTexture(W);
    // own[r][c] = (r+1)*10 + c, i.e. rows: {10,11,12,13} {20,21,22,23} {30,31,32,33} {40,41,42,43}

    std::array<std::shared_ptr<Texture>, 9> neighbors;
    // A couple of real neighbors present (must be copied through untouched)...
    neighbors[gridIndex(0, 1)] = makeConstTexture(W, 77);   // S
    neighbors[gridIndex(-1, 0)] = makeConstTexture(W, 88);  // W
    // ...N, E, and NE deliberately left missing (nullptr) to exercise the mirror fallback.

    auto mosaic = stitchElevationMosaic(own, neighbors);
    REQUIRE(mosaic != nullptr);

    // Present neighbors copied through unchanged (S cell = bottom row band [0, W), since
    // buffer rows run south -> north).
    for (int r = 0; r < W; r++) {
        for (int c = 0; c < W; c++) {
            REQUIRE(mosaicPixel(*mosaic, r, W + c) == 77);  // S cell
            REQUIRE(mosaicPixel(*mosaic, W + r, c) == 88);  // W cell
        }
    }

    // N (dx=0, dy=-1, top row band [2W, 3W)): vertical mirror of own, no column flip --
    // cell row r == own row (W-1-r).
    for (int r = 0; r < W; r++) {
        for (int c = 0; c < W; c++) {
            uint8_t expect = uint8_t((W - 1 - r + 1) * 10 + c);
            uint8_t actual = mosaicPixel(*mosaic, 2 * W + r, W + c);
            REQUIRE(actual == expect);
            REQUIRE(actual != 0);  // not garbage/zero
        }
    }

    // E (dx=1, dy=0): horizontal mirror of own, no row flip -- cell col c == own col (W-1-c).
    for (int r = 0; r < W; r++) {
        for (int c = 0; c < W; c++) {
            uint8_t expect = uint8_t((r + 1) * 10 + (W - 1 - c));
            uint8_t actual = mosaicPixel(*mosaic, W + r, 2 * W + c);
            REQUIRE(actual == expect);
            REQUIRE(actual != 0);
        }
    }

    // NE (dx=1, dy=-1): mirrored both ways -- row (W-1-r), col (W-1-c) -- in the top-right cell.
    for (int r = 0; r < W; r++) {
        for (int c = 0; c < W; c++) {
            uint8_t expect = uint8_t((W - 1 - r + 1) * 10 + (W - 1 - c));
            uint8_t actual = mosaicPixel(*mosaic, 2 * W + r, 2 * W + c);
            REQUIRE(actual == expect);
            REQUIRE(actual != 0);
        }
    }
}

TEST_CASE("Elevation mosaic - node-registered (odd W) tiles stitch into a globally seamless field", "[RasterMosaic]") {
    // Odd-width tiles (e.g. real 257x257 elevation tiles) are node-registered: a tile's
    // outermost row/col duplicates the adjacent tile's edge. Model a global linear field
    // f(rowCoord, colCoord) sampled by a 3x3 block of such tiles: tile (dx, dy) covers
    // rowCoord = r - dy*(W-1), colCoord = c + dx*(W-1) (buffer rows run opposite to tile y,
    // and adjacent tiles share their edge node, hence the W-1 stride). A correct stitch must
    // reproduce f exactly across the whole mosaic interior - any orientation flip, cell
    // misplacement, or duplicated shared edge breaks the linear ramp somewhere.
    const int W = 5;
    auto f = [&](int rowCoord, int colCoord) { return uint8_t(100 + 10 * rowCoord + colCoord); };

    auto makeTile = [&](int dx, int dy) {
        auto tex = std::make_shared<Texture>(testOptions());
        std::vector<GLubyte> data(size_t(W) * size_t(W));
        for (int r = 0; r < W; r++) {
            for (int c = 0; c < W; c++) {
                data[size_t(r) * W + c] = f(r - dy * (W - 1), c + dx * (W - 1));
            }
        }
        tex->setPixelData(W, W, 1, data.data(), data.size());
        return tex;
    };

    auto own = makeTile(0, 0);
    std::array<std::shared_ptr<Texture>, 9> neighbors;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) { continue; }
            neighbors[gridIndex(dx, dy)] = makeTile(dx, dy);
        }
    }

    auto mosaic = stitchElevationMosaic(own, neighbors);
    REQUIRE(mosaic != nullptr);

    // Mosaic pixel (m, n) corresponds to (rowCoord, colCoord) = (m - W, n - W). The outermost
    // row/col on each far edge is a clamped duplicate (data past the neighbor ring doesn't
    // exist), so check the interior [1, 3W-1).
    for (int m = 1; m < 3 * W - 1; m++) {
        for (int n = 1; n < 3 * W - 1; n++) {
            INFO("mosaic row " << m << " col " << n);
            REQUIRE(mosaicPixel(*mosaic, m, n) == f(m - W, n - W));
        }
    }
}

TEST_CASE("Elevation mosaic - refuses to build without CPU buffer data or a square texture", "[RasterMosaic]") {
    std::array<std::shared_ptr<Texture>, 9> neighbors;

    SECTION("own texture with no pixel data set") {
        auto own = std::make_shared<Texture>(testOptions());
        REQUIRE(stitchElevationMosaic(own, neighbors) == nullptr);
    }

    SECTION("non-square own texture") {
        auto own = std::make_shared<Texture>(testOptions());
        std::vector<GLubyte> data(4 * 5, 1);
        own->setPixelData(4, 5, 1, data.data(), data.size());
        REQUIRE(stitchElevationMosaic(own, neighbors) == nullptr);
    }

    SECTION("null own texture") {
        REQUIRE(stitchElevationMosaic(nullptr, neighbors) == nullptr);
    }
}
