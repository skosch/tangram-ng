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
    // outermost row/col duplicates the adjacent tile's edge, and each mosaic cell is the
    // tile cropped to Wp = W-1 rows/cols (so the mosaic is 3*(W-1) wide and stays
    // mip-grid-aligned with neighboring tiles' mosaics). Model a global linear field
    // f(rowCoord, colCoord) sampled by a 3x3 block of such tiles: tile (dx, dy) covers
    // rowCoord = r - dy*(W-1), colCoord = c + dx*(W-1) (buffer rows run opposite to tile y,
    // and adjacent tiles share their edge node, hence the W-1 stride). A correct stitch must
    // reproduce f exactly across the ENTIRE mosaic - any orientation flip, cell
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

    // Cells are cropped to Wp = W-1, so the mosaic is 3*(W-1) wide...
    const int Wp = W - 1;
    REQUIRE(mosaic->width() == 3 * Wp);
    REQUIRE(mosaic->height() == 3 * Wp);

    // ...and pixel (m, n) corresponds to (rowCoord, colCoord) = (m - Wp, n - Wp), exact over
    // the entire mosaic (every pixel is a real sample of some tile - no clamped duplicates).
    for (int m = 0; m < 3 * Wp; m++) {
        for (int n = 0; n < 3 * Wp; n++) {
            INFO("mosaic row " << m << " col " << n);
            REQUIRE(mosaicPixel(*mosaic, m, n) == f(m - Wp, n - Wp));
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

// --- stitchElevationOverzoomMosaic (docs/terrain-illumination-plan.md "raster-zoom mismatch") ---
//
// Unlike stitchElevationMosaic()'s 3x3 neighbor RING (own centered, 8 same-zoom neighbors
// around it), this stitches an N x N grid of FINER descendant tiles that together cover the
// exact same [0,1] footprint as `own` -- a direct higher-resolution replacement, not a padded
// neighbor mosaic. `children` is row-major in TILE coordinate order: children[row*N+col], col
// west->east, row north->south (matching child TileID x/y order) -- see the doc comment on the
// function.

uint8_t compositePixel(const Texture& composite, int row, int col) {
    return composite.bufferData()[size_t(row) * size_t(composite.width()) + size_t(col)];
}

TEST_CASE("Overzoom mosaic - dimensions and real children land in the correct grid cell", "[RasterMosaic]") {
    const int W = 4;
    const int N = 2;
    auto own = makeConstTexture(W, 200);  // must never show through: every cell is a real child

    // children[row*N+col]: (row=0,col=0)=NW, (row=0,col=1)=NE, (row=1,col=0)=SW, (row=1,col=1)=SE
    std::vector<std::shared_ptr<Texture>> children(N * N);
    children[0 * N + 0] = makeConstTexture(W, 1);   // NW
    children[0 * N + 1] = makeConstTexture(W, 2);   // NE
    children[1 * N + 0] = makeConstTexture(W, 3);   // SW
    children[1 * N + 1] = makeConstTexture(W, 4);   // SE

    auto composite = stitchElevationOverzoomMosaic(own, children, N);
    REQUIRE(composite != nullptr);

    // W is even (no node-registration crop), so Wp = W and the composite is exactly N*W wide.
    REQUIRE(composite->width() == N * W);
    REQUIRE(composite->height() == N * W);

    // row=0 (north) children land in the TOP band [Wp, 2Wp) of the composite -- buffer rows run
    // south -> north like stitchElevationMosaic(), opposite to tile y -- and col=0 (west) lands
    // in the LEFT band [0, Wp), matching normal west->east column order.
    for (int r = 0; r < W; r++) {
        for (int c = 0; c < W; c++) {
            REQUIRE(compositePixel(*composite, W + r, c) == 1);          // NW
            REQUIRE(compositePixel(*composite, W + r, W + c) == 2);      // NE
            REQUIRE(compositePixel(*composite, r, c) == 3);              // SW
            REQUIRE(compositePixel(*composite, r, W + c) == 4);          // SE
        }
    }
}

TEST_CASE("Overzoom mosaic - missing children fall back to nearest-neighbor upsample of own, not garbage", "[RasterMosaic]") {
    const int W = 2;
    const int N = 2;
    // own[r][c] = (r+1)*10 + c -- own(0,0)=10, own(0,1)=11, own(1,0)=20, own(1,1)=21. Buffer
    // rows run south -> north (row 0 = south), so own row 1 is the "northern" half, row 0 the
    // "southern" half, matching stitchElevationMosaic()'s convention.
    auto own = makeRampTexture(W);

    std::vector<std::shared_ptr<Texture>> children(N * N);  // all missing -> full fallback

    auto composite = stitchElevationOverzoomMosaic(own, children, N);
    REQUIRE(composite != nullptr);
    REQUIRE(composite->width() == N * W);
    REQUIRE(composite->height() == N * W);

    // Every fallback-filled quadrant should be a CONSTANT nearest-neighbor sample of own (W=2,
    // N=2 means each quadrant's whole Wp x Wp footprint maps to a single own texel), and it
    // must be the geographically-correct texel: the NW quadrant of the composite should sample
    // own's own north-west corner, etc.
    for (int r = 0; r < W; r++) {
        for (int c = 0; c < W; c++) {
            REQUIRE(compositePixel(*composite, W + r, c) == 20);          // NW <- own(row=1,col=0)
            REQUIRE(compositePixel(*composite, W + r, W + c) == 21);      // NE <- own(row=1,col=1)
            REQUIRE(compositePixel(*composite, r, c) == 10);              // SW <- own(row=0,col=0)
            REQUIRE(compositePixel(*composite, r, W + c) == 11);          // SE <- own(row=0,col=1)
        }
    }
}

TEST_CASE("Overzoom mosaic - partial availability: present cells are real, missing cells fall back", "[RasterMosaic]") {
    const int W = 2;
    const int N = 2;
    auto own = makeRampTexture(W);  // own(0,0)=10, own(0,1)=11, own(1,0)=20, own(1,1)=21

    std::vector<std::shared_ptr<Texture>> children(N * N);
    children[0 * N + 0] = makeConstTexture(W, 99);  // NW is real data
    // NE, SW, SE deliberately left null -> mirrored fallback from own

    auto composite = stitchElevationOverzoomMosaic(own, children, N);
    REQUIRE(composite != nullptr);

    for (int r = 0; r < W; r++) {
        for (int c = 0; c < W; c++) {
            REQUIRE(compositePixel(*composite, W + r, c) == 99);          // NW: real child data
            REQUIRE(compositePixel(*composite, W + r, W + c) == 21);      // NE: fallback <- own(1,1)
            REQUIRE(compositePixel(*composite, r, c) == 10);              // SW: fallback <- own(0,0)
            REQUIRE(compositePixel(*composite, r, W + c) == 11);          // SE: fallback <- own(0,1)
        }
    }
}

TEST_CASE("Overzoom mosaic - a child with mismatched dimensions is treated as missing (fallback, not garbage)", "[RasterMosaic]") {
    const int W = 4;
    const int N = 2;
    auto own = makeConstTexture(W, 55);

    std::vector<std::shared_ptr<Texture>> children(N * N);
    children[0 * N + 0] = makeConstTexture(W - 1, 1);  // wrong size -- must be rejected, not copied

    auto composite = stitchElevationOverzoomMosaic(own, children, N);
    REQUIRE(composite != nullptr);

    // Falls back to own (constant 55) rather than reading out-of-bounds or copying mismatched data.
    for (int r = 0; r < W; r++) {
        for (int c = 0; c < W; c++) {
            REQUIRE(compositePixel(*composite, W + r, c) == 55);
        }
    }
}

TEST_CASE("Overzoom mosaic - node-registered (odd W) own texture crops cells to Wp = W - 1", "[RasterMosaic]") {
    const int W = 5;
    const int N = 2;
    auto own = makeConstTexture(W, 7);
    std::vector<std::shared_ptr<Texture>> children(N * N);  // all fallback from own

    auto composite = stitchElevationOverzoomMosaic(own, children, N);
    REQUIRE(composite != nullptr);

    const int Wp = W - 1;
    REQUIRE(composite->width() == N * Wp);
    REQUIRE(composite->height() == N * Wp);
}

TEST_CASE("Overzoom mosaic - refuses invalid input", "[RasterMosaic]") {
    const int W = 4;
    const int N = 2;

    SECTION("null own texture") {
        std::vector<std::shared_ptr<Texture>> children(N * N);
        REQUIRE(stitchElevationOverzoomMosaic(nullptr, children, N) == nullptr);
    }

    SECTION("N <= 1 (no overzoom to compensate for)") {
        auto own = makeConstTexture(W, 1);
        std::vector<std::shared_ptr<Texture>> children(1);
        REQUIRE(stitchElevationOverzoomMosaic(own, children, 1) == nullptr);
        REQUIRE(stitchElevationOverzoomMosaic(own, children, 0) == nullptr);
    }

    SECTION("children.size() != N*N") {
        auto own = makeConstTexture(W, 1);
        std::vector<std::shared_ptr<Texture>> children(N * N - 1);
        REQUIRE(stitchElevationOverzoomMosaic(own, children, N) == nullptr);
    }

    SECTION("own texture with no pixel data set") {
        auto own = std::make_shared<Texture>(testOptions());
        std::vector<std::shared_ptr<Texture>> children(N * N);
        REQUIRE(stitchElevationOverzoomMosaic(own, children, N) == nullptr);
    }

    SECTION("non-square own texture") {
        auto own = std::make_shared<Texture>(testOptions());
        std::vector<GLubyte> data(4 * 5, 1);
        own->setPixelData(4, 5, 1, data.data(), data.size());
        std::vector<std::shared_ptr<Texture>> children(N * N);
        REQUIRE(stitchElevationOverzoomMosaic(own, children, N) == nullptr);
    }
}
