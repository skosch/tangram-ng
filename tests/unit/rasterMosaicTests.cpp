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
// around it), this stitches an N x N grid of cells covering the exact same [0,1] footprint as
// `own` -- a direct higher-resolution replacement, not a padded neighbor mosaic. `cells` is
// row-major in TILE coordinate order: cells[row*N+col], col west->east, row north->south
// (matching child TileID x/y order) -- see OverzoomCell's doc comment.

uint8_t compositePixel(const Texture& composite, int row, int col) {
    return composite.bufferData()[size_t(row) * size_t(composite.width()) + size_t(col)];
}

// An exact same-target-zoom cell: `texture` IS the real data for grid position (col, row).
OverzoomCell exactCell(std::shared_ptr<Texture> texture, int col, int row) {
    OverzoomCell cell;
    cell.texture = std::move(texture);
    cell.blockColOriginWest = col;
    cell.blockRowOriginNorth = row;
    cell.blockSize = 1;
    return cell;
}

// A cell with no source at all named -- stitchElevationOverzoomMosaic falls back to PL-upsampling
// `own` itself over the whole footprint for it.
OverzoomCell missingCell() { return OverzoomCell{}; }

TEST_CASE("Overzoom mosaic - dimensions and real children land in the correct grid cell", "[RasterMosaic]") {
    const int W = 4;
    const int N = 2;
    auto own = makeConstTexture(W, 200);  // must never show through: every cell is a real child

    // cells[row*N+col]: (row=0,col=0)=NW, (row=0,col=1)=NE, (row=1,col=0)=SW, (row=1,col=1)=SE
    std::vector<OverzoomCell> cells(N * N);
    cells[0 * N + 0] = exactCell(makeConstTexture(W, 1), 0, 0);   // NW
    cells[0 * N + 1] = exactCell(makeConstTexture(W, 2), 1, 0);   // NE
    cells[1 * N + 0] = exactCell(makeConstTexture(W, 3), 0, 1);   // SW
    cells[1 * N + 1] = exactCell(makeConstTexture(W, 4), 1, 1);   // SE

    auto composite = stitchElevationOverzoomMosaic(own, cells, N);
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

TEST_CASE("Overzoom mosaic - missing children fall back to triangle-PL upsample of own's terrain-mesh "
          "surface, not garbage", "[RasterMosaic]") {
    const int W = 2;
    const int N = 2;
    // own[r][c] = (r+1)*10 + c -- own(0,0)=10, own(0,1)=11, own(1,0)=20, own(1,1)=21. Buffer
    // rows run south -> north (row 0 = south), so own row 1 is the "northern" half, row 0 the
    // "southern" half, matching stitchElevationMosaic()'s convention. own is far coarser than
    // the fixed 64-cell lattice fillLatticeUpsampledCell evaluates, so every sampled composite
    // texel below falls deep inside one quadrant (not on the exact lattice split at u=v=0.5) and
    // the PL surface there is flat, degenerating to exactly the same value nearest-neighbor
    // sampling would have given -- this test is really about the fallback still reaching the
    // geographically-correct quadrant, not about literal nearest-vs-PL distinction (see the
    // dedicated PL-exactness tests below for that).
    auto own = makeRampTexture(W);

    std::vector<OverzoomCell> cells(N * N, missingCell());  // all missing -> full fallback

    auto composite = stitchElevationOverzoomMosaic(own, cells, N);
    REQUIRE(composite != nullptr);
    REQUIRE(composite->width() == N * W);
    REQUIRE(composite->height() == N * W);

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

    std::vector<OverzoomCell> cells(N * N, missingCell());
    cells[0 * N + 0] = exactCell(makeConstTexture(W, 99), 0, 0);  // NW is real data
    // NE, SW, SE deliberately left missing -> PL fallback from own

    auto composite = stitchElevationOverzoomMosaic(own, cells, N);
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

    std::vector<OverzoomCell> cells(N * N, missingCell());
    cells[0 * N + 0] = exactCell(makeConstTexture(W - 1, 1), 0, 0);  // wrong size -- must be rejected, not copied

    auto composite = stitchElevationOverzoomMosaic(own, cells, N);
    REQUIRE(composite != nullptr);

    // Falls back to own (constant 55) rather than reading out-of-bounds or copying mismatched data.
    for (int r = 0; r < W; r++) {
        for (int c = 0; c < W; c++) {
            REQUIRE(compositePixel(*composite, W + r, c) == 55);
        }
    }
}

TEST_CASE("Overzoom mosaic - node-registered (odd W) own texture: interior cells crop to Wp = W - 1, "
          "composite keeps the far (north/east) edge line", "[RasterMosaic]") {
    const int W = 5;
    const int N = 2;
    auto own = makeConstTexture(W, 7);
    std::vector<OverzoomCell> cells(N * N, missingCell());  // all fallback from own

    auto composite = stitchElevationOverzoomMosaic(own, cells, N);
    REQUIRE(composite != nullptr);

    // Composite width is (N-1)*Wp + W, NOT N*Wp: unlike stitchElevationMosaic()'s neighbor-ring
    // mosaic (where every cell borders another real cell on all sides), this composite's own
    // outer boundary has no further cell beyond it, so its north-most/east-most band keeps the
    // full W (not cropped to Wp) to preserve the true world-1.0 edge sample -- see
    // stitchElevationOverzoomMosaic's doc comment ("missing far-edge line" fix).
    const int Wp = W - 1;
    REQUIRE(composite->width() == (N - 1) * Wp + W);
    REQUIRE(composite->height() == (N - 1) * Wp + W);
}

TEST_CASE("Overzoom mosaic - real children: the composite's far (north/east) edge is the true "
          "child edge sample, not one texel short of it", "[RasterMosaic]") {
    // Node-registered (odd W) children with distinguishable per-tile ramps, so the true edge
    // texel (child-local column/row W-1) is distinct from the next-to-edge one (W-2) -- proves
    // the composite's outer boundary is the REAL edge sample, not a cropped/dropped one.
    const int W = 5;
    const int N = 2;
    auto own = makeConstTexture(W, 200);  // must never show through: every cell is a real child

    auto makeChild = [&](uint8_t base) {
        auto tex = std::make_shared<Texture>(testOptions());
        std::vector<GLubyte> data(size_t(W) * size_t(W));
        for (int r = 0; r < W; r++) {
            for (int c = 0; c < W; c++) {
                data[size_t(r) * W + c] = uint8_t(base + r * 10 + c);
            }
        }
        tex->setPixelData(W, W, 1, data.data(), data.size());
        return tex;
    };

    std::vector<OverzoomCell> cells(N * N);
    cells[0 * N + 0] = exactCell(makeChild(1), 0, 0);      // NW (row=0, col=0)
    cells[0 * N + 1] = exactCell(makeChild(101), 1, 0);    // NE (row=0, col=N-1) -- north- AND east-most
    cells[1 * N + 0] = exactCell(makeChild(51), 0, 1);     // SW (row=1, col=0)
    cells[1 * N + 1] = exactCell(makeChild(151), 1, 1);    // SE (row=1, col=N-1) -- east-most

    auto composite = stitchElevationOverzoomMosaic(own, cells, N);
    REQUIRE(composite != nullptr);

    const int Wp = W - 1;
    const int compositeW = (N - 1) * Wp + W;
    REQUIRE(composite->width() == compositeW);

    // NE cell (row=0, col=N-1) occupies the composite's absolute top-right corner. Its own
    // top-right texel (child-local row=W-1, col=W-1) is the true world corner (north AND east
    // edge at once) and must land at the composite's own top-right corner, unmodified.
    uint8_t expectedCorner = uint8_t(101 + (W - 1) * 10 + (W - 1));
    REQUIRE(compositePixel(*composite, compositeW - 1, compositeW - 1) == expectedCorner);

    // SE cell (row=1, col=N-1): east-most but NOT north-most -- its far EAST column (child-local
    // col=W-1) must be present at the composite's east edge, at composite row Wp - 1 (SE cell's
    // band is rows [0, Wp), cropped in height since row=1 isn't the north-most band; local row
    // Wp-1 is its highest kept row).
    uint8_t expectedE = uint8_t(151 + (Wp - 1) * 10 + (W - 1));
    REQUIRE(compositePixel(*composite, Wp - 1, compositeW - 1) == expectedE);

    // NW cell (row=0, col=0): north-most but NOT east-most -- its far NORTH row (child-local
    // row=W-1) must be present at the composite's north edge, at composite col Wp - 1 (NW cell's
    // band is cols [0, Wp), cropped in width since col=0 isn't the east-most band).
    uint8_t expectedN = uint8_t(1 + (W - 1) * 10 + (Wp - 1));
    REQUIRE(compositePixel(*composite, compositeW - 1, Wp - 1) == expectedN);
}

TEST_CASE("Overzoom mosaic - refuses invalid input", "[RasterMosaic]") {
    const int W = 4;
    const int N = 2;

    SECTION("null own texture") {
        std::vector<OverzoomCell> cells(N * N);
        REQUIRE(stitchElevationOverzoomMosaic(nullptr, cells, N) == nullptr);
    }

    SECTION("N <= 1 (no overzoom to compensate for)") {
        auto own = makeConstTexture(W, 1);
        std::vector<OverzoomCell> cells(1);
        REQUIRE(stitchElevationOverzoomMosaic(own, cells, 1) == nullptr);
        REQUIRE(stitchElevationOverzoomMosaic(own, cells, 0) == nullptr);
    }

    SECTION("cells.size() != N*N") {
        auto own = makeConstTexture(W, 1);
        std::vector<OverzoomCell> cells(N * N - 1);
        REQUIRE(stitchElevationOverzoomMosaic(own, cells, N) == nullptr);
    }

    SECTION("own texture with no pixel data set") {
        auto own = std::make_shared<Texture>(testOptions());
        std::vector<OverzoomCell> cells(N * N);
        REQUIRE(stitchElevationOverzoomMosaic(own, cells, N) == nullptr);
    }

    SECTION("non-square own texture") {
        auto own = std::make_shared<Texture>(testOptions());
        std::vector<GLubyte> data(4 * 5, 1);
        own->setPixelData(4, 5, 1, data.data(), data.size());
        std::vector<OverzoomCell> cells(N * N);
        REQUIRE(stitchElevationOverzoomMosaic(own, cells, N) == nullptr);
    }
}

// --- Triangle-PL upsampling exactness (docs/terrain-illumination-plan.md "triangle-PL upsampling")
//
// The whole point of replacing nearest-neighbor fallback fill with triangle-PL upsampling is that
// a fine lattice with the SAME anti-diagonal split nests exactly inside a coarser one, so a
// fallback-filled region reproduces the coarser real tile's own terrain-mesh surface EXACTLY, not
// just approximately. These tests use FLOAT elevation textures (real production pixel format) with
// W = 64 exactly, so nearest-sampling the 65-node lattice maps node index i (0..63) to texel i
// with NO quantization (see buildLatticeNodes: floor(i/64*64) == i) -- letting a texel-linear field
// value(row, col) = A + B*row + C*col be reproduced by PL interpolation EXACTLY (not merely
// approximately) at any interior fractional lattice position, since the PL surface of an exactly
// linear node field equals that field everywhere regardless of which way the anti-diagonal split
// runs. This isolates the fill's own geometry (world-position -> ancestor-local (u,v) mapping,
// axis orientation, block-origin math) from lattice-quantization noise.

TextureOptions floatOptions() {
    TextureOptions options;
    options.pixelFormat = PixelFormat::FLOAT;
    return options;
}

// value(row, col) = A + B*row + C*col, stored as float, row 0 = south (matching every other raw
// elevation buffer's convention in this file).
std::shared_ptr<Texture> makeLinearFloatTexture(int W, double A, double B, double C) {
    auto tex = std::make_shared<Texture>(floatOptions());
    std::vector<float> data(size_t(W) * size_t(W));
    for (int r = 0; r < W; r++) {
        for (int c = 0; c < W; c++) {
            data[size_t(r) * size_t(W) + size_t(c)] = float(A + B * double(r) + C * double(c));
        }
    }
    tex->setPixelData(W, W, 4, reinterpret_cast<const GLubyte*>(data.data()), data.size() * sizeof(float));
    return tex;
}

float compositeFloatPixel(const Texture& composite, int row, int col) {
    const float* data = reinterpret_cast<const float*>(composite.bufferData());
    return data[size_t(row) * size_t(composite.width()) + size_t(col)];
}

TEST_CASE("Overzoom mosaic - triangle-PL fallback fill reproduces a coarser real ancestor's own "
          "terrain-mesh surface exactly, not a flat nearest-neighbor block", "[RasterMosaic][PLUpsample]") {
    const int W = 64;    // 1:1 node<->texel correspondence, see file-level comment above
    const int N = 2;
    auto own = makeLinearFloatTexture(W, 0.0, 0.0, 0.0);  // dimension reference (Wp/compositeW derive from own->width()); FLOAT format to match the real cells below (a composite always inherits own->getOptions())

    // A single ancestor tile (blockSize == N, covering the WHOLE footprint, exactly like the
    // "own"-as-ultimate-fallback path but exercised explicitly via a distinct ancestor texture)
    // with a texel-linear field distinguishable per axis (B != C).
    const double A = 1000.0, B = 3.0, C = 0.25;
    auto ancestor = makeLinearFloatTexture(W, A, B, C);

    std::vector<OverzoomCell> cells(N * N);
    for (int row = 0; row < N; row++) {
        for (int col = 0; col < N; col++) {
            OverzoomCell& cell = cells[size_t(row) * size_t(N) + size_t(col)];
            cell.texture = ancestor;
            cell.blockColOriginWest = 0;
            cell.blockRowOriginNorth = 0;
            cell.blockSize = N;
        }
    }

    auto composite = stitchElevationOverzoomMosaic(own, cells, N);
    REQUIRE(composite != nullptr);
    REQUIRE(composite->getOptions().pixelFormat == PixelFormat::FLOAT);

    int compW = composite->width();
    // Sample a handful of interior points (avoiding the very last row/col, which the W=64 -> 63
    // clamp turns non-linear) and check exact reproduction of the linear field at the
    // corresponding ancestor-local position: composite texel m's ancestor-local coordinate along
    // an axis of extent compW is m / (compW - 1) * 64 (world fraction * lattice resolution, since
    // blockSize == N here means "ancestor spans the whole footprint").
    auto expected = [&](int row, int col) {
        double v = double(row) / double(compW - 1) * 64.0;  // south-based, matches value()'s row
        double u = double(col) / double(compW - 1) * 64.0;
        return float(A + B * v + C * u);
    };

    for (int row : {0, compW / 4, compW / 2, 3 * compW / 4}) {
        for (int col : {0, compW / 4, compW / 2, 3 * compW / 4}) {
            INFO("row=" << row << " col=" << col << " compW=" << compW);
            REQUIRE(compositeFloatPixel(*composite, row, col) == Approx(expected(row, col)).margin(1e-3));
        }
    }
}

TEST_CASE("Overzoom mosaic - mixed-zoom coverage: a real exact cell and a triangle-PL-filled "
          "coarser-ancestor cell coexist correctly in the same composite", "[RasterMosaic][PLUpsample]") {
    // This is the actual "intra-footprint zoom variation" scenario the composite exists to solve
    // (docs/terrain-illumination-plan.md): under a tilted camera, one coarser vector tile's
    // footprint can have some regions where the terrain tileset renders the FINEST zoom (real
    // exact data available) and others where it only ever reaches one zoom coarser (a real
    // ancestor tile, PL-upsampled) -- both must appear correctly, at the right positions, in ONE
    // composite, rather than the whole composite being forced to one uniform "achieved" level.
    const int W = 64;
    const int N = 4;  // 2 levels of overzoom: a 4x4 grid of exact-zoom cells
    auto own = makeLinearFloatTexture(W, 0.0, 0.0, 0.0);  // dimension reference only; FLOAT format to match the real cells below

    const double A = 500.0, B = 2.0, C = 5.0;
    auto ancestorNE = makeLinearFloatTexture(W, A, B, C);  // covers the NE 2x2 block (blockSize=2)
    auto exactSW = makeLinearFloatTexture(W, 42.0, 0.0, 0.0);  // one real exact cell, constant

    std::vector<OverzoomCell> cells(N * N, missingCell());
    // NE 2x2 block (grid cols 2-3, rows 0-1) all fall back to one coarser real ancestor.
    for (int row = 0; row < 2; row++) {
        for (int col = 2; col < 4; col++) {
            OverzoomCell& cell = cells[size_t(row) * size_t(N) + size_t(col)];
            cell.texture = ancestorNE;
            cell.blockColOriginWest = 2;
            cell.blockRowOriginNorth = 0;
            cell.blockSize = 2;
        }
    }
    // One real, exact-zoom cell at grid (col=0, row=3) -- the SW-most cell.
    cells[3 * N + 0] = exactCell(exactSW, 0, 3);

    auto composite = stitchElevationOverzoomMosaic(own, cells, N);
    REQUIRE(composite != nullptr);
    int compW = composite->width();
    int Wp = W;  // W=64 is even -> no node-registration crop

    // The exact cell (grid col=0, row=3 -- south-west-most) occupies composite band
    // rows [0, Wp), cols [0, Wp): must be byte-for-byte the ancestor's own constant value (42),
    // NOT any PL-interpolated approximation.
    REQUIRE(compositeFloatPixel(*composite, 0, 0) == Approx(42.0).margin(1e-3));
    REQUIRE(compositeFloatPixel(*composite, Wp - 1, Wp - 1) == Approx(42.0).margin(1e-3));

    // The NE-block cells (grid cols 2-3, rows 0-1) occupy composite band rows [2*Wp, 4*Wp) x cols
    // [2*Wp, 4*Wp) (row=0 is the north-most grid row -> top band of the composite, independently
    // verified by the "dimensions and real children land in correct grid cell" tests above via
    // the pre-existing, unchanged copyOverzoomGridCell band placement): must reproduce
    // ancestorNE's linear field exactly at the corresponding ancestor-local position. Ancestor-
    // local (u, v) uses the SAME node-registered (compositeW - 1) world convention as
    // fillLatticeUpsampledCell itself (docs/terrain-illumination-plan.md contract point 6:
    // composite texel m sits at world m / (width - 1)) -- NOT a naive Wp-band fraction, which is
    // off by the sub-percent but non-negligible compositeW vs. (compositeW - 1) difference.
    auto expectedNE = [&](int row, int col) {
        double southFracGlobal = double(row) / double(compW - 1);
        double westFracGlobal = double(col) / double(compW - 1);
        double blockRowOriginSouth = double(N) - /*blockRowOriginNorth*/ 0.0 - /*blockSize*/ 2.0;
        double vLocal = (southFracGlobal * double(N) - blockRowOriginSouth) / 2.0;
        double uLocal = (westFracGlobal * double(N) - /*blockColOriginWest*/ 2.0) / 2.0;
        return float(A + B * (vLocal * 64.0) + C * (uLocal * 64.0));
    };
    for (int row : {2 * Wp, 3 * Wp - 1}) {
        for (int col : {2 * Wp, 3 * Wp - 1}) {
            INFO("row=" << row << " col=" << col);
            REQUIRE(compositeFloatPixel(*composite, row, col) == Approx(expectedNE(row, col)).margin(1e-2));
        }
    }
}

// --- GOLD STANDARD: node-for-node drape-vs-terrain equality (2026-07 white-patch audit) ---
//
// The drape (getTerrainMeshElevation, elevation.yaml) and the terrain mesh (RasterStyle 64-cell
// lattice + `filtering: nearest`) must read the IDENTICAL elevation value at every lattice node,
// or draped land cover floats above/below the rendered terrain (the "white patch" symptom).
// These tests simulate both shader reads bit-exactly, on production-shaped node-registered
// W=257 float tiles (so the floor(i/64*W) nearest-quantization is really exercised), against a
// composite stitched exactly the way RasterSource::buildOverzoomElevationMosaic does it:
//  - terrain read: rendered tile's own 64-lattice, node (i,j) -> texel floor(i/64*W) clamped;
//  - drape read (OFF path): composite attached directly, cells = 64*(compositeW-1)/256,
//    node (k,l) -> texel floor(k/cells * compositeW) clamped;
//  - drape read (ON path): composite embedded as a 3Wp x 3Wp mosaic center
//    (stitchElevationMosaic), uv remapped by elevationMosaicUV with ELEVATION_TILE_OVERLAP=1
//    and snapped to the texel center (sampleElevationTex's snap).
// Synthetic data comes from a GLOBAL coordinate function so shared node-registered tile edges
// agree across children exactly as real-world data does. Values are small integers, exactly
// representable in float, and deliberately non-planar (catches diagonal/axis mixups that a
// linear field would forgive).

namespace {

const int kGoldW = 257;   // production node-registered tile width (256 + 1 shared line)

// Non-planar, integer-exact global field; (R, C) are global sample indices (row south-based).
double goldField(int R, int C) {
    return 1000.0 + double(R) * 3.0 + double(C) * 7.0 + double((R * C) % 13);
}

// Child tile at grid (col, rowNorth) of an N x N zt-grid: child-local sample (r, c) sits at
// global sample (rowSouthBand * (W-1) + r, col * (W-1) + c) -- node-registered tiles share
// their boundary lines, so adjacent children agree on them by construction.
std::shared_ptr<Texture> makeGoldChild(int N, int col, int rowNorth) {
    auto tex = std::make_shared<Texture>(floatOptions());
    std::vector<float> data(size_t(kGoldW) * size_t(kGoldW));
    int rowSouth = N - 1 - rowNorth;
    for (int r = 0; r < kGoldW; r++) {
        for (int c = 0; c < kGoldW; c++) {
            data[size_t(r) * kGoldW + c] =
                float(goldField(rowSouth * (kGoldW - 1) + r, col * (kGoldW - 1) + c));
        }
    }
    tex->setPixelData(kGoldW, kGoldW, 4, reinterpret_cast<const GLubyte*>(data.data()),
                      data.size() * sizeof(float));
    return tex;
}

// The parent (own) tile of that N x N grid: its sample (r, c) covers the same world span, so it
// sits at global position (r*N, c*N) in child-sample units -- i.e. the parent's own field is
// goldField evaluated on an N-times-coarser world grid. (Values need NOT be consistent with the
// children for the fallback test -- the drape must reproduce the PARENT's surface there.)
std::shared_ptr<Texture> makeGoldParent(int N) {
    auto tex = std::make_shared<Texture>(floatOptions());
    std::vector<float> data(size_t(kGoldW) * size_t(kGoldW));
    for (int r = 0; r < kGoldW; r++) {
        for (int c = 0; c < kGoldW; c++) {
            data[size_t(r) * kGoldW + c] = float(goldField(r * N, c * N));
        }
    }
    tex->setPixelData(kGoldW, kGoldW, 4, reinterpret_cast<const GLubyte*>(data.data()),
                      data.size() * sizeof(float));
    return tex;
}

// Terrain mesh vertex read: RasterStyle's fixed 64-cell lattice node (i west->east, j
// south->north in buffer-row order) of one rendered tile, GL_NEAREST + CLAMP_TO_EDGE.
float terrainNodeRead(const Texture& tile, int i, int j) {
    int W = tile.width();
    int col = std::min(W - 1, int(double(i) / 64.0 * double(W)));
    int row = std::min(W - 1, int(double(j) / 64.0 * double(W)));
    return reinterpret_cast<const float*>(tile.bufferData())[size_t(row) * W + col];
}

// Drape lattice-node read, texture-shading OFF: composite attached directly at the vector
// tile's own footprint (u_raster_offsets identity), getTerrainMeshElevation's node uv k/cells,
// GL_NEAREST (`filtering: nearest` is inherited by the composite).
float drapeNodeRead(const Texture& composite, int k, int l, int cells) {
    int W = composite.width();
    int col = std::min(W - 1, int(double(k) / double(cells) * double(W)));
    int row = std::min(W - 1, int(double(l) / double(cells) * double(W)));
    return reinterpret_cast<const float*>(composite.bufferData())[size_t(row) * W + col];
}

// Drape lattice-node read, texture-shading ON: the composite is the CENTER cell of a 3Wp x 3Wp
// neighbor mosaic; the shader remaps tile-local uv via elevationMosaicUV (ELEVATION_TILE_OVERLAP
// = 1) then snaps to the texel center (sampleElevationTex) -- the snap makes the LINEAR-filtered
// mosaic read the exact texel value, simulated here as the plain floor.
float drapeMosaicNodeRead(const Texture& mosaic, int k, int l, int cells) {
    int msize = mosaic.width();
    double Wp = double(msize) / 3.0;
    double fu = (double(k) / double(cells) * (Wp + 1.0) + Wp) / double(msize);
    double fv = (double(l) / double(cells) * (Wp + 1.0) + Wp) / double(msize);
    int col = std::min(msize - 1, int(fu * double(msize)));
    int row = std::min(msize - 1, int(fv * double(msize)));
    return reinterpret_cast<const float*>(mosaic.bufferData())[size_t(row) * msize + col];
}

// Reference implementation of the shader's PL interpolation (getTerrainMeshElevation /
// rasterStyle.cpp anti-diagonal split) over one rendered tile's 65x65 terrain-node grid.
double terrainSurfaceRef(const Texture& tile, double u, double v) {
    double stx = std::min(1.0, std::max(0.0, u)) * 64.0;
    double sty = std::min(1.0, std::max(0.0, v)) * 64.0;
    int cx = std::min(63, int(stx));
    int cy = std::min(63, int(sty));
    double fx = stx - cx, fy = sty - cy;
    double h00 = terrainNodeRead(tile, cx, cy);
    double h10 = terrainNodeRead(tile, cx + 1, cy);
    double h01 = terrainNodeRead(tile, cx, cy + 1);
    double h11 = terrainNodeRead(tile, cx + 1, cy + 1);
    return (fx + fy <= 1.0) ? h00 + fx * (h10 - h00) + fy * (h01 - h00)
                            : h11 + (1.0 - fx) * (h01 - h11) + (1.0 - fy) * (h10 - h11);
}

} // namespace

TEST_CASE("Gold standard - direct-copy composite: drape lattice nodes read the IDENTICAL value "
          "the rendered child tiles' own terrain meshes read", "[RasterMosaic][GoldStandard]") {
    const int N = 2;
    auto own = makeGoldParent(N);

    std::vector<OverzoomCell> cells(N * N);
    std::shared_ptr<Texture> children[2][2];  // [rowNorth][col]
    for (int rowN = 0; rowN < N; rowN++) {
        for (int col = 0; col < N; col++) {
            children[rowN][col] = makeGoldChild(N, col, rowN);
            cells[size_t(rowN) * N + col] = exactCell(children[rowN][col], col, rowN);
        }
    }

    auto composite = stitchElevationOverzoomMosaic(own, cells, N);
    REQUIRE(composite != nullptr);
    const int compositeW = (N - 1) * (kGoldW - 1) + kGoldW;   // 513
    REQUIRE(composite->width() == compositeW);
    const int cells64 = 64 * (compositeW - 1) / 256;          // shader: 64 * effectiveWp / 256
    REQUIRE(cells64 == 128);

    int mismatches = 0;
    int firstI = -1, firstJ = -1, firstCol = -1, firstRow = -1;
    float firstTerrain = 0, firstDrape = 0;
    for (int rowN = 0; rowN < N; rowN++) {
        int bandSouth = N - 1 - rowN;
        for (int col = 0; col < N; col++) {
            for (int j = 0; j <= 64; j++) {
                for (int i = 0; i <= 64; i++) {
                    float terrain = terrainNodeRead(*children[rowN][col], i, j);
                    int k = 64 * col + i;
                    int l = 64 * bandSouth + j;
                    float drape = drapeNodeRead(*composite, k, l, cells64);
                    if (terrain != drape) {
                        if (mismatches == 0) {
                            firstI = i; firstJ = j; firstCol = col; firstRow = rowN;
                            firstTerrain = terrain; firstDrape = drape;
                        }
                        mismatches++;
                    }
                }
            }
        }
    }
    INFO("first mismatch: child(col=" << firstCol << ",rowN=" << firstRow << ") node(i=" << firstI
         << ",j=" << firstJ << ") terrain=" << firstTerrain << " drape=" << firstDrape);
    REQUIRE(mismatches == 0);
}

TEST_CASE("Gold standard - PL-upsampled composite (coarse-displayed / capped cells): drape lattice "
          "nodes lie exactly ON the parent tile's rendered terrain-mesh surface",
          "[RasterMosaic][GoldStandard]") {
    const int N = 2;
    auto own = makeGoldParent(N);

    // All cells missing -> every cell falls back to PL-upsampling `own` over the whole footprint,
    // exactly the displayed-zoom-capped case (cellTargetZoom == _primary.z) after the caps fix.
    std::vector<OverzoomCell> cells(N * N, missingCell());

    auto composite = stitchElevationOverzoomMosaic(own, cells, N);
    REQUIRE(composite != nullptr);
    const int compositeW = composite->width();
    const int cells64 = 64 * (compositeW - 1) / 256;
    REQUIRE(cells64 == 128);

    // (a) Drape nodes that coincide with the parent's own terrain nodes (even k, l) must read the
    //     IDENTICAL value the parent's terrain mesh reads there.
    int mismatches = 0;
    for (int j = 0; j <= 64; j++) {
        for (int i = 0; i <= 64; i++) {
            float terrain = terrainNodeRead(*own, i, j);
            float drape = drapeNodeRead(*composite, 2 * i, 2 * j, cells64);
            if (terrain != drape) { mismatches++; }
        }
    }
    REQUIRE(mismatches == 0);

    // (b) Every remaining (finer) drape node must lie ON the parent's rendered PL surface --
    //     evaluated by an independent reimplementation of the shader formula -- so the drape's
    //     nested finer lattice reproduces the parent's rendered surface exactly, not just at
    //     shared nodes.
    int surfaceMisses = 0;
    int firstK = -1, firstL = -1;
    double firstExpect = 0, firstGot = 0;
    for (int l = 0; l <= cells64; l++) {
        for (int k = 0; k <= cells64; k++) {
            double expect = terrainSurfaceRef(*own, double(k) / cells64, double(l) / cells64);
            double got = drapeNodeRead(*composite, k, l, cells64);
            if (std::abs(expect - got) > 1e-2) {
                if (surfaceMisses == 0) { firstK = k; firstL = l; firstExpect = expect; firstGot = got; }
                surfaceMisses++;
            }
        }
    }
    INFO("first surface miss: node(k=" << firstK << ",l=" << firstL << ") expect=" << firstExpect
         << " got=" << firstGot);
    REQUIRE(surfaceMisses == 0);
}

TEST_CASE("Gold standard - composite embedded as texture-shading mosaic center: the ON-path drape "
          "read equals the OFF-path drape read at every interior lattice node",
          "[RasterMosaic][GoldStandard]") {
    const int N = 2;
    auto own = makeGoldParent(N);

    std::vector<OverzoomCell> cells(N * N);
    for (int rowN = 0; rowN < N; rowN++) {
        for (int col = 0; col < N; col++) {
            cells[size_t(rowN) * N + col] = exactCell(makeGoldChild(N, col, rowN), col, rowN);
        }
    }
    auto composite = stitchElevationOverzoomMosaic(own, cells, N);
    REQUIRE(composite != nullptr);

    std::array<std::shared_ptr<Texture>, 9> noNeighbors;   // all mirrored -- center third untouched
    auto mosaic = stitchElevationMosaic(composite, noNeighbors);
    REQUIRE(mosaic != nullptr);
    REQUIRE(mosaic->width() == 3 * (composite->width() - 1));   // 3 * Wp = 1536

    const int cells64 = 64 * (composite->width() - 1) / 256;    // 128, same as OFF path (msize/3)

    // NOTE k, l <= cells64 - 1: the lattice node at uv exactly 1.0 is a KNOWN separate defect in
    // the ON path (elevationMosaicUV maps uv=1.0 one texel past the world-1.0 line into the
    // neighbor band; the OFF path clamps to the composite's kept far-edge line instead). Tracked
    // separately; this test pins down interior equality, which is what the toggle-independence
    // of the live patches is about.
    int mismatches = 0;
    int firstK = -1, firstL = -1;
    float firstOff = 0, firstOn = 0;
    for (int l = 0; l < cells64; l++) {
        for (int k = 0; k < cells64; k++) {
            float off = drapeNodeRead(*composite, k, l, cells64);
            float on = drapeMosaicNodeRead(*mosaic, k, l, cells64);
            if (off != on) {
                if (mismatches == 0) { firstK = k; firstL = l; firstOff = off; firstOn = on; }
                mismatches++;
            }
        }
    }
    INFO("first mismatch: node(k=" << firstK << ",l=" << firstL << ") off=" << firstOff
         << " on=" << firstOn);
    REQUIRE(mismatches == 0);
}
