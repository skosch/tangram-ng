#include "catch.hpp"

#include "util/builders.h"

#include <map>

using namespace Tangram;

namespace {

struct GridFixture {
    PolygonBuilder builder;
    std::vector<glm::vec3> coords;
    // chunk-rebased indices, directly usable against `coords` regardless of whether
    //  buildPolygonGrid flushed multiple uint16-sized sub-ranges (PolygonBuilder::chunkOffsets) -
    //  mirrors the rebasing MeshBase::compileIndices (gl/mesh.cpp) does at render time
    std::vector<uint32_t> globalIndices;

    GridFixture(int gridRes) {
        builder.useTexCoords = false;
        builder.gridRes = gridRes;
        builder.addVertex = [&](const glm::vec3& _coord, const glm::vec3& _normal, const glm::vec2& _uv) {
            coords.push_back(_coord);
        };
    }

    void build(const Polygon& _polygon) {
        Builders::buildPolygon(_polygon, 0.f, builder);

        globalIndices.clear();
        size_t src = 0, vertBase = 0;
        for (auto& chunk : builder.chunkOffsets) {
            for (size_t i = 0; i < chunk.first; i++) { globalIndices.push_back(uint32_t(builder.indices[src++]) + vertBase); }
            vertBase += chunk.second;
        }
        for (; src < builder.indices.size(); src++) { globalIndices.push_back(uint32_t(builder.indices[src]) + vertBase); }
    }

    // signed area of the triangulation (positive for CCW triangles)
    double signedArea() {
        double area = 0;
        for (size_t i = 0; i + 2 < globalIndices.size(); i += 3) {
            const glm::vec3& a = coords[globalIndices[i]];
            const glm::vec3& b = coords[globalIndices[i + 1]];
            const glm::vec3& c = coords[globalIndices[i + 2]];
            area += 0.5 * (double(b.x - a.x) * (c.y - a.y) - double(c.x - a.x) * (b.y - a.y));
        }
        return area;
    }

    bool indicesValid() {
        for (auto i : globalIndices) {
            if (i >= coords.size()) { return false; }
        }
        size_t totalVerts = builder.numVertices;
        for (auto& chunk : builder.chunkOffsets) { totalVerts += chunk.second; }
        return totalVerts == coords.size();
    }

    // no triangle may span a grid cell boundary (all its points within one cell, allowing edges)
    bool trianglesRespectGrid() {
        float res = float(builder.gridRes);
        for (size_t i = 0; i + 2 < globalIndices.size(); i += 3) {
            glm::vec2 lo(1e9f), hi(-1e9f);
            for (int k = 0; k < 3; k++) {
                const glm::vec3& p = coords[globalIndices[i + k]];
                lo.x = std::min(lo.x, p.x); lo.y = std::min(lo.y, p.y);
                hi.x = std::max(hi.x, p.x); hi.y = std::max(hi.y, p.y);
            }
            // triangle bbox must fit within one cell: floor of (lo*res) and ceil of (hi*res) differ by <= 1
            if (std::ceil(double(hi.x) * res) - std::floor(double(lo.x) * res) > 1.0001) { return false; }
            if (std::ceil(double(hi.y) * res) - std::floor(double(lo.y) * res) > 1.0001) { return false; }
        }
        return true;
    }
};

}  // namespace

TEST_CASE("grid tessellation of a large CCW square", "[builders]") {
    GridFixture fix(64);
    // 0.8 x 0.8 square -> 51x51 cells, mostly full cells
    fix.build({{ {0.1f, 0.1f}, {0.9f, 0.1f}, {0.9f, 0.9f}, {0.1f, 0.9f} }});

    REQUIRE(fix.indicesValid());
    REQUIRE(fix.builder.indices.size() % 3 == 0);
    REQUIRE(fix.builder.indices.size() / 3 > 2 * 50 * 50);  // at least two triangles per interior cell
    REQUIRE(std::abs(fix.signedArea()) == Approx(0.64).epsilon(1e-3));
    REQUIRE(fix.trianglesRespectGrid());
}

TEST_CASE("grid tessellation of a CW square matches plain earcut winding", "[builders]") {
    // earcut normalizes the output winding regardless of the input ring orientation, and face
    //  culling is enabled at render time, so what matters is that the grid path emits the same
    //  orientation as the plain earcut path does for the same polygon
    Polygon square = {{ {0.1f, 0.1f}, {0.1f, 0.9f}, {0.9f, 0.9f}, {0.9f, 0.1f} }};

    GridFixture grid(64);
    grid.build(square);
    GridFixture plain(0);
    plain.build(square);

    REQUIRE(grid.indicesValid());
    REQUIRE(grid.signedArea() == Approx(plain.signedArea()).epsilon(1e-3));
}

TEST_CASE("grid tessellation of a square with a hole", "[builders]") {
    GridFixture fix(64);
    fix.build({{ {0.f, 0.f}, {1.f, 0.f}, {1.f, 1.f}, {0.f, 1.f} },
               { {0.25f, 0.25f}, {0.25f, 0.75f}, {0.75f, 0.75f}, {0.75f, 0.25f} }});  // hole, opposite winding

    REQUIRE(fix.indicesValid());
    REQUIRE(std::abs(fix.signedArea()) == Approx(1.0 - 0.25).epsilon(1e-3));
    REQUIRE(fix.trianglesRespectGrid());
}

TEST_CASE("grid tessellation of a small polygon falls back to plain earcut", "[builders]") {
    GridFixture fix(64);
    // fits inside one cell -> plain earcut path (2 triangles from 4 points)
    fix.build({{ {0.501f, 0.501f}, {0.505f, 0.501f}, {0.505f, 0.505f}, {0.501f, 0.505f} }});

    REQUIRE(fix.indicesValid());
    REQUIRE(fix.builder.indices.size() == 6);
    REQUIRE(std::abs(fix.signedArea()) == Approx(4e-3 * 4e-3).epsilon(1e-3));
}

TEST_CASE("grid tessellation matches plain earcut area for a concave polygon", "[builders]") {
    // L-shaped polygon spanning many cells
    Polygon lshape = {{ {0.1f, 0.1f}, {0.9f, 0.1f}, {0.9f, 0.5f}, {0.5f, 0.5f}, {0.5f, 0.9f}, {0.1f, 0.9f} }};

    GridFixture grid(64);
    grid.build(lshape);
    GridFixture plain(0);
    plain.build(lshape);

    REQUIRE(grid.indicesValid());
    REQUIRE(grid.signedArea() == Approx(plain.signedArea()).epsilon(1e-3));
    REQUIRE(grid.trianglesRespectGrid());
}

TEST_CASE("grid tessellation flushes multiple index-16 chunks for a very fine grid", "[builders]") {
    // full-tile square at gridRes=256 needs 257*257 ~= 66049 unique vertices, comfortably over
    //  the uint16 headroom buildPolygonGrid flushes at (65000) - forces at least one chunk split
    //  (PolygonBuilder::chunkOffsets), the mechanism that lets terrain_grid be raised well past 64
    //  (see stylus-osm.yaml's unlit-polygons) without falling back to a coarse undraped triangle.
    GridFixture fix(256);
    fix.build({{ {0.0f, 0.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}, {0.0f, 1.0f} }});

    REQUIRE(fix.coords.size() > 65000);
    REQUIRE(fix.builder.chunkOffsets.size() >= 1);
    for (auto& chunk : fix.builder.chunkOffsets) {
        REQUIRE(chunk.second <= 65535);
    }
    REQUIRE(fix.builder.numVertices <= 65535);
    REQUIRE(fix.indicesValid());
    REQUIRE(fix.builder.indices.size() % 3 == 0);
    REQUIRE(std::abs(fix.signedArea()) == Approx(1.0).epsilon(1e-3));
    REQUIRE(fix.trianglesRespectGrid());
}
