#include "catch.hpp"

#include "util/areaShape.h"

#include <glm/geometric.hpp>
#include <chrono>
#include <cmath>
#include <vector>

using namespace Tangram;
using D = glm::dvec2;

namespace {

AreaRing rect(double x0, double y0, double x1, double y1) {
    return { D(x0, y0), D(x1, y0), D(x1, y1), D(x0, y1) };
}

AreaRing disc(double cx, double cy, double r, int n = 64) {
    AreaRing ring;
    for (int i = 0; i < n; i++) {
        double a = 2.0 * M_PI * i / n;
        ring.push_back(D(cx + r * std::cos(a), cy + r * std::sin(a)));
    }
    return ring;
}

float spineLength(const AreaPart& p) {
    float len = 0.f;
    for (size_t i = 1; i < p.spine.size(); i++) { len += glm::length(p.spine[i] - p.spine[i - 1]); }
    return len;
}

}  // namespace

TEST_CASE("fragments meeting at a seam fuse into one part", "[Core][AreaShape]") {
    // a 60 km x 2 km lake cut into three "tiles"; (offset far from the origin like real meters)
    const double ox = -13700000.0, oy = 6300000.0;
    std::vector<AreaPolygon> frags = {
        { rect(ox, oy, ox + 20000, oy + 2000) },
        { rect(ox + 20000, oy, ox + 40000, oy + 2000) },
        { rect(ox + 40000, oy, ox + 60000, oy + 2000) },
    };
    AreaShape s;
    REQUIRE(analyzeArea(frags, AreaShapeParams(), s));
    REQUIRE(s.parts.size() == 1);
    CHECK(s.necks.empty());
    CHECK_FALSE(s.parts[0].horizontal);
    CHECK(spineLength(s.parts[0]) > 0.8f * 60000.f);
    CHECK(s.parts[0].maxRadius == Approx(1000.f).margin(150.f));
}

TEST_CASE("a long thin strip between two bodies is a neck", "[Core][AreaShape]") {
    std::vector<AreaPolygon> one = {
        { disc(0, 0, 3000) }, { rect(2500, -150, 17500, 150) }, { disc(20000, 0, 3000) }
    };
    AreaShape s;
    REQUIRE(analyzeArea(one, AreaShapeParams(), s));
    CHECK(s.parts.size() == 2);
    CHECK(s.necks.size() >= 1);
}

TEST_CASE("a short pinch does not split a lake", "[Core][AreaShape]") {
    std::vector<AreaPolygon> one = {
        { disc(0, 0, 3000) }, { rect(2500, -600, 3500, 600) }, { disc(6000, 0, 3000) }
    };
    AreaShape s;
    REQUIRE(analyzeArea(one, AreaShapeParams(), s));
    CHECK(s.parts.size() == 1);
}

TEST_CASE("a round lake gets a horizontal chord", "[Core][AreaShape]") {
    std::vector<AreaPolygon> one = { { disc(5000, 5000, 2000) } };
    AreaShape s;
    REQUIRE(analyzeArea(one, AreaShapeParams(), s));
    REQUIRE(s.parts.size() == 1);
    CHECK(s.parts[0].horizontal);
    CHECK(std::fabs(s.parts[0].spine.front().y - s.parts[0].spine.back().y) < 1e-3f);
    CHECK(spineLength(s.parts[0]) > 3000.f);
}

TEST_CASE("island holes are respected", "[Core][AreaShape]") {
    AreaPolygon withIsland = { rect(0, 0, 10000, 2000), disc(5000, 1000, 300) };
    AreaShape s;
    REQUIRE(analyzeArea({ withIsland }, AreaShapeParams(), s));
    REQUIRE(s.parts.size() == 1);
    CHECK(s.parts[0].area == Approx(20e6f - 3.1416f * 300.f * 300.f).epsilon(0.02));
    CHECK(s.parts[0].area < 19.9e6f);
}

TEST_CASE("a 120 km wiggly fjord with a 20k-vertex coast analyses quickly", "[Core][AreaShape][.perf]") {
    AreaRing ring;
    const int n = 10000;
    for (int i = 0; i <= n; i++) {
        double t = double(i) / n;
        double x = 120000.0 * t, y = 20000.0 * std::sin(t * 6.0) + 1500.0 + 300.0 * std::sin(t * 900.0);
        ring.push_back(D(x, y));
    }
    for (int i = n; i >= 0; i--) {
        double t = double(i) / n;
        double x = 120000.0 * t, y = 20000.0 * std::sin(t * 6.0) - 1500.0 + 300.0 * std::sin(t * 700.0);
        ring.push_back(D(x, y));
    }
    AreaShape s;
    auto t0 = std::chrono::steady_clock::now();
    bool ok = analyzeArea({ { ring } }, AreaShapeParams(), s);
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    printf("analyzeArea: %.2f ms, %d cells, cell %.0f m, parts %zu, necks %zu\n", ms, s.cells, s.cell,
           s.parts.size(), s.necks.size());
    REQUIRE(ok);
}
