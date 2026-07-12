#include "catch.hpp"

#include "util/geom.h"

#include <cmath>
#include <vector>

using namespace Tangram;

namespace {

// Approximate a regular N-gon (a stand-in for a "near-circle" polygon), CCW-wound.
std::vector<glm::vec2> makeRegularPolygon(glm::vec2 center, float radius, int sides) {
    std::vector<glm::vec2> pts;
    pts.reserve(sides);
    for (int i = 0; i < sides; i++) {
        float a = float(i) / sides * 2.f * float(PI);
        pts.push_back(center + radius * glm::vec2(std::cos(a), std::sin(a)));
    }
    return pts;
}

} // namespace

TEST_CASE("polygonPrincipalAxis on a long thin rectangle is strongly elongated along its length",
          "[Core][Geom]") {
    // Axis-aligned rectangle, width 10 (x) x height 1 (y), centered at the origin, CCW-wound.
    std::vector<glm::vec2> rect = {
        {-5.f, -0.5f}, {5.f, -0.5f}, {5.f, 0.5f}, {-5.f, 0.5f}
    };

    PolygonAxis axis = polygonPrincipalAxis(rect.begin(), rect.end());

    // For a rectangle the area-covariance elongation ratio equals the true width/height
    // aspect ratio (see the derivation in util/geom.h): 10/1 = 10.
    REQUIRE(axis.elongation == Approx(10.f).epsilon(0.01));

    // Center should be at the origin.
    REQUIRE(axis.center.x == Approx(0.f).margin(1e-4));
    REQUIRE(axis.center.y == Approx(0.f).margin(1e-4));

    // Direction should be (near-)horizontal, i.e. along the long (x) axis; sign is
    // arbitrary (a principal axis has no inherent orientation), so compare |direction.x|.
    REQUIRE(std::abs(axis.direction.x) == Approx(1.f).epsilon(0.01));
    REQUIRE(std::abs(axis.direction.y) < 0.01f);
}

TEST_CASE("polygonPrincipalAxis on a near-circular polygon reports low elongation",
          "[Core][Geom]") {
    auto circle = makeRegularPolygon({100.f, -40.f}, 5.f, 32);

    PolygonAxis axis = polygonPrincipalAxis(circle.begin(), circle.end());

    // A regular polygon has no preferred axis; elongation should be very close to 1 (not
    // exactly 1 due to the polygon-vs-true-circle discretization, but well under any
    // reasonable curving threshold).
    REQUIRE(axis.elongation < 1.05f);
    REQUIRE(axis.elongation >= 1.f);

    REQUIRE(axis.center.x == Approx(100.f).margin(0.05));
    REQUIRE(axis.center.y == Approx(-40.f).margin(0.05));
}

TEST_CASE("polygonPrincipalAxis on an L-shape reports sane elongation and a unit direction",
          "[Core][Geom]") {
    // An L-shaped polygon (union of a wide short arm and a tall narrow arm) in a 4x4
    // bounding box, CCW-wound.
    std::vector<glm::vec2> lshape = {
        {0.f, 0.f}, {4.f, 0.f}, {4.f, 1.f}, {1.f, 1.f}, {1.f, 4.f}, {0.f, 4.f}
    };

    PolygonAxis axis = polygonPrincipalAxis(lshape.begin(), lshape.end());

    // The L-shape is elongated overall (it's clearly not compact/round), well above the
    // curving threshold used by TextStyleBuilder (~1.4-1.6).
    REQUIRE(axis.elongation > 1.3f);

    // Direction must be a sane unit vector.
    REQUIRE(glm::length(axis.direction) == Approx(1.f).epsilon(0.01));

    // The centroid of an L-shape need not be a nice round number, but it must lie within
    // the shape's bounding box.
    REQUIRE(axis.center.x >= 0.f);
    REQUIRE(axis.center.x <= 4.f);
    REQUIRE(axis.center.y >= 0.f);
    REQUIRE(axis.center.y <= 4.f);
}

TEST_CASE("polygonPrincipalAxis is winding-order independent", "[Core][Geom]") {
    std::vector<glm::vec2> ccw = {
        {-5.f, -0.5f}, {5.f, -0.5f}, {5.f, 0.5f}, {-5.f, 0.5f}
    };
    std::vector<glm::vec2> cw(ccw.rbegin(), ccw.rend());

    PolygonAxis axisCcw = polygonPrincipalAxis(ccw.begin(), ccw.end());
    PolygonAxis axisCw = polygonPrincipalAxis(cw.begin(), cw.end());

    REQUIRE(axisCcw.elongation == Approx(axisCw.elongation).epsilon(0.01));
    REQUIRE(std::abs(axisCcw.direction.x) == Approx(std::abs(axisCw.direction.x)).epsilon(0.01));
}

TEST_CASE("clipLineToPolygon clips the principal axis of a rectangle to its boundary",
          "[Core][Geom]") {
    std::vector<glm::vec2> rect = {
        {-5.f, -0.5f}, {5.f, -0.5f}, {5.f, 0.5f}, {-5.f, 0.5f}
    };

    PolygonAxis axis = polygonPrincipalAxis(rect.begin(), rect.end());

    glm::vec2 start, end;
    bool ok = clipLineToPolygon(axis.center, axis.direction, rect.begin(), rect.end(), start, end);
    REQUIRE(ok);

    // The clipped segment should span the full length of the rectangle along x (order of
    // start/end depends on the arbitrary sign of the principal-axis direction).
    float minX = std::min(start.x, end.x);
    float maxX = std::max(start.x, end.x);
    REQUIRE(minX == Approx(-5.f).epsilon(0.01));
    REQUIRE(maxX == Approx(5.f).epsilon(0.01));
    REQUIRE(start.y == Approx(0.f).margin(1e-3));
    REQUIRE(end.y == Approx(0.f).margin(1e-3));
}

TEST_CASE("clipLineToPolygon fails gracefully for a degenerate polygon", "[Core][Geom]") {
    // A zero-area (collinear) "polygon" -- callers must fall back to a point label.
    std::vector<glm::vec2> degenerate = { {0.f, 0.f}, {1.f, 0.f}, {2.f, 0.f} };

    glm::vec2 start, end;
    bool ok = clipLineToPolygon({0.5f, 0.f}, {1.f, 0.f}, degenerate.begin(), degenerate.end(), start, end);
    REQUIRE_FALSE(ok);
}
