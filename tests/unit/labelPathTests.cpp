#include "catch.hpp"

#include "util/geom.h"
#include "util/labelPath.h"

#include <glm/glm.hpp>

#include <cmath>
#include <vector>

using namespace Tangram;
using V = glm::vec2;

namespace {

// A banana: points on an arc of radius R spanning `span` radians, thickened by +-halfWidth
// (outer ring then inner ring reversed), i.e. a lake that bends.
std::vector<V> makeBanana(float R, float span, float halfWidth, int n) {
    std::vector<V> ring;
    for (int i = 0; i <= n; i++) {
        float a = -span * 0.5f + span * float(i) / float(n);
        ring.push_back(V(std::sin(a), std::cos(a)) * (R + halfWidth));
    }
    for (int i = n; i >= 0; i--) {
        float a = -span * 0.5f + span * float(i) / float(n);
        ring.push_back(V(std::sin(a), std::cos(a)) * (R - halfWidth));
    }
    return ring;
}

} // namespace

TEST_CASE("smoothPolyline keeps endpoints and removes jitter", "[Core][LabelPath]") {
    // A straight horizontal line with +-0.02 zig-zag noise every 0.01 units.
    std::vector<V> noisy;
    for (int i = 0; i <= 200; i++) {
        noisy.push_back(V(0.01f * float(i), (i % 2 ? 0.02f : -0.02f)));
    }
    auto smooth = smoothPolyline(noisy, 0.01f, 0.1f);

    REQUIRE(smooth.size() >= 2);
    REQUIRE(smooth.front().x == Approx(noisy.front().x).margin(1e-5));
    REQUIRE(smooth.back().x == Approx(noisy.back().x).margin(1e-5));
    // Endpoints are pinned to the input's (so the neighbourhood of each keeps a trace of
    // its noise); the middle of the path must be flat.
    for (const auto& p : smooth) {
        if (p.x > 0.4f && p.x < 1.6f) { REQUIRE(std::abs(p.y) < 0.005f); }
    }
    // Simplification must collapse a straight line to very few points.
    REQUIRE(smooth.size() < 12);
}

TEST_CASE("smoothPolyline preserves a gentle curve's overall extent", "[Core][LabelPath]") {
    std::vector<V> arc;
    for (int i = 0; i <= 100; i++) {
        float a = float(PI) * float(i) / 100.f;
        arc.push_back(V(std::cos(a), std::sin(a)));
    }
    auto smooth = smoothPolyline(arc, 0.02f, 0.05f);
    float len = polylineLength(smooth);
    REQUIRE(len == Approx(float(PI)).epsilon(0.02));
    // Mid-arc apex stays near (0,1) (a Gaussian pulls a curve inward only slightly).
    float maxY = 0.f;
    for (const auto& p : smooth) { maxY = std::max(maxY, p.y); }
    REQUIRE(maxY > 0.98f);
}

TEST_CASE("trimPolyline removes arc length from both ends", "[Core][LabelPath]") {
    std::vector<V> line = { V(0, 0), V(1, 0), V(1, 1) };  // length 2
    auto t = trimPolyline(line, 0.5f, 0.25f);
    REQUIRE(t.size() == 3);
    REQUIRE(t.front().x == Approx(0.5f));
    REQUIRE(t.back().y == Approx(0.75f));
    REQUIRE(polylineLength(t) == Approx(1.25f));
    REQUIRE(trimPolyline(line, 1.5f, 1.5f).empty());
}

TEST_CASE("polygonCenterline follows a banana-shaped polygon and stays inside", "[Core][LabelPath]") {
    const float R = 10.f, halfW = 0.6f;
    std::vector<std::vector<V>> rings = { makeBanana(R, 1.2f, halfW, 40) };

    // Principal axis of a gently bent banana is roughly the chord direction (horizontal).
    PolygonCenterline cl;
    REQUIRE(polygonCenterline(rings, V(1, 0), 1.0f, 3.f, 0.5f, cl));

    // The path follows the arc, i.e. its distance to the arc's centre (0,0) stays ~R,
    // which a straight axis chord (sagitta ~ R(1-cos(0.6)) = 1.7) could not do.
    for (const auto& p : cl.points) {
        REQUIRE(std::abs(glm::length(p) - R) < 0.25f);
        REQUIRE(pointInPolygon(rings, p));
    }
    // Spine is about as long as the arc (R * 1.2 = 12) minus the narrow-tip losses.
    REQUIRE(cl.length > 9.f);
    REQUIRE(cl.length < 12.5f);
    REQUIRE(cl.minWidth >= 1.0f);
}

TEST_CASE("polygonCenterline respects the minimum width and length", "[Core][LabelPath]") {
    // 10 x 0.5 rectangle: too narrow for a strip 1.0 wide.
    std::vector<std::vector<V>> rect = {{ V(0, 0), V(10, 0), V(10, 0.5f), V(0, 0.5f) }};
    PolygonCenterline cl;
    REQUIRE_FALSE(polygonCenterline(rect, V(1, 0), 1.0f, 3.f, 0.f, cl));
    // ... but fine for 0.3 wide.
    REQUIRE(polygonCenterline(rect, V(1, 0), 0.3f, 3.f, 0.f, cl));
    REQUIRE(cl.length > 9.f);
    // Too short for a long label.
    REQUIRE_FALSE(polygonCenterline(rect, V(1, 0), 0.3f, 50.f, 0.f, cl));
}

TEST_CASE("polygonCenterline picks the continuous interval across an island", "[Core][LabelPath]") {
    // 20 x 4 rectangle with a 2 x 2 hole off-centre (bottom half): slicing through the
    // hole yields two intervals; the spine must not jump between them.
    std::vector<std::vector<V>> poly = {
        { V(0, 0), V(20, 0), V(20, 4), V(0, 4) },
        { V(9, 0.5f), V(9, 2.5f), V(11, 2.5f), V(11, 0.5f) },
    };
    PolygonCenterline cl;
    REQUIRE(polygonCenterline(poly, V(1, 0), 1.0f, 3.f, 0.f, cl));
    for (size_t i = 1; i < cl.points.size(); i++) {
        REQUIRE(std::abs(cl.points[i].y - cl.points[i - 1].y) < 2.5f);  // no lobe-to-lobe leap beyond the hole
        REQUIRE(pointInPolygon(poly, cl.points[i]));
    }
}
