#pragma once

#include "glm/vec2.hpp"
#include "glm/vec4.hpp"
#include "glm/mat4x4.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace Tangram {

constexpr double PI = 3.14159265358979323846;
constexpr double TWO_PI = 6.28318530717958647693;
constexpr double FOUR_PI = 12.56637061435917295385;
constexpr double HALF_PI = 1.57079632679489661923;

/// Multiply degrees by DEG_TO_RAD to get radians.
constexpr double DEG_TO_RAD = 0.01745329251994329576;

/// Multiply radians by RAD_TO_DEG to get degrees.
constexpr double RAD_TO_DEG = 57.2957795130823208767;

/// Minimum value between two variables that support < comparison.
template<typename T>
constexpr T min(T a, T b) {
    return a < b ? a : b;
}

/// Maximum value between two values that support > comparison.
template<typename T>
constexpr T max(T a, T b) {
    return a > b ? a : b;
}

/// Clamp a value between a min and max value.
template<typename T>
constexpr T clamp(T value, T min, T max) {
    return value < min ? min : (value > max ? max : value);
}

/// Clamp a value to the range [0, 1].
template<typename T>
constexpr T clamp01(T value) {
    return clamp(value, 0.f, 1.f);
}

/// Absolute value of a numeric variable.
template<typename T>
constexpr T abs(T value) {
    return value < 0 ? -value : value;
}

// glm::round() (uses std::round()) is slow!
// nearbyint uses current FPU rounding mode (should be nearest) and optimizes to `roundps` on x64
inline glm::vec2 nearbyint(glm::vec2 in) {
    return {std::nearbyint(in.x), std::nearbyint(in.y)};
}

inline glm::vec3 nearbyint(glm::vec3 in) {
    return {std::nearbyint(in.x), std::nearbyint(in.y), std::nearbyint(in.z)};
}

inline glm::vec4 nearbyint(glm::vec4 in) {
    return {std::nearbyint(in.x), std::nearbyint(in.y), std::nearbyint(in.z), std::nearbyint(in.w)};
}

struct BoundingBox {

    glm::dvec2 min;
    glm::dvec2 max;

    double width() const { return glm::abs(max.x - min.x); }
    double height() const { return glm::abs(max.y - min.y); }
    glm::dvec2 center() const { return 0.5 * (min + max); }
    bool containsX(double x) const { return x >= min.x && x <= max.x; }
    bool containsY(double y) const { return y >= min.y && y <= max.y; }
    bool contains(double x, double y) const { return containsX(x) && containsY(y); }
    void expand(double x, double y) {
        min = { glm::min(min.x, x), glm::min(min.y, y) };
        max = { glm::max(max.x, x), glm::max(max.y, y) };
    }
};

/// Return the signed area of the polygon defined by the points in the range [begin, end).
/// The sign of the area is positive if the vertices of the polygon are counter-clockwise.
template<class InputIt>
float signedArea(InputIt begin, InputIt end) {
    if (begin == end) { return 0; }
    float area = 0;
    auto prev = end - 1;
    for (auto curr = begin; curr != end; ++curr) {
        area += static_cast<float>(curr->x * prev->y - curr->y * prev->x);
        prev = curr;
    }
    return 0.5f * area;
}

/// Calculate the area centroid of a closed polygon given as a sequence of vectors.
/// If the polygon has no area, the coordinates returned are NaN.
template<class InputIt, class Vector = typename InputIt::value_type>
Vector centroid(InputIt begin, InputIt end) {
    Vector centroid{};
    float area = 0.f;
    const Vector offset(begin->x, begin->y);

    for (auto curr = begin, prev = end - 1; curr != end; prev = curr, ++curr) {
        const Vector prevPoint(prev->x - offset.x, prev->y - offset.y);
        const Vector currPoint(curr->x - offset.x, curr->y - offset.y);
        float a = static_cast<float>(prevPoint.x * currPoint.y - currPoint.x * prevPoint.y);
        centroid.x += (prevPoint.x + currPoint.x) * a;
        centroid.y += (prevPoint.y + currPoint.y) * a;
        area += a;
    }
    centroid.x = centroid.x / (3.f * area) + offset.x;
    centroid.y = centroid.y / (3.f * area) + offset.y;
    return centroid;
}

/// Result of analyzing a polygon's area-weighted principal axis (see polygonPrincipalAxis).
struct PolygonAxis {
    glm::vec2 center{0.f, 0.f};     // area centroid
    glm::vec2 direction{1.f, 0.f};  // unit vector along the major (principal) axis
    float elongation = 1.f;         // sqrt(largest / smallest eigenvalue) of the area
                                     // covariance matrix; equals width/height for a
                                     // rectangle, and 1 for a circle. >= 1.
};

/// Compute the area-weighted principal axis and elongation ratio of the polygon ring
/// [begin, end) (a single closed ring; the first point need not be repeated at the end).
///
/// Uses the analytic second-moment-of-area formulas for a polygon (integrated over the
/// polygon's *interior* via Green's theorem) rather than a naive covariance of the
/// boundary vertices -- the latter would be skewed by uneven vertex density along the
/// ring (e.g. a finely-subdivided shoreline segment next to a few long straight edges).
///
/// For a degenerate (near-zero-area, or fewer than 3 point) ring, returns elongation = 1
/// and an arbitrary direction; callers should treat that as "not elongated".
template<class InputIt, class Vector = typename InputIt::value_type>
PolygonAxis polygonPrincipalAxis(InputIt begin, InputIt end) {
    PolygonAxis result;
    if (begin == end || (end - begin) < 3) { return result; }

    // Translate to local coordinates relative to the first vertex for numeric stability
    // -- same trick as centroid() above, more important here since we accumulate
    // products of coordinates up to cubic terms.
    const double ox = begin->x, oy = begin->y;

    double area2 = 0.0; // sum of cross_i == 2 * signed polygon area
    double sumCx = 0.0, sumCy = 0.0;
    double sumXX = 0.0, sumYY = 0.0, sumXY = 0.0;

    auto prev = end - 1;
    for (auto curr = begin; curr != end; prev = curr, ++curr) {
        double x0 = prev->x - ox, y0 = prev->y - oy;
        double x1 = curr->x - ox, y1 = curr->y - oy;
        double cross = x0 * y1 - x1 * y0;

        area2 += cross;
        sumCx += (x0 + x1) * cross;
        sumCy += (y0 + y1) * cross;
        sumXX += (x0 * x0 + x0 * x1 + x1 * x1) * cross;
        sumYY += (y0 * y0 + y0 * y1 + y1 * y1) * cross;
        sumXY += (x0 * y1 + 2.0 * x0 * y0 + 2.0 * x1 * y1 + x1 * y0) * cross;
    }

    if (std::abs(area2) < 1e-12) { return result; }

    double cx = sumCx / (3.0 * area2);
    double cy = sumCy / (3.0 * area2);
    result.center = Vector(float(cx + ox), float(cy + oy));

    // Second moments about the origin, shifted to the centroid (parallel axis theorem).
    // Dividing by the same signed 'area2' throughout keeps Var(x)/Var(y)/Cov(x,y) correct
    // regardless of the ring's winding direction (moments and area both flip sign
    // together for a clockwise ring, so the ratio stays positive).
    double varX = sumXX / (6.0 * area2) - cx * cx;
    double varY = sumYY / (6.0 * area2) - cy * cy;
    double covXY = sumXY / (12.0 * area2) - cx * cy;

    // Eigen-decomposition of the symmetric 2x2 covariance matrix [[varX, covXY], [covXY, varY]].
    double trace = varX + varY;
    double halfDiff = (varX - varY) * 0.5;
    double disc = std::sqrt(std::max(0.0, halfDiff * halfDiff + covXY * covXY));
    double lambda1 = trace * 0.5 + disc; // larger eigenvalue
    double lambda2 = trace * 0.5 - disc; // smaller eigenvalue

    if (lambda1 <= 1e-12) { return result; } // degenerate (near-zero extent in every direction)

    glm::vec2 dir;
    if (std::abs(covXY) > 1e-9 * std::max(1.0, std::abs(trace))) {
        // Eigenvector of the symmetric matrix for eigenvalue lambda1: (lambda1 - varY, covXY).
        dir = glm::vec2(float(lambda1 - varY), float(covXY));
    } else {
        // Axis-aligned (no shear): pick the axis with the larger variance.
        dir = (varX >= varY) ? glm::vec2(1.f, 0.f) : glm::vec2(0.f, 1.f);
    }
    float dirLen = glm::length(dir);
    result.direction = (dirLen > 1e-12f) ? dir / dirLen : glm::vec2(1.f, 0.f);

    double lambda2c = std::max(lambda2, 0.0);
    // Cap the ratio for near-degenerate (essentially 1-dimensional) shapes rather than
    // returning +inf.
    result.elongation = (lambda2c > 1e-12) ? float(std::sqrt(lambda1 / lambda2c)) : 1000.f;

    return result;
}

/// Return the signed area of the triangle formed by the points a, b, and c in the plane.
template<class T>
float signedArea(const T& a, const T& b, const T& c) {
    return 0.5 * ((b.y - a.y) * (c.x - b.x) - (b.x - a.x) * (c.y - b.y));
}

/// Return the 'perpendicular dot product' of two vectors in the plane.
/// The result is equal to |a|*|b|*sin(theta), where theta is the angle between the vectors.
/// https://mathworld.wolfram.com/PerpDotProduct.html
inline float perpDotProduct(const glm::vec2& a, const glm::vec2& b) {
    return (a.x * b.y) - (a.y * b.x);
}

/// Intersect the infinite line through 'origin' with direction 'dir' against the polygon
/// ring [begin, end), returning in (outStart, outEnd) the sub-segment of that line lying
/// inside the ring and containing the origin's own projection (parametric t = 0).
///
/// Returns false if no such segment is found -- e.g. 'origin' lies outside the ring for a
/// concave polygon, or the ring is degenerate. Callers should fall back to a plain
/// centroid label in that case.
template<class InputIt>
bool clipLineToPolygon(glm::vec2 origin, glm::vec2 dir, InputIt begin, InputIt end,
                        glm::vec2& outStart, glm::vec2& outEnd) {
    if (begin == end) { return false; }

    std::vector<float> ts;
    auto prev = end - 1;
    for (auto curr = begin; curr != end; prev = curr, ++curr) {
        glm::vec2 a(prev->x, prev->y);
        glm::vec2 b(curr->x, curr->y);
        glm::vec2 edge = b - a;

        float denom = perpDotProduct(dir, edge);
        if (std::abs(denom) < 1e-9f) { continue; } // line parallel to this edge

        glm::vec2 diff = a - origin;
        float u = perpDotProduct(diff, dir) / denom;
        if (u < 0.f || u >= 1.f) { continue; } // intersection outside this edge segment

        float t = perpDotProduct(diff, edge) / denom;
        ts.push_back(t);
    }

    if (ts.size() < 2) { return false; }
    std::sort(ts.begin(), ts.end());

    // Pair up consecutive crossings (ray-casting parity: alternating gaps are 'inside'
    // the polygon). Return the pair that brackets t = 0, i.e. contains the origin.
    for (size_t i = 0; i + 1 < ts.size(); i += 2) {
        if (ts[i] <= 0.f && ts[i + 1] >= 0.f) {
            outStart = origin + dir * ts[i];
            outEnd = origin + dir * ts[i + 1];
            return true;
        }
    }
    return false;
}

/// Map a value from the range [inputMin, inputMax] into the range [0, 1].
/// Ex: mapRange01(6, 2, 10) == 0.5, mapRange01(10, 0, 5) == 1
float mapRange01(float value, float inputMin, float inputMax);

/// Computes the clip coordinates from position in world space and a model view matrix
inline glm::vec4 worldToClipSpace(const glm::mat4& mvp, const glm::vec4& worldPosition) {
    return mvp * worldPosition;
}

/// Return true if the given clip space point is behind the camera plane.
inline bool clipSpaceIsBehindCamera(const glm::vec4& clip) {
    return clip.w < 0;
}

/// Return the given clip space point converted to Normalized Device Coordinates.
inline glm::vec3 clipSpaceToNdc(const glm::vec4& clip) {
    return glm::vec3(clip) / clip.w;
}

/// Return the given point in Normalized Device Coordinates converted to screen space.
inline glm::vec2 ndcToScreenSpace(const glm::vec3& ndc, const glm::vec2& screenSize) {
    return glm::vec2(1 + ndc.x, 1 - ndc.y) * screenSize * 0.5f;
}

/// Compute the screen coordinates from a world position, a model view matrix and a screen size.
glm::vec4 worldToScreenSpace(const glm::mat4& mvp, const glm::vec4& worldPosition, const glm::vec2& screenSize, bool& behindCamera);

/// Return the rotation unit vector representing a counter-clockwise rotation in the plane of the given radians.
inline glm::vec2 rotation2dRad(float radians) {
    return { cos(radians), sin(radians) };
}

/// Return the rotation unit vector representing a counter-clockwise rotation in the plane of the given degrees.
inline glm::vec2 rotation2dDeg(float degrees) {
    return rotation2dRad(static_cast<float>(DEG_TO_RAD) * degrees);
}

/// Return the vector 'in' rotated in the plane by the rotation unit vector 'rotation'.
inline glm::vec2 rotate2d(const glm::vec2& in, const glm::vec2& rotation) {
    return { in.x * rotation.x + in.y * rotation.y, -in.x * rotation.y + in.y * rotation.x };
}

/// Return true if the value is exactly a power of two.
inline bool isPowerOfTwo(int value) {
    return (value & (value - 1)) == 0;
}

/// Return the square of the distance from the point 'p' to the nearest point on the segment from 'a' to 'b'.
float pointSegmentDistanceSq(const glm::vec2& p, const glm::vec2& a, const glm::vec2& b);

/// Return the distance from the point 'p' to the nearest point on the segment from 'a' to 'b'.
float pointSegmentDistance(const glm::vec2& p, const glm::vec2& a, const glm::vec2& b);

/// Clip line segment a -> b to rectangle defined by corners min, max using the Liang-Barsky algorithm
/// returns false if segment completely outside rectangle
bool clipLine(glm::vec2& a, glm::vec2& b, glm::vec2 min, glm::vec2 max);

}
