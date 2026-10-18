#pragma once

#include <glm/vec2.hpp>

#include <functional>
#include <vector>

namespace Tangram {

// Geometry helpers for labels that follow a path (docs/label-typography.md): ridges,
// valleys, rivers, and the "spine" of an elongated lake, glacier or protected area.
// Units are whatever the points are in (tile units at the call sites); every distance
// parameter is in those same units.

/// Total length of an open polyline.
float polylineLength(const std::vector<glm::vec2>& pts);

/// Resample an open polyline at ~`step` spacing, apply a Gaussian low-pass of standard
/// deviation `sigma` along its arc length, and simplify the result again (Douglas-Peucker,
/// tolerance `step * 0.1`). Endpoints stay fixed in position and tangent (point-reflection
/// padding) so smoothing never shortens a path. The point of this is typographic: glyphs
/// placed along digitized geometry (a ridge traced by hand, a river with a vertex every
/// few metres) jitter and fold; glyphs placed along the *large-scale shape* of the same
/// geometry read as one calm curve.
std::vector<glm::vec2> smoothPolyline(const std::vector<glm::vec2>& pts, float step, float sigma);

/// Douglas-Peucker simplification of an open polyline.
std::vector<glm::vec2> simplifyPolyline(const std::vector<glm::vec2>& pts, float tolerance);

/// Remove `fromStart` of arc length from the start and `fromEnd` from the end of an open
/// polyline. Returns an empty path if nothing (< 2 points / zero length) is left.
std::vector<glm::vec2> trimPolyline(const std::vector<glm::vec2>& pts, float fromStart, float fromEnd);

/// Even-odd point-in-polygon test over all rings (outer + holes) of one polygon.
bool pointInPolygon(const std::vector<std::vector<glm::vec2>>& rings, glm::vec2 p);

struct PolygonCenterline {
    std::vector<glm::vec2> points;  // smooth path, ordered along the principal axis
    float length = 0.f;             // arc length of `points`
    float minWidth = 0.f;           // narrowest polygon width measured across the path
};

/// Compute a smooth "spine" through an elongated polygon, suitable for setting a label
/// along it. The polygon (all rings, even-odd) is sliced perpendicular to `axisDir` (its
/// principal axis); the middle of the widest interior interval of each slice is a spine
/// point (preferring the interval that continues the previous point, so a polygon with
/// an island or a pinch doesn't make the spine jump). The longest stretch where the
/// polygon is at least `minWidth` wide survives -- that's where a text strip of that
/// height physically fits -- and is smoothed with a Gaussian of `smoothSigma`, falling
/// back to progressively less smoothing if smoothing would push the spine outside.
///
/// Returns false if no stretch at least `minLength` long exists.
bool polygonCenterline(const std::vector<std::vector<glm::vec2>>& rings, glm::vec2 axisDir,
                       float minWidth, float minLength, float smoothSigma,
                       PolygonCenterline& out);


/// Conservative raster of a polygon's interior (outer ring + holes, even-odd) with a summed-
/// area table, so "does this rectangle lie entirely inside the polygon?" is O(1). Cells the
/// outline passes through count as outside, i.e. everything it says is inside really is.
/// This is the workhorse of area labelling (docs/label-typography.md): a label may never
/// leave the lake / park / glacier it names, so every candidate (horizontal block, line-
/// broken block, strip along a spine) is tested against it before anything is optimised.
class InsideMask {
public:
    /// Rings in pixel units. The grid is at most `maxCells` per side (cell >= `minCell`).
    bool build(const std::vector<std::vector<glm::vec2>>& rings, int maxCells = 128, float minCell = 1.5f);

    float cell() const { return m_cell; }
    /// Mean position of the inside cells.
    glm::vec2 centroid() const { return m_centroid; }
    float insideArea() const { return m_insideCells * m_cell * m_cell; }

    bool pointInside(glm::vec2 p) const;
    /// Fraction (0..1) of the rectangle [lo, hi] that is inside the polygon; parts of the
    /// rectangle off the grid count as outside.
    float insideFraction(glm::vec2 lo, glm::vec2 hi) const;
    /// True if the whole rectangle [lo, hi] is inside the polygon.
    bool rectInside(glm::vec2 lo, glm::vec2 hi) const;

    /// Search every horizontal band of height `bandH` for the widest run that is inside the
    /// polygon over the whole band. Runs wider than `capW` count as `capW` (so the search
    /// prefers the spot nearest the polygon's centre among all roomy ones). Returns false
    /// if no run reaches `minW`.
    struct Band { float cx = 0, cy = 0, width = 0; };
    /// `busy(lo, hi)` (optional) returns 0..1: how much other map content (ferry line,
    /// shoreline, road) the box would sit on; the search then steers away from it. The box
    /// tested is the run's centre section `boxFill` of the run wide.
    bool widestBand(float bandH, float minW, float capW, Band& out,
                    const std::function<float(glm::vec2, glm::vec2)>& busy = nullptr,
                    float boxFill = 0.8f) const;

    /// Distance from `p` to the polygon's outline along the unit direction `dir` (stops at
    /// `maxDist`). Used to measure local width perpendicular to a spine.
    float rayLength(glm::vec2 p, glm::vec2 dir, float maxDist) const;

private:
    int m_w = 0, m_h = 0;
    float m_cell = 1.f;
    glm::vec2 m_origin{0.f};
    glm::vec2 m_centroid{0.f};
    int m_insideCells = 0;
    std::vector<unsigned char> m_in;
    std::vector<int> m_sat;  // (m_w + 1) * (m_h + 1)

    int sum(int x0, int y0, int x1, int y1) const;
};

/// Resample a path at `step` spacing (positions only; keeps both end points).
std::vector<glm::vec2> resamplePolyline(const std::vector<glm::vec2>& pts, float step);

}
