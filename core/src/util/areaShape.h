#pragma once

#include <glm/vec2.hpp>
#include <glm/common.hpp>

#include <memory>
#include <vector>

namespace Tangram {

// Shape analysis for area labels (lakes, glaciers, ...): given every tile fragment of one
// named feature in world meters, find the PARTS a label should be set in, and for each part a
// spine with a clearance profile. Everything runs on ONE small raster (<= AreaShapeParams::
// maxCells cells, in the frame of the feature's principal axis), so the cost is bounded
// by the grid, not by how many vertices the coastline has or how many tiles the feature spans:
//
//   1. rasterise every fragment (even-odd per polygon, cell centres, OR-ed) -- fragments that
//      meet at a tile edge fuse without any polygon union;
//   2. exact Euclidean distance transform -> inscribed radius r(x) at every inside cell;
//   3. watershed over r, merging basins unless their saddle is a NECK (a long, narrow
//      connection: saddle radius < neckRatio * the smaller basin radius AND the gap between
//      the two inscribed circles is >= neckGap * saddle). Surviving basins are the parts;
//   4. per part: a spine (centre-weighted shortest path between the two geodesic extremes),
//      smoothed, with r(x) sampled along it. Compact parts (elongation < `compact`) get a
//      straight horizontal chord through their widest point instead: a round lake has no
//      "main axis", and labels on those stay horizontal.
//
// Units: meters, relative to AreaShape::origin (so everything fits comfortably in float).

using AreaRing = std::vector<glm::dvec2>;
using AreaPolygon = std::vector<AreaRing>;  // outer ring + holes, even-odd

struct AreaShapeParams {
    int maxCells = 24000;       // total grid budget
    float minCell = 8.f;        // never finer than this (m)
    float neckRatio = 0.25f;    // saddle radius / smaller basin radius below which a connection is narrow
    float neckGap = 3.f;        // ... and the gap between the basins' circles is >= this * saddle radius
    float compact = 1.3f;       // spine length / (2 * max radius) below which the part is "compact"
    int chordRows = 0;              // >0: also list horizontal chords (about this many rows) as fallback label paths
    bool alwaysHorizontal = false;  // parks: every part gets the best horizontal chord, never a spine
    float smooth = 0.35f;       // spine smoothing sigma, in multiples of the part's median clearance
};

struct AreaChord {
    std::vector<glm::vec2> pts;       // horizontal run inside the part, rel. to AreaShape::origin
    std::vector<float> clearance;
};

struct AreaPart {
    int id = -1;                // raster label of the part (for AreaShape::bestChord)
    float area = 0.f;           // m^2
    float maxRadius = 0.f;      // m, largest inscribed circle
    glm::vec2 peak{0.f};        // centre of that circle
    float elongation = 1.f;     // spine length / (2 * maxRadius)
    bool horizontal = false;    // compact part: spine is a horizontal chord
    std::vector<glm::vec2> spine;     // smooth path, rel. to AreaShape::origin
    std::vector<float> clearance;     // inscribed radius at each spine point (m)
    std::vector<AreaChord> chords;    // fallback paths, longest first (AreaShapeParams::chordRows)
};

struct AreaNeck {
    glm::vec2 pos{0.f};         // where the two parts were kept apart
    float radius = 0.f;         // saddle radius (m)
};

struct AreaGrid;

struct AreaShape {
    glm::dvec2 origin{0.0};     // world position of local (0,0)
    float cell = 0.f;           // m
    int cells = 0;
    std::vector<AreaPart> parts;       // largest first
    std::vector<AreaNeck> necks;
    std::shared_ptr<const AreaGrid> grid;   // the raster the parts were found on
};

/// Returns false if the input has no area.
bool analyzeArea(const std::vector<AreaPolygon>& polys, const AreaShapeParams& params, AreaShape& out);

}  // namespace Tangram
