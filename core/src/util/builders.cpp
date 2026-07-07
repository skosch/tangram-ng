#include "util/builders.h"

#include "util/geom.h"
#include "log.h"
#include "glm/gtx/rotate_vector.hpp"
#include "glm/gtx/norm.hpp"

#include <cstring>
#include <unordered_map>

namespace mapbox { namespace util {
template <>
struct nth<0, Tangram::Point> {
    inline static float get(const Tangram::Point &t) { return t.x; };
};
template <>
struct nth<1, Tangram::Point> {
    inline static float get(const Tangram::Point &t) { return t.y; };
};
}}

namespace {
// Tests if a line segment (from point A to B) is outside the edge of a tile
bool isOutsideTile(const glm::vec2& _a, const glm::vec2& _b) {

    // tweak this adjust if catching too few/many line segments near tile edges
    // TODO: make tolerance configurable by source if necessary
    float tolerance = 0.0005f;
    float tile_min = 0.0f + tolerance;
    float tile_max = 1.0f - tolerance;

    if ( (_a.x < tile_min && _b.x < tile_min) ||
         (_a.x > tile_max && _b.x > tile_max) ||
         (_a.y < tile_min && _b.y < tile_min) ||
         (_a.y > tile_max && _b.y > tile_max) ) {
        return true;
    }

    return false;
}

bool isOutsideTile(const glm::vec2& p) {
    float tolerance = 0.0005f;
    float tile_min = 0.0f + tolerance;
    float tile_max = 1.0f - tolerance;
    return ((p.x < tile_min) || (p.x > tile_max) ||
            (p.y < tile_min) || (p.y > tile_max));
}

// --- grid tessellation for terrain-draped polygons (see PolygonBuilder::gridRes) ---

// Sutherland-Hodgman clip of a convex loop against an axis-aligned half-plane. Only valid for
//  CONVEX input (on non-convex rings SH generates degenerate "bridge" geometry along the clip
//  edge, which mis-triangulates downstream) - the grid tessellation below therefore only ever
//  clips earcut output triangles and their (convex) clipped pieces.
// Clip-generated points get exactly p[axis] == bound and are computed from the same original
//  segment on both sides of a shared cell edge, so shared vertices come out bit-identical across
//  cells (no cracks after the i16 position quantization in the styles).
void clipConvexAxis(const Tangram::Line& _in, Tangram::Line& _out, int _axis, float _bound, bool _keepBelow) {
    _out.clear();
    size_t n = _in.size();
    for (size_t i = 0; i < n; i++) {
        glm::vec2 a = _in[i];
        glm::vec2 b = _in[(i + 1) % n];
        float fa = a[_axis] - _bound, fb = b[_axis] - _bound;
        bool ina = _keepBelow ? fa <= 0.f : fa >= 0.f;
        bool inb = _keepBelow ? fb <= 0.f : fb >= 0.f;
        if (ina) { _out.push_back(a); }
        if (ina != inb) {
            float t = fa / (fa - fb);
            glm::vec2 p = a + t * (b - a);
            p[_axis] = _bound;
            _out.push_back(p);
        }
    }
    if (_out.size() < 3) { _out.clear(); }
}
}

namespace Tangram {

CapTypes CapTypeFromString(const std::string& str) {
    if (str == "square") { return CapTypes::square; }
    if (str == "round") { return CapTypes::round; }
    return CapTypes::butt;
}

JoinTypes JoinTypeFromString(const std::string& str) {
    if (str == "bevel") { return JoinTypes::bevel; }
    if (str == "round") { return JoinTypes::round; }
    return JoinTypes::miter;
}

// Tessellate a polygon on the _ctx.gridRes grid over tile-local [0,1]^2 (extending beyond for
//  the tile buffer): earcut the whole polygon exactly as the normal path does, then split each
//  output triangle along the grid lines. Each triangle (and every clipped piece of it) is convex,
//  so Sutherland-Hodgman clipping is exact and each piece fan-triangulates trivially with the
//  original winding preserved. (Clipping the polygon RINGS per cell instead does not work: SH on
//  non-convex rings emits degenerate "bridge" geometry along clip edges, which earcut then
//  triangulates with overlapping/reversed triangles - interiors get culled away.)
// With 3D terrain, per-vertex elevation then makes the polygon surface follow the terrain mesh
//  surface closely (exactly, on interior cells emitted with the terrain mesh's own diagonal), so
//  the small draw-order depth offset suffices to keep the polygon visible on the terrain at any
//  distance - big earcut triangles spanning ridges/valleys would otherwise let terrain poke
//  through them (or paint over ridges in front of them).
static void buildPolygonGrid(const Polygon& _polygon, float _height, glm::vec2 _min, glm::vec2 _max,
                             PolygonBuilder& _ctx) {

    float res = float(_ctx.gridRes);

    // Run earcut on the whole polygon, triangles are stored in _ctx.earcut.indices
    _ctx.earcut(_polygon);
    if (_ctx.earcut.indices.empty()) {
        if (_polygon[0].size() > 3) { LOGD("earcut failed for polygon!"); }
        return;
    }

    // earcut indices refer to the polygon points in linear order across all rings
    std::vector<glm::vec2> pts;
    for (auto& ring : _polygon) { pts.insert(pts.end(), ring.begin(), ring.end()); }

    // share vertices between pieces/cells/triangles via their exact float bit pattern - clip
    //  points on shared edges are computed bit-identically on both sides (see clipConvexAxis),
    //  so this also merges cell corners and grid-line crossings (no cracks, no duplicates)
    std::unordered_map<uint64_t, uint16_t> vertMap;
    bool truncated = false;
    auto emitVertex = [&](glm::vec2 p) -> uint16_t {
        uint32_t xbits, ybits;
        std::memcpy(&xbits, &p.x, sizeof(xbits));
        std::memcpy(&ybits, &p.y, sizeof(ybits));
        uint64_t key = (uint64_t(xbits) << 32) | ybits;
        auto it = vertMap.find(key);
        if (it != vertMap.end()) { return it->second; }
        glm::vec3 coord(p.x, p.y, _height);
        if (_ctx.useTexCoords) {
            glm::vec2 uv(mapRange01(p.x, _min.x, _max.x), mapRange01(p.y, _max.y, _min.y));
            _ctx.addVertex(coord, glm::vec3(0.f, 0.f, 1.f), uv);
        } else {
            _ctx.addVertex(coord, glm::vec3(0.f, 0.f, 1.f), glm::vec2(0));
        }
        uint16_t idx = uint16_t(_ctx.numVertices++);
        vertMap.emplace(key, idx);
        return idx;
    };
    auto emitTriangle = [&](glm::vec2 a, glm::vec2 b, glm::vec2 c) {
        // uint16 index headroom; big landuse polygons stay far below this in practice
        if (_ctx.numVertices > 65200) { truncated = true; return; }
        _ctx.indices.push_back(emitVertex(a));
        _ctx.indices.push_back(emitVertex(b));
        _ctx.indices.push_back(emitVertex(c));
    };

    Line tri(3), colPiece, piece, tmp;
    for (size_t t = 0; t + 2 < _ctx.earcut.indices.size(); t += 3) {
        glm::vec2 A = pts[_ctx.earcut.indices[t]];
        glm::vec2 B = pts[_ctx.earcut.indices[t + 1]];
        glm::vec2 C = pts[_ctx.earcut.indices[t + 2]];
        glm::vec2 lo = glm::vec2(std::min({A.x, B.x, C.x}), std::min({A.y, B.y, C.y}));
        glm::vec2 hi = glm::vec2(std::max({A.x, B.x, C.x}), std::max({A.y, B.y, C.y}));
        int tx0 = int(std::floor(lo.x * res)), tx1 = int(std::ceil(hi.x * res));
        int ty0 = int(std::floor(lo.y * res)), ty1 = int(std::ceil(hi.y * res));
        if (tx1 - tx0 <= 1 && ty1 - ty0 <= 1) {  // triangle already fits in a single cell
            emitTriangle(A, B, C);
            continue;
        }
        // winding of this triangle, for the fixed-diagonal full-cell case below
        bool ccw = (double(B.x - A.x) * (C.y - A.y) - double(C.x - A.x) * (B.y - A.y)) > 0;
        tri = { A, B, C };
        for (int gx = tx0; gx < tx1; gx++) {
            float x0f = gx / res, x1f = (gx + 1) / res;
            clipConvexAxis(tri, tmp, 0, x0f, false);
            clipConvexAxis(tmp, colPiece, 0, x1f, true);
            if (colPiece.empty()) { continue; }
            float ymin = std::numeric_limits<float>::max(), ymax = std::numeric_limits<float>::lowest();
            for (auto& p : colPiece) { ymin = std::min(ymin, p.y); ymax = std::max(ymax, p.y); }
            int cy0 = std::max(ty0, int(std::floor(ymin * res)));
            int cy1 = std::min(ty1, int(std::ceil(ymax * res)));
            for (int gy = cy0; gy < cy1; gy++) {
                float y0f = gy / res, y1f = (gy + 1) / res;
                clipConvexAxis(colPiece, tmp, 1, y0f, false);
                clipConvexAxis(tmp, piece, 1, y1f, true);
                if (piece.empty()) { continue; }
                // cell fully inside the triangle: emit the two triangles with the same diagonal
                //  orientation as the RasterStyle terrain mesh (rasterStyle.cpp), so the draped
                //  surface matches the terrain surface exactly on interior cells. The piece must
                //  contain each of the 4 distinct cell corners - just checking "4 points on the
                //  cell bounds" is fooled by half-cell pieces with a duplicated corner when a
                //  polygon edge crosses the cell exactly corner-to-corner (SH emits the corner
                //  twice), which would double-cover the cell from both adjacent triangles
                bool c00 = false, c10 = false, c01 = false, c11 = false;
                bool fullCell = piece.size() == 4;
                for (size_t k = 0; fullCell && k < piece.size(); k++) {
                    const glm::vec2& p = piece[k];
                    if      (p.x == x0f && p.y == y0f) { c00 = true; }
                    else if (p.x == x1f && p.y == y0f) { c10 = true; }
                    else if (p.x == x0f && p.y == y1f) { c01 = true; }
                    else if (p.x == x1f && p.y == y1f) { c11 = true; }
                    else { fullCell = false; }
                }
                if (fullCell && c00 && c10 && c01 && c11) {
                    if (ccw) {
                        emitTriangle({x0f, y0f}, {x1f, y0f}, {x0f, y1f});
                        emitTriangle({x1f, y0f}, {x1f, y1f}, {x0f, y1f});
                    } else {
                        emitTriangle({x0f, y0f}, {x0f, y1f}, {x1f, y0f});
                        emitTriangle({x1f, y0f}, {x0f, y1f}, {x1f, y1f});
                    }
                    continue;
                }
                // convex piece: fan triangulation, winding preserved by the clipping
                for (size_t k = 1; k + 1 < piece.size(); k++) {
                    emitTriangle(piece[0], piece[k], piece[k + 1]);
                }
            }
        }
    }
    if (truncated) { LOGW("buildPolygonGrid: vertex limit reached, polygon truncated"); }
}

void Builders::buildPolygon(const Polygon& _polygon, float _height, PolygonBuilder& _ctx) {

    if (_polygon.empty() || _polygon[0].empty()) { return; }

    glm::vec2 min, max;
    if (_ctx.useTexCoords || _ctx.gridRes > 0) {
        min = glm::vec2(std::numeric_limits<float>::max());
        max = glm::vec2(std::numeric_limits<float>::lowest());

        for (auto& p : _polygon[0]) {
            min.x = std::min(min.x, p.x);
            min.y = std::min(min.y, p.y);
            max.x = std::max(max.x, p.x);
            max.y = std::max(max.y, p.y);
        }
    }

    // grid tessellation for terrain draping - only if the polygon actually spans multiple grid cells
    if (_ctx.gridRes > 0 && (max.x - min.x > 1.f/_ctx.gridRes || max.y - min.y > 1.f/_ctx.gridRes)) {
        buildPolygonGrid(_polygon, _height, min, max, _ctx);
        return;
    }

    // Run earcut, triangles are stored in _ctx.earcut.indices
    _ctx.earcut(_polygon);

    size_t sumPoints = 0;
    for (auto& line : _polygon) {
        sumPoints += line.size();
    }

    // Mark the points that are referenced by indices as used.
    size_t sumVertices = 0;
    _ctx.used.assign(sumPoints, 0);
    for (auto i : _ctx.earcut.indices) {
        if (_ctx.used[i] == 0) {
            _ctx.used[i] = 1;
            sumVertices++;
        }
    }

    if(!sumVertices && sumPoints > 3) { LOGD("earcut failed for polygon!"); }

    uint16_t vertexDataOffset = _ctx.numVertices;
    _ctx.numVertices += sumVertices;

    size_t ring = 0;
    size_t offset = 0;

    // Go through all points of the polyon.
    for (size_t src = 0, dst = 0; src < sumPoints; src++) {
        // The points of the polygon rings are indexed linearly.
        // This maps the indices back to the original ring and point.
        if (src - offset >= _polygon[ring].size()) {
            offset += _polygon[ring].size();
            ring += 1;
        }

        // Add vertex only when the point is used.
        if (_ctx.used[src] == 0) { continue; }

        // Keep track of skipped points to update indices
        _ctx.used[src] = dst++;

        auto& p = _polygon[ring][src - offset];
        glm::vec3 coord(p.x, p.y, _height);

        if (_ctx.useTexCoords) {
            glm::vec2 uv(mapRange01(coord.x, min.x, max.x), mapRange01(coord.y, max.y, min.y));

            _ctx.addVertex(coord, glm::vec3(0.0, 0.0, 1.0), uv);
        } else {
            _ctx.addVertex(coord, glm::vec3(0.0, 0.0, 1.0), glm::vec2(0));
        }
    }

    for (auto i : _ctx.earcut.indices) {
        _ctx.indices.push_back(vertexDataOffset + _ctx.used[i]);
    }
}

void Builders::buildPolygonExtrusion(const Polygon& _polygon, float _minHeight, float _maxHeight, PolygonBuilder& _ctx) {

    auto vertexDataOffset = _ctx.numVertices;

    static const glm::vec3 upVector(0.0f, 0.0f, 1.0f);
    glm::vec3 normalVector;

    for (auto& line : _polygon) {

        size_t lineSize = line.size();

        for (size_t i = 0; i < lineSize - 1; i++) {

            glm::vec3 a(line[i], 0.f);
            glm::vec3 b(line[i+1], 0.f);

            if (!_ctx.keepTileEdges && isOutsideTile(a, b)) {
                continue;
            }
            normalVector = glm::cross(upVector, b - a);
            normalVector = glm::normalize(normalVector);

            if (std::isnan(normalVector.x)
             || std::isnan(normalVector.y)
             || std::isnan(normalVector.z)) {
                continue;
            }

            // 1st vertex top
            a.z = _maxHeight;
            _ctx.addVertex(a, normalVector, glm::vec2(1.,1.));

            // 2nd vertex top
            b.z = _maxHeight;
            _ctx.addVertex(b, normalVector, glm::vec2(0.,1.));

            // 1st vertex bottom
            a.z = _minHeight;
            _ctx.addVertex(a, normalVector, glm::vec2(1.,0.));

            // 2nd vertex bottom
            b.z = _minHeight;
            _ctx.addVertex(b, normalVector, glm::vec2(0.,0.));

            // Start the index from the previous state of the vertex Data
            _ctx.indices.push_back(vertexDataOffset);
            _ctx.indices.push_back(vertexDataOffset + 1);
            _ctx.indices.push_back(vertexDataOffset + 2);

            _ctx.indices.push_back(vertexDataOffset + 1);
            _ctx.indices.push_back(vertexDataOffset + 3);
            _ctx.indices.push_back(vertexDataOffset + 2);

            vertexDataOffset += 4;
        }

        _ctx.numVertices = vertexDataOffset;
    }
}

// Get 2D perpendicular of two points
static glm::vec2 perp2d(const glm::vec2& _v1, const glm::vec2& _v2 ){
    return glm::vec2(_v2.y - _v1.y, _v1.x - _v2.x);
}

// Helper function for polyline tesselation
static inline void addPolyLineVertex(const glm::vec2& _coord, const glm::vec2& _normal,
                                     const glm::vec2& _uv, PolyLineBuilder& _ctx) {
    _ctx.numVertices++;
    _ctx.addVertex(_coord, _normal, _uv);
}

// Helper function for polyline tesselation; adds indices for pairs of vertices arranged like a line strip
static void indexPairs( int _nPairs, int _nVertices, std::vector<uint16_t>& _indicesOut) {
    for (int i = 0; i < _nPairs; i++) {
        _indicesOut.push_back(_nVertices - 2*i - 4);
        _indicesOut.push_back(_nVertices - 2*i - 2);
        _indicesOut.push_back(_nVertices - 2*i - 3);

        _indicesOut.push_back(_nVertices - 2*i - 3);
        _indicesOut.push_back(_nVertices - 2*i - 2);
        _indicesOut.push_back(_nVertices - 2*i - 1);
    }
}

//  Tessalate a fan geometry between points A       B
//  using their normals from a center        \ . . /
//  and interpolating their UVs               \ p /
//                                             \./
//                                              C
static void addFan(const glm::vec2& _pC,
            const glm::vec2& _nA, const glm::vec2& _nB, const glm::vec2& _nC,
            const glm::vec2& _uA, const glm::vec2& _uB, const glm::vec2& _uC,
            int _numTriangles, PolyLineBuilder& _ctx) {

    // Find angle difference
    float cross = _nA.x * _nB.y - _nA.y * _nB.x; // z component of cross(_CA, _CB)
    float angle = atan2f(cross, glm::dot(_nA, _nB));

    int startIndex = _ctx.numVertices;

    // Add center vertex
    addPolyLineVertex(_pC, _nC, _uC, _ctx);

    // Add vertex for point A
    addPolyLineVertex(_pC, _nA, _uA, _ctx);

    // Add radial vertices
    glm::vec2 radial = _nA;
    for (int i = 0; i < _numTriangles; i++) {
        float frac = (i + 1)/(float)_numTriangles;
        radial = glm::rotate(_nA, angle * frac);

        glm::vec2 uv(0.0);
        if (_ctx.useTexCoords) {
            uv = (1.f - frac) * _uA + frac * _uB;
        }

        addPolyLineVertex(_pC, radial, uv, _ctx);

        // Add indices
        _ctx.indices.push_back(startIndex); // center vertex
        _ctx.indices.push_back(startIndex + i + (angle > 0 ? 1 : 2));
        _ctx.indices.push_back(startIndex + i + (angle > 0 ? 2 : 1));
    }

}

// Function to add the vertices for line caps
static void addCap(const glm::vec2& _coord, const glm::vec2& _normal, int _numCorners, bool _isBeginning, PolyLineBuilder& _ctx) {

    float v = _isBeginning ? 0.f : 1.f; // length-wise tex coord

    if (_numCorners < 1) {
        // "Butt" cap needs no extra vertices
        return;
    } else if (_numCorners == 2) {
        // "Square" cap needs two extra vertices
        glm::vec2 tangent(-_normal.y, _normal.x);
        addPolyLineVertex(_coord, _normal + tangent, {0.f, v}, _ctx);
        addPolyLineVertex(_coord, -_normal + tangent, {0.f, v}, _ctx);
        if (!_isBeginning) { // At the beginning of a line we can't form triangles with previous vertices
            indexPairs(1, _ctx.numVertices, _ctx.indices);
        }
        return;
    }

    // "Round" cap type needs a fan of vertices
    glm::vec2 nA(_normal), nB(-_normal), nC(0.f, 0.f), uA(1.f, v), uB(0.f, v), uC(0.5f, v);
    if (_isBeginning) {
        nA *= -1.f; // To flip the direction of the fan, we negate the normal vectors
        nB *= -1.f;
        uA.x = 0.f; // To keep tex coords consistent, we must reverse these too
        uB.x = 1.f;
    }
    addFan(_coord, nA, nB, nC, uA, uB, uC, _numCorners, _ctx);
}

static void buildPolyLineSegment(const Line& _line, PolyLineBuilder& _ctx, size_t _startIndex,
                          size_t _endIndex, bool startCap = true, bool endCap = true) {

    float distance = 0; // Cumulative distance along the polyline.

    size_t origLineSize = _line.size();

    // endIndex/startIndex could be wrapped values, calculate lineSize accordingly
    int lineSize = (int)((_endIndex > _startIndex) ?
                   (_endIndex - _startIndex) :
                   (origLineSize - _startIndex + _endIndex));
    if (lineSize < 2) { return; }

    glm::vec2 coordCurr(_line[_startIndex]);
    // get the Point using wrapped index in the original line geometry
    glm::vec2 coordNext(_line[(_startIndex + 1) % origLineSize]);
    glm::vec2 normPrev, normNext, miterVec;

    int cornersOnCap = (int)_ctx.cap;
    int trianglesOnJoin = (int)_ctx.join;

    // Process first point in line with an end cap
    normNext = glm::normalize(perp2d(coordCurr, coordNext));

    if (startCap) {
        addCap(coordCurr, normNext, cornersOnCap, true, _ctx);
    }
    addPolyLineVertex(coordCurr, normNext, {1.0f, 0.0f}, _ctx); // right corner
    addPolyLineVertex(coordCurr, -normNext, {0.0f, 0.0f}, _ctx); // left corner


    // Process intermediate points
    for (int i = 1; i < lineSize - 1; i++) {
        // get the Point using wrapped index in the original line geometry
        int nextIndex = (i + _startIndex + 1) % origLineSize;

        distance += glm::distance(coordCurr, coordNext);

        coordCurr = coordNext;
        coordNext = _line[nextIndex];

        if (coordCurr == coordNext) {
            continue;
        }

        normPrev = normNext;
        normNext = glm::normalize(perp2d(coordCurr, coordNext));

        // Compute "normal" for miter joint
        miterVec = normPrev + normNext;

        float scale = 1.f;

        // normPrev and normNext are in the opposite direction
        // in order to prevent NaN values, we use the perp
        // vector of those two vectors
        if (miterVec == glm::zero<glm::vec2>()) {
            miterVec = perp2d(glm::vec3(normNext, 0.f), glm::vec3(normPrev, 0.f));
        } else {
            scale = 2.f / glm::dot(miterVec, miterVec);
        }

        miterVec *= scale;

        if (glm::length2(miterVec) > glm::length2(_ctx.miterLimit)) {
            trianglesOnJoin = 1;
            miterVec *= _ctx.miterLimit / glm::length(miterVec);
        }

        float v = distance;

        if (trianglesOnJoin == 0) {
            // Join type is a simple miter

            addPolyLineVertex(coordCurr, miterVec, {1.0, v}, _ctx); // right corner
            addPolyLineVertex(coordCurr, -miterVec, {0.0, v}, _ctx); // left corner
            indexPairs(1, _ctx.numVertices, _ctx.indices);

        } else {

            // Join type is a fan of triangles

            bool isRightTurn = (normNext.x * normPrev.y - normNext.y * normPrev.x) > 0; // z component of cross(normNext, normPrev)

            if (isRightTurn) {

                addPolyLineVertex(coordCurr, miterVec, {1.0f, v}, _ctx); // right (inner) corner
                addPolyLineVertex(coordCurr, -normPrev, {0.0f, v}, _ctx); // left (outer) corner
                indexPairs(1, _ctx.numVertices, _ctx.indices);

                addFan(coordCurr, -normPrev, -normNext, miterVec, {0.f, v}, {0.f, v}, {1.f, v}, trianglesOnJoin, _ctx);

                addPolyLineVertex(coordCurr, miterVec, {1.0f, v}, _ctx); // right (inner) corner
                addPolyLineVertex(coordCurr, -normNext, {0.0f, v}, _ctx); // left (outer) corner

            } else {

                addPolyLineVertex(coordCurr, normPrev, {1.0f, v}, _ctx); // right (outer) corner
                addPolyLineVertex(coordCurr, -miterVec, {0.0f, v}, _ctx); // left (inner) corner
                indexPairs(1, _ctx.numVertices, _ctx.indices);

                addFan(coordCurr, normPrev, normNext, -miterVec, {1.f, v}, {1.f, v}, {0.0f, v}, trianglesOnJoin, _ctx);

                addPolyLineVertex(coordCurr, normNext, {1.0f, v}, _ctx); // right (outer) corner
                addPolyLineVertex(coordCurr, -miterVec, {0.0f, v}, _ctx); // left (inner) corner
            }
        }
    }

    distance += glm::distance(coordCurr, coordNext);

    // Process last point in line with a cap
    addPolyLineVertex(coordNext, normNext, {1.f, distance}, _ctx); // right corner
    addPolyLineVertex(coordNext, -normNext, {0.f, distance}, _ctx); // left corner
    indexPairs(1, _ctx.numVertices, _ctx.indices);
    if (endCap) {
        addCap(coordNext, normNext, cornersOnCap, false, _ctx);
    }

}

void Builders::buildPolyLine(const Line& _line, PolyLineBuilder& _ctx) {

    size_t lineSize = _line.size();

    if (_ctx.keepTileEdges) {

        buildPolyLineSegment(_line, _ctx, 0, lineSize);

    } else {

        size_t cut = 0;
        size_t firstCutEnd = 0;

        // Determine cuts
        for (size_t i = 0; i < lineSize - 1; i++) {
            const glm::vec2& coordCurr = _line[i];
            const glm::vec2& coordNext = _line[i+1];
            if (isOutsideTile(coordCurr, coordNext)) {
                if (cut == 0) {
                    firstCutEnd = i + 1;
                }
                buildPolyLineSegment(_line, _ctx, cut, i + 1);
                cut = i + 1;
            } else if (!_ctx.closedPolygon) {
                // note that both endpoints can be outside tile even if the line segment intersects tile!
                bool currOutside = isOutsideTile(coordCurr);
                bool nextOutside = isOutsideTile(coordNext);

                if (currOutside || nextOutside) {
                    if (!currOutside) {
                        buildPolyLineSegment(_line, _ctx, cut, i + 1, true, false);
                    }
                    glm::vec2 a = coordCurr, b = coordNext;
                    if (clipLine(a, b, {0, 0}, {1, 1})) {
                        buildPolyLineSegment({ a, b }, _ctx, 0, 2,
                                             !currOutside && i == 0, !nextOutside && i+1 == lineSize-1);
                    }
                    cut = i + 1;
                }
            }
        }

        if (_ctx.closedPolygon) {
            if (cut == 0) {
                // no tile edge cuts!
                // loop and close the polygon with no endcaps
                buildPolyLineSegment(_line, _ctx, 0, lineSize+2, false, false);
            } else {
                // merge first and last cut line-segments together
                buildPolyLineSegment(_line, _ctx, cut, firstCutEnd);
            }
        } else {
            buildPolyLineSegment(_line, _ctx, cut, lineSize);
        }

    }

}

void Builders::buildQuadAtPoint(const glm::vec2& _screenPosition, const glm::vec2& _size, const glm::vec2& _uvBL, const glm::vec2& _uvTR, SpriteBuilder& _ctx) {
    float halfWidth = _size.x * .5f;
    float halfHeight = _size.y * .5f;

    _ctx.addVertex(glm::vec2(-halfWidth, -halfHeight), _screenPosition, {_uvBL.x, _uvBL.y});
    _ctx.addVertex(glm::vec2(-halfWidth, halfHeight), _screenPosition, {_uvBL.x, _uvTR.y});
    _ctx.addVertex(glm::vec2(halfWidth, -halfHeight), _screenPosition, {_uvTR.x, _uvBL.y});
    _ctx.addVertex(glm::vec2(halfWidth, halfHeight), _screenPosition, {_uvTR.x, _uvTR.y});

    _ctx.indices.push_back(_ctx.numVerts + 2);
    _ctx.indices.push_back(_ctx.numVerts + 0);
    _ctx.indices.push_back(_ctx.numVerts + 1);
    _ctx.indices.push_back(_ctx.numVerts + 1);
    _ctx.indices.push_back(_ctx.numVerts + 3);
    _ctx.indices.push_back(_ctx.numVerts + 2);

    _ctx.numVerts += 4;

}

}
