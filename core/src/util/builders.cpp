#include "util/builders.h"
#include "util/waterMask.h"

#include "util/geom.h"
#include "log.h"
#include "glm/gtx/rotate_vector.hpp"
#include "glm/gtx/norm.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
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

// Distance to the polygon boundary (capped), via a uniform bucket grid over the polygon's bbox. Segments lying ON
//  the tile / tile-buffer clip lines are not real boundaries and are skipped.
struct EdgeDistField {
    float cap = 0.f, cell = 1.f;
    glm::vec2 org;
    int nx = 1, ny = 1;
    std::vector<std::pair<glm::vec2, glm::vec2>> segs;
    std::vector<std::vector<uint32_t>> buckets;

    void build(const Tangram::Polygon& _poly, glm::vec2 _min, glm::vec2 _max, float _cap) {
        cap = _cap;
        const float tol = 0.0005f;
        for (auto& ring : _poly) {
            size_t n = ring.size();
            for (size_t i = 0; i < n; i++) {
                glm::vec2 a = ring[i], b = ring[(i + 1) % n];
                if (a == b) { continue; }
                bool clipX = std::abs(a.x - b.x) < 1e-6f && (a.x <= tol || a.x >= 1.f - tol);
                bool clipY = std::abs(a.y - b.y) < 1e-6f && (a.y <= tol || a.y >= 1.f - tol);
                if (clipX || clipY) { continue; }
                segs.emplace_back(a, b);
            }
        }
        glm::vec2 ext = _max - _min + glm::vec2(2.f * cap);
        cell = std::max(cap, std::max(ext.x, ext.y) / 256.f);
        org = _min - glm::vec2(cap);
        nx = std::max(1, int(std::ceil(ext.x / cell)));
        ny = std::max(1, int(std::ceil(ext.y / cell)));
        buckets.assign(size_t(nx) * size_t(ny), {});
        for (uint32_t k = 0; k < segs.size(); k++) {
            glm::vec2 lo = glm::min(segs[k].first, segs[k].second) - glm::vec2(cap);
            glm::vec2 hi = glm::max(segs[k].first, segs[k].second) + glm::vec2(cap);
            int x0 = std::max(0, int(std::floor((lo.x - org.x) / cell))), x1 = std::min(nx - 1, int(std::floor((hi.x - org.x) / cell)));
            int y0 = std::max(0, int(std::floor((lo.y - org.y) / cell))), y1 = std::min(ny - 1, int(std::floor((hi.y - org.y) / cell)));
            for (int y = y0; y <= y1; y++) { for (int x = x0; x <= x1; x++) { buckets[size_t(y) * nx + x].push_back(k); } }
        }
    }
    float dist(glm::vec2 p) const {
        int cx = int(std::floor((p.x - org.x) / cell)), cy = int(std::floor((p.y - org.y) / cell));
        if (cx < 0 || cy < 0 || cx >= nx || cy >= ny) { return cap; }
        float best = cap * cap;
        for (uint32_t k : buckets[size_t(cy) * nx + cx]) {
            glm::vec2 a = segs[k].first, ab = segs[k].second - a, ap = p - a;
            float t = glm::clamp(glm::dot(ap, ab) / std::max(glm::dot(ab, ab), 1e-20f), 0.f, 1.f);
            glm::vec2 d = ap - t * ab;
            best = std::min(best, glm::dot(d, d));
        }
        return std::sqrt(best);
    }
};

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

// Same, against the cell's anti-diagonal (x - x0) + (y - y0) = _cell - the diagonal the terrain
//  mesh splits every lattice cell on (rasterStyle.cpp). Splitting partial pieces on it makes every
//  emitted triangle lie inside ONE terrain triangle, so a draped vertex-evaluated surface matches
//  the terrain exactly instead of chording across a steep bump within a cell (the residual
//  "gap at the bump" on near-vertical faces: pieces cut by polygon edges / earcut edges were
//  fan-triangulated with arbitrary diagonals, unlike the full cells).
void clipConvexDiag(const Tangram::Line& _in, Tangram::Line& _out, float _x0, float _y0, float _cell,
                    bool _keepBelow) {
    _out.clear();
    size_t n = _in.size();
    auto f = [&](const glm::vec2& p) { return (p.x - _x0) + (p.y - _y0) - _cell; };
    for (size_t i = 0; i < n; i++) {
        glm::vec2 a = _in[i];
        glm::vec2 b = _in[(i + 1) % n];
        float fa = f(a), fb = f(b);
        bool ina = _keepBelow ? fa <= 0.f : fa >= 0.f;
        bool inb = _keepBelow ? fb <= 0.f : fb >= 0.f;
        if (ina) { _out.push_back(a); }
        if (ina != inb) {
            float t = fa / (fa - fb);
            _out.push_back(a + t * (b - a));
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
    EdgeDistField edgeField;
    if (_ctx.edgeDistCap > 0.f) { edgeField.build(_polygon, _min, _max, _ctx.edgeDistCap); }
    // NOT necessarily 0: buildPolygonExtrusion (polygonStyle.cpp addPolygon) may already have
    //  pushed wall indices into _ctx.indices for this same feature before grid tessellation
    //  runs (extrude + terrain_grid on the same style) - chunk index counts must exclude those.
    //  No current style combines both, so this is currently always indices.size()==0, but the
    //  first flush's chunk.first would otherwise silently include the extrusion's index count.
    size_t chunkIndexStart = _ctx.indices.size();
    auto emitVertex = [&](glm::vec2 p) -> uint16_t {
        uint32_t xbits, ybits;
        std::memcpy(&xbits, &p.x, sizeof(xbits));
        std::memcpy(&ybits, &p.y, sizeof(ybits));
        uint64_t key = (uint64_t(xbits) << 32) | ybits;
        auto it = vertMap.find(key);
        if (it != vertMap.end()) { return it->second; }
        glm::vec3 coord(p.x, p.y, _height);
        if (_ctx.edgeDistCap > 0.f) {
            _ctx.addVertex(coord, glm::vec3(0.f, 0.f, 1.f), glm::vec2(edgeField.dist(p) / _ctx.edgeDistCap, 0.f));
        } else if (_ctx.useTexCoords) {
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
        _ctx.indices.push_back(emitVertex(a));
        _ctx.indices.push_back(emitVertex(b));
        _ctx.indices.push_back(emitVertex(c));
    };

    // Emit a convex piece lying inside grid cell (gx, gy), split on the cell's terrain-mesh
    //  diagonal so no emitted triangle straddles two terrain triangles (see clipConvexDiag).
    Line diagLo, diagHi;
    auto emitCellPiece = [&](const Line& pc, int gx, int gy) {
        float x0f = gx / res, y0f = gy / res, cell = 1.f / res;
        clipConvexDiag(pc, diagLo, x0f, y0f, cell, true);
        clipConvexDiag(pc, diagHi, x0f, y0f, cell, false);
        if (diagLo.empty() && diagHi.empty()) {
            for (size_t k = 1; k + 1 < pc.size(); k++) { emitTriangle(pc[0], pc[k], pc[k + 1]); }
            return;
        }
        for (const Line* part : { &diagLo, &diagHi }) {
            for (size_t k = 1; k + 1 < part->size(); k++) {
                emitTriangle((*part)[0], (*part)[k], (*part)[k + 1]);
            }
        }
    };

    // Fast path for interior lattice-corner vertices (tile-pipeline-perf-plan.md R1.1): the
    //  full-cell case below emits the SAME 4 grid corners for every cell, each shared by up to
    //  4 neighboring cells, so routing every corner reference through vertMap's hash costs 3
    //  redundant lookups per corner. Cache corners by direct (gx, gy) index instead, over just
    //  this polygon's own cell-index bounding box (not the whole tile grid) so the array stays
    //  small for the common small-polygon case. Falls through to emitVertex (still the sole
    //  vertex-creation authority, so the exact-bit-pattern sharing guarantee is unaffected) on
    //  first touch of each corner; only repeat touches skip the hash.
    int gxMin = int(std::floor(_min.x * res)), gxMax = int(std::ceil(_max.x * res));
    int gyMin = int(std::floor(_min.y * res)), gyMax = int(std::ceil(_max.y * res));
    int gridW = std::max(1, gxMax - gxMin + 1), gridH = std::max(1, gyMax - gyMin + 1);
    std::vector<uint16_t> latticeIndex(size_t(gridW) * size_t(gridH), 0xFFFFu);
    auto emitLatticeVertex = [&](int _gx, int _gy) -> uint16_t {
        size_t li = size_t(_gy - gyMin) * size_t(gridW) + size_t(_gx - gxMin);
        uint16_t idx = latticeIndex[li];
        if (idx != 0xFFFFu) { return idx; }
        idx = emitVertex(glm::vec2(float(_gx) / res, float(_gy) / res));
        latticeIndex[li] = idx;
        return idx;
    };
    auto emitLatticeTriangle = [&](int ax, int ay, int bx, int by, int cx, int cy) {
        _ctx.indices.push_back(emitLatticeVertex(ax, ay));
        _ctx.indices.push_back(emitLatticeVertex(bx, by));
        _ctx.indices.push_back(emitLatticeVertex(cx, cy));
    };
    // Flush the current sub-range whenever it's close to the uint16 ceiling (headroom for the
    //  handful of vertices one grid cell or fallback triangle can add) and start a fresh 0-based
    //  block; see PolygonBuilder::chunkOffsets. Checked once per grid CELL (not once per earcut
    //  triangle) because a single large earcut triangle can by itself span enough cells to
    //  overflow the budget before finishing.
    auto flushChunkIfNeeded = [&]() {
        if (_ctx.numVertices <= 65000) { return; }
        _ctx.chunkOffsets.emplace_back(_ctx.indices.size() - chunkIndexStart, _ctx.numVertices);
        chunkIndexStart = _ctx.indices.size();
        _ctx.numVertices = 0;
        vertMap.clear();
        // Cached indices refer to the chunk just flushed - stale (and wrongly reused) against
        //  the fresh 0-based block that follows, same reason vertMap is cleared above.
        std::fill(latticeIndex.begin(), latticeIndex.end(), uint16_t(0xFFFFu));
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
            flushChunkIfNeeded();
            tri = { A, B, C };
            emitCellPiece(tri, tx0, ty0);
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
                flushChunkIfNeeded();
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
                    // (gx, gy)/(gx+1, gy)/(gx, gy+1)/(gx+1, gy+1) are exactly x0f/y0f/x1f/y1f's
                    //  lattice indices - route through the direct-index cache (R1.1) instead of
                    //  emitTriangle's hash lookup, since every interior corner here is shared by
                    //  up to 4 cells.
                    if (ccw) {
                        emitLatticeTriangle(gx, gy, gx + 1, gy, gx, gy + 1);
                        emitLatticeTriangle(gx + 1, gy, gx + 1, gy + 1, gx, gy + 1);
                    } else {
                        emitLatticeTriangle(gx, gy, gx, gy + 1, gx + 1, gy);
                        emitLatticeTriangle(gx + 1, gy, gx, gy + 1, gx + 1, gy + 1);
                    }
                    continue;
                }
                // convex partial piece: split on the terrain diagonal, fan-triangulate each part
                //  (winding preserved by the clipping)
                emitCellPiece(piece, gx, gy);
            }
        }
    }
}

// ---- tree scatter (see PolygonBuilder::TreeScatter) ----
namespace {
inline uint32_t treePcg(uint32_t v) {
    uint32_t s = v * 747796405u + 2891336453u;
    uint32_t w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}
}

void Builders::buildTreeScatter(const Polygon& _polygon, PolygonBuilder& _ctx) {
    const auto& ts = _ctx.trees;
    if (_polygon.empty() || _polygon[0].size() < 3 || ts.cells <= 0) { return; }
    const int N = ts.cells;
    glm::vec2 mn(std::numeric_limits<float>::max()), mx(std::numeric_limits<float>::lowest());
    for (auto& p : _polygon[0]) { mn = glm::min(mn, p); mx = glm::max(mx, p); }
    int i0 = std::max(0, int(std::floor(mn.x * N))), i1 = std::min(N - 1, int(std::floor(mx.x * N)));
    int j0 = std::max(0, int(std::floor(mn.y * N))), j1 = std::min(N - 1, int(std::floor(mx.y * N)));
    if (i1 < i0 || j1 < j0) { return; }

    // edges bucketed per grid row (point-in-polygon by crossing parity, only the edges of the tree's own row)
    struct Seg { glm::vec2 a, b; };
    std::vector<Seg> segs;
    for (auto& ring : _polygon) {
        size_t n = ring.size();
        for (size_t k = 0; k < n; k++) {
            glm::vec2 a = ring[k], b = ring[(k + 1) % n];
            if (a.y != b.y) { segs.push_back({a, b}); }
        }
    }
    std::vector<std::vector<uint32_t>> rows(size_t(j1 - j0 + 1));
    for (uint32_t k = 0; k < segs.size(); k++) {
        float ylo = std::min(segs[k].a.y, segs[k].b.y), yhi = std::max(segs[k].a.y, segs[k].b.y);
        int r0 = std::max(j0, int(std::floor(ylo * N))), r1 = std::min(j1, int(std::floor(yhi * N)));
        for (int r = r0; r <= r1; r++) { rows[size_t(r - j0)].push_back(k); }
    }

    const float spacing = 1.f / float(N);                       // tile units
    const float feather = ts.featherM / ts.tileScale;           // tile units
    EdgeDistField field;
    field.build(_polygon, mn, mx, std::max(feather * 1.05f, spacing));
    const float crownR = 0.27f * spacing;
    const float spacingM = spacing * ts.tileScale;              // projected metres (shader scales the tree size by this)

    size_t chunkIndexStart = _ctx.indices.size();
    for (int j = j0; j <= j1; j++) {
        const auto& row = rows[size_t(j - j0)];
        if (row.empty()) { continue; }
        for (int i = i0; i <= i1; i++) {
            uint32_t gi = ts.tileX * uint32_t(N) + uint32_t(i), gj = ts.tileY * uint32_t(N) + uint32_t(j);
            uint32_t h0 = treePcg(gi ^ treePcg(gj + 0x9E3779B9u));
            float u[4];
            for (int k = 0; k < 4; k++) { u[k] = float(treePcg(h0 + uint32_t(k) * 0x85EBCA6Bu)) * (1.f / 4294967296.f); }
            if (u[2] > 0.92f) { continue; }                      // a few empty cells
            glm::vec2 p((float(i) + 0.5f + (u[0] - 0.5f) * 0.9f) * spacing, (float(j) + 0.5f + (u[1] - 0.5f) * 0.9f) * spacing);
            // inside? (even-odd over all rings: holes work)
            bool inside = false;
            for (uint32_t k : row) {
                const Seg& s = segs[k];
                if ((s.a.y > p.y) != (s.b.y > p.y)) {
                    float xc = s.a.x + (p.y - s.a.y) / (s.b.y - s.a.y) * (s.b.x - s.a.x);
                    if (xc > p.x) { inside = !inside; }
                }
            }
            if (!inside) { continue; }
            // thinning: this tree is kept iff its random id is below the density at ITS OWN position - whole trees only
            float d = field.dist(p);
            float rho = glm::clamp((d - crownR) / std::max(feather - crownR, 1e-6f), 0.f, 1.f);
            if (u[3] >= rho) { continue; }
            if (_ctx.numVertices + 12 > 65000) {
                _ctx.chunkOffsets.emplace_back(_ctx.indices.size() - chunkIndexStart, _ctx.numVertices);
                chunkIndexStart = _ctx.indices.size();
                _ctx.numVertices = 0;
            }
            float rnd = float(treePcg(h0 ^ 0x68E31DA4u)) * (1.f / 4294967296.f);
            glm::vec2 uv(rnd, std::min(spacingM / 256.f, 1.f));
            for (int pl = 0; pl < 3; pl++) {
                uint16_t b = uint16_t(_ctx.numVertices);
                const float cu[4] = { -1.f, 1.f, 1.f, -1.f }, cv[4] = { 0.f, 0.f, 1.f, 1.f };
                for (int c = 0; c < 4; c++) {
                    _ctx.addVertex(glm::vec3(p.x, p.y, 0.f), glm::vec3(cu[c], cv[c], float(pl) * 40.f / 127.f), uv);
                    _ctx.numVertices++;
                }
                // both windings: the crosses are seen from every side
                const uint16_t tri[12] = { 0, 1, 2, 0, 2, 3, 0, 2, 1, 0, 3, 2 };
                for (uint16_t t : tri) { _ctx.indices.push_back(uint16_t(b + t)); }
            }
        }
    }
}

void Builders::buildPolygonEdgeBand(const Polygon& _polygon, float _width, float _lift, const WaterMask* _water, PolygonBuilder& _ctx) {

    if (_polygon.empty() || _width <= 0.f) { return; }

    const float cell = 1.f / float(_ctx.gridRes > 0 ? _ctx.gridRes : 64);

    // --- rings, closing duplicate dropped
    std::vector<Line> rings;
    for (auto& r : _polygon) {
        Line l = r;
        if (l.size() > 1 && l.front() == l.back()) { l.pop_back(); }
        rings.push_back(std::move(l));
    }

    // --- segment buckets for the "thin polygon" cap
    struct Seg { glm::vec2 a, b; int ring; int idx; };
    std::vector<Seg> segs;
    glm::vec2 lo(std::numeric_limits<float>::max()), hi(std::numeric_limits<float>::lowest());
    for (size_t r = 0; r < rings.size(); r++) {
        const Line& l = rings[r];
        for (size_t i = 0; i < l.size(); i++) {
            lo = glm::min(lo, l[i]);
            hi = glm::max(hi, l[i]);
            // a tile-edge cut is not a real boundary: it must not shorten the band of the edges near it
            if (!_ctx.keepTileEdges && isOutsideTile(l[i], l[(i + 1) % l.size()])) { continue; }
            segs.push_back({ l[i], l[(i + 1) % l.size()], int(r), int(i) });
        }
    }
    if (segs.empty()) { return; }
    const int G = 48;
    const glm::vec2 ext = glm::max(hi - lo, glm::vec2(1e-6f));
    auto cellOf = [&](glm::vec2 p, int& cx, int& cy) {
        cx = std::max(0, std::min(G - 1, int((p.x - lo.x) / ext.x * G)));
        cy = std::max(0, std::min(G - 1, int((p.y - lo.y) / ext.y * G)));
    };
    std::vector<std::vector<uint32_t>> buckets(G * G);
    for (uint32_t k = 0; k < segs.size(); k++) {
        int x0, y0, x1, y1;
        cellOf(glm::min(segs[k].a, segs[k].b), x0, y0);
        cellOf(glm::max(segs[k].a, segs[k].b), x1, y1);
        for (int y = y0; y <= y1; y++) {
            for (int x = x0; x <= x1; x++) { buckets[y * G + x].push_back(k); }
        }
    }
    std::vector<uint32_t> stamp(segs.size(), 0);
    uint32_t stampId = 0;

    size_t chunkIndexStart = _ctx.indices.size();
    auto flushChunkIfNeeded = [&](size_t incoming) {
        if (_ctx.numVertices + incoming <= 65000) { return; }
        _ctx.chunkOffsets.emplace_back(_ctx.indices.size() - chunkIndexStart, _ctx.numVertices);
        chunkIndexStart = _ctx.indices.size();
        _ctx.numVertices = 0;
    };

    for (size_t r = 0; r < rings.size(); r++) {
        const Line& l = rings[r];
        const size_t n = l.size();
        if (n < 3) { continue; }

        float area2 = 0.f;
        for (size_t i = 0; i < n; i++) {
            const glm::vec2& a = l[i];
            const glm::vec2& b = l[(i + 1) % n];
            area2 += a.x * b.y - b.x * a.y;
        }
        if (std::fabs(area2) < 1e-12f) { continue; }
        // polygon material on the left of the travel direction for a CCW exterior / CW hole
        const float side = (area2 > 0.f ? 1.f : -1.f) * (r == 0 ? 1.f : -1.f);

        std::vector<char> skip(n, 0);       // edge i -> i+1 lies on a tile-edge cut
        std::vector<glm::vec2> nrm(n);      // inward unit normal of edge i
        std::vector<float> len(n);
        for (size_t i = 0; i < n; i++) {
            const glm::vec2& a = l[i];
            const glm::vec2& b = l[(i + 1) % n];
            skip[i] = !_ctx.keepTileEdges && isOutsideTile(a, b);
            glm::vec2 t = b - a;
            len[i] = glm::length(t);
            if (len[i] > 1e-12f) { t /= len[i]; }
            nrm[i] = side * glm::vec2(-t.y, t.x);
        }

        // per-vertex inward offset (miter-scaled) and available depth
        std::vector<glm::vec2> off(n, glm::vec2(0.f));
        std::vector<float> depth(n, 0.f);
        for (size_t i = 0; i < n; i++) {
            const size_t pe = (i + n - 1) % n;
            if (skip[pe] && skip[i]) { continue; }
            glm::vec2 dir;
            float scale = 1.f;
            if (skip[pe]) { dir = nrm[i]; }
            else if (skip[i]) { dir = nrm[pe]; }
            else {
                glm::vec2 m = nrm[pe] + nrm[i];
                float ml = glm::length(m);
                if (ml < 1e-6f) { dir = nrm[i]; }
                else {
                    dir = m / ml;
                    scale = std::min(2.5f, 1.f / std::max(0.4f, glm::dot(dir, nrm[i])));
                }
            }
            off[i] = dir * scale;

            // distance to the nearest non-adjacent boundary along the inward ray (looked for up to
            //  4 nominal widths away; no hit = no cap, the shader clamps to the real width)
            float hit = std::numeric_limits<float>::max();
            const float reach = _width * 4.f;
            ++stampId;
            const int steps = std::max(1, int(std::ceil(reach / (ext.x / G))));
            for (int s = 0; s <= std::min(steps, 4 * G); s++) {
                glm::vec2 q = l[i] + dir * (reach * float(s) / float(steps));
                if (q.x < lo.x - 1e-6f || q.y < lo.y - 1e-6f || q.x > hi.x + 1e-6f || q.y > hi.y + 1e-6f) { continue; }
                int cx, cy;
                cellOf(q, cx, cy);
                for (uint32_t k : buckets[cy * G + cx]) {
                    if (stamp[k] == stampId) { continue; }
                    stamp[k] = stampId;
                    const Seg& sg = segs[k];
                    if (sg.ring == int(r) && (sg.idx == int(i) || sg.idx == int(pe))) { continue; }
                    // ray (l[i], dir) vs segment (sg.a, sg.b)
                    glm::vec2 e = sg.b - sg.a;
                    float den = dir.x * e.y - dir.y * e.x;
                    if (std::fabs(den) < 1e-12f) { continue; }
                    glm::vec2 w = sg.a - l[i];
                    float tt = (w.x * e.y - w.y * e.x) / den;
                    float uu = (w.x * dir.y - w.y * dir.x) / den;
                    if (tt > 1e-7f && uu >= 0.f && uu <= 1.f && tt <= reach) { hit = std::min(hit, tt); }
                }
            }
            depth[i] = std::min(1.f, 0.5f * hit);
        }
        // the cap must not jump between neighbours or the band edge ripples
        for (int pass = 0; pass < 2; pass++) {
            for (size_t i = 0; i < n; i++) {
                const size_t pe = (i + n - 1) % n, ne = (i + 1) % n;
                if (depth[i] > 0.f) { depth[i] = std::min(depth[i], 1.5f * std::max(depth[pe], depth[ne]) + 1e-6f); }
            }
        }

        // strips, one run per stretch of an edge that is wholly land or wholly water
        std::vector<float> cuts;
        for (size_t i = 0; i < n; i++) {
            if (skip[i] || len[i] < 1e-9f) { continue; }
            const size_t j = (i + 1) % n;
            if (depth[i] <= 0.f && depth[j] <= 0.f) { continue; }

            std::vector<float> ts{ 0.f };
            std::vector<char> water;
            if (_water) {
                _water->crossings(l[i], l[j], cuts);
                for (float t : cuts) { if (t - ts.back() > 1e-5f) { ts.push_back(t); } }
            }
            ts.push_back(1.f);
            for (size_t k = 0; k + 1 < ts.size(); k++) {
                water.push_back(_water && _water->contains(l[i] + (l[j] - l[i]) * (0.5f * (ts[k] + ts[k + 1]))));
            }
            // merge neighbours of the same kind (a boundary that only touched the shore)
            for (size_t k = 1; k < water.size();) {
                if (water[k] == water[k - 1]) { water.erase(water.begin() + k); ts.erase(ts.begin() + k); }
                else { k++; }
            }

            // finer than the terrain lattice (cell/2) both ways so the strip hugs folds instead of chording
            const int rows = std::max(3, std::min(40, int(std::ceil(_width / (0.5f * cell))))) + 1;
            for (size_t k = 0; k < water.size(); k++) {
                const float s0 = ts[k], s1 = ts[k + 1];
                const int cols = std::max(1, int(std::ceil(len[i] * (s1 - s0) / (0.5f * cell)))) + 1;
                flushChunkIfNeeded(size_t(cols) * rows);
                const uint16_t base = uint16_t(_ctx.numVertices);
                for (int c = 0; c < cols; c++) {
                    const float tc = s0 + (s1 - s0) * float(c) / float(cols - 1);
                    const glm::vec2 p = l[i] + (l[j] - l[i]) * tc;
                    const glm::vec2 o = (off[i] + (off[j] - off[i]) * tc) * (1.f / 2.5f);
                    const float d = std::max(0.f, std::min(1.f, depth[i] + (depth[j] - depth[i]) * tc));
                    for (int q = 0; q < rows; q++) {
                        const float tr = float(q) / float(rows - 1);
                        _ctx.addVertex(glm::vec3(p.x, p.y, _lift), glm::vec3(o.x, o.y, water[k] ? -1.f : 1.f),
                                       glm::vec2(tr, d));
                        _ctx.numVertices++;
                    }
                }
                for (int c = 0; c + 1 < cols; c++) {
                    for (int q = 0; q + 1 < rows; q++) {
                        uint16_t a = base + c * rows + q, b = base + (c + 1) * rows + q;
                        uint16_t cc = b + 1, dd = a + 1;
                        if (side > 0.f) {
                            _ctx.indices.insert(_ctx.indices.end(), { a, b, cc, a, cc, dd });
                        } else {
                            _ctx.indices.insert(_ctx.indices.end(), { a, cc, b, a, dd, cc });
                        }
                    }
                }
            }
        }
    }
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

        if (_ctx.edgeDistCap > 0.f) {
            // not grid-tessellated: every vertex is a boundary point of a tiny polygon (gridRes > 0), or no terrain
            _ctx.addVertex(coord, glm::vec3(0.0, 0.0, 1.0), glm::vec2(_ctx.gridRes > 0 ? 0.f : 1.f, 0.f));
        } else if (_ctx.useTexCoords) {
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


int terrainMeshZoomBias() {
    static const int bias = [] {
        const char* e = std::getenv("ASCEND_TERRAIN_ZBIAS");
        return e ? std::atoi(e) : 100;
    }();
    return bias;
}

int terrainMeshMaxBoost() {
    static const int b = [] {
        const char* e = std::getenv("ASCEND_TERRAIN_BOOST");
        int v = e ? std::atoi(e) : 1;
        return v < 0 ? 0 : (v > 1 ? 1 : v);
    }();
    return b;
}

static std::atomic<int> s_terrainMeshBoost{0};
int terrainMeshBoost() { return s_terrainMeshBoost.load(std::memory_order_relaxed); }
void setTerrainMeshBoost(int boost) { s_terrainMeshBoost.store(boost, std::memory_order_relaxed); }

static std::atomic<int> s_terrainMeshRefZoom{100};

int terrainMeshRefZoom() { return s_terrainMeshRefZoom.load(std::memory_order_relaxed); }

void updateTerrainMeshRefZoom(double viewZoom) {
    // round(viewZoom) + bias, with HYSTERESIS: the (elevation-aware) view zoom jitters as DEM
    //  upgrades land, and a plain round() flips the reference zoom - and with it every tile's
    //  lattice, composite signature and drape - back and forth whenever the zoom sits near x.5.
    //  The shader reads this value through u_terrain_ref_zoom (never recomputes it from the zoom).
    constexpr double kHysteresis = 0.2;
    static int s_base = INT32_MIN;
    const int cand = int(std::floor(viewZoom + 0.5));
    if (s_base == INT32_MIN || std::fabs(viewZoom - double(s_base)) > 0.5 + kHysteresis) { s_base = cand; }
    s_terrainMeshRefZoom.store(s_base + terrainMeshZoomBias(), std::memory_order_relaxed);
}

}
