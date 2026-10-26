#include "util/areaShape.h"

#include "util/labelPath.h"

#include "glm/common.hpp"
#include "glm/geometric.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>

namespace Tangram {

namespace {

constexpr float kInf = 1e20f;

// Squared distance transform of one line (Felzenszwalb & Huttenlocher 2012): d[q] = min_p (q-p)^2 + f[p].
void edt1d(const float* f, float* d, int n, int* v, float* z) {
    int k = 0;
    v[0] = 0;
    z[0] = -kInf;
    z[1] = kInf;
    for (int q = 1; q < n; q++) {
        float s;
        for (;;) {
            int p = v[k];
            s = ((f[q] + float(q * q)) - (f[p] + float(p * p))) / float(2 * q - 2 * p);
            if (s <= z[k]) { k--; } else { break; }
        }
        k++;
        v[k] = q;
        z[k] = s;
        z[k + 1] = kInf;
    }
    k = 0;
    for (int q = 0; q < n; q++) {
        while (z[k + 1] < float(q)) { k++; }
        d[q] = float((q - v[k]) * (q - v[k])) + f[v[k]];
    }
}

}  // namespace

struct AreaGrid {
    int nx = 0, ny = 0;
    float cell = 1.f;
    double c = 1.0, s = 0.0;        // rotation: u = c*x + s*y, v = -s*x + c*y
    double u0 = 0.0, v0 = 0.0;      // grid corner in rotated frame (m)
    std::vector<unsigned char> in;  // inside mask
    std::vector<float> r;           // inscribed radius (m), 0 outside
    std::vector<int> part;          // part index per cell, -1 outside

    int idx(int x, int y) const { return y * nx + x; }

    glm::vec2 toGrid(glm::vec2 p) const {  // local meters -> cell coordinates (cell centres at +0.5)
        double u = c * p.x + s * p.y, v = -s * p.x + c * p.y;
        return { float((u - u0) / cell), float((v - v0) / cell) };
    }
    glm::vec2 toLocal(float gx, float gy) const {
        double u = u0 + gx * double(cell), v = v0 + gy * double(cell);
        return { float(c * u - s * v), float(s * u + c * v) };
    }
    glm::vec2 cellCenter(int x, int y) const { return toLocal(x + 0.5f, y + 0.5f); }

    bool inPart(glm::vec2 p, int part) const {
        glm::vec2 gp = toGrid(p);
        int x = int(std::floor(gp.x)), y = int(std::floor(gp.y));
        return gp.x >= 0.f && gp.y >= 0.f && x < nx && y < ny && this->part[idx(x, y)] == part;
    }

    // Bilinear inscribed radius at a local-meter position (0 outside the grid / the shape).
    float radiusAt(glm::vec2 p) const {
        glm::vec2 g = toGrid(p) - 0.5f;
        int x0 = int(std::floor(g.x)), y0 = int(std::floor(g.y));
        float fx = g.x - x0, fy = g.y - y0;
        auto at = [&](int x, int y) { return (x < 0 || y < 0 || x >= nx || y >= ny) ? 0.f : r[idx(x, y)]; };
        return (at(x0, y0) * (1 - fx) + at(x0 + 1, y0) * fx) * (1 - fy) +
               (at(x0, y0 + 1) * (1 - fx) + at(x0 + 1, y0 + 1) * fx) * fy;
    }
};

namespace {

struct UnionFind {
    std::vector<int> parent;
    int find(int a) {
        while (parent[a] != a) { parent[a] = parent[parent[a]]; a = parent[a]; }
        return a;
    }
};

constexpr int kDx[8] = { -1, 0, 1, -1, 1, -1, 0, 1 };
constexpr int kDy[8] = { -1, -1, -1, 0, 0, 1, 1, 1 };
constexpr float kLen[8] = { 1.41421356f, 1.f, 1.41421356f, 1.f, 1.f, 1.41421356f, 1.f, 1.41421356f };

}  // namespace

bool analyzeArea(const std::vector<AreaPolygon>& polys, const AreaShapeParams& P, AreaShape& out) {
    out = AreaShape();

    // --- bounds, principal axis ------------------------------------------------------------
    glm::dvec2 lo(1e300), hi(-1e300);
    size_t edges = 0;
    for (const auto& poly : polys) {
        for (const auto& ring : poly) {
            for (const auto& p : ring) { lo = glm::min(lo, p); hi = glm::max(hi, p); }
            edges += ring.size();
        }
    }
    if (edges < 3) { return false; }
    out.origin = (lo + hi) * 0.5;

    double sxx = 0, syy = 0, sxy = 0, sw = 0;
    for (const auto& poly : polys) {
        for (const auto& ring : poly) {
            const size_t n = ring.size();
            for (size_t i = 0; i < n; i++) {
                glm::dvec2 a = ring[i] - out.origin, b = ring[(i + 1) % n] - out.origin;
                double w = glm::length(b - a);
                glm::dvec2 m = (a + b) * 0.5;
                sxx += w * m.x * m.x; syy += w * m.y * m.y; sxy += w * m.x * m.y; sw += w;
            }
        }
    }
    if (sw <= 0) { return false; }
    const double theta = 0.5 * std::atan2(2.0 * sxy, sxx - syy);

    auto gridPtr = std::make_shared<AreaGrid>();
    AreaGrid& g = *gridPtr;
    g.c = std::cos(theta);
    g.s = std::sin(theta);

    double umin = 1e300, umax = -1e300, vmin = 1e300, vmax = -1e300;
    for (const auto& poly : polys) {
        for (const auto& ring : poly) {
            for (const auto& p : ring) {
                glm::dvec2 q = p - out.origin;
                double u = g.c * q.x + g.s * q.y, v = -g.s * q.x + g.c * q.y;
                umin = std::min(umin, u); umax = std::max(umax, u);
                vmin = std::min(vmin, v); vmax = std::max(vmax, v);
            }
        }
    }
    const double W = umax - umin, H = vmax - vmin;
    if (W <= 0 || H <= 0) { return false; }
    // Fine enough to resolve the feature's narrow dimension in ~24 cells, never finer than the
    // cell budget or minCell allow: a small round lake gets a few hundred cells, not the budget.
    g.cell = float(std::max({ double(P.minCell), std::sqrt(W * H / double(P.maxCells)), std::min(W, H) / 24.0 }));
    while ((W / g.cell + 5) * (H / g.cell + 5) > 4.0 * P.maxCells) { g.cell *= 1.25f; }
    g.nx = int(std::ceil(W / g.cell)) + 4;
    g.ny = int(std::ceil(H / g.cell)) + 4;
    g.u0 = umin - 2.0 * g.cell;
    g.v0 = vmin - 2.0 * g.cell;
    const int N = g.nx * g.ny;
    out.cell = g.cell;
    out.grid = gridPtr;
    out.cells = N;

    // --- rasterise (cell centres, even-odd per polygon, OR across polygons) ---------------------
    g.in.assign(N, 0);
    std::vector<std::vector<float>> rows(g.ny);
    for (const auto& poly : polys) {
        for (auto& r : rows) { r.clear(); }
        for (const auto& ring : poly) {
            const size_t n = ring.size();
            if (n < 3) { continue; }
            for (size_t i = 0; i < n; i++) {
                glm::dvec2 a = ring[i] - out.origin, b = ring[(i + 1) % n] - out.origin;
                double au = (g.c * a.x + g.s * a.y - g.u0) / g.cell, av = (-g.s * a.x + g.c * a.y - g.v0) / g.cell;
                double bu = (g.c * b.x + g.s * b.y - g.u0) / g.cell, bv = (-g.s * b.x + g.c * b.y - g.v0) / g.cell;
                if (av == bv) { continue; }
                if (av > bv) { std::swap(av, bv); std::swap(au, bu); }
                // rows whose centre (j + 0.5) lies in [av, bv)
                int j0 = std::max(0, int(std::ceil(av - 0.5)));
                int j1 = std::min(g.ny - 1, int(std::ceil(bv - 0.5)) - 1);
                for (int j = j0; j <= j1; j++) {
                    double t = (j + 0.5 - av) / (bv - av);
                    rows[j].push_back(float(au + t * (bu - au)));
                }
            }
        }
        for (int j = 0; j < g.ny; j++) {
            auto& xs = rows[j];
            if (xs.size() < 2) { continue; }
            std::sort(xs.begin(), xs.end());
            for (size_t k = 0; k + 1 < xs.size(); k += 2) {
                int i0 = std::max(0, int(std::ceil(xs[k] - 0.5f)));
                int i1 = std::min(g.nx - 1, int(std::ceil(xs[k + 1] - 0.5f)) - 1);
                for (int i = i0; i <= i1; i++) { g.in[g.idx(i, j)] = 1; }
            }
        }
    }

    // --- distance transform --------------------------------------------------------------------
    std::vector<float> d2(N);
    {
        const int m = std::max(g.nx, g.ny);
        std::vector<float> f(m), d(m), z(m + 1);
        std::vector<int> v(m);
        std::vector<float> tmp(N);
        for (int x = 0; x < g.nx; x++) {
            for (int y = 0; y < g.ny; y++) { f[y] = g.in[g.idx(x, y)] ? kInf : 0.f; }
            edt1d(f.data(), d.data(), g.ny, v.data(), z.data());
            for (int y = 0; y < g.ny; y++) { tmp[g.idx(x, y)] = d[y]; }
        }
        for (int y = 0; y < g.ny; y++) {
            for (int x = 0; x < g.nx; x++) { f[x] = tmp[g.idx(x, y)]; }
            edt1d(f.data(), d.data(), g.nx, v.data(), z.data());
            for (int x = 0; x < g.nx; x++) { d2[g.idx(x, y)] = d[x]; }
        }
    }
    g.r.assign(N, 0.f);
    int maxD2 = 0, insideCells = 0;
    std::vector<int> d2i(N, 0);
    for (int i = 0; i < N; i++) {
        if (!g.in[i]) { continue; }
        insideCells++;
        d2i[i] = std::min(int(d2[i] + 0.5f), 1 << 24);
        maxD2 = std::max(maxD2, d2i[i]);
        g.r[i] = std::max(0.5f, std::sqrt(float(d2i[i])) - 0.5f) * g.cell;
    }
    if (insideCells < 4) { return false; }

    // --- watershed with neck-aware merging -----------------------------------------------------
    // counting sort of inside cells by d2, descending
    std::vector<int> order(insideCells);
    {
        std::vector<int> count(maxD2 + 2, 0);
        for (int i = 0; i < N; i++) { if (g.in[i]) { count[maxD2 - d2i[i] + 1]++; } }
        for (int k = 1; k < int(count.size()); k++) { count[k] += count[k - 1]; }
        for (int i = 0; i < N; i++) { if (g.in[i]) { order[count[maxD2 - d2i[i]]++] = i; } }
    }
    UnionFind uf;
    uf.parent.assign(N, -1);
    std::vector<int> comp(N, -1);  // basin id at processing time
    std::vector<int> peakOf(N, -1);   // per root: cell of its highest point
    auto cellXY = [&](int i) { return glm::ivec2(i % g.nx, i / g.nx); };

    for (int idx : order) {
        glm::ivec2 xy = cellXY(idx);
        int roots[8], nr = 0;
        for (int k = 0; k < 8; k++) {
            int x = xy.x + kDx[k], y = xy.y + kDy[k];
            if (x < 0 || y < 0 || x >= g.nx || y >= g.ny) { continue; }
            int j = g.idx(x, y);
            if (comp[j] < 0) { continue; }
            int r = uf.find(comp[j]);
            bool seen = false;
            for (int q = 0; q < nr; q++) { seen |= (roots[q] == r); }
            if (!seen) { roots[nr++] = r; }
        }
        if (nr == 0) {
            comp[idx] = idx;
            uf.parent[idx] = idx;
            peakOf[idx] = idx;
            continue;
        }
        std::sort(roots, roots + nr, [&](int a, int b) {
            return d2i[peakOf[a]] != d2i[peakOf[b]] ? d2i[peakOf[a]] > d2i[peakOf[b]] : a < b;
        });
        const int T = roots[0];
        comp[idx] = T;
        const float S = g.r[idx];
        for (int q = 1; q < nr; q++) {
            const int O = uf.find(roots[q]);
            if (O == uf.find(T)) { continue; }
            const int Tr = uf.find(T);
            const float R1 = g.r[peakOf[Tr]], R2 = g.r[peakOf[O]];
            const float Rm = std::min(R1, R2);
            glm::vec2 p1 = g.cellCenter(cellXY(peakOf[Tr]).x, cellXY(peakOf[Tr]).y);
            glm::vec2 p2 = g.cellCenter(cellXY(peakOf[O]).x, cellXY(peakOf[O]).y);
            const float gap = glm::length(p1 - p2) - R1 - R2;
            const bool neck = S < P.neckRatio * Rm && gap >= P.neckGap * S;
            if (neck) {
                glm::vec2 pos = g.cellCenter(xy.x, xy.y);
                bool dup = false;
                for (const auto& n : out.necks) { dup |= glm::length(n.pos - pos) < 3.f * g.cell; }
                if (!dup) { out.necks.push_back({ pos, S }); }
            } else {
                uf.parent[O] = Tr;
            }
        }
    }

    // --- parts -------------------------------------------------------------------------------
    std::vector<int> rootToPart(N, -1);
    g.part.assign(N, -1);
    struct Acc { int cells = 0; int peak = -1; };
    std::vector<Acc> acc;
    for (int i : order) {
        int r = uf.find(comp[i]);
        if (rootToPart[r] < 0) { rootToPart[r] = int(acc.size()); acc.push_back({}); }
        int p = rootToPart[r];
        g.part[i] = p;
        acc[p].cells++;
        if (acc[p].peak < 0 || d2i[i] > d2i[acc[p].peak]) { acc[p].peak = i; }
    }
    // largest first
    std::vector<int> byArea(acc.size());
    for (size_t i = 0; i < acc.size(); i++) { byArea[i] = int(i); }
    std::sort(byArea.begin(), byArea.end(), [&](int a, int b) { return acc[a].cells > acc[b].cells; });

    std::vector<float> distA(N, kInf);
    std::vector<int> touched;
    struct QE { float d; int i; bool operator<(const QE& o) const { return d > o.d; } };

    // Dijkstra over cells of `part`; `weighted` => centre-hugging cost. Returns farthest cell.
    auto dijkstra = [&](int part, int src, bool weighted, float rmax, std::vector<int>* prev) {
        for (int i : touched) { distA[i] = kInf; }
        touched.clear();
        std::priority_queue<QE> pq;
        distA[src] = 0.f;
        touched.push_back(src);
        pq.push({ 0.f, src });
        int far = src;
        while (!pq.empty()) {
            QE e = pq.top(); pq.pop();
            if (e.d > distA[e.i]) { continue; }
            if (distA[e.i] > distA[far]) { far = e.i; }
            glm::ivec2 xy = cellXY(e.i);
            for (int k = 0; k < 8; k++) {
                int x = xy.x + kDx[k], y = xy.y + kDy[k];
                if (x < 0 || y < 0 || x >= g.nx || y >= g.ny) { continue; }
                int j = g.idx(x, y);
                if (g.part[j] != part) { continue; }
                float cost = kLen[k];
                if (weighted) {
                    float q = rmax / std::max(g.r[j], 0.5f * g.cell);
                    cost *= 1.f + 0.5f * q * q;
                }
                float nd = e.d + cost;
                if (nd < distA[j]) {
                    if (distA[j] == kInf) { touched.push_back(j); }
                    distA[j] = nd;
                    if (prev) { (*prev)[j] = e.i; }
                    pq.push({ nd, j });
                }
            }
        }
        return far;
    };

    std::vector<int> prev(N, -1);
    for (int pi : byArea) {
        const int part = pi;
        const Acc& a = acc[part];
        if (a.cells < 6) { continue; }
        AreaPart ap;
        ap.id = part;
        ap.area = float(a.cells) * g.cell * g.cell;
        ap.maxRadius = g.r[a.peak];
        ap.peak = g.cellCenter(cellXY(a.peak).x, cellXY(a.peak).y);

        int A = dijkstra(part, a.peak, false, 0.f, nullptr);
        int B = dijkstra(part, A, false, 0.f, nullptr);
        const float geodesic = distA[B] * g.cell;
        ap.elongation = geodesic / std::max(2.f * ap.maxRadius, g.cell);

        if (!P.alwaysHorizontal && ap.elongation >= P.compact) {
            dijkstra(part, A, true, ap.maxRadius, &prev);
            std::vector<glm::vec2> path;
            for (int c = B; c >= 0 && c != A; c = prev[c]) {
                glm::ivec2 xy = cellXY(c);
                path.push_back(g.cellCenter(xy.x, xy.y));
                if (path.size() > size_t(N)) { break; }
            }
            glm::ivec2 axy = cellXY(A);
            path.push_back(g.cellCenter(axy.x, axy.y));
            std::reverse(path.begin(), path.end());

            // median clearance sets the smoothing scale
            std::vector<float> rr;
            for (const auto& p : path) { rr.push_back(g.radiusAt(p)); }
            std::nth_element(rr.begin(), rr.begin() + rr.size() / 2, rr.end());
            const float med = rr[rr.size() / 2];
            ap.spine = smoothPolyline(resamplePolyline(path, g.cell), g.cell,
                                      std::max(2.f * g.cell, P.smooth * med));
        } else {
            ap.horizontal = true;
            // The row (local y) that holds the longest run of generous clearance: for a park or a
            // round lake that is the label's baseline. Compact parts have it through the peak.
            const float step = 0.5f * g.cell;
            const float reach = float(0.5 * std::hypot(W, H)) + g.cell;
            const float thr = std::max(0.35f * ap.maxRadius, 0.5f * g.cell);
            auto inPart = [&](glm::vec2 p) {
                glm::vec2 gp = g.toGrid(p);
                int x = int(std::floor(gp.x)), y = int(std::floor(gp.y));
                return x >= 0 && y >= 0 && x < g.nx && y < g.ny && g.part[g.idx(x, y)] == part;
            };
            float bestScore = -1.f, bestY = ap.peak.y, bestX = ap.peak.x;
            const int rowsHalf = P.alwaysHorizontal ? 12 : 0;
            for (int j = -rowsHalf; j <= rowsHalf; j++) {
                const float y = ap.peak.y + j * (0.75f * ap.maxRadius / std::max(rowsHalf, 1));
                float runA = 0.f, runBest = 0.f, runBestMid = ap.peak.x;
                bool open = false;
                for (float x = ap.peak.x - reach; x <= ap.peak.x + reach; x += step) {
                    glm::vec2 p(x, y);
                    bool ok = inPart(p) && g.radiusAt(p) >= thr;
                    if (ok && !open) { runA = x; open = true; }
                    if ((!ok || x + step > ap.peak.x + reach) && open) {
                        if (x - runA > runBest) { runBest = x - runA; runBestMid = 0.5f * (x + runA); }
                        open = false;
                    }
                }
                const float score = runBest - 0.25f * std::fabs(y - ap.peak.y);
                if (runBest > 0.f && score > bestScore) { bestScore = score; bestY = y; bestX = runBestMid; }
            }
            // the chord: everything connected to (bestX, bestY) inside the part
            const glm::vec2 seed(bestX, bestY);
            std::vector<glm::vec2> left, right;
            if (inPart(seed)) {
                for (int dir = -1; dir <= 1; dir += 2) {
                    auto& side = dir < 0 ? left : right;
                    for (int k = 1; k < 100000; k++) {
                        glm::vec2 p = seed + glm::vec2(dir * k * step, 0.f);
                        if (!inPart(p)) { break; }
                        side.push_back(p);
                    }
                }
            }
            std::reverse(left.begin(), left.end());
            ap.spine = left;
            ap.spine.push_back(seed);
            ap.spine.insert(ap.spine.end(), right.begin(), right.end());
        }
        if (ap.spine.size() < 2) { continue; }
        ap.clearance.reserve(ap.spine.size());
        for (const auto& p : ap.spine) { ap.clearance.push_back(g.radiusAt(p)); }
        if (P.chordRows > 0) {
            // every horizontal run of the part, row by row: the fallback paths a label can slide
            // to when the spine is off screen (or, for parks, the only paths there are)
            float lx = 1e30f, hx = -1e30f, ly = 1e30f, hy = -1e30f;
            for (int y = 0; y < g.ny; y++) {
                for (int x = 0; x < g.nx; x++) {
                    if (g.part[g.idx(x, y)] != part) { continue; }
                    glm::vec2 c = g.cellCenter(x, y);
                    lx = std::min(lx, c.x); hx = std::max(hx, c.x);
                    ly = std::min(ly, c.y); hy = std::max(hy, c.y);
                }
            }
            const float step = 0.5f * g.cell;
            const float dy = std::max(g.cell, (hy - ly) / float(P.chordRows));
            for (float y = ly; y <= hy + 0.01f; y += dy) {
                AreaChord ch;
                auto flush = [&]() {
                    if (ch.pts.size() >= 4) {
                        float mx = 0.f;
                        for (float r : ch.clearance) { mx = std::max(mx, r); }
                        if (mx >= 0.3f * ap.maxRadius) { ap.chords.push_back(std::move(ch)); }
                    }
                    ch = AreaChord();
                };
                for (float x = lx; x <= hx + step; x += step) {
                    glm::vec2 p(x, y);
                    if (g.inPart(p, part)) {
                        ch.pts.push_back(p);
                        ch.clearance.push_back(g.radiusAt(p));
                    } else {
                        flush();
                    }
                }
                flush();
            }
            std::sort(ap.chords.begin(), ap.chords.end(),
                      [](const AreaChord& a, const AreaChord& b) { return a.pts.size() > b.pts.size(); });
            if (ap.chords.size() > 60) { ap.chords.resize(60); }
        }
        out.parts.push_back(std::move(ap));
    }
    return !out.parts.empty();
}

}  // namespace Tangram
