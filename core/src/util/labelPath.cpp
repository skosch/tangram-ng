#include "util/labelPath.h"

#include "util/geom.h"

#include <glm/glm.hpp>

#include <algorithm>
#include <cmath>

namespace Tangram {

float polylineLength(const std::vector<glm::vec2>& pts) {
    float len = 0.f;
    for (size_t i = 1; i < pts.size(); i++) { len += glm::length(pts[i] - pts[i - 1]); }
    return len;
}

std::vector<glm::vec2> simplifyPolyline(const std::vector<glm::vec2>& pts, float tolerance) {
    const size_t n = pts.size();
    if (n < 3) { return pts; }

    std::vector<char> keep(n, 0);
    keep.front() = keep.back() = 1;

    std::vector<std::pair<size_t, size_t>> stack;
    stack.emplace_back(0, n - 1);
    const float tol2 = tolerance * tolerance;

    while (!stack.empty()) {
        size_t a = stack.back().first, b = stack.back().second;
        stack.pop_back();
        if (b <= a + 1) { continue; }

        glm::vec2 pa = pts[a], ab = pts[b] - pts[a];
        float abLen2 = glm::dot(ab, ab);
        float maxD2 = -1.f;
        size_t maxI = a;
        for (size_t i = a + 1; i < b; i++) {
            glm::vec2 ap = pts[i] - pa;
            float d2;
            if (abLen2 < 1e-20f) {
                d2 = glm::dot(ap, ap);
            } else {
                float t = Tangram::clamp(glm::dot(ap, ab) / abLen2, 0.f, 1.f);
                glm::vec2 q = ap - ab * t;
                d2 = glm::dot(q, q);
            }
            if (d2 > maxD2) { maxD2 = d2; maxI = i; }
        }
        if (maxD2 > tol2) {
            keep[maxI] = 1;
            stack.emplace_back(a, maxI);
            stack.emplace_back(maxI, b);
        }
    }

    std::vector<glm::vec2> out;
    for (size_t i = 0; i < n; i++) { if (keep[i]) { out.push_back(pts[i]); } }
    return out;
}

std::vector<glm::vec2> smoothPolyline(const std::vector<glm::vec2>& pts, float step, float sigma) {
    if (pts.size() < 3 || step <= 0.f || sigma <= 0.f) { return pts; }

    // Cumulative arc length.
    std::vector<float> cum(pts.size(), 0.f);
    for (size_t i = 1; i < pts.size(); i++) { cum[i] = cum[i - 1] + glm::length(pts[i] - pts[i - 1]); }
    const float total = cum.back();
    if (total < step * 2.f) { return pts; }

    // Uniform resample, endpoints included.
    const int n = std::max(3, int(std::ceil(total / step)) + 1);
    std::vector<glm::vec2> res(n);
    size_t seg = 0;
    for (int i = 0; i < n; i++) {
        float s = total * float(i) / float(n - 1);
        while (seg + 2 < pts.size() && cum[seg + 1] < s) { seg++; }
        float segLen = cum[seg + 1] - cum[seg];
        float t = segLen > 1e-20f ? (s - cum[seg]) / segLen : 0.f;
        res[i] = pts[seg] + (pts[seg + 1] - pts[seg]) * Tangram::clamp(t, 0.f, 1.f);
    }

    // Gaussian kernel over resampled indices.
    const float h = total / float(n - 1);               // actual spacing
    const float sigmaIdx = std::max(sigma / h, 0.5f);
    const int radius = std::max(1, int(std::ceil(3.f * sigmaIdx)));
    std::vector<float> kernel(radius + 1);
    float ksum = 0.f;
    for (int k = 0; k <= radius; k++) {
        kernel[k] = std::exp(-0.5f * float(k * k) / (sigmaIdx * sigmaIdx));
        ksum += (k == 0) ? kernel[k] : 2.f * kernel[k];
    }
    for (auto& k : kernel) { k /= ksum; }

    // Sample with point-reflection about the endpoints: p(-k) = 2 p(0) - p(k). Keeps the
    // end position and tangent, so the smoothed path is exactly as long as the span.
    auto at = [&](int i) -> glm::vec2 {
        if (i < 0) {
            int j = std::min(-i, n - 1);
            return 2.f * res[0] - res[j];
        }
        if (i > n - 1) {
            int j = std::max(2 * (n - 1) - i, 0);
            return 2.f * res[n - 1] - res[j];
        }
        return res[i];
    };

    std::vector<glm::vec2> out(n);
    for (int i = 0; i < n; i++) {
        glm::vec2 acc = res[i] * kernel[0];
        for (int k = 1; k <= radius; k++) { acc += (at(i - k) + at(i + k)) * kernel[k]; }
        out[i] = acc;
    }
    out.front() = res.front();
    out.back() = res.back();

    return simplifyPolyline(out, step * 0.1f);
}

std::vector<glm::vec2> trimPolyline(const std::vector<glm::vec2>& pts, float fromStart, float fromEnd) {
    if (pts.size() < 2) { return {}; }
    const float total = polylineLength(pts);
    const float a = std::max(fromStart, 0.f), b = total - std::max(fromEnd, 0.f);
    if (b - a < 1e-9f) { return {}; }

    std::vector<glm::vec2> out;
    float cum = 0.f;
    for (size_t i = 0; i + 1 < pts.size(); i++) {
        float segLen = glm::length(pts[i + 1] - pts[i]);
        float s0 = cum, s1 = cum + segLen;
        cum = s1;
        if (segLen < 1e-12f || s1 <= a || s0 >= b) { continue; }
        float t0 = std::max(a, s0), t1 = std::min(b, s1);
        glm::vec2 p0 = pts[i] + (pts[i + 1] - pts[i]) * ((t0 - s0) / segLen);
        glm::vec2 p1 = pts[i] + (pts[i + 1] - pts[i]) * ((t1 - s0) / segLen);
        if (out.empty()) { out.push_back(p0); }
        out.push_back(p1);
    }
    return out.size() >= 2 ? out : std::vector<glm::vec2>{};
}

bool pointInPolygon(const std::vector<std::vector<glm::vec2>>& rings, glm::vec2 p) {
    bool inside = false;
    for (const auto& ring : rings) {
        const size_t m = ring.size();
        for (size_t i = 0, j = m - 1; i < m; j = i++) {
            const glm::vec2& a = ring[i];
            const glm::vec2& b = ring[j];
            if ((a.y > p.y) != (b.y > p.y)) {
                float x = a.x + (p.y - a.y) / (b.y - a.y) * (b.x - a.x);
                if (p.x < x) { inside = !inside; }
            }
        }
    }
    return inside;
}

bool polygonCenterline(const std::vector<std::vector<glm::vec2>>& rings, glm::vec2 axisDir,
                       float minWidth, float minLength, float smoothSigma,
                       PolygonCenterline& out) {
    if (rings.empty() || rings.front().size() < 3) { return false; }

    float dl = glm::length(axisDir);
    if (dl < 1e-9f) { return false; }
    const glm::vec2 d = axisDir / dl;
    const glm::vec2 nrm(-d.y, d.x);

    float tmin = 1e30f, tmax = -1e30f;
    for (const auto& v : rings.front()) {
        float t = glm::dot(v, d);
        tmin = std::min(tmin, t);
        tmax = std::max(tmax, t);
    }
    const float extent = tmax - tmin;
    if (extent < minLength) { return false; }

    // Slice density: fine enough that a strip `minWidth` wide can't slip between slices.
    const int K = Tangram::clamp(int(std::ceil(extent / std::max(minWidth * 0.6f, extent / 160.f))), 24, 160);

    struct Slice { glm::vec2 p; float width; bool ok; };
    std::vector<Slice> slices(K, Slice{glm::vec2(0.f), 0.f, false});

    bool havePrev = false;
    float prevS = 0.f;
    std::vector<float> ss;
    for (int i = 0; i < K; i++) {
        const float t = tmin + extent * (float(i) + 0.5f) / float(K);
        ss.clear();
        for (const auto& ring : rings) {
            const size_t m = ring.size();
            for (size_t a = 0, b = m - 1; a < m; b = a++) {
                float ta = glm::dot(ring[a], d), tb = glm::dot(ring[b], d);
                if ((ta <= t) != (tb <= t)) {
                    float u = (t - ta) / (tb - ta);
                    glm::vec2 q = ring[a] + (ring[b] - ring[a]) * u;
                    ss.push_back(glm::dot(q, nrm));
                }
            }
        }
        if (ss.size() < 2) { havePrev = false; continue; }
        std::sort(ss.begin(), ss.end());

        // Interior intervals are (ss[0],ss[1]), (ss[2],ss[3]), ... (even-odd).
        int best = -1;
        float bestScore = -1.f;
        for (size_t k = 0; k + 1 < ss.size(); k += 2) {
            float w = ss[k + 1] - ss[k];
            bool cont = havePrev && prevS >= ss[k] && prevS <= ss[k + 1];
            float score = w * (cont ? 2.f : 1.f);
            if (score > bestScore) { bestScore = score; best = int(k); }
        }
        if (best < 0) { havePrev = false; continue; }

        float sMid = 0.5f * (ss[best] + ss[best + 1]);
        slices[i] = { d * t + nrm * sMid, ss[best + 1] - ss[best], true };
        havePrev = true;
        prevS = sMid;
    }

    // Longest run of consecutive slices wide enough for the text strip.
    int bestStart = -1, bestEnd = -1;
    float bestLen = 0.f;
    for (int i = 0; i < K;) {
        if (!slices[i].ok || slices[i].width < minWidth) { i++; continue; }
        int j = i;
        float len = 0.f;
        while (j + 1 < K && slices[j + 1].ok && slices[j + 1].width >= minWidth) {
            len += glm::length(slices[j + 1].p - slices[j].p);
            j++;
        }
        if (len > bestLen) { bestLen = len; bestStart = i; bestEnd = j; }
        i = j + 1;
    }
    if (bestStart < 0 || bestLen < minLength) { return false; }

    std::vector<glm::vec2> raw;
    float minW = 1e30f;
    for (int i = bestStart; i <= bestEnd; i++) {
        raw.push_back(slices[i].p);
        minW = std::min(minW, slices[i].width);
    }

    // Smooth the run. Slice spacing is the natural resampling step.
    const float step = extent / float(K);
    std::vector<glm::vec2> path = raw;
    float sigma = smoothSigma;
    for (int attempt = 0; attempt < 3; attempt++) {
        std::vector<glm::vec2> cand = (sigma > 0.f) ? smoothPolyline(raw, step, sigma) : raw;
        bool inside = true;
        for (const auto& c : cand) {
            if (!pointInPolygon(rings, c)) { inside = false; break; }
        }
        if (inside) { path = std::move(cand); break; }
        sigma *= 0.5f;
        if (attempt == 2) { path = raw; }
    }

    out.points = std::move(path);
    out.length = polylineLength(out.points);
    out.minWidth = minW;
    return out.length >= minLength && out.points.size() >= 2;
}


std::vector<glm::vec2> resamplePolyline(const std::vector<glm::vec2>& pts, float step) {
    std::vector<glm::vec2> out;
    if (pts.size() < 2 || step <= 0.f) { return pts; }
    out.push_back(pts.front());
    float carry = 0.f;
    for (size_t i = 1; i < pts.size(); i++) {
        glm::vec2 a = pts[i - 1], b = pts[i];
        float len = glm::length(b - a);
        if (len < 1e-9f) { continue; }
        float d = step - carry;
        while (d <= len) {
            out.push_back(a + (b - a) * (d / len));
            d += step;
        }
        carry = len - (d - step);
    }
    if (glm::length(out.back() - pts.back()) > step * 0.25f) { out.push_back(pts.back()); }
    else { out.back() = pts.back(); }
    return out;
}

bool InsideMask::build(const std::vector<std::vector<glm::vec2>>& rings, int maxCells, float minCell) {
    glm::vec2 lo(1e30f), hi(-1e30f);
    for (const auto& ring : rings) {
        for (const auto& p : ring) { lo = glm::min(lo, p); hi = glm::max(hi, p); }
    }
    if (!(hi.x > lo.x) || !(hi.y > lo.y)) { return false; }

    glm::vec2 ext = hi - lo;
    m_cell = std::max(minCell, std::max(ext.x, ext.y) / float(maxCells));
    m_w = int(std::ceil(ext.x / m_cell)) + 2;
    m_h = int(std::ceil(ext.y / m_cell)) + 2;
    m_origin = lo - glm::vec2(m_cell);
    m_in.assign(size_t(m_w) * m_h, 0);

    // Even-odd scanline fill at cell-row centres.
    std::vector<float> xs;
    for (int j = 0; j < m_h; j++) {
        float y = m_origin.y + (j + 0.5f) * m_cell;
        xs.clear();
        for (const auto& ring : rings) {
            const size_t n = ring.size();
            for (size_t i = 0, k = n - 1; i < n; k = i++) {
                glm::vec2 a = ring[k], b = ring[i];
                if ((a.y > y) != (b.y > y)) {
                    xs.push_back(a.x + (y - a.y) / (b.y - a.y) * (b.x - a.x));
                }
            }
        }
        std::sort(xs.begin(), xs.end());
        for (size_t s = 0; s + 1 < xs.size(); s += 2) {
            int i0 = int(std::ceil((xs[s] - m_origin.x) / m_cell - 0.5f));
            int i1 = int(std::floor((xs[s + 1] - m_origin.x) / m_cell - 0.5f));
            for (int i = std::max(i0, 0); i <= std::min(i1, m_w - 1); i++) { m_in[size_t(j) * m_w + i] = 1; }
        }
    }

    // The outline itself is outside: clear every cell an edge passes through.
    for (const auto& ring : rings) {
        const size_t n = ring.size();
        for (size_t i = 0, k = n - 1; i < n; k = i++) {
            glm::vec2 a = ring[k], b = ring[i];
            float len = glm::length(b - a);
            int steps = std::max(1, int(std::ceil(len / (m_cell * 0.5f))));
            for (int s = 0; s <= steps; s++) {
                glm::vec2 p = a + (b - a) * (float(s) / steps) - m_origin;
                int ci = int(p.x / m_cell), cj = int(p.y / m_cell);
                if (ci >= 0 && ci < m_w && cj >= 0 && cj < m_h) { m_in[size_t(cj) * m_w + ci] = 0; }
            }
        }
    }

    m_sat.assign(size_t(m_w + 1) * (m_h + 1), 0);
    m_insideCells = 0;
    glm::dvec2 acc(0.0);
    for (int j = 0; j < m_h; j++) {
        int row = 0;
        for (int i = 0; i < m_w; i++) {
            int v = m_in[size_t(j) * m_w + i];
            row += v;
            if (v) { acc += glm::dvec2(i + 0.5, j + 0.5); m_insideCells++; }
            m_sat[size_t(j + 1) * (m_w + 1) + (i + 1)] = m_sat[size_t(j) * (m_w + 1) + (i + 1)] + row;
        }
    }
    if (m_insideCells == 0) { return false; }
    m_centroid = m_origin + glm::vec2(acc / double(m_insideCells)) * m_cell;
    return true;
}

int InsideMask::sum(int x0, int y0, int x1, int y1) const {
    const size_t s = m_w + 1;
    return m_sat[size_t(y1) * s + x1] - m_sat[size_t(y0) * s + x1] -
           m_sat[size_t(y1) * s + x0] + m_sat[size_t(y0) * s + x0];
}

bool InsideMask::pointInside(glm::vec2 p) const {
    p -= m_origin;
    int i = int(std::floor(p.x / m_cell)), j = int(std::floor(p.y / m_cell));
    if (i < 0 || j < 0 || i >= m_w || j >= m_h) { return false; }
    return m_in[size_t(j) * m_w + i] != 0;
}

float InsideMask::insideFraction(glm::vec2 lo, glm::vec2 hi) const {
    lo -= m_origin; hi -= m_origin;
    float area = (hi.x - lo.x) * (hi.y - lo.y);
    if (area <= 0.f) { return 0.f; }
    int x0 = std::max(0, int(std::floor(lo.x / m_cell))), y0 = std::max(0, int(std::floor(lo.y / m_cell)));
    int x1 = std::min(m_w, int(std::ceil(hi.x / m_cell))), y1 = std::min(m_h, int(std::ceil(hi.y / m_cell)));
    if (x1 <= x0 || y1 <= y0) { return 0.f; }
    return Tangram::clamp(float(sum(x0, y0, x1, y1)) * m_cell * m_cell / area, 0.f, 1.f);
}

bool InsideMask::rectInside(glm::vec2 lo, glm::vec2 hi) const {
    lo -= m_origin; hi -= m_origin;
    int x0 = int(std::floor(lo.x / m_cell)), y0 = int(std::floor(lo.y / m_cell));
    int x1 = int(std::ceil(hi.x / m_cell)), y1 = int(std::ceil(hi.y / m_cell));
    if (x0 < 0 || y0 < 0 || x1 > m_w || y1 > m_h || x1 <= x0 || y1 <= y0) { return false; }
    return sum(x0, y0, x1, y1) == (x1 - x0) * (y1 - y0);
}

bool InsideMask::widestBand(float bandH, float minW, float capW, Band& out,
                            const std::function<float(glm::vec2, glm::vec2)>& busy, float boxFill) const {
    const int rows = std::max(1, int(std::ceil(bandH / m_cell)));
    if (rows > m_h) { return false; }
    float bestScore = -1e30f;
    bool found = false;
    for (int j = 0; j + rows <= m_h; j++) {
        int i = 0;
        while (i < m_w) {
            if (sum(i, j, i + 1, j + rows) != rows) { i++; continue; }
            int s = i;
            while (i < m_w && sum(i, j, i + 1, j + rows) == rows) { i++; }
            float runW = (i - s) * m_cell;
            if (runW < minW) { continue; }
            glm::vec2 c = m_origin + glm::vec2((s + i) * 0.5f * m_cell, (j + rows * 0.5f) * m_cell);
            // Prefer roomy spots; among equally roomy ones the one nearest the centre of mass.
            float score = std::min(runW, capW) - 0.35f * glm::length(c - m_centroid);
            if (busy) {
                float bw = std::max(minW, std::min(runW, capW) * boxFill);
                float cost = busy({c.x - bw * 0.5f, c.y - bandH * 0.5f}, {c.x + bw * 0.5f, c.y + bandH * 0.5f});
                score -= 0.6f * std::min(runW, capW) * cost;
            }
            if (score > bestScore) {
                bestScore = score; found = true;
                out.cx = c.x; out.cy = c.y; out.width = runW;
            }
        }
    }
    return found;
}

float InsideMask::rayLength(glm::vec2 p, glm::vec2 dir, float maxDist) const {
    const float stepLen = m_cell * 0.5f;
    float d = 0.f;
    while (d < maxDist && pointInside(p + dir * (d + stepLen))) { d += stepLen; }
    return d;
}

}

