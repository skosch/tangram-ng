#pragma once

#include "data/tileData.h"

#include "glm/vec2.hpp"
#include "glm/common.hpp"
#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

namespace Tangram {

// The water polygons of one tile (all of the tile's raw `water` layer), indexed on a coarse grid so a
// builder of ANOTHER layer can ask "which parts of this edge run over water". Built once per tile in
// TileBuilder::build() before any style runs (same reason as AnchorOccupancyGrid: a style's
// addFeature() only sees the layer its own draw rule matched). Used to turn the protected-area
// edge band blue where it crosses the sea / a lake. Coordinates are tile-local [0,1]; water polygons
// are clipped at the tile edge, which is a real (closed) boundary for the parity test.
class WaterMask {
public:
    static std::shared_ptr<WaterMask> build(const TileData& _tileData) {
        auto mask = std::make_shared<WaterMask>();
        for (const auto& layer : _tileData.layers) {
            if (layer.name != "water") { continue; }
            for (const auto& feature : layer.features) {
                for (const auto& polygon : feature.polygons) {
                    for (const auto& ring : polygon) { mask->addRing(ring); }
                }
            }
        }
        if (mask->m_segs.empty()) { return nullptr; }
        mask->index();
        return mask;
    }

    // Parameters 0 < t < 1 at which segment a->b crosses a water boundary, sorted.
    void crossings(glm::vec2 a, glm::vec2 b, std::vector<float>& _out) const {
        _out.clear();
        std::vector<uint32_t> cand;
        int x0, y0, x1, y1;
        cellOf(glm::min(a, b), x0, y0);
        cellOf(glm::max(a, b), x1, y1);
        for (int y = y0; y <= y1; y++) {
            for (int x = x0; x <= x1; x++) {
                const auto& c = m_cells[y * G + x];
                cand.insert(cand.end(), c.begin(), c.end());
            }
        }
        std::sort(cand.begin(), cand.end());
        cand.erase(std::unique(cand.begin(), cand.end()), cand.end());
        const glm::vec2 d = b - a;
        for (uint32_t k : cand) {
            const Seg& s = m_segs[k];
            glm::vec2 e = s.b - s.a;
            float den = d.x * e.y - d.y * e.x;
            if (std::fabs(den) < 1e-14f) { continue; }
            glm::vec2 w = s.a - a;
            float t = (w.x * e.y - w.y * e.x) / den;
            float u = (w.x * d.y - w.y * d.x) / den;
            if (t > 1e-6f && t < 1.f - 1e-6f && u >= 0.f && u < 1.f) { _out.push_back(t); }
        }
        std::sort(_out.begin(), _out.end());
    }

    // Even-odd parity over every water ring of the tile.
    bool contains(glm::vec2 p) const {
        int cx, cy;
        cellOf(p, cx, cy);
        std::vector<uint32_t> cand;
        for (int x = cx; x < G; x++) {
            const auto& c = m_cells[cy * G + x];
            cand.insert(cand.end(), c.begin(), c.end());
        }
        std::sort(cand.begin(), cand.end());
        cand.erase(std::unique(cand.begin(), cand.end()), cand.end());
        bool in = false;
        for (uint32_t k : cand) {
            const Seg& s = m_segs[k];
            if ((s.a.y > p.y) == (s.b.y > p.y)) { continue; }
            float x = s.a.x + (p.y - s.a.y) / (s.b.y - s.a.y) * (s.b.x - s.a.x);
            if (x > p.x) { in = !in; }
        }
        return in;
    }

private:
    struct Seg { glm::vec2 a, b; };
    static constexpr int G = 32;

    void addRing(const Line& _ring) {
        const size_t n = _ring.size();
        if (n < 3) { return; }
        for (size_t i = 0; i < n; i++) {
            const glm::vec2& a = _ring[i];
            const glm::vec2& b = _ring[(i + 1) % n];
            if (a != b) { m_segs.push_back({ a, b }); }
        }
    }

    void cellOf(glm::vec2 p, int& cx, int& cy) const {
        cx = std::max(0, std::min(G - 1, int(std::floor(p.x * G))));
        cy = std::max(0, std::min(G - 1, int(std::floor(p.y * G))));
    }

    void index() {
        m_cells.assign(G * G, {});
        for (uint32_t k = 0; k < m_segs.size(); k++) {
            int x0, y0, x1, y1;
            cellOf(glm::min(m_segs[k].a, m_segs[k].b), x0, y0);
            cellOf(glm::max(m_segs[k].a, m_segs[k].b), x1, y1);
            for (int y = y0; y <= y1; y++) {
                for (int x = x0; x <= x1; x++) { m_cells[y * G + x].push_back(k); }
            }
        }
    }

    std::vector<Seg> m_segs;
    std::vector<std::vector<uint32_t>> m_cells;
};

} // namespace Tangram
