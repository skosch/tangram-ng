#include "labels/anchorOccupancyGrid.h"

#include "glm/geometric.hpp"
#include "glm/glm.hpp"
#include <algorithm>
#include <cmath>

namespace Tangram {

namespace {
// Layer-name -> weight table for what counts as "visually busy" underneath a label.
// Matches the vector layer names produced by this app's tile schema (see
// assets/scenes/stylus-osm.yaml `data: { source: osm, layer: ... }` blocks). Water gets
// the highest weight (a label sitting on a lake/river reads worst); transportation
// (which includes both roads and trails/paths in this schema -- there is no separate
// "trail" layer) gets a lower weight since roads/trails are thinner and less visually
// disruptive than a filled water body.
float layerWeight(const std::string& _layerName) {
    if (_layerName == "water") { return 1.0f; }
    if (_layerName == "transportation") { return 0.5f; }
    return 0.f;
}
} // namespace

std::shared_ptr<AnchorOccupancyGrid> AnchorOccupancyGrid::build(const TileData& _tileData) {
    auto grid = std::make_shared<AnchorOccupancyGrid>();
    bool any = false;

    for (const auto& layer : _tileData.layers) {
        float weight = layerWeight(layer.name);
        if (weight <= 0.f) { continue; }

        for (const auto& feature : layer.features) {
            for (const auto& line : feature.lines) {
                grid->rasterizeLine(line, weight);
                any = true;
            }
            for (const auto& polygon : feature.polygons) {
                grid->rasterizePolygonBoundary(polygon, weight);
                any = true;
            }
            // Feature points (e.g. isolated water features) aren't rasterized -- a
            // single point contributes negligible "busy-ness" at this grid resolution.
        }
    }

    if (!any) { return nullptr; }
    return grid;
}

void AnchorOccupancyGrid::rasterizeLine(const Line& _line, float _weight) {
    if (_line.empty()) { return; }
    if (_line.size() == 1) {
        markCell(_line[0], _weight);
        return;
    }
    const float cellSize = 1.f / kResolution;
    for (size_t i = 0; i + 1 < _line.size(); i++) {
        glm::vec2 a = _line[i];
        glm::vec2 b = _line[i + 1];
        float len = glm::length(b - a);
        // Step at roughly half a cell width so no cell along the segment is skipped.
        int steps = std::max(1, int(std::ceil(len / (cellSize * 0.5f))));
        for (int s = 0; s <= steps; s++) {
            float t = float(s) / float(steps);
            markCell(glm::mix(a, b, t), _weight);
        }
    }
}

void AnchorOccupancyGrid::rasterizePolygonBoundary(const Polygon& _polygon, float _weight) {
    // Rasterize each ring's boundary as a closed line loop. See class comment: this is a
    // deliberate simplification (boundary only, no scanline fill) that is good enough at
    // this grid's coarse resolution.
    for (const auto& ring : _polygon) {
        rasterizeLine(ring, _weight);
        if (ring.size() > 1 && ring.front() != ring.back()) {
            // Close the loop if the source ring isn't already closed.
            markCell(glm::mix(ring.back(), ring.front(), 0.5f), _weight);
        }
    }
}

void AnchorOccupancyGrid::markCell(glm::vec2 _tileLocalPos, float _weight) {
    float cx = glm::clamp(_tileLocalPos.x, 0.f, 0.999999f);
    float cy = glm::clamp(_tileLocalPos.y, 0.f, 0.999999f);
    int ix = glm::clamp(int(cx * kResolution), 0, kResolution - 1);
    int iy = glm::clamp(int(cy * kResolution), 0, kResolution - 1);
    float& cell = m_cells[iy * kResolution + ix];
    // Take the max rather than accumulating: this is meant to answer "how busy is this
    // cell", not count crossings -- a cell with 10 minor trail segments through it
    // shouldn't outrank a cell with a single lake edge.
    cell = std::max(cell, _weight);
}

float AnchorOccupancyGrid::sample(glm::vec2 _tileLocalPos) const {
    float cx = glm::clamp(_tileLocalPos.x, 0.f, 0.999999f);
    float cy = glm::clamp(_tileLocalPos.y, 0.f, 0.999999f);
    int ix = glm::clamp(int(cx * kResolution), 0, kResolution - 1);
    int iy = glm::clamp(int(cy * kResolution), 0, kResolution - 1);
    return m_cells[iy * kResolution + ix];
}

} // namespace Tangram
