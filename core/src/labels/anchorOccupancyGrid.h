#pragma once

#include "data/tileData.h"

#include "glm/vec2.hpp"
#include <array>
#include <memory>

namespace Tangram {

// Coarse per-tile spatial index of "visually busy" vector geometry (water and
// transportation/road/trail features), used by label builders to steer anchor
// placement away from map features a label would otherwise sit on top of.
//
// See docs/label-placement-plan.md, Phase 3: a single style's StyleBuilder::addFeature
// pass (e.g. building `peak` labels) only ever sees the features of the layer(s) its own
// draw rules matched -- it has no visibility into the tile's *other* raw layers (e.g.
// `water`, `transportation`). This class is the answer to that phase's "one real open
// architecture question": a small shared per-tile pre-pass, built once from the tile's
// full TileData in TileBuilder::build() (which does see every layer) before any style's
// setup()/addFeature() runs, and handed to interested StyleBuilders via Tile.
//
// This is intentionally coarse and cheap: a fixed-resolution grid over the tile's
// normalized [0,1] x [0,1] local coordinate space (the same space Feature geometry is
// already expressed in -- see the coordinate-system comment atop tileData.h), storing a
// single scalar "occupancy" strength per cell. Only line/polygon-boundary geometry is
// rasterized (no polygon fill) -- see rasterizeLine()/rasterizePolygonBoundary() -- which
// is a deliberate simplification: at this grid resolution it is good enough to steer a
// label anchor away from a lake shoreline or a river, without the cost/complexity of a
// real scanline polygon fill.
class AnchorOccupancyGrid {
public:
    // Cells per side. 16x16 over a normalized tile is intentionally coarse -- this only
    // needs to answer "is there something visually busy near here", not give an exact
    // footprint.
    static constexpr int kResolution = 16;

    AnchorOccupancyGrid() { m_cells.fill(0.f); }

    // Builds a grid from a tile's full raw layer data (all layers, not just one style's
    // matched layer). Returns nullptr if no water/transportation geometry was found in
    // this tile, so callers can skip the term entirely with a single pointer check.
    static std::shared_ptr<AnchorOccupancyGrid> build(const TileData& _tileData);

    // Returns the occupancy strength (roughly [0,1], but see markCell()) at a
    // normalized tile-local position. Positions outside [0,1] are clamped to the nearest
    // edge cell -- this grid has no visibility into neighboring tiles, so a footprint
    // that extends past this tile's edge is approximated using this tile's own edge data
    // (the same kind of per-tile-only simplification the texture-shading mosaic project
    // made before its neighbor-stitching phase; not attempted here, see plan doc).
    float sample(glm::vec2 _tileLocalPos) const;

    void rasterizeLine(const Line& _line, float _weight);
    void rasterizePolygonBoundary(const Polygon& _polygon, float _weight);

private:
    void markCell(glm::vec2 _tileLocalPos, float _weight);

    std::array<float, kResolution * kResolution> m_cells;
};

} // namespace Tangram
