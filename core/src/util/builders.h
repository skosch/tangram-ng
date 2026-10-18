#pragma once

#include "data/tileData.h"

#include "glm/vec3.hpp"
#include "earcut.hpp"
#include <functional>
#include <vector>


namespace Tangram {

enum class CapTypes {
    butt = 0, // No points added to end of line
    square = 2, // Two points added to make a square extension
    round = 6 // Six points added in a fan to make a round cap
};

CapTypes CapTypeFromString(const std::string& str);

enum class JoinTypes {
    miter = 0, // No points added at line join
    bevel = 1, // One point added to flatten the corner of a join
    round = 5 // Five points added in a fan to make a round outer join
};

JoinTypes JoinTypeFromString(const std::string& str);

// Native grid resolution of the RasterStyle terrain mesh (rasterStyle.cpp) that 3D-terrain-draped
//  polygons must tessellate on to match the ground exactly when they are NOT overzoomed relative
//  to their data source (see PolygonBuilder::gridRes and PolygonStyleBuilder::setup, polygonStyle.cpp).
//  Shared by both files so their grid densities can never drift apart.
constexpr int kTerrainMeshGridRes = 64;

// Adaptive terrain mesh density (2026-10-11 perf work: a fixed 64x64 cells per tile put a z16
//  leaf that covers ~16 px on screen at 8192 triangles, and the heavy hillshade fragment shader
//  paid for 2x2-quad overshading of sub-pixel triangles - hillshade GPU 80 -> 34 ms in the
//  ASCEND_TERRAIN_GRID probe). A terrain tile of zoom z is drawn with
//      cells(z) = max(kTerrainMeshMinCells, 64 >> max(0, z - refZoom))
//  cells per edge, refZoom = round(view zoom) + bias. So every tile at or finer than refZoom has
//  the SAME world-space cell size (tile width(refZoom) / 64; ~4 px * 2^-bias on screen), and
//  coarser tiles keep 64. World cell size is therefore non-increasing with zoom and each coarser
//  lattice is a power-of-two multiple of every finer one, which is the nesting property the drape
//  lattice (getTerrainMeshElevation, elevation.yaml) and the triangle-PL composite upsampling
//  (rasterSource.cpp) rely on. The shader recomputes the identical formula from u_map_position.z
//  and the TERRAIN_MESH_ZBIAS define, so CPU and GPU must keep this definition in sync.
constexpr int kTerrainMeshMinCells = 4;
// `boost` (0 or 1, terrainMeshBoost()) doubles every tile's cell count while the camera is at
//  rest (settle-time refinement): world cell size stays a power-of-two ladder, so the finer
//  lattice nests exactly inside the coarser one (composites built at boost 0 stay valid: their
//  texels hold the coarse PL surface, which the finer lattice reproduces node for node).
inline int terrainMeshCellsAt(int tileZ, int refZoom, int boost = 0) {
    int d = tileZ - refZoom;
    if (d < 0) { d = 0; }
    if (d > 6) { d = 6; }
    int c = (kTerrainMeshGridRes << boost) >> d;
    return c < kTerrainMeshMinCells ? kTerrainMeshMinCells : c;
}
// bias from ASCEND_TERRAIN_ZBIAS (read once; default 100 = adaptive density off, i.e. always 64;
//  the app sets 1). Must equal global.terrain_mesh_zbias pushed to the scene.
int terrainMeshZoomBias();
// Current reference zoom (atomic; written once per frame by TileManager::updateTileSets).
int terrainMeshRefZoom();
void updateTerrainMeshRefZoom(double viewZoom);
// Settle-time density boost (ASCEND_TERRAIN_BOOST = max boost, default 1, 0 = off). The current
//  boost is written by Map::update (0 while moving, max once the camera has been still) and read
//  by the terrain mesh selection and, via u_terrain_mesh_boost, by the drape lattice shader.
int terrainMeshMaxBoost();
int terrainMeshBoost();
void setTerrainMeshBoost(int boost);

/* Callback function for PolygonBuilder:
 *
 * @coord  tesselated output coordinate
 * @normal triangle plane normal
 * @uv     texture coordinate of the output coordinate
 */
typedef std::function<void(const glm::vec3& coord, const glm::vec3& normal, const glm::vec2& uv)> PolygonVertexFn;

/* PolygonBuilder context,
 * see Builders::buildPolygon() and Builders::buildPolygonExtrusion()
 */
class WaterMask;

struct PolygonBuilder {
    std::vector<uint16_t> indices; // indices for drawing the polyon as triangles are added to this vector
    std::vector<int> used;

    PolygonVertexFn addVertex;
    size_t numVertices = 0;
    bool keepTileEdges;
    bool useTexCoords;
    // if > 0, tessellate polygons on a gridRes x gridRes grid over tile-local [0,1]^2 so that, with 3D
    //  terrain, per-vertex elevation makes the polygon follow the terrain surface instead of spanning
    //  it with a few huge earcut triangles (which lets terrain poke through the polygon interior, or the
    //  polygon bleed through ridges in front of it). Vertices are placed on the terrain mesh's own
    //  piecewise-linear surface (getTerrainMeshElevation, elevation.yaml), so the grid's only job is
    //  ensuring no triangle spans more than one terrain-lattice cell: use a power-of-two multiple of
    //  kTerrainMeshGridRes (the RasterStyle mesh resolution), scaled up per-tile for however overzoomed
    //  that tile actually is relative to its data source, capped at the style's configured worst-case
    //  ceiling - see terrain_grid on unlit-polygons in stylus-osm.yaml and PolygonStyleBuilder::setup.
    //  See Builders::buildPolygon().
    int gridRes = 0;

    // A single feature's grid tessellation can need more vertices than a uint16 index can address
    //  (large/complex polygon at high gridRes); buildPolygonGrid (builders.cpp) then flushes what
    //  it has as its own indexed sub-range and keeps going with a fresh 0-based vertex block,
    //  recording each flushed sub-range's (indexCount, vertexCount) here in append order. The
    //  final (usually only) chunk is NOT included - PolygonStyleBuilder::addPolygon appends it the
    //  same way it always has, so single-chunk features (the overwhelming majority) are unaffected.
    //  Mirrors the multi-batch mechanism MeshBase::compileIndices (mesh.cpp) already uses across
    //  separate features of one tile - just applied within a single feature.
    std::vector<std::pair<size_t, size_t>> chunkOffsets;

    mapbox::detail::Earcut<uint16_t> earcut;

    PolygonBuilder(PolygonVertexFn _addVertex = [](auto&,auto&,auto&){},
                   bool _kte = true, bool _useTexCoords = true)
        : addVertex(_addVertex), keepTileEdges(_kte), useTexCoords(_useTexCoords){}

    void clear() {
        numVertices = 0;
        indices.clear();
        chunkOffsets.clear();
    }
};


/* Callback function for PolyLineBuilder:
 *
 * @coord   tesselated output coordinate
 * @enormal extrusion vector of the output coordinate
 * @uv      texture coordinate of the output coordinate
 */
typedef std::function<void(const glm::vec2& coord, const glm::vec2& enormal, const glm::vec2& uv)> PolyLineVertexFn;

/* PolyLineBuilder context,
 * see Builders::buildPolyLine()
 */
struct PolyLineBuilder {
    std::vector<uint16_t> indices; // indices for drawing the polyline as triangles are added to this vector
    PolyLineVertexFn addVertex;
    size_t numVertices = 0;
    float miterLimit = 3.f;
    CapTypes cap;
    JoinTypes join;
    bool keepTileEdges;
    bool closedPolygon;
    bool useTexCoords = false;

    PolyLineBuilder(PolyLineVertexFn _addVertex = [](auto&,auto&,auto&){},
                    CapTypes _cap = CapTypes::butt,
                    JoinTypes _join = JoinTypes::bevel,
                    bool _kte = true, bool _closedPoly = false)
        : addVertex(_addVertex), cap(_cap), join(_join), keepTileEdges(_kte), closedPolygon(_closedPoly) {}

    void clear() {
        numVertices = 0;
        indices.clear();
    }
};

/* Callback function for SpriteBuilder
 * @coord tesselated coordinates of the sprite quad in screen space
 * @screenPos the screen position
 * @uv texture coordinate of the ouptput coordinate
 */
typedef std::function<void(const glm::vec2& coord, const glm::vec2& screenPos, const glm::vec2& uv)> SpriteBuilderFn;

/* SpriteBuidler context
 */
struct SpriteBuilder {
    std::vector<uint16_t> indices;
    SpriteBuilderFn addVertex;
    size_t numVerts = 0;

    SpriteBuilder(SpriteBuilderFn _addVertex) : addVertex(_addVertex) {}
};

class Builders {

public:

    /* Build a tesselated polygon
     * @_polygon input coordinates describing the polygon
     * @_ctx output vectors, see <PolygonBuilder>
     */
    static void buildPolygon(const Polygon& _polygon, float _height, PolygonBuilder& _ctx);

    /* Build a soft band along the INSIDE of every ring of a polygon (park / protected-area edge).
     * The mesh is a strip per boundary edge whose vertices all lie ON the boundary; the vertex shader
     * (polygon.vs, TANGRAM_EDGE_BAND) pushes them inward by the band's screen-pixel width, so the
     * width is constant on screen and identical across tiles whatever zoom each was built at:
     *   a_normal.xy = inward offset direction * miter / 2.5, a_normal.z = +1 land / -1 water,
     *   a_texcoord.x = fraction across the band (0 boundary .. 1 inner edge),
     *   a_texcoord.y = depth cap in tile units (1 = none) where a far boundary is close.
     * _width (tile units) is only the nominal width, used for sampling density and how far to look
     * for the far side. Rows are sampled at the terrain grid spacing so every vertex drapes on 3D
     * terrain individually. Edges are split where they cross _water (may be null) so each piece is
     * wholly land or wholly water. Unless keepTileEdges, tile-edge cuts are omitted.
     */
    static void buildPolygonEdgeBand(const Polygon& _polygon, float _width, float _lift, const WaterMask* _water, PolygonBuilder& _ctx);

    /* Build extruded 'walls' from a polygon
     * @_polygon input coordinates describing the polygon
     * @_minHeight the extrusion will extend from this z coordinate to the z of the polygon points
     * @_ctx output vectors, see <PolygonBuilder>
     */
    static void buildPolygonExtrusion(const Polygon& _polygon, float _minHeight, float _maxHeight, PolygonBuilder& _ctx);

    /* Build a tesselated polygon line of fixed width from line coordinates
     * @_line input coordinates describing the line
     * @_options parameters for polyline construction
     * @_ctx output vectors, see <PolyLineBuilder>
     */
    static void buildPolyLine(const Line& _line, PolyLineBuilder& _ctx);

    /* Build a tesselated quad centered on _screenOrigin
     * @_screenOrigin the sprite origin in screen space
     * @_size the size of the sprite in pixels
     * @_uvBL the bottom left UV coordinate of the quad
     * @_uvTR the top right UV coordinate of the quad
     * @_ctx output vectors, see <SpriteBuilder>
     */
    static void buildQuadAtPoint(const glm::vec2& _screenOrigin, const glm::vec2& _size, const glm::vec2& _uvBL, const glm::vec2& _uvTR, SpriteBuilder& _ctx);

};

}
