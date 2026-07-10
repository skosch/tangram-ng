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
    //  the RasterStyle mesh resolution (64) covering the worst-case overzoom of the vector source
    //  relative to the elevation source - see terrain_grid on unlit-polygons in stylus-osm.yaml.
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
