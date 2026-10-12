#include "style/polygonStyle.h"
#include <cstdlib>

#include "gl/mesh.h"
#include "gl/shaderProgram.h"
#include "map.h"
#include "scene/scene.h"
#include "marker/marker.h"
#include "material.h"
#include "platform.h"
#include "scene/drawRule.h"
#include "tile/tile.h"
#include "util/builders.h"
#include "util/color.h"
#include "util/extrude.h"

#include "glm/vec2.hpp"
#include "glm/vec3.hpp"
#include "glm/gtc/type_precision.hpp"
#include <algorithm>
#include <cmath>

#include "polygon_fs.h"
#include "polygon_vs.h"

constexpr float position_scale = 8192.0f;
constexpr float texture_scale = 65535.0f;
constexpr float normal_scale = 127.0f;

namespace Tangram {


struct PolygonVertexNoUVs {

    PolygonVertexNoUVs(glm::vec3 position, uint32_t order, glm::vec3 normal, glm::vec2 uv, GLuint abgr, GLuint selection)
        : pos(glm::i16vec4{ nearbyint(position * position_scale), order }),
          norm(normal * normal_scale),
          abgr(abgr),
          selection(selection) {}

    glm::i16vec4 pos; // pos.w contains layer (params.order)
    glm::i8vec3 norm;
    uint8_t padding = 0;
    GLuint abgr;
    GLuint selection;
};

struct PolygonVertex : PolygonVertexNoUVs {

    PolygonVertex(glm::vec3 position, uint32_t order, glm::vec3 normal, glm::vec2 uv, GLuint abgr, GLuint selection)
        : PolygonVertexNoUVs(position, order, normal, uv, abgr, selection), texcoord(uv * texture_scale) {}

    glm::u16vec2 texcoord;
};

PolygonStyle::PolygonStyle(std::string _name, Blending _blendMode, GLenum _drawMode, bool _selection)
    : Style(_name, _blendMode, _drawMode, _selection) {
    m_type = StyleType::polygon;
    m_material.material = std::make_shared<Material>();
}

void PolygonStyle::constructVertexLayout() {

    if (m_texCoordsGeneration) {
        m_vertexLayout = std::shared_ptr<VertexLayout>(new VertexLayout({
            {"a_position", 4, GL_SHORT, false, 0},
            {"a_normal", 4, GL_BYTE, true, 0}, // The 4th byte is for padding
            {"a_color", 4, GL_UNSIGNED_BYTE, true, 0},
            {"a_selection_color", 4, GL_UNSIGNED_BYTE, true, 0},
            {"a_texcoord", 2, GL_UNSIGNED_SHORT, true, 0},
        }));
    } else {
        m_vertexLayout = std::shared_ptr<VertexLayout>(new VertexLayout({
            {"a_position", 4, GL_SHORT, false, 0},
            {"a_normal", 4, GL_BYTE, true, 0},
            {"a_color", 4, GL_UNSIGNED_BYTE, true, 0},
            {"a_selection_color", 4, GL_UNSIGNED_BYTE, true, 0},
        }));
    }

}

void PolygonStyle::build(const Scene& _scene) {
    Style::build(_scene);
    // grid tessellation is only useful when polygons are draped over 3D terrain; without an
    //  ElevationManager plain earcut output is used regardless of the terrain_grid style param
    m_activeTerrainGrid = _scene.elevationManager() ? m_terrainGrid : 0;
}

void PolygonStyle::constructShaderProgram() {

    m_shaderSource->setSourceStrings(polygon_fs, polygon_vs);

    if (m_texCoordsGeneration) {
        m_shaderSource->addSourceBlock("defines", "#define TANGRAM_USE_TEX_COORDS\n");
    }
}

template <class V>
struct PolygonStyleBuilder : public StyleBuilder {

public:

    struct Parameters {
        uint32_t order = 0;
        uint32_t color = 0xffffffff;
        float height;
        float minHeight;
        uint32_t selectionColor = 0;
        bool keepTileEdges = false;
    };

    void setup(const Tile& _tile) override {
        m_tileUnitsPerMeter = _tile.getInverseScale();
        auto id = _tile.getID();
        m_zoom = id.z;
        m_meshData.clear();
        // activeTerrainGrid() is sized for the worst-case overzoom (style's data source vs. the
        //  elevation source's max_zoom), but most tiles aren't overzoomed at all - id.s > id.z only
        //  when this tile is being displayed finer than its own data zoom. Scale the grid down to
        //  match how overzoomed THIS tile actually is, so the common (non-overzoomed) case ends up
        //  exactly at kTerrainMeshGridRes, matching the terrain mesh's own native grid density
        //  (rasterStyle.cpp) instead of needlessly tessellating finer than the ground it sits on.
        //  Never exceeds the configured ceiling, so the worst-case tiles are unaffected.
        int overzoomLevels = std::max(0, int(id.s) - int(id.z));
        // Restored 2026-10-10 (perf): the rock-texture-v2 round-20 forced-density TEMP that
        //  bypassed this scale-down (always using the full 256 ceiling) was a test of whether
        //  mesh density caused the torn-triangle symptom; rounds 17-19 then found the real
        //  causes (OSM coverage gap, tile-cache thrash) and the density hypothesis was
        //  independently refuted in round 16, so the forced ceiling bought nothing and cost up
        //  to 16x the polygon vertices (256^2 vs 64^2 cells per tile) on every draped
        //  land-cover tile - a real trace had unlit-polygons at 77 ms/frame with a near-empty
        //  fragment shader. Non-overzoomed tiles now match the terrain mesh's own 64-cell grid
        //  (the design intent above); overzoomed ones still scale up to the style's ceiling.
        // The polygon grid must be at least as fine as the terrain lattice it drapes on, or
        //  triangles chord across convex relief and the hillshade overlay fails its depth test
        //  there (unshaded "torn" patches on steep faces - confirmed 2026-10-12: forcing 256
        //  removed them). The lattice per vector-tile edge is terrainMeshCellsAt(zt, ref) <<
        //  (zt - z) with zt up to z + 2, i.e. 64 << clamp(ref - z, 0, 2) cells, where ref is
        //  the CURRENT reference zoom (round(view zoom) + bias; util/builders.h). The tile's own
        //  styling zoom `s` is only a per-tile screen-area heuristic and routinely lags the view
        //  (far tiles, tiles built before a zoom-in), so size the grid from the reference zoom
        //  too (+1 level of margin for zooming in after the build), whichever is finer.
        const int refLevels = std::max(0, std::min(2, terrainMeshRefZoom() - int(id.z) + 1));
        const int gridLevels = std::max(overzoomLevels, refLevels);
        m_builder.gridRes = m_style.activeTerrainGrid() > 0
            ? std::min(m_style.activeTerrainGrid(), kTerrainMeshGridRes << gridLevels)
            : 0;
        // DIAGNOSTIC (2026-10-12, drape patches): ASCEND_POLY_GRID=N forces the draped polygon grid
        //  to min(style ceiling, N) regardless of the tile's styling zoom, to test whether polygon
        //  triangles chording across the terrain lattice (grid coarser than lattice) causes the
        //  unshaded/torn patches. Costs vertices; remove once settled.
        if (m_style.activeTerrainGrid() > 0) {
            static const int forced = [] {
                const char* e = getenv("ASCEND_POLY_GRID");
                return e ? atoi(e) : 0;
            }();
            if (forced > 0) { m_builder.gridRes = std::min(m_style.activeTerrainGrid(), forced); }
        }
        // TEMP DIAGNOSTIC (rock-texture-v2 round 18, "torn holes while panning at zoom 16+,
        //  angle-dependent" investigation): rounds 14-17 refuted the composite-data-mismatch,
        //  gridRes-staleness, elevation-shelf-pinning, and OSM-coverage-gap hypotheses in
        //  turn. This tests the one mesh-density angle only reasoned about abstractly so far,
        //  not measured: whether gridRes actually reaches the "256, hand-verified sufficient"
        //  value during ORDINARY (not deeply overzoomed) zoom-16 panning at tilt 0, or is
        //  landing lower (64/128) because overzoomLevels (id.s - id.z, THIS tile's own
        //  build-time styling zoom) undercounts the true resolution gap - id.s reflects the
        //  vector tile's own screen-area heuristic, not what the elevation tileset
        //  independently ends up displaying under this same footprint (the same divergence
        //  round 14/15 fixed for composite DATA, but never for mesh geometry - see
        //  docs/rock-texture-v2-plan.md round 15's "accept bounded chording" deferral).
        //  Fires once per tile build (not per-frame), so LOGD is fine at that frequency.
        if (m_style.activeTerrainGrid() > 0) {
            LOGD("PolygonStyleBuilder::setup gridRes: tile %s overzoomLevels=%d gridRes=%d (ceiling=%d) sourceID=%d sourceGen=%lld style=%s",
                 id.toString().c_str(), overzoomLevels, m_builder.gridRes, m_style.activeTerrainGrid(),
                 _tile.sourceID(), (long long)_tile.sourceGeneration(), m_style.getName().c_str());
        }
    }

    void setup(const Marker& _marker, int zoom) override {
        m_zoom = zoom;
        m_tileUnitsPerMeter = 1.f / _marker.extent();
        m_meshData.clear();
        // markers use their own local coordinate extent, not tile-local [0,1]^2, so the terrain
        //  grid (which must align with the tile grid) does not apply
        m_builder.gridRes = 0;
    }

    bool addPolygon(const Polygon& _polygon, const Properties& _props, const DrawRule& _rule) override;

    const Style& style() const override { return m_style; }

    std::unique_ptr<StyledMesh> build() override;

    PolygonStyleBuilder(const PolygonStyle& _style) : m_style(_style) {}

    Parameters parseRule(const DrawRule& _rule, const Properties& _props);

    PolygonBuilder& polygonBuilder() { return m_builder; }

private:

    const PolygonStyle& m_style;

    PolygonBuilder m_builder;

    MeshData<V> m_meshData;

    float m_tileUnitsPerMeter = 0;
    int m_zoom = 0;

};

template <class V>
std::unique_ptr<StyledMesh> PolygonStyleBuilder<V>::build() {
    if (m_meshData.vertices.empty()) { return nullptr; }

    auto mesh = std::make_unique<Mesh<V>>(m_style.vertexLayout(),
                                                      m_style.drawMode());
    mesh->compile(m_meshData);
    m_meshData.clear();

    return std::move(mesh);
}

template <class V>
auto PolygonStyleBuilder<V>::parseRule(const DrawRule& _rule, const Properties& _props) -> Parameters {
    Parameters p;
    _rule.get(StyleParamKey::color, p.color);
    float alpha = 1;
    if (_rule.get(StyleParamKey::alpha, alpha)) {
        p.color = Color(p.color).withAlpha(alpha).abgr;
    }

    _rule.get(StyleParamKey::order, p.order);
    _rule.get(StyleParamKey::tile_edges, p.keepTileEdges);

    if (Tangram::getDebugFlag(Tangram::DebugFlags::proxy_colors)) {
        p.color <<= (m_zoom % 6);
    }

    auto& extrude = _rule.findParameter(StyleParamKey::extrude);
    p.minHeight = getLowerExtrudeMeters(extrude, _props) * m_tileUnitsPerMeter;
    p.height = getUpperExtrudeMeters(extrude, _props) * m_tileUnitsPerMeter;

    p.selectionColor = _rule.selectionColor;
    return p;
}

template <class V>
bool PolygonStyleBuilder<V>::addPolygon(const Polygon& _polygon, const Properties& _props, const DrawRule& _rule) {

    auto p = parseRule(_rule, _props);

    m_builder.keepTileEdges = p.keepTileEdges;

    m_builder.addVertex = [this, p](const glm::vec3& coord,
                                 const glm::vec3& normal,
                                 const glm::vec2& uv) {
        m_meshData.vertices.push_back({ coord, p.order, normal, uv, p.color, p.selectionColor });
    };

    if (p.minHeight != p.height) {
        Builders::buildPolygonExtrusion(_polygon, p.minHeight,
                                        p.height, m_builder);
    }

    Builders::buildPolygon(_polygon, p.height, m_builder);

    m_meshData.indices.insert(m_meshData.indices.end(),
                              m_builder.indices.begin(),
                              m_builder.indices.end());

    // terrain-grid tessellation (see PolygonBuilder::chunkOffsets) may have flushed this
    //  feature into multiple indexed sub-ranges when it needed more vertices than a uint16
    //  index can address; push each flushed chunk first, then the final (usually only) one -
    //  m_builder.indices holds all of them concatenated, so the last chunk's index count is
    //  whatever remains after the already-flushed ones.
    size_t chunkedIndices = 0;
    for (auto& chunk : m_builder.chunkOffsets) {
        m_meshData.offsets.push_back(chunk);
        chunkedIndices += chunk.first;
    }
    m_meshData.offsets.emplace_back(m_builder.indices.size() - chunkedIndices,
                                    m_builder.numVertices);
    m_builder.clear();

    return true;
}

std::unique_ptr<StyleBuilder> PolygonStyle::createBuilder() const {
    if (m_texCoordsGeneration) {
        auto builder = std::make_unique<PolygonStyleBuilder<PolygonVertex>>(*this);
        builder->polygonBuilder().useTexCoords = true;
        return std::move(builder);
    } else {
        auto builder = std::make_unique<PolygonStyleBuilder<PolygonVertexNoUVs>>(*this);
        builder->polygonBuilder().useTexCoords = false;
        return std::move(builder);
    }
}

}
