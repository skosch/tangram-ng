#pragma once

#include "style/style.h"

namespace Tangram {

class PolygonStyle : public Style {

public:

    PolygonStyle(std::string _name, Blending _blendMode = Blending::opaque, GLenum _drawMode = GL_TRIANGLES, bool _selection = true);

    virtual void constructVertexLayout() override;
    virtual void constructShaderProgram() override;
    virtual std::unique_ptr<StyleBuilder> createBuilder() const override;
    virtual void build(const Scene& _scene) override;
    virtual ~PolygonStyle() {}

    // 'terrain_grid' style parameter: tessellate polygons on this grid over tile-local [0,1]^2 so
    //  per-vertex elevation drapes them accurately over 3D terrain (see PolygonBuilder::gridRes);
    //  only takes effect when the scene has an ElevationManager (3D terrain enabled)
    void setTerrainGrid(int _res) { m_terrainGrid = _res; }
    int activeTerrainGrid() const { return m_activeTerrainGrid; }

    // 'edge_distance' style parameter (projected metres, 0 = off): vertex texcoord.x = distance to the polygon
    //  boundary / this cap (see PolygonBuilder::edgeDistCap). Implies texcoords.
    void setEdgeDistance(float _m) { m_edgeDistance = _m; }
    float edgeDistance() const { return m_edgeDistance; }

    // 'tree_scatter' style parameter (projected metres, 0 = off): draw whole trees scattered inside the polygon instead of its fill
    //  (Builders::buildTreeScatter); the value is the feather length over which the tree density ramps up from the outline.
    void setTreeScatter(float _m) { m_treeScatter = _m; }
    float treeScatter() const { return m_treeScatter; }

protected:

    int m_terrainGrid = 0;
    int m_activeTerrainGrid = 0;
    float m_edgeDistance = 0.f;
    float m_treeScatter = 0.f;

};

}
