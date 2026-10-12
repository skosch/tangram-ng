#pragma once

#include "style/polygonStyle.h"

namespace Tangram {

// derive from PolygonStyle just to avoid duplicating polygon shader source
class RasterStyle : public PolygonStyle {

public:

    RasterStyle(std::string _name, Blending _blendMode = Blending::opaque);
    void build(const Scene& _scene) override;
    void constructVertexLayout() override;
    void constructShaderProgram() override;
    std::unique_ptr<StyleBuilder> createBuilder() const override;

    // finest (64-cell) mesh; kept for callers that don't know the tile zoom
    StyledMesh* rasterMesh() const { return m_rasterMeshes[0].get(); }
    // mesh with the adaptive cell count for a terrain tile of zoom `tileZ` (terrainMeshCellsAt)
    // `boost` < 0 = the current settle-time boost (terrainMeshBoost()); the terrain depth pass passes 0
    //  so the camera's depth-derived zoom can never depend on the boost (feedback loop otherwise)
    StyledMesh* rasterMeshFor(int tileZ, int boost = -1) const;

protected:
    // index i = 128 >> i cells per edge (128, 64, 32, 16, 8, 4); only [0] exists without 3D
    //  terrain (index 0 is the boosted, settle-time density)
    static constexpr int kNumRasterMeshes = 6;
    std::unique_ptr<StyledMesh> m_rasterMeshes[kNumRasterMeshes];

    UniformLocation m_uColor{"u_color"};
    UniformLocation m_uOrder{"u_order"};
    friend struct SharedMesh;

};

}
