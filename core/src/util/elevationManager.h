#pragma once

#include <memory>
#include <mutex>
#include <atomic>
#include "util/mapProjection.h"

namespace Tangram {

class RasterSource;
class Style;
class FrameBuffer;
class RenderState;
class View;
class Tile;
class AsyncWorker;
class Texture;

class ElevationManager
{
public:
  struct DepthData {
    std::vector<float> depth;
    int w = 0, h = 0;
    float zoom = 0;
    glm::dvec2 viewPos;
    glm::mat4 viewProj;
  };

  ElevationManager(std::shared_ptr<RasterSource> src, Style& style);
  ~ElevationManager();
  double getElevation(ProjectedMeters pos, bool& ok);
  glm::vec2 getMinMaxElev(TileID tileId, int ancestors = 0);

  // Per-tile "detail descriptor" for saliency-driven DEM LOD (docs/dem-saliency-lod-plan.md,
  // Phase 1). lapRms = RMS of the 4-neighbour discrete Laplacian of the tile's own texels, in
  // meters. Offline calibration on cached Esri WorldElevation tiles (Alps + BC + Europe,
  // z8-z15, ~7.7k parent->child quadrants): the 99th-percentile height error between a tile's
  // bilinear upsample and its true children is ~1.0-1.1 x lapRms, log-spread x1.6 (1 sigma).
  // Cached on the texture alongside min/max; own texture only (no ancestor climb).
  //
  // `facing` (may be null) is the per-texel salient-detail table described at DemFacingStats:
  // lets the LOD selector ask "how many texels of this tile that still hold unresolved detail
  // are oriented toward the camera?" instead of relying on a tile-wide mean.
  struct DemFacingStats;
  struct TerrainDetail { float relief = 0, lapRms = 0; bool valid = false; const DemFacingStats* facing = nullptr; };
  TerrainDetail getTerrainDetail(TileID tileId);

  // Salient-texel table (2026-10-11). Every interior texel with slope-change-per-texel
  // q = |4h - sum(4 neighbours)| / texelMeters >= kDemTierLo[0] is binned by (q tier, gradient
  // azimuth octant); per bin we keep the texel count and the summed gradient (dimensionless
  // rise/run, world axes: +x east, +y north). A query for any camera direction then evaluates
  // the projected-area factor P = max(0, g.v_h - v_z) per bin (see View::demRefineMass) without
  // touching texels again.
  static constexpr int kDemTiers = 6;
  static constexpr int kDemOctants = 8;
  struct DemFacingStats {
    float texelMeters = 0.f;   // ground size of one texel of THIS tile (true meters)
    uint32_t texels = 0;       // interior texel count (normalizer)
    uint32_t cnt[kDemTiers][kDemOctants] = {};
    float sgx[kDemTiers][kDemOctants] = {};
    float sgy[kDemTiers][kDemOctants] = {};
  };
  // Lower bound of each tier in q units; tier t covers [kDemTierLo[t], kDemTierLo[t+1]).
  static constexpr float kDemTierLo[kDemTiers + 1] = {0.05f, 0.1f, 0.2f, 0.4f, 0.8f, 1.6f, 1e9f};

  // Compute and cache the elevation statistics (min/max, Laplacian RMS, facing table) on
  // `tex` for tile `id`. Idempotent; thread-safe only against OTHER textures (call it from the
  // worker that owns a freshly decoded, not yet published texture).
  static void precomputeTileInfo(Texture& tex, TileID id);
  int32_t elevationSourceId() const;
  float getDepth(glm::vec2 screenpos);

  // Conservative occlusion query against the most recent terrain depth buffer (the same CPU
  // copy labels use). `lo`/`hi` are the box's projected-meter x/y extents, [zMin, zMax] its
  // elevation range. True only when EVERY sampled pixel under the box's screen projection
  // already shows terrain clearly nearer than the box's nearest corner - i.e. the box lies
  // entirely behind terrain drawn last frame. False (not occluded) whenever there is no usable
  // depth data, the box reaches behind/near the camera, or any sampled pixel is sky/unknown.
  bool isBoxOccluded(glm::dvec2 lo, glm::dvec2 hi, float zMin, float zMax);
  DepthData& getDepthData() { return m_depthData[0]; }
  bool hasTile(TileID tileId);
  void setMinZoom(int z) { m_minZoom = z; }
  int minZoom() const { return m_minZoom; }

  void renderTerrainDepth(RenderState& _rs, const View& _view,
                          const std::vector<std::shared_ptr<Tile>>& _tiles);

  static double elevationLerp(const Texture& tex, glm::vec2 pos, glm::vec2* gradOut = nullptr);
  static double elevationLerp(const Texture& tex, TileID tileId, ProjectedMeters meters);

  void drawDepthDebug(RenderState& _rs, const View& _view);

  std::shared_ptr<RasterSource> m_elevationSource;
  std::unique_ptr<Style> m_style;
  std::unique_ptr<FrameBuffer> m_frameBuffer;
  DepthData m_depthData[2];
  int m_minZoom = 0;
  float m_terrainScale = 1.0f;

  static std::unique_ptr<RenderState> m_renderState;
  static std::unique_ptr<AsyncWorker> offscreenWorker;
};

}
