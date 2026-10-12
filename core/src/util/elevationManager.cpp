#include "util/elevationManager.h"

#include "debug/profiler.h"
#include "scene/scene.h"
#include "util/asyncWorker.h"
#include "data/rasterSource.h"
#include "style/rasterStyle.h"
#include "gl/framebuffer.h"
#include "gl/renderState.h"
#include "gl/shaderProgram.h"
#include "gl/primitives.h"
#include "marker/marker.h"
#include "log.h"
#include "debug/frameInfo.h"

#include "../../platforms/common/platform_gl.h"

// with depth test enabled and no blending, final value of output should be correct depth (if larger depth
//  written first, will be overwritten; if smaller depth written first, depth test will discard larger depth)
const static char* terrain_depth_fs = R"RAW_GLSL(#version 300 es
#ifdef GL_ES
precision highp float;
#endif

in vec4 v_position;

layout (location = 0) out highp uint depthOut;

void main(void) {
  depthOut = floatBitsToUint(-v_position.z);  // if needed, ndcZ = (-proj[2][2]*clipW + proj[3][2])/clipW)
}
)RAW_GLSL";

namespace Tangram {

// smaller target greatly improves FPS
static float bufferScale = 2;
std::unique_ptr<RenderState> ElevationManager::m_renderState;
std::unique_ptr<AsyncWorker> ElevationManager::offscreenWorker;

class TerrainStyle : public RasterStyle
{
public:
  using RasterStyle::RasterStyle;

  void build(const Scene& _scene) override {
      RasterStyle::build(_scene);
      m_shaderProgram = std::make_shared<ShaderProgram>(
          m_shaderProgram->vertexShaderSource(), terrain_depth_fs, vertexLayout().get());
  }

  bool draw(RenderState& rs, const Tile& _tile) override {

      // need to check for mesh of cloned style to determine if we should draw tile
      auto& styleMesh = _tile.getMesh(*this);
      if (!styleMesh) { return false; }

      int prevTexUnit = rs.currentTextureUnit();
      setupTileShaderUniforms(rs, _tile, *m_shaderProgram, m_mainUniforms);
      m_shaderProgram->setUniformf(rs, m_uOrder, 0.f);

      bool styleMeshDrawn = rasterMeshFor(_tile.getID().z, 0)->draw(rs, *m_shaderProgram);
      if (!styleMeshDrawn) {
          LOGN("Mesh built by style %s cannot be drawn", m_name.c_str());
      }

      rs.resetTextureUnit(prevTexUnit);
      return styleMeshDrawn;
  }
};

static double readElevTex(const Texture& tex, int x, int y)
{
  // see getElevation() in hillshade.yaml and https://github.com/tilezen/joerd
  if(tex.getOptions().pixelFormat == PixelFormat::FLOAT)
    return ((float*)tex.bufferData())[y*tex.width() + x];
  GLubyte* p = tex.bufferData() + y*tex.width()*4 + x*4;
  //(red * 256 + green + blue / 256) - 32768
  return (p[0]*256 + p[1] + p[2]/256.0) - 32768;
}

double ElevationManager::elevationLerp(const Texture& tex, glm::vec2 pos, glm::vec2* gradOut)
{
  double x0 = pos.x*tex.width() - 0.5, y0 = pos.y*tex.height() - 0.5;  // -0.5 to adjust for pixel centers
  // we should extrapolate at edges instead of clamping - see shader in raster_contour.yaml
  int ix0 = std::max(0, int(std::floor(x0)));
  int iy0 = std::max(0, int(std::floor(y0)));
  int ix1 = std::min(int(std::ceil(x0)), tex.width()-1);
  int iy1 = std::min(int(std::ceil(y0)), tex.height()-1);
  double fx = x0 - ix0, fy = y0 - iy0;
  double t00 = readElevTex(tex, ix0, iy0);
  double t01 = readElevTex(tex, ix0, iy1);
  double t10 = readElevTex(tex, ix1, iy0);
  double t11 = readElevTex(tex, ix1, iy1);

  if(gradOut) {
    double dx0 = t10 - t00, dx1 = t11 - t01;
    double dy0 = t01 - t00, dy1 = t11 - t10;
    gradOut->x = (dx0 + fy*(dx1 - dx0))*tex.width();
    gradOut->y = (dy0 + fx*(dy1 - dy0))*tex.height();
  }

  double t0 = t00 + fx*(t10 - t00);
  double t1 = t01 + fx*(t11 - t01);
  return t0 + fy*(t1 - t0);
}

double ElevationManager::elevationLerp(const Texture& tex, TileID tileId, ProjectedMeters meters)
{
  double scale = MapProjection::metersPerTileAtZoom(tileId.z);
  ProjectedMeters tileOrigin = MapProjection::tileSouthWestCorner(tileId);
  //ProjectedMeters meters = MapProjection::lngLatToProjectedMeters(pos);  //glm::dvec2(tileCoord) * scale + tileOrigin;
  ProjectedMeters offset = meters - tileOrigin;
  double ox = offset.x/scale, oy = offset.y/scale;
  if(ox < 0 || ox > 1 || oy < 0 || oy > 1)
    return 0;  //LOGE("Elevation tile position out of range");
  return elevationLerp(tex, glm::vec2(ox, oy));
}

double ElevationManager::getElevation(ProjectedMeters pos, bool& ok)
{
  static std::weak_ptr<Texture> prevTex;
  static TileID prevTileId = {0, 0, 0, 0};

  ok = true;
  if (prevTileId.z >= m_minZoom) {
    TileID tileId = MapProjection::projectedMetersTile(pos, prevTileId.z);
    if(tileId == prevTileId) {
      if(auto tex = prevTex.lock())
        return elevationLerp(*tex.get(), tileId, pos);
    }
  }

  Raster raster = m_elevationSource->getRaster(pos);
  if(raster.texture) {
    prevTileId = raster.tileID;
    prevTex = raster.texture;
    ok = raster.tileID.z + 4 > m_minZoom;
    return elevationLerp(*raster.texture, raster.tileID, pos);
  }

  ok = false;
  return 0;
}

bool ElevationManager::hasTile(TileID tileId)
{
  bool ok = false;
  getElevation(MapProjection::tileCenter(tileId), ok);
  return ok;
}

// C++14: odr-used static constexpr members need an out-of-class definition.
constexpr int ElevationManager::kDemTiers;
constexpr int ElevationManager::kDemOctants;
constexpr float ElevationManager::kDemTierLo[];

namespace {
// Cached on the elevation Texture's userData (see ElevationManager::getMinMaxElev).
struct ElevTexInfo {
  uint32_t magic = 0xE1E7E1E7u;  // != kElevationMosaicInfoMagic (data/rasterSource.h)
  float min, max;
  float lapRms;
  ElevationManager::DemFacingStats facing;
};

// Gradient-azimuth octant, centered on the axes (0 = +x, 1 = +x+y, 2 = +y, ... counter-
// clockwise), without atan2.
inline int gradOctant(float gx, float gy)
{
  float ax = std::fabs(gx), ay = std::fabs(gy);
  if(ay <= 0.41421356f*ax) { return gx >= 0 ? 0 : 4; }
  if(ax <= 0.41421356f*ay) { return gy >= 0 ? 2 : 6; }
  if(gx >= 0) { return gy >= 0 ? 1 : 7; }
  return gy >= 0 ? 3 : 5;
}

ElevTexInfo* elevTexInfo(Texture& tex, TileID id)
{
  if(!tex.userData) {
    auto infoPtr = std::make_shared<ElevTexInfo>();
    ElevTexInfo& info = *infoPtr;
    info.min = FLT_MAX; info.max = -FLT_MAX; info.lapRms = 0;
    const int w = tex.width(), h = tex.height();
    const size_t npix = size_t(w)*h;
    const bool isFloat = tex.getOptions().pixelFormat == PixelFormat::FLOAT;
    const float* fbuf = isFloat ? (const float*)tex.bufferData() : nullptr;
    const uint8_t* bbuf = isFloat ? nullptr : tex.bufferData();
    auto at = [&](size_t i) -> float {
      if(isFloat) { return fbuf[i]; }
      const uint8_t* p = &bbuf[i*4];
      return (p[0]*256 + p[1] + p[2]/256.0f) - 32768;
    };
    for(size_t ii = 0; ii < npix; ++ii) {
      float e = at(ii);
      info.min = std::min(e, info.min);
      info.max = std::max(e, info.max);
    }
    if(w > 2 && h > 2) {
      // true ground size of one texel (256 texels span a tile; node-registered 257-wide tiles)
      LngLat ll = MapProjection::projectedMetersToLngLat(MapProjection::tileCenter(id));
      double cosLat = std::cos(ll.latitude*M_PI/180.0);
      double texM = MapProjection::metersPerTileAtZoom(id.z)*cosLat/double(w - (w % 2));
      info.facing.texelMeters = float(texM);
      info.facing.texels = uint32_t((w - 2)*(h - 2));
      const float invH = float(1.0/texM);
      const float invH2 = 0.5f*invH;
      double sum = 0;
      for(int y = 1; y < h - 1; ++y) {
        for(int x = 1; x < w - 1; ++x) {
          size_t i = size_t(y)*w + x;
          float l = 4*at(i) - at(i - 1) - at(i + 1) - at(i - w) - at(i + w);
          sum += double(l)*l;
          float q = std::fabs(l)*invH;
          if(q >= ElevationManager::kDemTierLo[0]) {
            int t = 0;
            while(t + 1 < ElevationManager::kDemTiers && q >= ElevationManager::kDemTierLo[t + 1]) { ++t; }
            // rows run south -> north, columns west -> east (= world +y, +x)
            float gx = (at(i + 1) - at(i - 1))*invH2;
            float gy = (at(i + w) - at(i - w))*invH2;
            int o = gradOctant(gx, gy);
            info.facing.cnt[t][o]++;
            info.facing.sgx[t][o] += gx;
            info.facing.sgy[t][o] += gy;
          }
        }
      }
      info.lapRms = float(std::sqrt(sum / (double(w - 2)*(h - 2))));
    }
    tex.userData = infoPtr;
  }
  return static_cast<ElevTexInfo*>(tex.userData.get());
}
}  // namespace

void ElevationManager::precomputeTileInfo(Texture& tex, TileID id)
{
  if(tex.bufferData()) { elevTexInfo(tex, TileID(id.x, id.y, id.z)); }
}

glm::vec2 ElevationManager::getMinMaxElev(TileID tileId, int ancestors)
{
  while(tileId.z > m_elevationSource->zoomOptions().maxZoom) { tileId = tileId.getParent(); }
  auto tex = m_elevationSource->getTexture(tileId);
  while(!tex && (tileId.z > 14 || (ancestors-- > 0 && tileId.z > 1))) {
    tileId = tileId.getParent();
    tex = m_elevationSource->getTexture(tileId);
  }
  if(!tex) { return {0, 9000}; }

  auto* info = elevTexInfo(*tex, tileId);
  return {info->min, info->max};
}

ElevationManager::TerrainDetail ElevationManager::getTerrainDetail(TileID tileId)
{
  while(tileId.z > m_elevationSource->zoomOptions().maxZoom) { tileId = tileId.getParent(); }
  auto tex = m_elevationSource->getTexture(tileId);
  if(!tex) { return {}; }
  auto* info = elevTexInfo(*tex, tileId);
  TerrainDetail d;
  d.relief = info->max - info->min;
  d.lapRms = info->lapRms;
  d.valid = true;
  d.facing = info->facing.texels > 0 ? &info->facing : nullptr;
  return d;
}

int32_t ElevationManager::elevationSourceId() const
{
  return m_elevationSource ? m_elevationSource->id() : -1;
}

void ElevationManager::renderTerrainDepth(RenderState& _rs, const View& _view,
                                          const std::vector<std::shared_ptr<Tile>>& _tiles)
{
  static std::atomic_bool isRunning{false};
  FrameInfo::scope _trace("renderTerrainDepth");

  //offscreenWorker.reset();
  if(!offscreenWorker) {
    LOGE("Offscreen worker has not been created!");

    int w = _view.getWidth()/bufferScale, h = _view.getHeight()/bufferScale;
    if (!m_frameBuffer || m_frameBuffer->getWidth() != w || m_frameBuffer->getHeight() != h) {
      m_frameBuffer = std::make_unique<FrameBuffer>(w, h, false, GL_R32UI);
      //_rs.m_terrainDepthTexture = m_frameBuffer->getTextureHandle();
    }

    auto& d = m_depthData[0];
    d.w = w;
    d.h = h;
    d.zoom = _view.getBaseZoom();
    if (int(d.depth.size()) != w * h)
      d.depth.resize(w * h, 0.0f);

    _rs.cacheDefaultFramebuffer();
    m_frameBuffer->applyAsRenderTarget(_rs);
    m_style->draw(_rs, _view, _tiles, {});

    GL::readPixels(0, 0, w, h, GL_RED_INTEGER, GL_UNSIGNED_INT, d.depth.data());
    _rs.framebuffer(_rs.defaultFrameBuffer());

    return;
  }

  // don't delay frame to wait for terrain depth
  if (isRunning) { return; }  //LOGD("Terrain depth frame %d skipped", int(Scene::frameCount));

  // glReadPixels() blocks on mobile, even when reading to a pixel buffer object, so render and read depth
  //  data on a separate thread using a shared GL context

  if(!m_renderState)
    m_renderState = std::make_unique<RenderState>();

  std::mutex drawMutex;
  std::condition_variable drawCond;
  bool drawFinished = false;
  std::unique_lock<std::mutex> mainLock(drawMutex);

  offscreenWorker->enqueue([&](){
    // CPU-only zone: this runs on the offscreen elevation worker thread, so PROFILE_GPU (main
    // GL context only) must not be used here even though this does issue GL calls.
    PROFILE_SCOPE("ElevationOffscreenDraw");
    std::unique_lock<std::mutex> workerLock(drawMutex);
    std::swap(m_depthData[0], m_depthData[1]);

    isRunning = true;
    m_renderState->flushResourceDeletion();
    int w = _view.getWidth()/bufferScale, h = _view.getHeight()/bufferScale;
    if (!m_frameBuffer || m_frameBuffer->getWidth() != w || m_frameBuffer->getHeight() != h)
      m_frameBuffer = std::make_unique<FrameBuffer>(w, h, false, GL_R32UI);
    m_frameBuffer->applyAsRenderTarget(*m_renderState);  // this does the glClear()

    auto& d = m_depthData[1];
    d.w = w;
    d.h = h;
    d.zoom = _view.getBaseZoom();
    d.viewPos = _view.getPosition();
    d.viewProj = _view.getViewProjectionMatrix();
    if (int(d.depth.size()) != w * h)
      d.depth.resize(w * h, 0.0f);

    // originally, we were reusing mesh from another style, but this will use the uniform location for the
    //  other style (since SharedMesh saves Style*); also creates problems when deleting Scene if
    //  first raster tile was drawn by offscreen worker; also, VAOs can't be shared between contexts
    m_style->draw(*m_renderState, _view, _tiles, {});

    drawFinished = true;
    drawCond.notify_all();  // notify before unlocking since drawCond can be destroyed after unlock
    workerLock.unlock();

    GL::readPixels(0, 0, w, h, GL_RED_INTEGER, GL_UNSIGNED_INT, d.depth.data());
    isRunning = false;
  });

  // wait for draw to finish to avoid changes to tiles, duplicate texture uploads, etc.
  drawCond.wait(mainLock, [&]{ return drawFinished; });
}

float ElevationManager::getDepth(glm::vec2 screenpos)
{
  auto& d = m_depthData[0];
  if(d.depth.empty()) { return 0; }
  glm::vec2 pos = glm::floor(screenpos/bufferScale);
  if(pos.x < 0 || pos.y < 0 || pos.x >= d.w || pos.y >= d.h) { return 0; }
  //return 2*depth[int(pos.x) + int(h - pos.y - 1)*w] - 1;  // convert from 0..1 (glDepthRange) to -1..1 (NDC)
  return m_depthData[0].depth[int(pos.x) + int(d.h - pos.y - 1)*d.w];
}

bool ElevationManager::isBoxOccluded(glm::dvec2 lo, glm::dvec2 hi, float zMin, float zMax)
{
  auto& d = m_depthData[0];
  if(d.depth.empty() || d.w <= 0 || d.h <= 0 || d.viewProj == glm::mat4(0)) { return false; }

  // corners relative to the camera position the depth pass was rendered from (same convention
  // as the label code: model translation = origin - viewPos, absolute elevation)
  glm::dvec2 c = MapProjection::wrapProjectedMeters(0.5*(lo + hi) - d.viewPos);
  glm::dvec2 half = 0.5*(hi - lo);
  float minW = FLT_MAX;
  float sx0 = FLT_MAX, sy0 = FLT_MAX, sx1 = -FLT_MAX, sy1 = -FLT_MAX;
  for(int i = 0; i < 8; ++i) {
    glm::vec4 p(float(c.x + ((i & 1) ? half.x : -half.x)),
                float(c.y + ((i & 2) ? half.y : -half.y)),
                (i & 4) ? zMax : zMin, 1.f);
    glm::vec4 clip = d.viewProj*p;
    if(!(clip.w > 1e-3f)) { return false; }   // reaches the camera plane: don't claim occlusion
    minW = std::min(minW, clip.w);
    float sx = (clip.x/clip.w*0.5f + 0.5f)*float(d.w);
    float sy = (clip.y/clip.w*0.5f + 0.5f)*float(d.h);   // GL row order (0 = bottom), as stored
    sx0 = std::min(sx0, sx); sx1 = std::max(sx1, sx);
    sy0 = std::min(sy0, sy); sy1 = std::max(sy1, sy);
  }
  if(sx1 < 0 || sy1 < 0 || sx0 >= d.w || sy0 >= d.h) { return false; }  // off-screen: not our call
  sx0 = std::max(sx0, 0.f); sy0 = std::max(sy0, 0.f);
  sx1 = std::min(sx1, float(d.w - 1)); sy1 = std::min(sy1, float(d.h - 1));

  // margin: depth buffer was rendered from last frame's camera and coarser terrain
  const float limit = minW*0.97f - 5.f;
  if(limit <= 0.f) { return false; }
  const int kN = 6;
  for(int j = 0; j < kN; ++j) {
    int py = int(sy0 + (sy1 - sy0)*float(j)/float(kN - 1));
    for(int i = 0; i < kN; ++i) {
      int px = int(sx0 + (sx1 - sx0)*float(i)/float(kN - 1));
      float z = d.depth[size_t(py)*size_t(d.w) + size_t(px)];
      if(!(z > 0.f && z < limit)) { return false; }  // sky / no data / not nearer: visible
    }
  }
  return true;
}

void ElevationManager::drawDepthDebug(RenderState& _rs, const View& _view)
{
  TextureOptions texoptions;
  texoptions.pixelFormat = PixelFormat::FLOAT;
  texoptions.magFilter = TextureMagFilter::NEAREST;
  texoptions.minFilter = TextureMinFilter::NEAREST;
  Texture tex(texoptions);
  auto& d = m_depthData[0];
  tex.setPixelData(d.w, d.h, 4, (GLubyte*)d.depth.data(), d.depth.size()*4);

  float worldTileSize = MapProjection::EARTH_CIRCUMFERENCE_METERS * std::exp2(-d.zoom);
  float maxTileDistance = worldTileSize * (std::exp2(7.0f) - 1.0f);
  float scale = std::min(maxTileDistance, 2*float(_view.getPosition().z));

  auto vp = _view.getViewport();
  _rs.blending(GL_FALSE);
  Primitives::drawTexture(_rs, tex, {0, 0}, {vp.z, vp.w}, 1/scale);
}

ElevationManager::ElevationManager(std::shared_ptr<RasterSource> src, Style& style) : m_elevationSource(src)
{
  //m_elevationSource->m_keepTextureData = true;  -- now done in Scene::load()

  // default blending mode is opaque, as desired
  m_style = std::make_unique<TerrainStyle>("__terrain");
  m_style->getShaderSource() = style.getShaderSource();
  // direct assignment doesn't work (operator= deleted on std:pair!?)
  for(auto& uniform : style.styleUniforms()) {
    m_style->styleUniforms().emplace_back(uniform.first.name, uniform.second);
  }
  m_style->setID(style.getID());  // use same mesh
  m_style->setRasterType(RasterType::custom);
}

ElevationManager::~ElevationManager()
{
  if(!offscreenWorker) { return; }
  offscreenWorker->enqueue([_style=m_style.release(), _fb=m_frameBuffer.release()](){
    // make sure RenderState doesn't cache handle of deleted framebuffer
    if(m_renderState) { m_renderState->framebuffer(0); }
    delete _style;
    delete _fb;
    //delete _rs;
  });
}

} // namespace Tangram
