#include "map.h"

#include "debug/textDisplay.h"
#include "debug/frameInfo.h"
#include "util/builders.h"
#include "debug/profiler.h"
#include "gl.h"
#include "gl/glError.h"
#include "gl/framebuffer.h"
#include "gl/hardware.h"
#include "gl/primitives.h"
#include "gl/renderState.h"
#include "gl/shaderProgram.h"
#include "labels/labelManager.h"
#include "marker/marker.h"
#include "marker/markerManager.h"
#include "platform.h"
#include "scene/scene.h"
#include "scene/sceneLoader.h"
#include "selection/selectionQuery.h"
#include "style/material.h"
#include "style/style.h"
#include "text/fontContext.h"
#include "tile/tile.h"
#include "tile/tileCache.h"
#include "util/asyncWorker.h"
#include "util/elevationManager.h"
#include "util/fastmap.h"
#include "util/inputHandler.h"
#include "util/ease.h"
#include "util/jobQueue.h"
#include "view/flyTo.h"
#include "view/view.h"

#include <algorithm>
#include <bitset>
#include <chrono>
#include <cstdlib>
#include <cmath>

namespace Tangram {

struct CameraEase {
    struct {
        glm::dvec2 pos;
        float zoom = 0;
        float rotation = 0;
        float tilt = 0;
    } start, end;
};

using CameraAnimator = std::function<uint32_t(float dt)>;

struct ClientTileSource {
    std::shared_ptr<TileSource> tileSource;
    bool added = false;
    bool clear = false;
    bool remove = false;
};


class Map::Impl {
public:
    explicit Impl(Platform& _platform) :
        platform(_platform),
        inputHandler(view),
        scene(std::make_unique<Scene>(_platform)) {}

    void setPixelScale(float _pixelsPerPoint);
    SceneID loadScene(SceneOptions&& _sceneOptions);
    SceneID loadSceneAsync(SceneOptions&& _sceneOptions);
    void syncClientTileSources(bool _firstUpdate);
    bool updateCameraEase(float _dt);
    LngLat getLngLat();
    CameraEase getCameraEase(const CameraPosition& _camera);

    Platform& platform;
    RenderState renderState;
    JobQueue jobQueue;
    View view;

    std::unique_ptr<AsyncWorker> asyncWorker = std::make_unique<AsyncWorker>("Map worker");
    InputHandler inputHandler;

    std::unique_ptr<Ease> ease;

    std::unique_ptr<Scene> scene;

    std::unique_ptr<FrameBuffer> selectionBuffer = std::make_unique<FrameBuffer>(0, 0);

    // Background-adaptive halo (docs/label-placement-plan.md Phase 4): full-resolution
    // (unlike the half-res selectionBuffer above - per-pixel accuracy under label halos
    // matters here) snapshot of the default framebuffer's color contents, blitted right
    // before text styles draw each frame (Scene::render) and sampled by sdf.fs's stroke
    // pass. Texture-backed (colorRenderBuffer=false) like selectionBuffer is once resized,
    // since it must be bound as a sampler2D.
    std::unique_ptr<FrameBuffer> backgroundBuffer = std::make_unique<FrameBuffer>(0, 0);

    // Dynamic resolution (2026-10-11 perf work): while the camera is moving the scene is rasterized
    //  into this offscreen target at `renderScale` of the viewport size, then upscaled to the
    //  default framebuffer; once the camera has been still for `settleSeconds` the scale returns
    //  to 1 and the next frame is full resolution. renderScaleMoving comes from
    //  ASCEND_RENDER_SCALE (1 = off, the core default; the app sets 0.75).
    std::unique_ptr<FrameBuffer> sceneBuffer;
    bool renderScaleEnvRead = false;
    float renderScaleMoving = 1.f;
    float settleSeconds = 0.25f;
    float renderScale = 1.f;
    // adaptive part of the dynamic resolution: the moving-state scale is driven by measured frame
    //  time (see Map::update), between renderScaleMin and 1; <0 = not initialized yet
    float adaptiveScale = -1.f;
    float renderScaleMin = 0.5f;
    float targetFrameSeconds = 0.022f;
    float frameEma = 0.f;
    bool adaptiveEnabled = true;
    double lastMotionTime = -1e9;
    bool hasLastCamera = false;
    glm::mat4 lastViewMatrix{1.f};
    glm::mat4 lastProjMatrix{1.f};
    float lastZoom = 0.f;
    glm::dvec3 lastCamPos{0.0};
    float lastBaseZoom = 0.f, lastYaw = 0.f, lastPitch = 0.f, lastMaxPitch = 0.f;
    float lastW = 0.f, lastH = 0.f;

    bool cacheGlState = false;
    float pickRadius = .5f;
    bool isAnimating = false;

    std::vector<SelectionQuery> selectionQueries;

    SceneReadyCallback onSceneReady = nullptr;
    CameraAnimationCallback cameraAnimationListener = nullptr;

    std::mutex tileSourceMutex;

    std::map<int32_t, ClientTileSource> clientTileSources;

    // TODO MapOption
    Color background{0xffffffff};
};


static std::bitset<10> g_flags = 0;

Map::Map(std::unique_ptr<Platform> _platform) : platform(std::move(_platform)) {
    LOGTOInit();
    impl = std::make_unique<Impl>(*platform);
}

Map::~Map() {
    // Let the platform stop all outstanding tasks:
    // Send cancel to UrlRequests so any thread blocking on a response can join,
    // and discard incoming UrlRequest directly.
    //
    // In any case after shutdown Platform may not call back into Map!
    platform->shutdown();

    // Impl will be automatically destroyed by unique_ptr, but threads owned by AsyncWorker and
    // Scene need to be destroyed before JobQueue stops.
    impl->asyncWorker.reset();
    impl->scene.reset();

    // Make sure other threads are stopped before calling stop()!
    // All jobs will be executed immediately on add() afterwards.
    impl->jobQueue.stop();

    g_flags = 0;
    TextDisplay::Instance().deinit();
    Primitives::deinit();
}


SceneID Map::loadScene(SceneOptions&& _sceneOptions, bool _async) {
    if (_async) {
        return impl->loadSceneAsync(std::move(_sceneOptions));
    } else {
        return impl->loadScene(std::move(_sceneOptions));
    }
}

SceneID Map::Impl::loadScene(SceneOptions&& _sceneOptions) {

    Scene* oldScene = scene.release();
    oldScene->cancelTasks();
    scene = std::make_unique<Scene>(platform, std::move(_sceneOptions), nullptr, oldScene);
    // oldScene may have been loaded async, so dispose on worker thread (after loading complete)
    view.m_elevationManager = nullptr;
    asyncWorker->enqueue([oldScene](){ delete oldScene; });
    scene->load();

    if (onSceneReady) {
        onSceneReady(scene->id, scene->errors());
    }

    return scene->id;
}

SceneID Map::Impl::loadSceneAsync(SceneOptions&& _sceneOptions) {

    Scene* oldScene = scene.release();
    oldScene->cancelTasks();
    view.m_elevationManager = nullptr;

    // Add callback for tile prefetching
    auto prefetchCallback = [&](Scene* _scene) {
        jobQueue.add([&, _scene]() {
            if (_scene == scene.get()) {
                scene->prefetchTiles(view);
                background = scene->backgroundColor(view.getIntegerZoom());
            }});
        platform.requestRender();
    };

    scene = std::make_unique<Scene>(platform, std::move(_sceneOptions), prefetchCallback, oldScene);

    // This async task gets a raw pointer to the new scene and the following task takes ownership of the shared_ptr to
    // the old scene. Tasks in the async queue are executed one at a time in FIFO order, so even if another scene starts
    // to load before this loading task finishes, the current scene won't be freed until after this task finishes.
    asyncWorker->enqueue([this, newScene = scene.get()]() {
        newScene->load();

        if (onSceneReady) {
            onSceneReady(newScene->id, newScene->errors());
        }

        platform.requestRender();
    });

    asyncWorker->enqueue([oldScene](){ delete oldScene; });

    return scene->id;
}

Scene* Map::getScene() {
    return impl->scene.get();
}

View& Map::getView() {
    return impl->view;
}

void Map::updateGlobals(const std::vector<SceneUpdate>& _sceneUpdates, bool _rebuildTiles)
{
  auto& config = const_cast<YAML::Node&>(impl->scene->config());
  SceneLoader::applyUpdates(config, _sceneUpdates);
  impl->scene->globalsGeneration++;
  if (_rebuildTiles) { impl->scene->tileManager()->clearTileSets(); }
  impl->platform.requestRender();
}

void Map::setSceneReadyListener(SceneReadyCallback _onSceneReady) {
    impl->onSceneReady = _onSceneReady;
}

void Map::setCameraAnimationListener(CameraAnimationCallback _cb) {
    impl->cameraAnimationListener = _cb;
}

Platform& Map::getPlatform() {
    return *platform;
}

void Map::resize(int _newWidth, int _newHeight) {
  setViewport(0, 0, _newWidth, _newHeight);
}

void Map::setViewport(int _newX, int _newY, int _newWidth, int _newHeight) {

    //LOGS("resize: %d x %d", _newWidth, _newHeight);
    LOGV("resize: %d x %d", _newWidth, _newHeight);

    impl->view.setViewport(_newX, _newY, _newWidth, _newHeight);
    // force texture (instead of render buffer) so drawDebug() works
    impl->selectionBuffer = std::make_unique<FrameBuffer>(_newWidth/2, _newHeight/2, false);
    // full resolution (see backgroundBuffer's declaration comment above)
    impl->backgroundBuffer = std::make_unique<FrameBuffer>(_newWidth, _newHeight, false);
}

bool Map::isViewDirty() const {
    return impl->view.isDirty();
}

MapState Map::update(float _dt) {

    FrameInfo::beginUpdate();
    FrameInfo::begin("Update");

    impl->jobQueue.runJobs();

    bool isEasing = impl->updateCameraEase(_dt);
    bool isFlinging = impl->inputHandler.update(_dt);

    uint32_t state = 0;
    if (isEasing || isFlinging) {
        state |= MapState::view_changing;
        state |= MapState::is_animating;
    }

    auto& scene = *impl->scene;
    bool wasReady = scene.isReady();

    if (!scene.completeScene(impl->view)) {
        state |= MapState::scene_loading;

    } else {

        // Sync ClientTileSource changes with TileManager
        bool firstUpdate = !wasReady;
        {
            PROFILE_SCOPE("SyncClientTileSources");
            impl->syncClientTileSources(firstUpdate);
        }

        Scene::UpdateState sceneState;
        {
            PROFILE_SCOPE("SceneUpdate");
            sceneState = scene.update(impl->renderState, impl->view, _dt);
        }

        Profiler::counter("activeUrlRequests", double(platform->activeUrlRequests()));

        if (sceneState.animateLabels || sceneState.animateMarkers) {
            state |= MapState::labels_changing;
            state |= MapState::is_animating;
        }
        if (sceneState.tilesLoading) {
            state |= MapState::tiles_loading;
        }
        // budget-deferred tile work (churn/upgrade caps, unpublished background stitches)
        //  needs further updates even when nothing else would request a frame
        if (scene.tileManager()->needsFollowUpFrame()) {
            state |= MapState::is_animating;
        }
    }

    // Dynamic resolution: camera motion drives the scene render scale
    {
        if (!impl->renderScaleEnvRead) {
            // read lazily: the app sets its defaults (setenv) after core static init
            impl->renderScaleEnvRead = true;
            if (const char* e = getenv("ASCEND_RENDER_SCALE")) {
                impl->renderScaleMoving = std::min(1.f, std::max(0.25f, float(atof(e))));
            }
            if (const char* e = getenv("ASCEND_RENDER_SCALE_MIN")) {
                impl->renderScaleMin = std::min(1.f, std::max(0.25f, float(atof(e))));
            }
            if (const char* e = getenv("ASCEND_RENDER_TARGET_MS")) {
                impl->targetFrameSeconds = std::max(8.f, float(atof(e))) * 0.001f;
            }
            if (const char* e = getenv("ASCEND_RENDER_ADAPT")) {
                impl->adaptiveEnabled = atoi(e) != 0;
            }
            if (const char* e = getenv("ASCEND_RENDER_SETTLE_MS")) {
                impl->settleSeconds = std::max(0.f, float(atof(e)) * 0.001f);
            }
        }
        auto& view = impl->view;
        // tolerance, not exact equality: the elevation-aware view re-derives its zoom/eye height
        //  as DEM tiles refine, which must not count as the user moving the camera
        auto matNear = [](const glm::mat4& a, const glm::mat4& b) {
            for (int c = 0; c < 4; c++) {
                for (int r = 0; r < 4; r++) {
                    float x = a[c][r], y = b[c][r];
                    if (std::fabs(x - y) > 1e-6f * std::max(1.f, std::max(std::fabs(x), std::fabs(y)))) { return false; }
                }
            }
            return true;
        };
        // Motion is judged from the USER-CONTROLLED camera parameters only. The view matrices and
        //  getZoom() are elevation-aware (zoom is derived from the terrain depth at screen center),
        //  so they drift whenever terrain data or mesh density changes and must not count as the
        //  camera moving (that fed back into the render scale / mesh boost and never settled).
        const glm::dvec3 camPos = view.getPosition();
        // pitch is clamped to a max-pitch curve evaluated at the (elevation-derived) zoom, so while
        //  it sits on that clamp it wobbles with the zoom without the user touching anything
        const float maxPitchRad = glm::radians(view.getMaxPitch());
        const bool onClampNow = view.getPitch() >= maxPitchRad - 2e-3f;
        const bool onClampPrev = impl->lastPitch >= impl->lastMaxPitch - 2e-3f;
        const bool pitchMoved = std::fabs(view.getPitch() - impl->lastPitch) > 1e-6 &&
            !(onClampNow && onClampPrev);
        impl->lastMaxPitch = maxPitchRad;
        bool moved = !impl->hasLastCamera ||
            camPos.x != impl->lastCamPos.x || camPos.y != impl->lastCamPos.y ||
            std::fabs(view.getBaseZoom() - impl->lastBaseZoom) > 1e-5 ||
            std::fabs(view.getYaw() - impl->lastYaw) > 1e-6 ||
            pitchMoved ||
            float(view.getWidth()) != impl->lastW || float(view.getHeight()) != impl->lastH;
        (void)matNear;
        impl->lastCamPos = camPos;
        impl->lastBaseZoom = view.getBaseZoom();
        impl->lastYaw = view.getYaw();
        impl->lastPitch = view.getPitch();
        impl->lastZoom = view.getZoom();
        impl->lastW = float(view.getWidth());
        impl->lastH = float(view.getHeight());
        const bool first = !impl->hasLastCamera;
        impl->hasLastCamera = true;

        double now = std::chrono::duration<double>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        if (moved && !first) { impl->lastMotionTime = now; }
        bool recentlyMoved = (now - impl->lastMotionTime) < double(impl->settleSeconds);
        // Adaptive dynamic resolution: while the camera moves, steer the scene scale so the
        //  frame time (time between updates) stays near the target - heavy views (big zoomed-out
        //  alpine panoramas: millions of hillshade/rock fragments) drop toward renderScaleMin,
        //  light ones climb back to full resolution. At rest the scale is always 1.
        if (impl->adaptiveScale < 0.f) { impl->adaptiveScale = impl->renderScaleMoving; }
        if (impl->adaptiveEnabled && moved && !first && _dt > 0.f && _dt < 0.5f) {
            impl->frameEma = impl->frameEma <= 0.f ? _dt : 0.8f * impl->frameEma + 0.2f * _dt;
            if (impl->frameEma > impl->targetFrameSeconds * 1.1f) {
                impl->adaptiveScale = std::max(impl->renderScaleMin, impl->adaptiveScale - 0.03f);
            } else if (impl->frameEma < impl->targetFrameSeconds * 0.8f) {
                impl->adaptiveScale = std::min(1.f, impl->adaptiveScale + 0.015f);
            }
        }
        const float movingScale = impl->adaptiveEnabled ? impl->adaptiveScale : impl->renderScaleMoving;
        float target = (movingScale < 0.999f && recentlyMoved) ? movingScale : 1.f;
        impl->renderScale = target;
        // keep frames coming until the settle delay elapses so the full-resolution frame happens
        //  even though no input event will arrive
        if (movingScale < 0.999f && recentlyMoved && !moved) {
            state |= MapState::is_animating;
        }
        // Settle-time mesh refinement: the terrain mesh (and the drape lattice) doubles its
        //  density once the camera has been still a little longer than the full-resolution
        //  switch, so panning stays snappy and a resting view sharpens (util/builders.h)
        {
            const double boostDelay = std::max(double(impl->settleSeconds), 0.4);
            const bool boostSettled = (now - impl->lastMotionTime) >= boostDelay;
            setTerrainMeshBoost(boostSettled ? terrainMeshMaxBoost() : 0);
            if (terrainMeshMaxBoost() > 0 && !boostSettled && !moved) {
                state |= MapState::is_animating;
            }
        }
        Profiler::counter("renderScale", double(impl->renderScale));
    }

    FrameInfo::endUpdate();
    FrameInfo::end("Update");

    return { state };
}

void Map::render() {

    auto& scene = *impl->scene;
    auto& view = impl->view;
    auto& renderState = impl->renderState;

    glm::vec4 viewport = view.getViewport();

    // Delete batch of gl resources
    renderState.flushResourceDeletion();

    // Invalidate render states for new frame
    if (!impl->cacheGlState) {
        renderState.invalidateStates();
    }

    // Cache default framebuffer handle used for rendering
    renderState.cacheDefaultFramebuffer();

    // Do not render while scene is loading
    if (!scene.isReady()) {
        FrameBuffer::apply(renderState, renderState.defaultFrameBuffer(),
                           viewport, impl->background.toColorF());
        return;
    }

    PROFILE_GPU("Render");

    Primitives::setResolution(renderState, view.getWidth(), view.getHeight());
    FrameInfo::beginFrame();

    scene.renderBeginFrame(renderState);

    // Render feature selection pass to offscreen framebuffer
    bool drawSelectionDebug = getDebugFlag(DebugFlags::selection_buffer);
    bool drawDepthDebug = scene.elevationManager() && getDebugFlag(DebugFlags::depth_buffer);
    bool drawSelectionBuffer = !impl->selectionQueries.empty();

    if (drawSelectionBuffer || drawSelectionDebug) {
        PROFILE_SCOPE("RenderSelectionPass");

        impl->selectionBuffer->applyAsRenderTarget(impl->renderState);

        scene.renderSelection(renderState, view,
                              *impl->selectionBuffer,
                              impl->selectionQueries);

        impl->selectionQueries.clear();
    }

    // Get background color for frame based on zoom level, if there are stops
    impl->background = (drawSelectionDebug || drawDepthDebug) ?
            Color(0, 0, 0, 255) : scene.backgroundColor(view.getIntegerZoom());

    // Setup the frame's render target: the default framebuffer, or (dynamic resolution, camera in
    //  motion) a scaled offscreen target that is upscaled onto the default framebuffer below.
    const GLuint realDefaultFB = renderState.defaultFrameBuffer();
    bool scaledTarget = false;
    int scaledW = 0, scaledH = 0;
    if (impl->renderScale < 0.999f && !drawSelectionDebug && !drawDepthDebug) {
        scaledW = std::max(1, int(std::lround(viewport.z * impl->renderScale)));
        scaledH = std::max(1, int(std::lround(viewport.w * impl->renderScale)));
        if (!impl->sceneBuffer || impl->sceneBuffer->getWidth() != scaledW ||
            impl->sceneBuffer->getHeight() != scaledH) {
            // 24-bit depth: the default framebuffer's depth precision is relied on by draped
            //  polygons coinciding with the terrain surface
            impl->sceneBuffer = std::make_unique<FrameBuffer>(scaledW, scaledH, false, GL_RGBA8,
                                                              GL_DEPTH24_STENCIL8, true);
        }
        scaledTarget = impl->sceneBuffer->applyAsRenderTarget(renderState, impl->background.toColorF());
        if (scaledTarget) {
            renderState.setDefaultFramebufferOverride(impl->sceneBuffer->getHandle());
            renderState.setRenderScale(float(scaledW) / viewport.z);
        }
    }
    if (!scaledTarget) {
        FrameBuffer::apply(renderState, renderState.defaultFrameBuffer(),
                           viewport, impl->background.toColorF());
    }

    if (drawSelectionDebug) {
        impl->selectionBuffer->drawDebug(renderState, {viewport.z, viewport.w});
    } else if (drawDepthDebug) {
        scene.elevationManager()->drawDepthDebug(renderState, view);
    } else {
        // Render scene
        PROFILE_SCOPE("RenderDraw");
        bool drawnAnimatedStyle = scene.render(renderState, view, impl->backgroundBuffer.get());

        if (scene.animated() != Scene::animate::no &&
            drawnAnimatedStyle != platform->isContinuousRendering()) {
            platform->setContinuousRendering(drawnAnimatedStyle);
        }

        if (scaledTarget) {
            // upscale the low-resolution scene onto the real default framebuffer. A textured quad,
            //  not glBlitFramebuffer: the window framebuffer is 4x multisampled, and a scaling
            //  blit from a single-sample target into it is GL_INVALID_OPERATION (map never updated).
            renderState.setDefaultFramebufferOverride(0);
            renderState.setRenderScale(1.f);
            FrameBuffer::apply(renderState, realDefaultFB, viewport, impl->background.toColorF());
            impl->sceneBuffer->drawFullscreen(renderState, {view.getWidth(), view.getHeight()});
        }

        scene.labelManager()->drawDebug(renderState, view);
    }

    // never leave the override set (e.g. early-outs above)
    renderState.setDefaultFramebufferOverride(0);
    renderState.setRenderScale(1.f);

    FrameInfo::draw(renderState, view, *this);

    // if almost out of font atlas textures, reset
    if (scene.fontContext()->glyphTextureCount() > FontContext::max_textures - 2) {
        LOGW("Rebuilding tiles due to font atlas exhaustion!");
        scene.tileManager()->clearTileSets();
        scene.markerManager()->clearMeshes();
        scene.fontContext()->releaseFonts();
        platform->requestRender();
    }
}

int Map::getViewportHeight() {
    return impl->view.getHeight();
}

int Map::getViewportWidth() {
    return impl->view.getWidth();
}

float Map::getPixelScale() {
    return impl->view.pixelScale();
}

void Map::captureSnapshot(unsigned int* _data) {
    GL::readPixels(0, 0, impl->view.getWidth(), impl->view.getHeight(), GL_RGBA,
                   GL_UNSIGNED_BYTE, (GLvoid*)_data);
}

CameraPosition Map::getCameraPosition(bool force2D) {
    CameraPosition camera;

    if (force2D && impl->view.m_elevationManager) {
        camera.setLngLat(MapProjection::projectedMetersToLngLat(glm::dvec2(impl->view.getPosition())));
        camera.zoom = impl->view.getBaseZoom();
    } else {
        getPosition(camera.longitude, camera.latitude);
        camera.zoom = getZoom();
    }

    camera.rotation = getRotation();
    camera.tilt = getTilt();

    return camera;
}

void Map::cancelCameraAnimation() {
    impl->inputHandler.cancelFling();

    impl->ease.reset();

    if (impl->cameraAnimationListener) {
        impl->cameraAnimationListener(false);
    }
}

void Map::setCameraPosition(const CameraPosition& _camera) {
    cancelCameraAnimation();

    impl->view.setZoom(_camera.zoom);
    impl->view.setYaw(_camera.rotation);
    impl->view.setPitch(_camera.tilt);

    bool elevOk;
    auto target = MapProjection::lngLatToProjectedMeters(_camera.lngLat());
    auto pos = impl->view.positionToLookAt(target, elevOk);
    impl->view.setPosition(pos);
    if (!elevOk) {
        if (_camera.tilt > M_PI/4) { impl->view.setPitch(M_PI/4); }
        if (impl->view.getBaseZoom() > 14.5f) { impl->view.setBaseZoom(14.5f); }
    }

    impl->platform.requestRender();
}

LngLat Map::Impl::getLngLat() {
    glm::vec2 center(view.getWidth()/2, view.getHeight()/2);
    auto padding = view.getPadding();
    if (!padding.isVisible) {
        center += glm::vec2(padding.right - padding.left, padding.top - padding.bottom)/2.0f;
    }
    return view.screenPositionToLngLat(center.x, center.y);
}

CameraEase Map::Impl::getCameraEase(const CameraPosition& _camera) {

    CameraEase e;

    e.start.zoom = view.getBaseZoom();
    float endBaseZoom = -std::log2(std::exp2(-_camera.zoom) - std::exp2(-view.getZoom()) + std::exp2(-e.start.zoom));
    e.end.zoom = glm::clamp(endBaseZoom, view.getMinZoom(), view.getMaxZoom());

    // Ease over the smallest angular distance needed
    float radiansStart = view.getYaw();
    float radiansDelta = _camera.rotation - radiansStart;
    // trying to get better numerical behavior, esp. final roll == commanded roll; both mod and floor
    //  produce issues w/ very small deltas
    if (radiansDelta < -float(PI)) { radiansDelta += float(TWO_PI); }
    if (radiansDelta > float(PI)) { radiansDelta -= float(TWO_PI); }

    e.start.rotation = radiansStart;
    e.end.rotation = radiansStart + radiansDelta;

    e.start.tilt = view.getPitch();
    e.end.tilt = _camera.tilt;

    LngLat llStart = getLngLat(), llEnd = _camera.lngLat();
    double dLongitude = llEnd.longitude - llStart.longitude;
    if (dLongitude > 180.0) { llEnd.longitude -= 360.0; }
    else if (dLongitude < -180.0) { llEnd.longitude += 360.0; }

    auto target = MapProjection::lngLatToProjectedMeters(llEnd);
    e.start.pos = view.getPosition();
    if (e.end.zoom != e.start.zoom) { view.setBaseZoom(e.end.zoom); }
    if (e.end.rotation != e.start.rotation) { view.setYaw(e.end.rotation); }
    if (e.end.tilt != e.start.tilt) { view.setPitch(e.end.tilt); }
    bool elevOk;
    e.end.pos = view.positionToLookAt(target, elevOk);
    // if elevation not available, set zoom and tilt to make sure we don't end up inside terrain
    if (!elevOk && (e.end.zoom > 14.5f || e.end.tilt > float(M_PI/4))) {
        e.end.tilt = std::min(e.end.tilt, float(M_PI/4));
        e.end.zoom = std::min(e.end.zoom, 14.5f);
        // get correct position for updated zoom and tilt
        view.setBaseZoom(e.end.zoom);
        view.setPitch(e.end.tilt);
        e.end.pos = view.positionToLookAt(target, elevOk);
    }

    return e;
}

void Map::setCameraPositionEased(const CameraPosition& _camera, float _duration, EaseType _e) {
    cancelCameraAnimation();

    CameraEase e = impl->getCameraEase(_camera);

    impl->ease = std::make_unique<Ease>(_duration,
        [=](float t) {
            impl->view.setPosition(ease(e.start.pos.x, e.end.pos.x, t, _e),
                                   ease(e.start.pos.y, e.end.pos.y, t, _e));
            impl->view.setBaseZoom(ease(e.start.zoom, e.end.zoom, t, _e));
            impl->view.setYaw(ease(e.start.rotation, e.end.rotation, t, _e));
            impl->view.setPitch(ease(e.start.tilt, e.end.tilt, t, _e));
        });

    platform->requestRender();
}

void Map::flyTo(const CameraPosition& _camera, float _duration, float _speed) {
    cancelCameraAnimation();

    CameraEase e = impl->getCameraEase(_camera);

    double distance = 0.0;
    glm::dvec3 xyz0(e.start.pos, e.start.zoom), xyz1(e.end.pos, e.end.zoom);
    auto fn = getFlyToFunction(impl->view, xyz0, xyz1, distance);

    auto cb =
        [=](float t) {
            glm::dvec3 pos = fn(t);
            impl->view.setPosition(pos.x, pos.y);
            impl->view.setBaseZoom(pos.z);
            impl->view.setYaw(ease(e.start.rotation, e.end.rotation, t, EaseType::cubic));
            impl->view.setPitch(ease(e.start.tilt, e.end.tilt, t, EaseType::cubic));
            impl->platform.requestRender();
        };

    float duration = _duration >= 0 ? _duration : (distance / (_speed > 0 ? _speed : 1.f));

    impl->ease = std::make_unique<Ease>(duration, cb);

    platform->requestRender();
}

bool Map::Impl::updateCameraEase(float _dt) {
    if (!ease) { return false; }

    ease->update(_dt);

    if (ease->finished()) {
        if (cameraAnimationListener) {
            cameraAnimationListener(true);
        }
        ease.reset();
        return false;
    }
    return true;
}

void Map::updateCameraPosition(const CameraUpdate& _update, float _duration, EaseType _e) {

    CameraPosition camera{};
    if ((_update.set & CameraUpdate::set_camera) != 0) {
        camera = getCameraPosition();
    }
    if ((_update.set & CameraUpdate::set_bounds) != 0) {
        camera = getEnclosingCameraPosition(_update.bounds[0], _update.bounds[1], _update.padding);
    }
    if ((_update.set & CameraUpdate::set_lnglat) != 0) {
        camera.setLngLat(_update.lngLat);
    }
    if ((_update.set & CameraUpdate::set_zoom) != 0) {
        camera.zoom = _update.zoom;
    }
    if ((_update.set & CameraUpdate::set_rotation) != 0) {
        camera.rotation = _update.rotation;
    }
    if ((_update.set & CameraUpdate::set_tilt) != 0) {
        camera.tilt = _update.tilt;
    }
    if ((_update.set & CameraUpdate::set_zoom_by) != 0) {
        camera.zoom += _update.zoomBy;
    }
    if ((_update.set & CameraUpdate::set_rotation_by) != 0) {
        camera.rotation += _update.rotationBy;
    }
    if ((_update.set & CameraUpdate::set_tilt_by) != 0) {
        camera.tilt += _update.tiltBy;
    }

    if (_duration == 0.f) {
        setCameraPosition(camera);
        // The animation listener needs to be called even when the update has no animation duration
        // because this is how our Android MapController passes updates to its MapChangeListener.
        if (impl->cameraAnimationListener) {
            impl->cameraAnimationListener(true);
        }
    } else {
        setCameraPositionEased(camera, _duration, _e);
    }
}

void Map::setPosition(double _lon, double _lat) {
    cancelCameraAnimation();

    bool elevOk;
    glm::dvec2 meters = MapProjection::lngLatToProjectedMeters({_lon, _lat});
    impl->view.setPosition(impl->view.positionToLookAt(meters, elevOk));
    impl->platform.requestRender();
}

void Map::getPosition(double& _lon, double& _lat) {
    LngLat degrees = impl->getLngLat();
    _lon = degrees.longitude;
    _lat = degrees.latitude;
}

void Map::setZoom(float _z) {
    cancelCameraAnimation();

    impl->view.setZoom(_z);
    impl->platform.requestRender();
}

float Map::getZoom() {
    return impl->view.getZoom();
}

void Map::setMinZoom(float _minZoom) {
    impl->view.setMinZoom(_minZoom);
}

float Map::getMinZoom() const {
    return impl->view.getMinZoom();
}

void Map::setMaxZoom(float _maxZoom) {
    impl->view.setMaxZoom(_maxZoom);
}

float Map::getMaxZoom() const {
    return impl->view.getMaxZoom();
}

void Map::setRotation(float _radians) {
    cancelCameraAnimation();

    impl->view.setYaw(_radians);
    impl->platform.requestRender();
}

float Map::getRotation() {
    return impl->view.getYaw();
}

void Map::setTilt(float _radians) {
    cancelCameraAnimation();

    impl->view.setPitch(_radians);
    impl->platform.requestRender();
}

float Map::getTilt() {
    return impl->view.getPitch();
}

void Map::setPadding(const EdgePadding& padding) {
    impl->view.setPadding(padding);
}

EdgePadding Map::getPadding() const {
    return impl->view.getPadding();
}

CameraPosition Map::getEnclosingCameraPosition(LngLat a, LngLat b) const {
    return getEnclosingCameraPosition(a, b, getPadding());
}

CameraPosition Map::getEnclosingCameraPosition(LngLat a, LngLat b, EdgePadding padding) const {
    const View& view = impl->view;

    // Convert the bounding coordinates into Mercator meters.
    ProjectedMeters aMeters = MapProjection::lngLatToProjectedMeters(a);
    ProjectedMeters bMeters = MapProjection::lngLatToProjectedMeters(b);
    ProjectedMeters dMeters = glm::abs(aMeters - bMeters);

    // Calculate the inner size of the view that the bounds must fit within.
    glm::dvec2 innerSize(view.getWidth(), view.getHeight());
    innerSize -= glm::dvec2((padding.left + padding.right), (padding.top + padding.bottom));
    innerSize /= view.pixelScale();

    // Calculate the map scale that fits the bounds into the inner size in each dimension.
    glm::dvec2 metersPerPixel = dMeters / innerSize;

    // Take the value from the larger dimension to calculate the final zoom.
    double maxMetersPerPixel = std::max(metersPerPixel.x, metersPerPixel.y);
    double zoom = MapProjection::zoomAtMetersPerPixel(maxMetersPerPixel);
    double finalZoom = glm::clamp(zoom, (double)getMinZoom(), (double)getMaxZoom());
    double finalMetersPerPixel = MapProjection::metersPerPixelAtZoom(finalZoom);

    // Adjust the center of the final visible region using the padding converted to Mercator meters.
    glm::dvec2 paddingMeters = padding.isVisible ?
          glm::dvec2(padding.right - padding.left, padding.top - padding.bottom) * finalMetersPerPixel :
          glm::dvec2(0,0);
    glm::dvec2 centerMeters = 0.5 * (aMeters + bMeters + paddingMeters);

    LngLat centerLngLat = MapProjection::projectedMetersToLngLat(centerMeters);

    CameraPosition camera;
    camera.zoom = float(finalZoom);
    camera.setLngLat(centerLngLat);
    return camera;
}

bool Map::screenPositionToLngLat(double _x, double _y, double* _lng, double* _lat) {

    float elev = 0;
    bool intersection = false;
    LngLat lngLat = impl->view.screenPositionToLngLat(_x, _y, &elev, &intersection);
    *_lng = lngLat.longitude;
    *_lat = lngLat.latitude;

    return intersection;
}

bool Map::lngLatToScreenPosition(double _lng, double _lat, double* _x, double* _y, bool clipToViewport) {
    bool outsideViewport = false;
    glm::vec2 screenPosition = impl->view.lngLatToScreenPosition(_lng, _lat, outsideViewport, clipToViewport);

    if(_x) *_x = screenPosition.x;
    if(_y) *_y = screenPosition.y;

    return !outsideViewport;
}

void Map::setPixelScale(float _pixelsPerPoint) {
    impl->setPixelScale(_pixelsPerPoint);
}

void Map::Impl::setPixelScale(float _pixelsPerPoint) {

    // If the pixel scale changes we need to re-build all the tiles.
    // This is expensive, so first check whether the new value is different.
    if (_pixelsPerPoint == view.pixelScale()) {
        // Nothing to do!
        return;
    }
    view.setPixelScale(_pixelsPerPoint);
    scene->setPixelScale(_pixelsPerPoint);
}

void Map::setCameraType(int _type) {
    impl->view.setCameraType(static_cast<CameraType>(_type));
    platform->requestRender();
}

int Map::getCameraType() {
    return static_cast<int>(impl->view.cameraType());
}

void Map::addTileSource(std::shared_ptr<TileSource> _source) {
    std::lock_guard<std::mutex> lock(impl->tileSourceMutex);

    auto& tileSources = impl->clientTileSources;
    auto& entry = tileSources[_source->id()];

    entry.tileSource = _source;
    entry.added = true;
}

bool Map::removeTileSource(TileSource& _source) {
    std::lock_guard<std::mutex> lock(impl->tileSourceMutex);

    auto& tileSources = impl->clientTileSources;
    auto it = tileSources.find(_source.id());
    if (it != tileSources.end()) {
        it->second.remove = true;
        return true;
    }
    return false;
}

bool Map::clearTileSource(TileSource& _source, bool _data, bool _tiles) {
    std::lock_guard<std::mutex> lock(impl->tileSourceMutex);

    if (_data) { _source.clearData(); }
    if (!_tiles) { return true; }

    auto& tileSources = impl->clientTileSources;
    auto it = tileSources.find(_source.id());
    if (it != tileSources.end()) {
        it->second.clear = true;
        return true;
    }
    return false;
}

void Map::Impl::syncClientTileSources(bool _firstUpdate) {
    std::lock_guard<std::mutex> lock(tileSourceMutex);

    auto& tileManager = *scene->tileManager();
    for (auto it = clientTileSources.begin();
         it != clientTileSources.end(); ) {
        auto& ts = it->second;
        if (ts.remove) {
            tileManager.removeClientTileSource(it->first);
            it = clientTileSources.erase(it);
            continue;
        }
        if (ts.added || _firstUpdate) {
            ts.added = false;
            tileManager.addClientTileSource(ts.tileSource);
        }
        if (ts.clear) {
            ts.clear = false;
            tileManager.clearTileSet(it->first);
        }
        ++it;
    }
}

MarkerID Map::markerAdd() {
    return impl->scene->markerManager()->add();
}

bool Map::markerRemove(MarkerID _marker) {
    bool success = impl->scene->markerManager()->remove(_marker);
    platform->requestRender();
    return success;
}

bool Map::markerSetPoint(MarkerID _marker, LngLat _lngLat) {
    bool success = impl->scene->markerManager()->setPoint(_marker, _lngLat);
    platform->requestRender();
    return success;
}

bool Map::markerSetPointEased(MarkerID _marker, LngLat _lngLat, float _duration, EaseType ease) {
    bool success = impl->scene->markerManager()->setPointEased(_marker, _lngLat, _duration, ease);
    platform->requestRender();
    return success;
}

bool Map::markerSetPolyline(MarkerID _marker, LngLat* _coordinates, int _count) {
    bool success = impl->scene->markerManager()->setPolyline(_marker, _coordinates, _count);
    platform->requestRender();
    return success;
}

bool Map::markerSetPolygon(MarkerID _marker, LngLat* _coordinates, int* _counts, int _rings) {
    bool success = impl->scene->markerManager()->setPolygon(_marker, _coordinates, _counts, _rings);
    platform->requestRender();
    return success;
}

bool Map::markerSetProperties(MarkerID _marker, Properties&& _properties) {
    bool success = impl->scene->markerManager()->setProperties(_marker, std::move(_properties));
    platform->requestRender();
    return success;
}

bool Map::markerSetAlternate(MarkerID _marker, MarkerID _alt) {
    bool success = impl->scene->markerManager()->setAlternate(_marker, _alt);
    platform->requestRender();
    return success;
}

bool Map::markerSetStylingFromString(MarkerID _marker, const char* _styling) {
    bool success = impl->scene->markerManager()->setStylingFromString(_marker, _styling);
    platform->requestRender();
    return success;
}

bool Map::markerSetStylingFromPath(MarkerID _marker, const char* _path) {
    bool success = impl->scene->markerManager()->setStylingFromPath(_marker, _path);
    platform->requestRender();
    return success;
}

bool Map::markerSetBitmap(MarkerID _marker, int _width, int _height, const unsigned int* _data, float _density) {
    bool success = impl->scene->markerManager()->setBitmap(_marker, _width, _height, _density, _data);
    platform->requestRender();
    return success;
}

bool Map::markerSetVisible(MarkerID _marker, bool _visible) {
    bool success = impl->scene->markerManager()->setVisible(_marker, _visible);
    platform->requestRender();
    return success;
}

bool Map::markerSetDrawOrder(MarkerID _marker, int _drawOrder) {
    bool success = impl->scene->markerManager()->setDrawOrder(_marker, _drawOrder);
    platform->requestRender();
    return success;
}

void Map::markerRemoveAll() {
    impl->scene->markerManager()->removeAll();
    platform->requestRender();
}

void Map::setPickRadius(float _radius) {
    impl->pickRadius = _radius;
}

void Map::pickFeatureAt(float _x, float _y, FeaturePickCallback _onFeaturePickCallback) {
    impl->selectionQueries.push_back({{_x, _y}, impl->pickRadius, _onFeaturePickCallback});
    platform->requestRender();
}

void Map::pickLabelAt(float _x, float _y, LabelPickCallback _onLabelPickCallback) {
    impl->selectionQueries.push_back({{_x, _y}, impl->pickRadius, _onLabelPickCallback});
    platform->requestRender();
}

void Map::pickMarkerAt(float _x, float _y, MarkerPickCallback _onMarkerPickCallback) {
    impl->selectionQueries.push_back({{_x, _y}, impl->pickRadius, _onMarkerPickCallback});
    platform->requestRender();
}

void Map::handleTapGesture(float _posX, float _posY) {
    cancelCameraAnimation();
    impl->inputHandler.handleTapGesture(_posX, _posY);
    impl->platform.requestRender();
}

void Map::handleDoubleTapGesture(float _posX, float _posY) {
    cancelCameraAnimation();
    // We want tapped map position to remain at same screen position throughout zoom; using a camera ease
    //  gives correct final state but causes tapped position to wobble during zoom.
    //impl->inputHandler.handleDoubleTapGesture(_posX, _posY);  -- doesn't do any animation!
    float startZoom = impl->view.getZoom();
    impl->ease = std::make_unique<Ease>(0.35f, [=](float t) {
        float z0 = impl->view.getZoom();
        float z1 = ease(startZoom, startZoom+1, t, EaseType::linear);
        impl->inputHandler.handlePinchGesture(_posX, _posY, std::exp2(z1 - z0), 0);
    });
    impl->platform.requestRender();
}

void Map::handlePanGesture(float _startX, float _startY, float _endX, float _endY) {
    cancelCameraAnimation();
    impl->inputHandler.handlePanGesture(_startX, _startY, _endX, _endY);
    impl->platform.requestRender();
}

void Map::handleFlingGesture(float _posX, float _posY, float _velocityX, float _velocityY) {
    cancelCameraAnimation();
    impl->inputHandler.handleFlingGesture(_posX, _posY, _velocityX, _velocityY);
    impl->platform.requestRender();
}

void Map::handlePinchGesture(float _posX, float _posY, float _scale, float _velocity) {
    cancelCameraAnimation();
    impl->inputHandler.handlePinchGesture(_posX, _posY, _scale, _velocity);
    impl->platform.requestRender();
}

void Map::handleRotateGesture(float _posX, float _posY, float _radians) {
    cancelCameraAnimation();
    impl->inputHandler.handleRotateGesture(_posX, _posY, _radians);
    impl->platform.requestRender();
}

void Map::handleShoveGesture(float _distance) {
    cancelCameraAnimation();
    impl->inputHandler.handleShoveGesture(_distance);
    impl->platform.requestRender();
}

void Map::setupGL() {

    LOG("setup GL");

    impl->renderState.invalidate();

    //impl->scene->tileManager()->clearTileSets();
    impl->scene->markerManager()->rebuildAll();

    if (impl->selectionBuffer->valid()) {
        impl->selectionBuffer = std::make_unique<FrameBuffer>(impl->selectionBuffer->getWidth(),
                                                              impl->selectionBuffer->getHeight(), false);
    }

    if (impl->backgroundBuffer->valid()) {
        impl->backgroundBuffer = std::make_unique<FrameBuffer>(impl->backgroundBuffer->getWidth(),
                                                                impl->backgroundBuffer->getHeight(), false);
    }

    // Load GL extensions and capabilities
    Hardware::loadCapabilities();
    Hardware::loadExtensions();
    // Hardware::printAvailableExtensions();
}

void Map::useCachedGlState(bool _useCache) {
    impl->cacheGlState = _useCache;
}

void Map::runAsyncTask(std::function<void()> _task) {
    if (impl->asyncWorker) {
        impl->asyncWorker->enqueue(std::move(_task));
    }
}

void Map::onMemoryWarning() {
    if(!impl->scene) return;
    impl->scene->tileManager()->clearTileSets(true);
    if (impl->scene->fontContext()) {
        impl->scene->fontContext()->releaseFonts();
    }
}

void Map::setDefaultBackgroundColor(float r, float g, float b) {
    impl->renderState.defaultOpaqueClearColor(r, g, b);
}

void setDebugFlag(DebugFlags _flag, bool _on) {

    g_flags.set(_flag, _on);
    // m_view.setZoom(m_view.getZoom()); // Force the view to refresh

}

bool getDebugFlag(DebugFlags _flag) {

    return g_flags.test(_flag);

}

void toggleDebugFlag(DebugFlags _flag) {

    g_flags.flip(_flag);
    // m_view.setZoom(m_view.getZoom()); // Force the view to refresh

    // Rebuild tiles for debug modes that needs it
    // if (_flag == DebugFlags::proxy_colors
    //  || _flag == DebugFlags::draw_all_labels
    //  || _flag == DebugFlags::tile_bounds
    //  || _flag == DebugFlags::tile_infos) {
    //     if (m_tileManager) {
    //         m_tileManager->clearTileSets();
    //     }
    // }
}

}
