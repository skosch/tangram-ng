#include "labels/labelManager.h"

#include "debug/profiler.h"
#include "data/tileSource.h"
#include "gl/primitives.h"
#include "gl/shaderProgram.h"
#include "labels/curvedLabel.h"
#include "labels/labelSet.h"
#include "labels/obbBuffer.h"
#include "labels/textLabel.h"
#include "map.h"
#include "marker/marker.h"
#include "platform.h"
#include "scene/scene.h"
#include "style/pointStyle.h"
#include "style/style.h"
#include "style/textStyle.h"
#include "tile/tile.h"
#include "tile/tileCache.h"
#include "tile/tileManager.h"
#include "view/view.h"
#include "util/elevationManager.h"

#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtx/rotate_vector.hpp"
#include "glm/gtx/norm.hpp"

#include <cassert>
#include <cfloat>
#include <chrono>

namespace Tangram {

namespace {
// Tune by eye: wall-clock milliseconds per frame that may be spent inside
// Label::refineAnchor() calls -- see LabelManager::m_anchorRefineTimeUsedMs's doc comment
// (labelManager.h) for why this replaced a fixed per-frame label COUNT (round 9). A budget
// in real time, rather than label count, behaves the same in both Debug and Release
// without needing two separately-tuned constants.
constexpr double kAnchorRefineBudgetMs = 4.0;

// Round 16: same idiom as kAnchorRefineBudgetMs above, for Label::refinePriority() - see
// LabelManager::m_priorityRefineTimeUsedMs's doc comment (labelManager.h) for why this was
// missing entirely until now. A single sampleTextureShading() call is cheap once its
// mosaic's pyramid is cached, but a fresh pyramid build is not, and refinePriority() (unlike
// refineAnchor()) was never gated at all - one call per not-yet-refined label, every frame,
// no ceiling.
constexpr double kPriorityRefineBudgetMs = 4.0;

// docs/label-placement-v2-spec.md's "Deterministic-but-not-jittery collision resolution"
// (Option A) -- see LabelManager::priorityComparator()'s own comment for the full
// reasoning. Tune by eye: frames a label must stay continuously unoccluded before its
// history is allowed to outweigh a competitor's tier/score. Small enough to settle
// quickly (a few animation frames), large enough that a single lucky frame during initial
// tile load can't lock in a permanent winner.
constexpr int kStableFramesThreshold = 5;
}

LabelManager::LabelManager()
    : m_needUpdate(false),
      m_lastZoom(0.0f) {}

LabelManager::~LabelManager() {}

void LabelManager::processLabelUpdate(const ViewState& _viewState, const LabelSet* _labelSet, Style* _style,
                                const Tile* _tile, const Marker* _marker, ElevationManager* _elevManager,
                                float _dt, bool _onlyRender, float _viewportMaxElev, bool _haveViewportMaxElev) {

    assert(_tile || _marker);
    // TODO appropriate buffer to filter out-of-screen labels
    float border = 256.0f;
    AABB extendedBounds(-border, -border,
                        _viewState.viewportSize.x + border,
                        _viewState.viewportSize.y + border);

    AABB screenBounds(0, 0,
                      _viewState.viewportSize.x,
                      _viewState.viewportSize.y);

    bool drawAllLabels = Tangram::getDebugFlag(DebugFlags::draw_all_labels);

    // use blendOrder == INT_MAX to indicate debug style
    glm::mat4 elevMVP;
    bool recalcPos = false, behindCam = false;
    bool useElev = _elevManager && _style->blendOrder() < INT_MAX;
    bool setElev = useElev && (_marker || _elevManager->hasTile(_tile->getID()));
    if (useElev) {
        if (_tile) {
            _elevManager->setMinZoom(!_tile->rasters().empty() ?
                                     _tile->rasters().back().tileID.z : std::min(int(_tile->getID().z), 14));
        }
        // if depth data is older than last frame, we have to recalculate label positions
        auto& depthData = _elevManager->getDepthData();
        if (depthData.viewPos != m_lastViewPos || depthData.viewProj != m_lastViewProj) {
            glm::mat4 modelMat = _tile ? _tile->getModelMatrix() : _marker->modelMatrix();
            glm::dvec2 origin = _tile ? _tile->getOrigin() : _marker->origin();
            double offset = _tile ? _tile->getScale()/2 : 0;
            auto relorigin = MapProjection::wrapProjectedMeters((origin + offset) - depthData.viewPos) - offset;
            modelMat[3][0] = float(relorigin.x);
            modelMat[3][1] = float(relorigin.y);
            elevMVP = depthData.viewProj * modelMat;
            recalcPos = true;
        }
    }

    for (auto& label : _labelSet->getLabels()) {
        if (!drawAllLabels && (label->state() == Label::State::dead) ) {
            continue;
        }

        // docs/label-placement-plan.md Phase 2: keep a linked text label's priority in sync
        // with its icon's (see Label::syncRelativePriority()'s doc comment for why this can't
        // just be the one-time copy setRelative() already does). Cheap no-op for any label that
        // isn't tracking a relative's priority; independent of elevation/terrain being enabled
        // at all.
        label->syncRelativePriority();

        if (setElev && !label->m_elevationSet) {
            if (_tile) {
                label->m_elevationSet = label->setElevation(*_elevManager, _tile->getOrigin(), _tile->getScale());
            } else if (_marker) {
                double scale = _marker->extent();  // see Marker::setMesh()
                if (scale <= 0) { scale = MapProjection::metersPerTileAtZoom(_marker->builtZoomLevel()); }
                label->m_elevationSet = label->setElevation(*_elevManager, _marker->origin(), scale);
            }
        }

        // docs/label-placement-plan.md Phase 2: refine peak icon priority using the CPU
        // texture-shading sampler, main-thread only (see textureShading.h's threading note --
        // this can't safely happen at tile-build time on a TileWorker thread). No-op unless the
        // label opted in (Label::Options::refinePriorityWithTextureShading); retried every
        // frame, like m_elevationSet/setElevation above, until the elevation mosaic becomes
        // available at this label's position.
        //
        // Round 16: budget-gated the same way refineAnchor() below already is (see
        // m_priorityRefineTimeUsedMs's doc comment, labelManager.h) -- this was previously
        // unbounded, one call per not-yet-refined label every frame with no ceiling, despite
        // calling into the same sampleTextureShading() whose per-mosaic pyramid cache can miss
        // (and pay a real rebuild) on the first peak of every newly-streamed-in tile.
        if (_elevManager && !label->m_prominenceRefined) {
            if (m_priorityRefineTimeUsedMs < kPriorityRefineBudgetMs) {
                auto t0 = std::chrono::steady_clock::now();
                if (_tile) {
                    label->m_prominenceRefined =
                        label->refinePriority(*_elevManager, _tile->getOrigin(), _tile->getScale(),
                                              _viewportMaxElev, _haveViewportMaxElev);
                } else if (_marker) {
                    double scale = _marker->extent();
                    if (scale <= 0) { scale = MapProjection::metersPerTileAtZoom(_marker->builtZoomLevel()); }
                    label->m_prominenceRefined = label->refinePriority(*_elevManager, _marker->origin(), scale,
                                                                        _viewportMaxElev, _haveViewportMaxElev);
                }
                auto t1 = std::chrono::steady_clock::now();
                m_priorityRefineTimeUsedMs += std::chrono::duration<double, std::milli>(t1 - t0).count();
            } else {
                // Budget exhausted this frame - same "keep rendering until it gets its
                // turn" reasoning as refineAnchor()'s budget-exhausted branch below.
                m_needUpdate = true;
            }
        }

        // docs/label-placement-plan.md Task 4: steer this label's anchor away from both a
        // strong ridge/canyon texture-shading response and nearby vector clutter (via the
        // tile's AnchorOccupancyGrid), main-thread only -- same retry pattern as
        // refinePriority() above. No-op unless the label opted in
        // (Label::Options::refineAnchorWithTextureShading).
        //
        // Round 9 / round 15: gated on a measured wall-clock budget (see
        // m_anchorRefineTimeUsedMs's doc comment, labelManager.h) -- this is the expensive
        // one (~45 texture-shading samples per label), so once a frame's budget is spent,
        // remaining labels simply wait for a later frame rather than all paying the cost
        // in the same frame.
        if (_elevManager && !label->m_anchorRefined) {
            if (m_anchorRefineTimeUsedMs < kAnchorRefineBudgetMs) {
                auto t0 = std::chrono::steady_clock::now();
                if (_tile) {
                    label->m_anchorRefined =
                        label->refineAnchor(*_elevManager, _tile->getOrigin(), _tile->getScale(),
                                            _tile->anchorOccupancyGrid().get());
                } else if (_marker) {
                    double scale = _marker->extent();
                    if (scale <= 0) { scale = MapProjection::metersPerTileAtZoom(_marker->builtZoomLevel()); }
                    label->m_anchorRefined = label->refineAnchor(*_elevManager, _marker->origin(), scale, nullptr);
                }
                auto t1 = std::chrono::steady_clock::now();
                m_anchorRefineTimeUsedMs += std::chrono::duration<double, std::milli>(t1 - t0).count();
            } else {
                // Budget exhausted this frame -- make sure the app keeps rendering frames
                // until this label (and any others still waiting) gets its turn, exactly like
                // m_needUpdate already does for in-progress fade transitions (see
                // Scene::update()'s use of LabelManager::needUpdate()). Without this, a session
                // that stops moving the camera right as the budget runs out could leave labels
                // permanently stuck unrefined.
                m_needUpdate = true;
            }
        }

        // terrain depth is from previous frame, so we must compare label position before Label::update()
        glm::vec4 screenCoord = label->screenCoord();   //glm::vec4(0);
        if (recalcPos) {
            auto worldPos = glm::vec4(label->modelCenter(), 1.f);
            screenCoord = worldToScreenSpace(elevMVP, worldPos, _viewState.viewportSize, behindCam);
        }

        Range transformRange;
        ScreenTransform transform { m_transforms, transformRange };

        // Use extendedBounds when labels take part in collision detection.
        auto bounds = (_onlyRender || !label->canOcclude()) ? screenBounds : extendedBounds;

        const glm::mat4& mvp = _tile ? _tile->mvp() : _marker->modelViewProjectionMatrix();
        if (!label->update(mvp, _viewState, &bounds, transform)) {
            continue;
        }

        if (useElev) {
            // clamp to viewport to get reasonable behavior for labels entering and leaving screen
            glm::vec2 xyclamped = glm::clamp(glm::vec2(screenCoord), {0,0}, screenBounds.max - 1.0f);
            float labelz = 1/screenCoord.w;
            float zdn = _elevManager->getDepth({xyclamped.x, xyclamped.y+2});
            float zup = _elevManager->getDepth({xyclamped.x, xyclamped.y-2});

            // need some hysteresis to reduce label flashing; hysteresis on thresh does not work well
            bool wasBehind = label->state() == Label::State::out_of_screen;
            float terrainz = wasBehind ? zdn : zup;
            float thresh = std::max(200.f, 0.002f*zup);  //std::max(200.0f, std::abs(zup - z00)) * (wasBehind ? 1 : 2);

            if (terrainz != 0 && screenCoord.w != 0 && labelz > terrainz + thresh) {
                label->enterState(Label::State::out_of_screen);
                continue;
            } else if (wasBehind) {
                label->enterState(Label::State::sleep);
            }
            //isBehindTerrain = coord.z > terrainz + 0.005f;  -- 0.005 NDC ~ 1000 - 1500m
            //LOGW("'%s' - Camera: label %f, terrain %f; delta: %f", label->debugTag.c_str(), labelz, terrainz, labelz - terrainz);
        }

        bool isProxy = _tile && _tile->isProxy();
        if (_onlyRender) {
            if (label->occludedLastFrame()) { label->occlude(); }

            if (label->visibleState() || !label->canOcclude()) {
                m_needUpdate |= label->evalState(_dt);
                label->addVerticesToMesh(transform, _viewState.viewportSize);
            }
        } else if (label->canOcclude()) {
            m_labels.emplace_back(label.get(), _style, _tile, _marker, isProxy, transformRange);
        } else {
            m_needUpdate |= label->evalState(_dt);
            label->addVerticesToMesh(transform, _viewState.viewportSize);
        }
        if (label->selectionColor()) {
            m_selectionLabels.emplace_back(label.get(), _style, _tile, _marker, isProxy, transformRange);
        }
    }

    if (useElev && _tile) { _elevManager->setMinZoom(0); }
}

std::pair<Label*, const Tile*> LabelManager::getLabel(uint32_t _selectionColor) const {

    for (auto& entry : m_selectionLabels) {

        if (entry.label->visibleState() &&
            entry.label->selectionColor() == _selectionColor) {

            return { entry.label, entry.tile };
        }
    }
    return {nullptr, nullptr};
}

static Style* getStyleById(const Scene& _scene, uint32_t id) {
    for (const auto& style : _scene.styles()) {
        if (style->getID() == id) { return style.get(); }
    }
    return nullptr;
}

void LabelManager::updateLabels(const ViewState& _viewState, float _dt, const Scene& _scene,
                          const std::vector<std::shared_ptr<Tile>>& _tiles,
                          const std::vector<std::unique_ptr<Marker>>& _markers,
                          bool _onlyRender) {

    if (!_onlyRender) { m_labels.clear(); }

    m_selectionLabels.clear();

    m_needUpdate = false;
    m_anchorRefineTimeUsedMs = 0.0;
    m_priorityRefineTimeUsedMs = 0.0;

    auto* elevManager = _scene.elevationManager();

    // docs/label-placement-plan.md round 7 (corrected): the highest elevation found anywhere
    // among the tiles already resident for THIS frame's rendering -- i.e. the actual current
    // viewport, no speculative coarser tile involved (see Label::refinePriority()'s doc
    // comment for why the original ancestor-climb version never actually found data). Computed
    // once here, not once per peak; ElevationManager::getMinMaxElev(tileId, 0) with ancestors=0
    // only ever checks the tile's own already-resident texture, so this is as cheap as it looks
    // -- no new fetch, no climbing. A tile with no elevation raster attached (flat style,
    // terrain toggled off, or not yet loaded) is simply skipped, matching
    // getMinMaxElev()'s own documented "nothing resident" sentinel {0, 9000}.
    float viewportMaxElev = -FLT_MAX;
    if (elevManager) {
        for (const auto& tile : _tiles) {
            glm::vec2 minMax = elevManager->getMinMaxElev(tile->getID(), 0);
            if (minMax == glm::vec2(0.f, 9000.f)) { continue; }  // not resident for this tile
            viewportMaxElev = std::max(viewportMaxElev, minMax.y);
        }
    }
    bool haveViewportMaxElev = viewportMaxElev > -FLT_MAX;

    const auto& _styles = _scene.styles();
    for (const auto& tile : _tiles) {
        for (const auto& style : _styles) {
            const auto& mesh = tile->getMesh(*style);
            auto labels = dynamic_cast<const LabelSet*>(mesh.get());
            if (!labels) { continue; }

            processLabelUpdate(_viewState, labels, style.get(), tile.get(), nullptr,
                               elevManager, _dt, _onlyRender, viewportMaxElev, haveViewportMaxElev);
        }
    }

    // Marker::builtZoomLevel() does not account for 3D LOD; more correct approach would be to figure out
    //  which tile each marker sits on (needs one pass over _tiles per marker)
    if (elevManager && !_tiles.empty()) {
        elevManager->setMinZoom(std::min(int(_tiles.back()->getID().z), 14));
    }

    for (const auto& marker : _markers) {

        if (!marker->isVisible() || !marker->mesh()) { continue; }

        if (marker->isAltMarker) {
            if (!_onlyRender) { marker->altMeshAdded = false; }
            if (!marker->altMeshAdded) { continue; }
        }

        Style* style = getStyleById(_scene, marker->styleId());
        const auto& mesh = marker->mesh();
        auto labels = dynamic_cast<const LabelSet*>(mesh);
        if (!style || !labels) { continue; }

        processLabelUpdate(_viewState, labels, style, nullptr, marker.get(),
                           elevManager, _dt, _onlyRender, viewportMaxElev, haveViewportMaxElev);
    }

    if (elevManager) { elevManager->setMinZoom(0); }
#if 0  //def DEBUG
    if (_scene.elevationManager()) {
        auto& depthData = _scene.elevationManager()->getDepthData();
        if (depthData.viewPos != m_lastViewPos || depthData.viewProj != m_lastViewProj) {
            LOGD("updateLabels frame %d: terrain depth data stale", Scene::frameCount);
        }
    }
#endif
}

void LabelManager::skipTransitions(const std::vector<const Style*>& _styles, Tile& _tile, Tile& _proxy) const {

    for (const auto& style : _styles) {

        auto* mesh0 = dynamic_cast<const LabelSet*>(_tile.getMesh(*style).get());
        if (!mesh0) { continue; }

        auto* mesh1 = dynamic_cast<const LabelSet*>(_proxy.getMesh(*style).get());
        if (!mesh1) { continue; }

        for (auto& l0 : mesh0->getLabels()) {
            if (!l0->canOcclude()) { continue; }
            if (l0->state() != Label::State::none) { continue; }

            for (auto& l1 : mesh1->getLabels()) {
                if (!l1->visibleState()) { continue; }
                if (!l1->canOcclude()) { continue;}

                // Using repeat group to also handle labels with dynamic style properties
                if (l0->options().repeatGroup != l1->options().repeatGroup) { continue; }
                // if (l0->hash() != l1->hash()) { continue; }

                float d2 = glm::distance2(l0->screenCenter(), l1->screenCenter());

                // The new label lies within the circle defined by the bbox of l0
                if (sqrt(d2) < std::max(l0->dimension().x, l0->dimension().y)) {
                    l0->skipTransitions();
                }
            }
        }
    }
}

std::shared_ptr<Tile> findProxy(int32_t _sourceID, const TileID& _proxyID,
                                const std::vector<std::shared_ptr<Tile>>& _tiles,
                                TileCache& _cache) {

    auto proxy = _cache.contains(_sourceID, _proxyID);
    if (proxy) { return proxy; }

    for (auto& tile : _tiles) {
        if (tile->getID() == _proxyID && tile->sourceID() == _sourceID) {
            return tile;
        }
    }
    return nullptr;
}

void LabelManager::skipTransitions(const Scene& _scene, const std::vector<std::shared_ptr<Tile>>& _tiles,
                                   TileManager& _tileManager, float _currentZoom) const {

    std::vector<const Style*> styles;

    for (const auto& style : _scene.styles()) {
        if (dynamic_cast<const TextStyle*>(style.get()) ||
            dynamic_cast<const PointStyle*>(style.get())) {
            styles.push_back(style.get());
        }
    }

    for (const auto& tile : _tiles) {
        TileID tileID = tile->getID();
        std::shared_ptr<Tile> proxy;

        // TileManager has all geometry (e.g. labels) generating sources
        auto source = _tileManager.getTileSource(tile->sourceID());
        if (!source) { assert(source); continue; }  // should never happen

        if (m_lastZoom < _currentZoom) {
            // zooming in, add the one cached parent tile
            proxy = findProxy(tile->sourceID(), tileID.getParent(source->zoomBias()), _tiles,
                              *_tileManager.getTileCache());
            if (proxy) { skipTransitions(styles, *tile, *proxy); }
        } else {
            // zooming out, add the 4 cached children tiles
            for (int i = 0; i < 4; i++) {
                proxy = findProxy(tile->sourceID(), tileID.getChild(i, source->maxZoom()), _tiles,
                                  *_tileManager.getTileCache());
                if (proxy) { skipTransitions(styles, *tile, *proxy); }
            }
        }
    }
}

bool LabelManager::priorityComparator(const LabelEntry& _a, const LabelEntry& _b) {
    if (_a.proxy != _b.proxy) {
        return _b.proxy;  // non-proxy over proxy
    }
    // docs/label-placement-v2-spec.md's "nearbyint tier-bucketing risk": priority is
    // encoded as tier + fraction (e.g. peak = [14.0, 14.9]), and nearbyint() rounds to the
    // nearest integer using round-half-to-even -- an unnamed peak's band ([14.5, 14.9])
    // could round UP into 15 (saddle's tier) for any fraction above the .5 tie point,
    // silently misclassifying it into the wrong coarse category. floor() has no such
    // edge case: every value in a tier's [N.0, N.9] band floors to exactly N, matching how
    // Label::refinePriority() already extracts a tier floor for its own half-tier bands
    // (`std::floor(m_options.priority * 2.f) / 2.f`) -- this is the direct fix, not a
    // rewrite of the whole priority/tier representation, which the risk write-up also
    // offers as an equally valid alternative to a full data-model split.
    float aprio = std::floor(_a.priority);
    float bprio = std::floor(_b.priority);
    if (aprio != bprio) {
        return aprio < bprio;
    }
    if (_a.tile && _b.tile) {
        if (_a.tile->getID().z != _b.tile->getID().z) {
            return _a.tile->getID().z > _b.tile->getID().z;  // higher zoom over lower zoom
        }
    } else if (_a.tile || _b.tile) {
        return (bool)_a.tile;  // tile labels over marker labels (maybe reverse this?)
    }

    auto l1 = _a.label;
    auto l2 = _b.label;

    // give priority to labels closer to camera
    float z1 = l1->screenCoord().z, z2 = l2->screenCoord().z;
    if (z1 != z2) {
        return z1 < z2;
    }

    // for tile labels, we want fractional priority over history
    if (_a.tile) {
      if (_a.priority != _b.priority) { return _a.priority < _b.priority; }  // already know int parts equal
      if (l1->isChild() != l2->isChild()) { return l2->isChild(); }  // non-child over child
    }

    // docs/label-placement-v2-spec.md's "Deterministic-but-not-jittery collision
    // resolution", Option A. This USED to be a raw `occludedLastFrame()` compare -- "did
    // I win the single previous frame" -- which meant the final settled winner between
    // two near-tied competitors depended on whichever happened to render first (tile
    // arrival order, network timing), and then perpetuated itself indefinitely once it
    // won even one frame by chance: a real, confirmed, non-deterministic-placement risk.
    // Requiring a label to have stayed unoccluded for kStableFramesThreshold consecutive
    // frames before its history counts fixes this: for as long as NEITHER competitor is
    // yet stable (e.g. every frame during initial load), this check is a no-op and the
    // tiebreak falls through to the fully deterministic (tier, score, hash, id) chain
    // below, which gives the SAME answer every single frame given fixed inputs -- so
    // whichever candidate is genuinely better by that chain wins from frame one and
    // accumulates stability consistently, not incidentally. Once a label DOES reach
    // stability, this still protects it from flickering out if a competitor's score
    // later improves (e.g. texture-shading refinement completing) -- the actual
    // anti-flicker property this check exists for, now decoupled from incidental history.
    bool l1Stable = l1->stableFrames() >= kStableFramesThreshold;
    bool l2Stable = l2->stableFrames() >= kStableFramesThreshold;
    if (l1Stable != l2Stable) {
        return l1Stable;  // prefer the already-settled label
    }

    // marker labels can be very dense since they are not prefiltered by LabelCollider, so to avoid flashing
    //  we want history over fractional priority
    if (!_a.tile) {
        if (_a.priority != _b.priority) { return _a.priority < _b.priority; }  // already know int parts equal
        if (l1->isChild() != l2->isChild()) { return l2->isChild(); }  // non-child over child
    }

    // This prefers labels within screen over out_of_screen. Important for repeat groups!
    // Must be after sort on child because parent could be out of screen when child is not
    if (l1->visibleState() != l2->visibleState()) {
        return l1->visibleState();
    }

    if (l1->options().repeatGroup != l2->options().repeatGroup) {
        return l1->options().repeatGroup < l2->options().repeatGroup;
    }

    if (l1->type() == l2->type()) {
        return l1->candidatePriority() < l2->candidatePriority();
    }

    if (l1->hash() != l2->hash()) {
        return l1->hash() < l2->hash();
    }

    // Deterministic creation-order tiebreak (was `l1 < l2`, comparing raw addresses -- heap
    // layout/allocation order isn't consistent across builds, e.g. Debug's DEBUG-only debugTag
    // member changes sizeof(Label) outright, so which same-priority label won here -- and then
    // kept winning via the occludedLastFrame hysteresis above -- was silently build-dependent.
    // See Label::id()'s doc comment.
    return l1->id() < l2->id();
}

bool LabelManager::zOrderComparator(const LabelEntry& _a, const LabelEntry& _b) {

    if (_a.style != _b.style) {
        return _a.style < _b.style;
    }

    if (_a.marker && _b.marker) {
        if (_a.marker->drawOrder() != _b.marker->drawOrder()) {
            return _a.marker->drawOrder() < _b.marker->drawOrder();
        }
    }

    // Sort by texture to reduce draw calls (increase batching)
    if (_a.label->texture() != _b.label->texture()) {
        return _a.label->texture() < _b.label->texture();
    }

    // Sort Markers by id
    if (_a.marker && _b.marker) {
        return _a.marker->id() < _b.marker->id();
    }

    // Just keep tile label order consistent
    if (_a.tile && _b.tile) {
        return _a.label < _b.label;
    }

    // Add tile labels before markers
    return bool(_a.tile);
}

void LabelManager::handleOcclusions(const ViewState& _viewState, bool _hideExtraLabels) {

    m_isect2d.clear();
    m_repeatGroups.clear();

    using iterator = decltype(m_labels)::const_iterator;

    // Find the label to which the obb belongs
    auto findLabel = [](iterator begin, iterator end, int obb) {
        for (auto it = begin; it != end; it++) {
            if (obb >= it->obbsRange.start && obb < it->obbsRange.end()) {
                return it->label;
            }
        }
        assert(false);
        return static_cast<Label*>(nullptr);
    };

    for (auto it = m_labels.begin(); it != m_labels.end(); ++it) {
        auto& entry = *it;
        auto* l = entry.label;

        // note that bounds needed even if label is occluded by repeat group (for example) to determine if
        //  label is on screen - could still be drawn if fading out
        ScreenTransform transform { m_transforms, entry.transformRange };
        OBBBuffer obbs { m_obbs, entry.obbsRange };

        l->obbs(transform, obbs);

        // if requested, hide extra labels indicated by transition.selected < 0
        if (_hideExtraLabels && l->options().selectTransition.time < 0) {
          l->occlude();
          l->skipTransitions();
          continue;
        }

        // Parent must have been processed earlier so at this point its
        // occlusion and anchor position is determined for the current frame.
        if (l->isChild()) {
            if (l->relative()->isOccluded()) {
                l->occlude();
                if (l->relative()->state() == Label::State::skip_transition) {
                    l->skipTransitions();
                }
                continue;
            }
        }

        // Skip label if another label of this repeatGroup is
        // within repeatDistance.
        if (l->options().repeatDistance > 0.f) {
            if (withinRepeatDistance(l)) {
                l->occlude();
                // If this label is not marked optional, then mark the relative label as occluded
                if (l->relative() && !l->options().optional) {
                    l->relative()->occlude();
                }
                continue;
            }
        }

        int anchorIndex = l->anchorIndex();

        // For each anchor
        do {
            if (l->isOccluded()) {
                // Update OBB for anchor fallback
                obbs.clear();

                l->obbs(transform, obbs);

                if (anchorIndex == l->anchorIndex()) {
                    // Reached first anchor again
                    break;
                }
            }

            l->occlude(false);

            // Occlude label when its obbs intersect with a previous label.
            for (auto& obb : obbs) {
                m_isect2d.intersect(obb.getExtent(), [&](auto& a, auto& b) {
                        size_t other = reinterpret_cast<size_t>(b.m_userData);

                        if (!intersect(obb, m_obbs[other])) {
                            return true;
                        }
                        // Ignore intersection with relative label
                        //
                        // Deliberately NOT extended to exempt siblings sharing the same
                        // relative (unlike the analogous fix in LabelCollider::process()):
                        // round 11 tried that here too and it was wrong. Unlike the
                        // tile-build-time pass (one-shot, no anchor retry -- exempting
                        // siblings there is the only way to avoid a permanent kill), THIS
                        // loop already has a real, graceful resolution mechanism for two
                        // labels landing on the same anchor: `nextAnchor()` cycling (below),
                        // driven by exactly this kind of detected intersection. Exempting
                        // siblings here suppressed that cycling entirely, so a peak's name
                        // and elevation sub-label -- which do start on the same icon-relative
                        // anchor before being spread apart -- could sit permanently
                        // overlapping instead of one of them moving to its next candidate
                        // anchor. Confirmed via live user report ("name and elevation
                        // overlapping") after round 11 shipped; see docs/label-placement-
                        // plan.md's Phase 7 round 12.
                        Label* other_label = findLabel(std::begin(m_labels), it, other);
                        if (l->relative() && l->relative() == other_label) {
                            return true;
                        }
                        l->occlude();
                        // for now, we're using selection transition time > 0 (previously unused style
                        //  param) to indicate a marker which should immediately hide all colliding labels
                        // in the future, we could use it to specify a (faster) hide transition in this case
                        if (l->state() != Label::State::sleep && other_label->options().selectTransition.time > 0) {
                            l->skipTransitions();
                        }
                        return false;

                    }, false);

                if (l->isOccluded()) { break; }
            }
        } while (l->isOccluded() && l->nextAnchor());

        // At this point, the label has a relative that is visible,
        // if it is not an optional label, turn the relative to occluded
        if (l->isOccluded()) {
            if (l->relative() && !l->options().optional) {
                l->relative()->occlude();
                if (l->state() == Label::State::skip_transition) {
                    l->relative()->skipTransitions();
                }
            }
        } else {
            // Insert into ISect2D grid
            int obbPos = entry.obbsRange.start;
            for (auto& obb : obbs) {
                auto aabb = obb.getExtent();
                aabb.m_userData = reinterpret_cast<void*>(obbPos++);
                m_isect2d.insert(aabb);
            }

            if (l->options().repeatDistance > 0.f) {
                m_repeatGroups[l->options().repeatGroup].push_back(l);
            }
        }
    }
}

bool LabelManager::withinRepeatDistance(Label *_label) {
    float threshold2 = pow(_label->options().repeatDistance, 2);

    auto it = m_repeatGroups.find(_label->options().repeatGroup);
    if (it != m_repeatGroups.end()) {
        for (auto* ll : it->second) {
            float d2 = glm::distance2(_label->screenCenter(), ll->screenCenter());
            if (d2 < threshold2) {
                return true;
            }
        }
    }
    return false;
}

void LabelManager::updateLabelSet(const View& _view, float _dt, const Scene& _scene,
                            const std::vector<std::shared_ptr<Tile>>& _tiles,
                            const std::vector<std::unique_ptr<Marker>>& _markers,
                            bool _onlyRender) {

    m_transforms.clear();
    m_obbs.clear();

    auto _viewState = _view.state();
    /// Collect and update labels from visible tiles
    {
        PROFILE_SCOPE("LabelsCollect");
        updateLabels(_viewState, _dt, _scene, _tiles, _markers, _onlyRender);
    }
    if (_onlyRender) { return; }

    std::sort(m_labels.begin(), m_labels.end(), LabelManager::priorityComparator);

    /// Mark labels to skip transitions

    if (int(m_lastZoom) != int(_viewState.zoom)) {
        skipTransitions(_scene, _tiles, *_scene.tileManager(), _viewState.zoom);
    }

    m_isect2d.resize({_viewState.viewportSize.x / 256, _viewState.viewportSize.y / 256},
                     {_viewState.viewportSize.x, _viewState.viewportSize.y});

    {
        PROFILE_SCOPE("LabelsCollision");
        handleOcclusions(_viewState, _scene.hideExtraLabels);
    }

    // Update label state
    for (auto& entry : m_labels) {
        //if (m_lastZoom < _viewState.zoom && entry.label->isOccluded() && !entry.label->occludedLastFrame()) {
        //    LOGD("Label hidden on zoom in: %s (%d labels)", entry.label->debugTag.c_str(), int(m_labels.size()));
        //}
        m_needUpdate |= entry.label->evalState(_dt);
    }

    std::sort(m_labels.begin(), m_labels.end(), LabelManager::zOrderComparator);

    Label::AABB screenBounds{0, 0, _viewState.viewportSize.x, _viewState.viewportSize.y};

    auto* elevManager = _scene.elevationManager();
    int markerMinZ = elevManager && !_tiles.empty() ? std::min(int(_tiles.back()->getID().z), 14) : 0;
    // Update label meshes
    for (auto& entry : m_labels) {

        // show alt marker if (non-optional part of) marker is occluded
        if (entry.label->isOccluded() && entry.marker
                && entry.marker->altMarker && !entry.label->options().optional) {
            Marker* alt = entry.marker->altMarker;
            if (alt->altMeshAdded) { continue; }  // already shown
            alt->altMeshAdded = true;
            Style* style = getStyleById(_scene, alt->styleId());
            auto labels = dynamic_cast<const LabelSet*>(alt->mesh());
            if (!style || !labels) { continue; }
            for (auto& label : labels->getLabels()) {
                if (label->canOcclude()) {
                    LOGE("Alt marker styling must set collide: false");
                    const_cast<Label::Options&>(label->options()).collide = false;  // fix invalid state
                }
            }

            if (elevManager) { elevManager->setMinZoom(markerMinZ); }
            // Alt markers aren't peak labels and don't opt into refinePriorityWithTextureShading,
            // so the viewport-max isolation term is irrelevant here -- pass "unavailable".
            processLabelUpdate(_viewState, labels, style, nullptr, alt, elevManager, _dt, false, 0.f, false);
            if (elevManager) { elevManager->setMinZoom(0); }
            continue;
        }

        if (!entry.label->visibleState()) { continue; }

        ScreenTransform transform { m_transforms, entry.transformRange };

        for (auto& obb : OBBBuffer{ m_obbs, entry.obbsRange }) {

            if (obb.getExtent().intersect(screenBounds)) {
                entry.label->addVerticesToMesh(transform, _viewState.viewportSize);
                break;
            }
        }
    }
    m_lastZoom = _viewState.zoom;
    m_lastViewPos = _view.getPosition();
    m_lastViewProj = _view.getViewProjectionMatrix();

    Profiler::counter("labelsActive", double(m_labels.size()));
}

void LabelManager::drawDebug(RenderState& rs, const View& _view) {

    if (!Tangram::getDebugFlag(Tangram::DebugFlags::labels)) {
        return;
    }

    for (auto& entry : m_labels) {
        auto* label = entry.label;

        if (label->type() == Label::Type::debug) { continue; }

        glm::vec2 sp = label->screenCenter();

        // draw bounding box
        switch (label->state()) {
        case Label::State::sleep:
            Primitives::setColor(rs, 0x0000ff);
            break;
        case Label::State::visible:
            Primitives::setColor(rs, 0x000000);
            break;
        case Label::State::none:
            Primitives::setColor(rs, 0x0000ff);
            break;
        case Label::State::dead:
            Primitives::setColor(rs, 0xff00ff);
            break;
        case Label::State::fading_in:
            Primitives::setColor(rs, 0xffff00);
            break;
        case Label::State::fading_out:
            Primitives::setColor(rs, 0xff0000);
            break;
        default:
            Primitives::setColor(rs, 0x999999);
        }

#if DEBUG_OCCLUSION
        if (label->isOccluded()) {
            Primitives::setColor(rs, 0xff0000);
            if (label->occludedLastFrame()) {
                Primitives::setColor(rs, 0xffff00);
            }
        } else if (label->occludedLastFrame()) {
            Primitives::setColor(rs, 0x00ff00);
        } else {
            Primitives::setColor(rs, 0x000000);
        }
#endif

        for (auto& obb : OBBBuffer{ m_obbs, entry.obbsRange }) {
            Primitives::drawPoly(rs, &(obb.getQuad())[0], 4);
        }

        if (label->relative() && label->relative()->visibleState() && !label->relative()->isOccluded()) {
            Primitives::setColor(rs, 0xff0000);
            Primitives::drawLine(rs, m_obbs[entry.obbsRange.start].getCentroid(),
                                 label->relative()->screenCenter());
        }

        if (label->type() == Label::Type::curved) {
            //for (int i = entry.transform.start; i < entry.transform.end()-2; i++) {
            for (int i = entry.transformRange.start; i < entry.transformRange.end()-1; i++) {
                if (i % 2 == 0) {
                    Primitives::setColor(rs, 0xff0000);
                } else {
                    Primitives::setColor(rs, 0x0000ff);

                }
                Primitives::drawLine(rs, glm::vec2(m_transforms.points[i]),
                                     glm::vec2(m_transforms.points[i+1]));
            }
        }
#if 0
        // draw offset
        glm::vec2 rot = label->screenTransform().rotation;
        glm::vec2 offset = label->options().offset;
        if (label->relative()) { offset += label->relative()->options().offset; }
        offset = rotateBy(offset, rot);

        Primitives::setColor(rs, 0x000000);
        Primitives::drawLine(rs, sp, sp - glm::vec2(offset.x, -offset.y));
#endif

        // draw projected anchor point
        Primitives::setColor(rs, 0x0000ff);
        Primitives::drawRect(rs, sp - glm::vec2(1.f), sp + glm::vec2(1.f));

#if 0
        if (label->options().repeatGroup != 0 && label->state() == Label::State::visible) {
            size_t seed = 0;
            hash_combine(seed, label->options().repeatGroup);
            float repeatDistance = label->options().repeatDistance;

            Primitives::setColor(rs, seed);
            Primitives::drawLine(rs, label->screenCenter(),
                                 glm::vec2(repeatDistance, 0.f) + label->screenCenter());

            float off = M_PI / 6.f;
            for (float pad = 0.f; pad < M_PI * 2.f; pad += off) {
                glm::vec2 p0 = glm::vec2(cos(pad), sin(pad)) * repeatDistance
                    + label->screenCenter();
                glm::vec2 p1 = glm::vec2(cos(pad + off), sin(pad + off)) * repeatDistance
                    + label->screenCenter();
                Primitives::drawLine(rs, p0, p1);
            }
        }
#endif
    }

    glm::vec2 split(_view.getWidth() / 256, _view.getHeight() / 256);
    glm::vec2 res(_view.getWidth(), _view.getHeight());
    const short xpad = short(ceilf(res.x / split.x));
    const short ypad = short(ceilf(res.y / split.y));

    Primitives::setColor(rs, 0x7ef586);
    short x = 0, y = 0;
    for (int j = 0; j < split.y; ++j) {
        for (int i = 0; i < split.x; ++i) {
            AABB cell(x, y, x + xpad, y + ypad);
            Primitives::drawRect(rs, {x, y}, {x + xpad, y + ypad});
            x += xpad;
            if (x >= res.x) {
                x = 0;
                y += ypad;
            }
        }
    }
}

}
