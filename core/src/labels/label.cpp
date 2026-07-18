#include "labels/label.h"

#include "labels/anchorOccupancyGrid.h"
#include "log.h"
#include "platform.h"
#include "tile/tile.h"
#include "util/elevationManager.h"
#include "util/geom.h"
#include "util/mapProjection.h"
#include "util/textureShading.h"
#include "view/view.h"

#include <algorithm>
#include <cmath>

namespace Tangram {

namespace {
// Mirrors the sqrt-compression-into-fractional-priority idiom already used by the scene's own
// place_fn (assets/scenes/stylus-osm.yaml, "Math.round(Math.sqrt(population))/65536") -- keeps
// the whole adjustment inside the fractional part of a tier-based priority (see
// docs/label-placement-plan.md Phase 2, design step 3). `shade` is already a dimensionless,
// roughly [0,1] value (Brown's contrast curve saturates towards 0/1), so unlike place_fn's
// population/prominence-in-meters input, no extra normalization constant is needed before the
// sqrt.
float compressTextureShading(float shade) {
    float x = std::max(0.f, shade);
    return std::round(std::sqrt(x) * 4096.f) / 65536.f;
}
}

const float Label::activation_distance_threshold = 2;

Label::Label(glm::vec2 _size, Type _type, Options _options)
    : m_type(_type),
      m_dim(_size + _options.buffer),
      m_options(_options),
      m_state(State::none) {

    if (m_type == Type::debug) {
        m_options.collide = false;
    }

    if (m_options.collide) {
        enterState(State::none, 0.0);
    } else {
        enterState(State::visible, 1.0);
    }

    m_occludedLastFrame = false;
    m_occluded = false;
    m_relative = nullptr;
    m_anchorIndex = 0;
}

Label::~Label() {}

void Label::setRelative(Label& _relative, bool _definePriority, bool _defineCollide) {
    m_relative = &_relative;
    m_trackRelativePriority = _definePriority;

    if (_definePriority) {
        m_options.priority = _relative.options().priority;  //+ 0.5f;
    }

    if (_defineCollide) {
        m_options.collide = _relative.options().collide;
    }

    applyAnchor(m_options.anchors[m_anchorIndex]);
}

void Label::syncRelativePriority() {
    // docs/label-placement-plan.md Phase 2: setRelative() above only copies priority once, at
    // tile-build time. A peak's icon label can have its priority updated later (main thread,
    // see refinePriority() below) once the elevation mosaic becomes available, so a linked text
    // label that tracks it (_definePriority was true) needs to keep re-copying it every frame
    // rather than relying on that one-time copy going stale.
    if (m_trackRelativePriority && m_relative) {
        m_options.priority = m_relative->options().priority;
    }
}

namespace {
// Tune by eye, like every other constant in this pipeline: how much of the priority
// fraction's [0, 0.4]-wide band the isolation term is allowed to claim.
constexpr float kIsolationWeight = 0.15f;
constexpr float kIsolationMarginMeters = 300.f;
}

float Label::isolationScore(double peakElev, float regionMaxElev, float marginMeters) {
    if (marginMeters <= 0.f) { return peakElev >= regionMaxElev ? 1.f : 0.f; }
    float diff = float(peakElev - regionMaxElev);
    return clamp01(diff / marginMeters + 1.f);
}

bool Label::refinePriority(ElevationManager& elevMgr, glm::dvec2 origin, double scale,
                           float viewportMaxElev, bool haveViewportMaxElev) {
    if (!m_options.refinePriorityWithTextureShading) { return true; }
    if (!elevMgr.m_elevationSource) { return false; }

    ProjectedMeters worldPos = origin + glm::dvec2(modelCenter()) * scale;
    bool ok = false;
    float shade = sampleTextureShading(*elevMgr.m_elevationSource, worldPos, ok);
    if (!ok) { return false; }

    // Regional isolation term -- see this method's doc comment in label.h. Best-effort: only
    // contributes if both the current frame's viewport-wide max elevation (computed once by
    // LabelManager::updateLabels(), from tiles already resident for rendering -- see its own
    // comment) and this peak's own elevation are available; otherwise 0, same as today's
    // shade-only behavior.
    float isolation = 0.f;
    if (haveViewportMaxElev) {
        bool elevOk = false;
        double peakElev = elevMgr.getElevation(worldPos, elevOk);
        if (elevOk) {
            isolation = isolationScore(peakElev, viewportMaxElev, kIsolationMarginMeters);
        }
    }

    // Original priority (computed at tile-build time, see the peak draw rule's `priority:`
    // function in stylus-osm.yaml) is always `<half-tier band> + <fractional in [0, 0.4]>`,
    // where the peak tier is split into two HALF-integer bands -- named peaks get
    // [global.priority.peak, +0.4], unnamed/elevation-only ones get
    // [global.priority.peak+0.5, +0.9] -- so that an unnamed peak can never outrank a named
    // one regardless of texture-shading timing (revised 2026-07-16, user report: a famous
    // named summit was losing to nearby anonymous elevation nodes). Recovering the coarser
    // `floor(priority)` here (as this used to) would collapse both bands back to the same
    // integer and erase that separation the moment refinement runs; floor(priority*2)/2
    // recovers the correct HALF-tier band instead, preserving it. Whichever band this
    // started in, replace the fractional part with the texture-shading-derived one -- see
    // docs/label-placement-plan.md Phase 2, design step 3, and Task 3's 2026-07-16 revision.
    // The isolation term above is added on top (clamped so the combined compression can't
    // spill past the 0.4-wide band into the next integer tier).
    float bandFloor = std::floor(m_options.priority * 2.f) / 2.f;
    float compression = std::min(compressTextureShading(shade) + kIsolationWeight * isolation, 0.39f);
    m_options.priority = bandFloor - compression;
    return true;
}

bool Label::refineAnchor(ElevationManager& elevMgr, glm::dvec2 origin, double scale,
                         const AnchorOccupancyGrid* occupancyGrid) {
    if (!m_options.refineAnchorWithTextureShading) { return true; }
    if (m_options.anchors.count <= 1) { return true; } // nothing to choose between
    if (!elevMgr.m_elevationSource) { return false; }

    // "Ridge/canyon response" is distance from the texture-shading formula's neutral value
    // (0.5, see textureShading.cpp/hillshade.yaml's contrast curve) -- a gentle slope reads
    // close to neutral in either direction, while a strong ridge or canyon pushes the shade
    // value away from it. Lower cost = gentler ground to stamp the label across.
    auto ridgeCost = [](float shade) { return std::fabs(shade - 0.5f); };

    // Candidate footprint geometry, mirroring TextStyleBuilder::anchorCandidateCost()
    // (tile-build-time equivalent of this function): icon radius + half this label's own
    // measured extent, offset from the icon position in the anchor's direction. Icon
    // radius is approximated the same way PointStyleBuilder does (half the icon's larger
    // dimension) -- 0 if this label isn't an icon-linked child (e.g. a standalone text
    // label), which just means the footprint starts right at the label's own position.
    glm::vec2 iconDim = isChild() ? m_relative->dimension() : glm::vec2(0.f);
    float iconRadiusPx = 0.5f * std::max(iconDim.x, iconDim.y);
    glm::vec2 halfExtentPx = dimension() * 0.5f;

    // Convert pixel-space offsets to the tile's normalized [0,1] local space modelCenter()
    // and AnchorOccupancyGrid are both expressed in. MapProjection::tileSize() (pixels per
    // full tile width) is a fixed constant, not the current view's screen projection -- see
    // this method's doc comment in label.h for why that's the correct conversion to reuse
    // here even though this runs at an arbitrary view zoom/tilt.
    float tileSizePx = float(MapProjection::tileSize());
    glm::vec2 halfExtentTile = halfExtentPx / tileSizePx;
    glm::vec2 iconPosTile = glm::vec2(modelCenter());

    // Sample a handful of points spread across each candidate's footprint (not just its
    // center), matching anchorCandidateCost()'s pattern.
    constexpr int kNumSamples = 5;
    const glm::vec2 sampleOffsets[kNumSamples] = {
        glm::vec2(0.f, 0.f),
        glm::vec2(-1.f, -1.f), glm::vec2(1.f, -1.f),
        glm::vec2(-1.f, 1.f), glm::vec2(1.f, 1.f),
    };

    // Equal weight between "don't stamp across a ridge" and "don't stamp across a river/
    // road/trail" -- both costs are roughly [0, 0.5]-[0,1] in practice, comparable
    // magnitude; tune by eye if one term should dominate.
    const float kOccupancyWeight = 1.0f;

    // Samples kNumSamples points around footprintCenterTile (already tile-local, already
    // including the icon's own position) and returns the combined ridge+occupancy cost, or
    // false if any sample isn't available yet (caller should retry next frame).
    auto sampleFootprintCost = [&](glm::vec2 footprintCenterTile, float& outCost) -> bool {
        float shadeCostSum = 0.f;
        float occupancyCostSum = 0.f;
        for (const auto& sampleOffset : sampleOffsets) {
            glm::vec2 sampleTile = footprintCenterTile + sampleOffset * halfExtentTile;
            ProjectedMeters sampleWorld = origin + glm::dvec2(sampleTile) * scale;

            bool ok = false;
            float shade = sampleTextureShading(*elevMgr.m_elevationSource, sampleWorld, ok);
            if (!ok) { return false; }

            shadeCostSum += ridgeCost(shade);
            if (occupancyGrid) {
                occupancyCostSum += occupancyGrid->sample(sampleTile);
            }
        }
        outCost = (shadeCostSum + kOccupancyWeight * occupancyCostSum) / float(kNumSamples);
        return true;
    };

    std::vector<float> costs(m_options.anchors.count);
    for (int i = 0; i < m_options.anchors.count; i++) {
        glm::vec2 dir = LabelProperty::anchorDirection(m_options.anchors.anchor[i]);
        float offsetPx = iconRadiusPx + glm::length(halfExtentPx);
        glm::vec2 footprintCenterTile = iconPosTile + (dir * offsetPx) / tileSizePx;

        if (!sampleFootprintCost(footprintCenterTile, costs[i])) {
            return false; // retry next frame once every candidate is sampleable
        }
    }

    // Fully re-sort the compass anchors by ascending cost -- this only ever runs once per
    // label (see m_anchorRefined), so there's no per-frame flicker/churn risk to guard
    // against with a "only switch if meaningfully better" threshold. No more "stack
    // directly under a target label" special case (removed 2026-07-16, see this method's
    // header comment) -- every label, including a peak's elevation sub-label, always picks
    // among these same icon-relative compass anchors.
    if (!costs.empty()) {
        std::vector<int> order = sortAnchorIndicesByCost(costs);
        std::array<LabelProperty::Anchor, LabelProperty::max_anchors> sorted;
        for (size_t i = 0; i < order.size(); i++) {
            sorted[i] = m_options.anchors.anchor[order[i]];
        }
        for (size_t i = 0; i < order.size(); i++) {
            m_options.anchors.anchor[i] = sorted[i];
        }
        setAnchorIndex(0);
    }

    return true;
}

std::vector<int> Label::sortAnchorIndicesByCost(const std::vector<float>& _costs) {
    std::vector<int> order(_costs.size());
    for (size_t i = 0; i < order.size(); i++) { order[i] = int(i); }

    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        return _costs[a] < _costs[b];
    });

    return order;
}

bool Label::visibleState() const {
    int visibleFlags = (State::visible |
                        State::fading_in |
                        State::fading_out |
                        State::skip_transition);

    return (visibleFlags & m_state);
}

void Label::skipTransitions() {
    enterState(State::skip_transition, 0.0);
}

void Label::enterState(const State& _state, float _alpha) {
    if (m_state == State::dead) { return; }

    m_state = _state;
    setAlpha(_alpha);

    if (m_state == State::sleep) {
        // Reset anchor fallback index
        m_anchorIndex = 0;
    }
}

void Label::setAlpha(float _alpha) {
    m_alpha = clamp01(_alpha);
}

void Label::resetState() {

    if (m_state == State::dead) { return; }

    m_occludedLastFrame = false;
    m_occluded = false;
    m_anchorIndex = 0;
    enterState(State::none, 0.0);
}

void Label::print() const {
    LOG("Label - %p", this);
    LOG("\tm_occludedLastFrame: %d", m_occludedLastFrame);
    LOG("\tm_occluded: %d", m_occluded);
    std::string state;
    switch (m_state) {
        case State::none: state = "none"; break;
        case State::visible: state = "visible"; break;
        case State::fading_in: state = "fading_in"; break;
        case State::fading_out: state = "fading_out"; break;
        case State::skip_transition: state = "skip_transition"; break;
        case State::sleep: state = "sleep"; break;
        case State::dead: state = "dead"; break;
        case State::out_of_screen: state = "out_of_screen"; break;
    }
    LOG("\tm_state: %s", state.c_str());
    LOG("\tm_anchorIndex: %d", m_anchorIndex);
}

bool Label::nextAnchor() {
    // Guard against count == 0 (or negative): `% 0` is undefined behavior, found via a real
    // crash (SIGSEGV inside Anchors::operator[] with a garbage index) in Phase 6/7 headless
    // testing -- some label's Options.anchors reached here still default-constructed (count
    // == 0, LabelProperty::Anchors's default ctor only fills the array, it does not set
    // count) rather than given at least one anchor. Mirrors the same `count <= 1` guard
    // refineAnchor() already had. Root cause of which label gets here with count == 0 is a
    // separate, deeper bug (not yet found) -- this guard just makes it inert instead of a
    // crash, matching "nothing to cycle to" behavior for any label with 0 or 1 anchors.
    if (m_options.anchors.count <= 1) { return false; }

    int index = m_anchorIndex;
    setAnchorIndex((index + 1) % m_options.anchors.count);

    return m_anchorIndex != index;
}

bool Label::setAnchorIndex(int _index) {
    if (_index >= int(m_options.anchors.count) || _index < 0) {
        return false;
    }
    m_anchorIndex = _index;

    applyAnchor(m_options.anchors[m_anchorIndex]);

    return true;
}

bool Label::update(const glm::mat4& _mvp, const ViewState& _viewState,
                   const AABB* _bounds, ScreenTransform& _transform) {

    m_occludedLastFrame = m_occluded;
    m_occluded = false;

    // See stableFrames()'s doc comment (label.h) -- tracks a track record of consecutive
    // wins, not just the single previous frame's outcome.
    if (m_occludedLastFrame) {
        m_stableFrames = 0;
    } else {
        m_stableFrames++;
    }

    bool valid = updateScreenTransform(_mvp, _viewState, _bounds, _transform);
    if (!valid) {
        enterState(State::sleep, 0.0);
        return false;
    }

    return true;
}

bool Label::evalState(float _dt) {

#ifdef DEBUG
    if (Tangram::getDebugFlag(DebugFlags::draw_all_labels)) {
        enterState(State::visible, 1.0);
        return false;
    }
#endif

    switch (m_state) {
        case State::none:
        case State::sleep:
            if (m_occluded) {
                enterState(State::sleep, 0.0);
                return false;
            }
            if (m_options.showTransition.time == 0.f) {
                enterState(State::visible, 1.0);
                return false;
            }
            m_fade.reset(true, m_options.showTransition.ease,
                         m_options.showTransition.time);
            enterState(State::fading_in, 0.0);
            return true;

        case State::visible:
            if (!m_occluded) { return false; }

            if (m_options.hideTransition.time == 0.f) {
                enterState(State::sleep, 0.0);
                return false;
            }
            m_fade.reset(false, m_options.hideTransition.ease,
                         m_options.hideTransition.time);

            enterState(State::fading_out, 1.0);
            return true;

        case State::fading_in:
            if (m_occluded) {
                enterState(State::sleep, 0.0);
                return false;
            }
            setAlpha(m_fade.update(_dt));
            if (m_fade.isFinished()) {
                enterState(State::visible, 1.0);
                return false;
            }
            return true;

        case State::fading_out:
            // if (!m_occluded) {
            //     enterState(State::fading_in, m_transform.state.alpha);
            //     m_fade.reset(false, m_options.hideTransition.ease,
            //                  m_options.showTransition.time);
            //     return true;
            // }
            setAlpha(m_fade.update(_dt));
            if (m_fade.isFinished()) {
                enterState(State::sleep, 0.0);
                return false;
            }
            return true;

        case State::skip_transition:
            if (m_occluded) {
                enterState(State::sleep, 0.0);
            } else {
                enterState(State::visible, 1.0);
            }
            return true;

        case State::dead:
        case State::out_of_screen:
            break;
    }

    return false;
}

}
