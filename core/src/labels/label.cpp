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

bool Label::refinePriority(ElevationManager& elevMgr, glm::dvec2 origin, double scale) {
    if (!m_options.refinePriorityWithTextureShading) { return true; }
    if (!elevMgr.m_elevationSource) { return false; }

    ProjectedMeters worldPos = origin + glm::dvec2(modelCenter()) * scale;
    bool ok = false;
    float shade = sampleTextureShading(*elevMgr.m_elevationSource, worldPos, ok);
    if (!ok) { return false; }

    // Original priority (computed at tile-build time, see the peak draw rule's `priority:`
    // function in stylus-osm.yaml) is always `global.priority.peak + <fractional in (0,1)>`
    // (today's pure-elevation weighting), so its integer part reliably recovers the tier;
    // replace the fractional part with the texture-shading-derived one -- see
    // docs/label-placement-plan.md Phase 2, design step 3.
    m_options.priority = std::floor(m_options.priority) - compressTextureShading(shade);
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

    std::vector<float> costs(m_options.anchors.count);

    for (int i = 0; i < m_options.anchors.count; i++) {
        glm::vec2 dir = LabelProperty::anchorDirection(m_options.anchors.anchor[i]);
        float offsetPx = iconRadiusPx + glm::length(halfExtentPx);
        glm::vec2 footprintCenterTile = (dir * offsetPx) / tileSizePx;

        float shadeCostSum = 0.f;
        float occupancyCostSum = 0.f;

        for (const auto& sampleOffset : sampleOffsets) {
            glm::vec2 sampleTile = glm::vec2(modelCenter()) + footprintCenterTile + sampleOffset * halfExtentTile;
            ProjectedMeters sampleWorld = origin + glm::dvec2(sampleTile) * scale;

            bool ok = false;
            float shade = sampleTextureShading(*elevMgr.m_elevationSource, sampleWorld, ok);
            if (!ok) { return false; } // retry next frame once every sample is available

            shadeCostSum += ridgeCost(shade);
            if (occupancyGrid) {
                occupancyCostSum += occupancyGrid->sample(sampleTile);
            }
        }

        costs[i] = (shadeCostSum + kOccupancyWeight * occupancyCostSum) / float(kNumSamples);
    }

    // Fully re-sort by ascending combined cost -- this only ever runs once per label (see
    // m_anchorRefined), so there's no per-frame flicker/churn risk to guard against with a
    // "only switch if meaningfully better" threshold the way an earlier version of this
    // method did.
    std::vector<int> order = sortAnchorIndicesByCost(costs);
    std::array<LabelProperty::Anchor, LabelProperty::max_anchors> sorted;
    for (size_t i = 0; i < order.size(); i++) {
        sorted[i] = m_options.anchors.anchor[order[i]];
    }
    for (size_t i = 0; i < order.size(); i++) {
        m_options.anchors.anchor[i] = sorted[i];
    }
    setAnchorIndex(0);

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
