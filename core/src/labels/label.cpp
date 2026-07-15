#include "labels/label.h"

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

bool Label::refineAnchor(ElevationManager& elevMgr, glm::dvec2 origin, double scale) {
    if (!m_options.refineAnchorWithTextureShading) { return true; }
    if (m_options.anchors.count <= 1) { return true; } // nothing to choose between
    if (!elevMgr.m_elevationSource) { return false; }

    ProjectedMeters worldPos = origin + glm::dvec2(modelCenter()) * scale;

    // Fixed real-world sampling radius, not the label's actual measured extent -- see this
    // method's doc comment in label.h for why a screen-pixel footprint can't be converted to
    // world meters here. 25m is in the right ballpark for how far a small peak label's text
    // actually sits from its icon on the ground at typical label-visible zooms; tune by eye.
    const double kSampleRadiusMeters = 25.0;

    // "Ridge/canyon response" is distance from the texture-shading formula's neutral value
    // (0.5, see textureShading.cpp/hillshade.yaml's contrast curve) -- a gentle slope reads
    // close to neutral in either direction, while a strong ridge or canyon pushes the shade
    // value away from it. Lower cost = gentler ground to stamp the label across.
    auto ridgeCost = [](float shade) { return std::fabs(shade - 0.5f); };

    std::vector<float> costs(m_options.anchors.count);

    for (int i = 0; i < m_options.anchors.count; i++) {
        glm::vec2 dir = LabelProperty::anchorDirection(m_options.anchors.anchor[i]);
        ProjectedMeters samplePos = worldPos + glm::dvec2(dir) * kSampleRadiusMeters;

        bool ok = false;
        float shade = sampleTextureShading(*elevMgr.m_elevationSource, samplePos, ok);
        if (!ok) { return false; } // retry next frame once every candidate is sampleable

        costs[i] = ridgeCost(shade);
    }

    // Only switch for a meaningful improvement -- avoid churn between near-tied candidates.
    const float kMinImprovement = 0.08f;
    int bestIndex = pickBetterAnchorIndex(costs, m_anchorIndex, kMinImprovement);
    if (bestIndex >= 0) {
        // Rotate the chosen anchor to the front of the fallback list (mirrors Phase 3's
        // salienceOrderedAnchors()'s "resort, hand to the existing anchor-cycling mechanism
        // unchanged" approach) and switch to it now.
        LabelProperty::Anchor best = m_options.anchors.anchor[bestIndex];
        for (int i = bestIndex; i > 0; i--) {
            m_options.anchors.anchor[i] = m_options.anchors.anchor[i - 1];
        }
        m_options.anchors.anchor[0] = best;
        setAnchorIndex(0);
    }

    return true;
}

int Label::pickBetterAnchorIndex(const std::vector<float>& _costs, int _currentIndex,
                                 float _minImprovement) {
    if (_costs.empty() || _currentIndex < 0 || _currentIndex >= (int)_costs.size()) {
        return -1;
    }

    int bestIndex = 0;
    for (size_t i = 1; i < _costs.size(); i++) {
        if (_costs[i] < _costs[bestIndex]) { bestIndex = (int)i; }
    }

    if (bestIndex == _currentIndex) { return -1; }
    if (_costs[bestIndex] + _minImprovement >= _costs[_currentIndex]) { return -1; }

    return bestIndex;
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
