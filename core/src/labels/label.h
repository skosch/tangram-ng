#pragma once

#include "labels/fadeEffect.h"
#include "labels/labelProperty.h"
#include "map.h"
#include "util/types.h"
#include "util/hash.h"

#include "glm/vec2.hpp"
#include "glm/vec3.hpp"
#include "glm/mat4x4.hpp"
#include <climits> // needed in aabb.h
#include "aabb.h"
#include "obb.h"
#include <string>
#include <limits>
#include <memory>

namespace Tangram {

struct ScreenTransform;
struct ViewState;
struct OBBBuffer;
class Texture;
class ElevationManager;

class Label {

public:

    using OBB = isect2d::OBB<glm::vec2>;
    using AABB = isect2d::AABB<glm::vec2>;

    enum class Type {
        point,
        line,
        curved,
        debug
    };

    enum State {
        none            = 1 << 0,
        fading_in       = 1 << 1,
        fading_out      = 1 << 2,
        visible         = 1 << 3,
        sleep           = 1 << 4,
        out_of_screen   = 1 << 5,
        skip_transition = 1 << 6,
        dead            = 1 << 7,
    };

    struct Transition {
        FadeEffect::Interpolation ease = FadeEffect::Interpolation::linear;
        float time = 0.0;
    };

    struct Options {
        glm::vec2 offset;
        glm::vec2 buffer;
        float priority = std::numeric_limits<float>::max();
        bool collide = true;
        Transition selectTransition;
        Transition hideTransition;
        Transition showTransition;
        size_t repeatGroup = 0;
        float repeatDistance = 0;
        size_t paramHash = 0; // the label hash based on its styling parameters
        LabelProperty::Anchors anchors;
        bool optional = false;
        bool flat = false;
        float angle = 0.f;
        uint32_t featureId = 0;
        // docs/label-placement-plan.md Phase 2: true for a label (currently only peak icons,
        // via the `priority_texture_shading` style param) whose `priority` above was computed
        // from a provisional fallback (pure-elevation weighting) at tile-build time because the
        // CPU texture-shading sampler can't safely run on the TileWorker thread
        // PointStyleBuilder::addFeature() executes on (see textureShading.h's threading note).
        // LabelManager refines such a label's priority once, on the main thread, the first time
        // the elevation mosaic is available at its position -- see Label::refinePriority().
        bool refinePriorityWithTextureShading = false;
    };

    static const float activation_distance_threshold;

    Label(glm::vec2 _size, Type _type, Options _options);

    virtual ~Label();

    // Add vertices for this label to its Style's shared Mesh
    virtual void addVerticesToMesh(ScreenTransform& _transform, const glm::vec2& _screenSize) = 0;

    virtual LabelType renderType() const = 0;

    virtual uint32_t selectionColor() = 0;

    virtual glm::vec3 modelCenter() const = 0;

    // Returns the candidate priority for a feature with multiple labels
    virtual float candidatePriority() const { return 0; }

    virtual const Texture* texture() const { return nullptr; }

    bool update(const glm::mat4& _mvp, const ViewState& _viewState,
                const AABB* _bounds, ScreenTransform& _transform);

    bool evalState(float _dt);

    // Update the screen position of the label
    virtual bool updateScreenTransform(const glm::mat4& _mvp, const ViewState& _viewState,
                                       const AABB* _bounds, ScreenTransform& _transform) = 0;

    virtual bool setElevation(ElevationManager& elevMgr, glm::dvec2 origin, double scale) = 0;

    // docs/label-placement-plan.md Phase 2: attempt to refine this label's `priority` using the
    // CPU texture-shading sampler (sampleTextureShading(), util/textureShading.h), main-thread
    // only. No-op (returns true, i.e. "nothing left to do") unless
    // m_options.refinePriorityWithTextureShading is set. Returns whether a usable value was
    // found (mirrors setElevation()'s ok-return idiom): callers should keep retrying every
    // frame (like m_elevationSet/setElevation already do) until this returns true, then stop --
    // see LabelManager::processLabelUpdate(). worldPos = origin + modelCenter() * scale, same
    // convention as SpriteLabel::setElevation().
    bool refinePriority(ElevationManager& elevMgr, glm::dvec2 origin, double scale);

    // If this label tracks a relative's priority (see setRelative()'s _definePriority), copy
    // the relative's current priority again. setRelative() only does this copy once, at
    // construction time; a peak's text label therefore needs this called every frame so it
    // stays in sync with its icon's priority as refinePriority() above updates it on the main
    // thread (label construction itself happens once, at tile-build time, on a worker thread).
    // No-op for a label that isn't tracking a relative, or wasn't set up with
    // _definePriority=true. See LabelManager::processLabelUpdate().
    void syncRelativePriority();

    // Current screen position of the label anchor
    glm::vec2 screenCenter() const { return glm::vec2(m_screenCenter); }
    glm::vec4 screenCoord() const { return m_screenCenter; }

    // Occlude the label
    void occlude(bool _occlusion = true) { m_occluded = _occlusion; }

    // Checks whether the label is in a state where it can occlusion
    bool canOcclude() const { return m_options.collide; }

    void skipTransitions();

    size_t hash() const { return m_options.paramHash; }

    glm::vec2 dimension() const { return m_dim; }

    // Gets for label options: color and offset
    const Options& options() const { return m_options; }

    // Adds the oriented bounding boxes of the label to _obbs, updates Range
    virtual void obbs(ScreenTransform& _transform, OBBBuffer& _obbs) const = 0;

    State state() const { return m_state; }

    bool isOccluded() const { return m_occluded; }

    bool occludedLastFrame() const { return m_occludedLastFrame; }

    Label* relative() const { return m_relative; }
    bool isChild() const { return bool(m_relative); }
    bool isSibling() const { return bool(m_relative) && bool(m_relative->relative()); }

    void setRelative(Label& _relative, bool _definePriority, bool _defineCollide);

    LabelProperty::Anchor anchorType() const {
        return m_options.anchors[m_anchorIndex];
    }

    int anchorIndex() { return m_anchorIndex; }

    bool nextAnchor();

    bool setAnchorIndex(int _index);

    void enterState(const State& _state, float _alpha = 1.0f);

    void resetState();

    // Checks whether the label is in a visible state
    bool visibleState() const;

    Type type() const { return m_type; }

    void print() const;

    void setAlpha(float _alpha);

    bool m_elevationSet = false;
    // docs/label-placement-plan.md Phase 2: set once refinePriority() has found a usable
    // texture-shading value (or immediately, if refinePriorityWithTextureShading isn't set at
    // all) -- mirrors m_elevationSet's "retry every frame until success, then stop" pattern.
    bool m_prominenceRefined = false;
#ifdef DEBUG
    std::string debugTag;
#endif

protected:

    virtual void applyAnchor(LabelProperty::Anchor _anchor) = 0;

    const Type m_type;
    const glm::vec2 m_dim;

    Options m_options;

    Label* m_relative;
    // docs/label-placement-plan.md Phase 2: whether this label's priority should be kept in
    // sync with m_relative's every frame -- set from setRelative()'s _definePriority; see
    // syncRelativePriority().
    bool m_trackRelativePriority = false;

    State m_state;
    FadeEffect m_fade;

    int m_anchorIndex;
    glm::vec2 m_anchor;

    bool m_occludedLastFrame;
    bool m_occluded;

    glm::vec4 m_screenCenter;
    float m_alpha;
};

}

namespace std {
    template <>
    struct hash<Tangram::Label::Options> {
        size_t operator() (const Tangram::Label::Options& o) const {
            std::size_t seed = 0;
            hash_combine(seed, o.offset.x);
            hash_combine(seed, o.offset.y);
            hash_combine(seed, o.priority);
            hash_combine(seed, o.collide);
            hash_combine(seed, o.refinePriorityWithTextureShading);
            hash_combine(seed, o.repeatDistance);
            hash_combine(seed, o.repeatGroup);
            hash_combine(seed, (int)o.selectTransition.ease);
            hash_combine(seed, o.selectTransition.time);
            hash_combine(seed, (int)o.hideTransition.ease);
            hash_combine(seed, o.hideTransition.time);
            hash_combine(seed, (int)o.showTransition.ease);
            hash_combine(seed, o.showTransition.time);
            for (int i = 0; i < o.anchors.count; ++i) {
                hash_combine(seed, (int)o.anchors[i]);
            }
            return seed;
        }
    };
}
