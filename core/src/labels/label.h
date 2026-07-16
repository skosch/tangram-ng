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
#include <vector>

namespace Tangram {

struct ScreenTransform;
struct ViewState;
struct OBBBuffer;
class Texture;
class ElevationManager;
class AnchorOccupancyGrid;

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
        // docs/label-placement-plan.md Task 1/user feedback 2026-07-15 ("labels too far
        // from peaks"): TextLabel::applyAnchor()/CurvedLabel::applyAnchor() add the full
        // relative's dimension() into the icon<->label gap (offset = m_dim +
        // relative.dimension(), then halved) -- scales that contribution, 1.0 = unchanged
        // default. A child label can set this below 1 to sit closer to its relative than
        // the standard "clear of the whole icon" gap.
        float anchorGapScale = 1.0f;
        // docs/label-placement-plan.md Phase 2: true for a label (currently only peak icons,
        // via the `priority_texture_shading` style param) whose `priority` above was computed
        // from a provisional fallback (pure-elevation weighting) at tile-build time because the
        // CPU texture-shading sampler can't safely run on the TileWorker thread
        // PointStyleBuilder::addFeature() executes on (see textureShading.h's threading note).
        // LabelManager refines such a label's priority once, on the main thread, the first time
        // the elevation mosaic is available at its position -- see Label::refinePriority().
        bool refinePriorityWithTextureShading = false;
        // docs/label-placement-plan.md Task 4: true for a label (currently only peak name
        // text, via the `text:anchor_texture_shading` style param) whose anchor fallback
        // order should be refined once, on the main thread, using the CPU texture-shading
        // sampler -- steering it away from a candidate anchor that would stamp the label
        // across a ridge/canyon, the same way Phase 3's salienceOrderedAnchors() already
        // steers it away from water/road/trail geometry at tile-build time (that term
        // couldn't include texture shading itself, since the sampler is main-thread-only --
        // see anchorCandidateCost()'s comment). See Label::refineAnchor().
        bool refineAnchorWithTextureShading = false;
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

    // docs/label-placement-plan.md Task 4 (revised 2026-07-16 after user review): steer
    // this label's anchor away from both a strong ridge/canyon texture-shading response
    // AND nearby vector clutter (water/road/trail, via the tile's AnchorOccupancyGrid --
    // the same structure Phase 3's tile-build-time anchorCandidateCost() uses), main-thread
    // only, the same "retry every frame until the mosaic is available, then do it once"
    // pattern as refinePriority() above. No-op (returns true) unless
    // m_options.refineAnchorWithTextureShading is set.
    //
    // Scores every candidate anchor in m_options.anchors (the Imhof/Yoeli-ordered default,
    // see TextStyleBuilder::applyRule, or whatever text_anchor was set explicitly) at
    // several points spread across that anchor's actual footprint (this label's own
    // measured dimension() plus its relative icon's radius, converted to the tile's
    // normalized [0,1] local space via MapProjection::tileSize() -- a fixed per-tile-
    // geometry conversion, not the current view's screen projection, valid here since this
    // relates a fixed-size footprint to its own tile's coordinate system, not the camera),
    // and fully re-sorts the list by ascending combined cost.
    //
    // If setStackFallbackTarget() was called (currently only the peak elevation sub-label,
    // targeting the name label), ALSO scores one more candidate: sitting just below the
    // target's own current position, aligned to whichever of its edges is nearest the icon
    // (see the horizontal-alignment switch in the .cpp). This is a *fallback default*, not
    // a hard link (per user direction: name and elevation don't have to be adjacent in real
    // topo-map practice -- OSM peaks routinely have one without the other -- so this label
    // is free to end up anywhere the 8 compass anchors say is best; stacking under the
    // target is simply preferred, via `kStackPreferenceMargin`, when it isn't meaningfully
    // worse than the best independent spot). The target pointer is read ONLY here, ONCE
    // (this whole method only ever runs once per label, see m_anchorRefined) -- its public
    // dimension()/anchorType()/options() are sampled synchronously to compute a final
    // absolute m_options.offset if stacking wins, and the pointer itself is never touched
    // again afterward. m_relative (used for the icon<->label gap and the occlusion-cascade
    // "die together" logic, needed every frame for the label's whole lifetime) stays the
    // icon throughout, exactly like every other icon-linked label in this engine -- the
    // stack-fallback target is a separate, much lower-stakes pointer than m_relative would
    // be if it pointed at a sibling label instead of the icon (an earlier version of this
    // mechanism did exactly that, via setRelative(), and was reverted after finding it
    // couldn't be shown safe against a sibling label dying independently).
    //
    // The normal per-frame LabelManager::handleOcclusions() collision pass that runs right
    // after this call re-validates the new position exactly like any other anchor (and
    // falls back further via nextAnchor() if it turns out to newly collide with something),
    // so no separate occlusion re-check needs to be triggered here. `_occupancyGrid` may be
    // null (e.g. a marker label, or a tile with no water/transportation geometry) -- the
    // vector-clutter term is simply skipped in that case.
    bool refineAnchor(ElevationManager& elevMgr, glm::dvec2 origin, double scale,
                      const AnchorOccupancyGrid* occupancyGrid);

    // Sets a *preferred fallback* position for refineAnchor() (see its doc comment above):
    // this label will prefer stacking directly under _target when that's cost-competitive
    // with the best independent compass anchor, but is not required to. Currently only used
    // for the peak elevation sub-label (targeting the name label) -- see
    // PointStyleBuilder::addFeature. Does not affect m_relative/occlusion cascading.
    void setStackFallbackTarget(Label& _target) { m_stackFallbackTarget = &_target; }

    // Real use-after-free bug found during Phase 7 verification (headless launch: garbage
    // anchor indices, occasionally a full SIGSEGV): a dead label is dropped -- and its
    // unique_ptr destroyed -- by TextStyleBuilder::build()'s "Add only alive labels" pass,
    // but nothing invalidated any OTHER label's m_stackFallbackTarget still pointing at it
    // (there's no cascade between siblings via this pointer -- see setStackFallbackTarget()'s
    // own comment: "does not affect m_relative/occlusion cascading" -- so a text2/elevation
    // label with `optional: true` and its own repeatGroup opted out (see stylus-osm.yaml's
    // peak draw rule) routinely survives a tile-build-time repeat-group/priority cut that
    // kills its target, the name label, in the very same pass). refineAnchor() then
    // dereferences the dangling pointer on some LATER frame (whenever the elevation mosaic
    // finally loads), reading freed/reused memory. Called from build() right before it drops
    // a dead label -- see build()'s own comment for why an O(n) scan per drop is fine here.
    void clearStackFallbackTargetIfEquals(const Label* _target) {
        if (m_stackFallbackTarget == _target) { m_stackFallbackTarget = nullptr; }
    }

    // Pure decision logic factored out of refineAnchor() so it's unit-testable without a
    // real ElevationManager/RasterSource (see tests/unit/labelTests.cpp): given a combined
    // cost per candidate anchor (same indexing as Options::anchors, lower = better), returns
    // the permutation of [0, costs.size()) that stably sorts anchors by ascending cost --
    // ties keep their original relative order, so an unmodified/all-tied cost list is a
    // no-op (preserves whatever base order -- e.g. the Imhof/Yoeli default, see
    // TextStyleBuilder::applyRule -- the caller started with).
    static std::vector<int> sortAnchorIndicesByCost(const std::vector<float>& _costs);

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
    // docs/label-placement-plan.md Task 4: mirrors m_prominenceRefined's "retry every frame
    // until success, then stop" pattern for refineAnchor().
    bool m_anchorRefined = false;
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
    // docs/label-placement-plan.md Task 1 (2026-07-16 redesign): see
    // setStackFallbackTarget()/refineAnchor()'s doc comments -- read at most once, inside
    // refineAnchor(), never stored anywhere else or dereferenced again afterward.
    Label* m_stackFallbackTarget = nullptr;

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
            hash_combine(seed, o.refineAnchorWithTextureShading);
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
