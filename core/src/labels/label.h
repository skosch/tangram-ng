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
#include <atomic>
#include <cstdint>
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
    //
    // Round 7 (2026-07-16, user report: Matterhorn -- by far the tallest peak in view --
    // still losing to neighbors of similar LOCAL curvature): the shade sample above only sees
    // a few hundred meters of context (the tile's own 3x3 mosaic), so it can't tell "the one
    // peak that dominates this whole valley" from "one of several similarly jagged
    // neighbors" -- it has no notion of actual regional height at all. This adds a
    // **viewport-relative** isolation term: compare this peak's own elevation against
    // viewportMaxElev, the highest elevation found anywhere in the CURRENT VIEWPORT --
    // computed once per frame by LabelManager::updateLabels() from tiles already resident for
    // rendering (see its own comment), not a fixed real-world search radius (which would need
    // new, unbuilt tile-fetch infrastructure and was rejected as overkill). This naturally
    // scales with zoom for free: zoomed out, the viewport spans a huge area, so only a truly
    // regionally-dominant peak scores well; zoomed in, it's a small area and local peaks
    // compete fairly -- exactly "relevant to current zoom level" without needing to explicitly
    // grow a search radius.
    //
    // Round 7 correction (2026-07-16, same day): the FIRST round-7 implementation instead had
    // each peak independently climb to a speculative coarser "ancestor" tile via
    // ElevationManager::getMinMaxElev()'s cache-only ancestor-climb, gambling that something
    // coarser than the current view's own zoom happened to already be resident. It never was
    // -- elevation tiles are only ever fetched matching the vector tile's own zoom (confirmed
    // via the `stopZoomByMaxZoom:elevation` profiler counter), so that ancestor tile was
    // reliably a 100% cache miss and the whole isolation term was silently a no-op the entire
    // time (found via temporary instrumentation after the user reported "absurdly slow to
    // load" led to profiling, which surfaced a real, unrelated, much bigger perf bug --
    // see textureShading.cpp's sampleTextureShading() pyramid cache -- and, while verifying
    // that fix, this dead-isolation-term bug too). Passing the already-computed viewport max
    // in directly, rather than each peak re-deriving its own possibly-nonexistent ancestor,
    // both fixes this and is cheaper (computed once per frame, not once per peak).
    //
    // Best-effort/additive throughout: if haveViewportMaxElev is false (no tile in view has
    // elevation data -- flat/non-terrain map style) or this peak's own elevation isn't
    // available, isolation contributes 0 (falls back to today's shade-only behavior) rather
    // than blocking finalization -- unlike the shade sample above, isolation is never a hard
    // requirement. Note this term is evaluated once, at whatever the viewport's max elevation
    // happened to be when refinement first succeeds, and never revisited after -- the same
    // one-time-snapshot behavior the shade term above already has (see the Frozen Interface
    // Contract in docs/texture-shading-plan.md: priority must not change as the camera moves).
    bool refinePriority(ElevationManager& elevMgr, glm::dvec2 origin, double scale,
                        float viewportMaxElev, bool haveViewportMaxElev);

    // Pure, unit-testable helper backing refinePriority()'s isolation term above. 1.0 when
    // peakElev is at or above regionMaxElev (this peak IS the highest point found in the
    // current viewport), ramping linearly down to 0.0 by the time peakElev is marginMeters
    // below regionMaxElev, and 0.0 beyond that. marginMeters <= 0 degenerates to a hard step
    // at regionMaxElev.
    static float isolationScore(double peakElev, float regionMaxElev, float marginMeters);

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
    // Revised 2026-07-16 (user report: "elevations are still too far away from the peak
    // dots sometimes -- their distance seems to be sometimes very close, sometimes quite
    // far"): this used to ALSO score a "stack directly under a target label" candidate
    // (setStackFallbackTarget(), removed) for the peak elevation sub-label. That computed
    // its offset from the TARGET's own dimension/position, so the effective icon-to-label
    // distance varied with the target's text length/wrapping instead of staying constant --
    // exactly the reported inconsistency. Removed entirely: every label now always uses
    // this same icon-relative formula (icon radius + own half-extent + anchorGapScale gap),
    // so two sibling labels of the same icon (e.g. a peak's name and elevation) are always
    // the same fixed distance from it, regardless of which anchors they end up on. A
    // "stacked under the name" look can still happen when both independently pick
    // vertically-adjacent anchors, but it's no longer a distinct mode with its own offset
    // math -- one less pointer relationship to keep safe (this was also the second
    // use-after-free source found this session, via the target-tracking mechanism this
    // replaced).
    //
    // The normal per-frame LabelManager::handleOcclusions() collision pass that runs right
    // after this call re-validates the new position exactly like any other anchor (and
    // falls back further via nextAnchor() if it turns out to newly collide with something),
    // so no separate occlusion re-check needs to be triggered here. `_occupancyGrid` may be
    // null (e.g. a marker label, or a tile with no water/transportation geometry) -- the
    // vector-clutter term is simply skipped in that case.
    bool refineAnchor(ElevationManager& elevMgr, glm::dvec2 origin, double scale,
                      const AnchorOccupancyGrid* occupancyGrid);

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

    // Deterministic creation-order serial, used as LabelManager::priorityComparator's final
    // tiebreak instead of comparing `this` pointers directly. Label addresses depend on heap
    // layout/allocation order, which is not consistent across builds -- e.g. `sizeof(Label)`
    // itself differs between Debug and Release (the DEBUG-only `debugTag` member below), and
    // -O0 vs -O2 codegen allocates everything else differently too. Two same-style peak labels
    // (identical hash()) with tied priority -- the common case before texture-shading refinement
    // completes, or when the proxy genuinely can't separate two similarly rugged summits, see
    // docs/label-placement-plan.md's Task 4c Matterhorn analysis -- fell through to that address
    // compare, so which one won the screen space (and then stayed, via the occludedLastFrame
    // hysteresis) was silently build-dependent. This serial is independent of both.
    uint32_t id() const { return m_id; }

    glm::vec2 dimension() const { return m_dim; }

    // Gets for label options: color and offset
    const Options& options() const { return m_options; }

    // Adds the oriented bounding boxes of the label to _obbs, updates Range
    virtual void obbs(ScreenTransform& _transform, OBBBuffer& _obbs) const = 0;

    State state() const { return m_state; }

    bool isOccluded() const { return m_occluded; }

    bool occludedLastFrame() const { return m_occludedLastFrame; }

    // docs/label-placement-v2-spec.md's "Deterministic-but-not-jittery collision
    // resolution", Option A: consecutive frames (as of the end of the last completed
    // handleOcclusions()) this label has been continuously visible/unoccluded, reset to 0
    // the instant it's occluded. Unlike occludedLastFrame() alone (which just asks "did I
    // win the SINGLE previous frame" -- true after winning by chance even once, including
    // during initial unsettled load), this tracks a track record, so
    // LabelManager::priorityComparator() can require a label to have actually PROVEN
    // itself over several consecutive frames before its history gets to outweigh a
    // competitor's tier/score -- see that function's own comment for the full reasoning.
    int stableFrames() const { return m_stableFrames; }

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

    // Backs m_id's default member initializer; a plain (not thread_local) atomic counter shared
    // by every Label subclass, since Labels are constructed from multiple TileWorker threads
    // concurrently. Only relative order matters, so relaxed ordering is sufficient.
    static uint32_t nextId() {
        static std::atomic<uint32_t> s_nextId{0};
        return s_nextId.fetch_add(1, std::memory_order_relaxed);
    }

    const Type m_type;
    const glm::vec2 m_dim;
    // Monotonic construction-order serial backing id() above; assigned once here so every
    // subclass constructor gets one for free.
    const uint32_t m_id = nextId();

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
    int m_stableFrames = 0;

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
