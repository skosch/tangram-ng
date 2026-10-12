#pragma once

#include "data/properties.h"
#include "labels/label.h"
#include "labels/screenTransform.h"
#include "labels/spriteLabel.h"
#include "tile/tileID.h"

#include "glm_vec.h" // for isect2d.h
#include "isect2d.h"

#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <unordered_map>
#include <vector>

namespace Tangram {

class FontContext;
class LabelSet;
class Marker;
class Tile;
class Style;
class Scene;
class TileManager;

class LabelManager {

public:
    LabelManager();

    virtual ~LabelManager();

    void drawDebug(RenderState& rs, const View& _view);

    // onlyRender: when the view and tiles have not changed one does not need to update the set of
    //  active labels. We just need to render these the labels in this case
    void updateLabelSet(const View& _view, float _dt, const Scene& _scene,
                        const std::vector<std::shared_ptr<Tile>>& _tiles,
                        const std::vector<std::unique_ptr<Marker>>& _markers,
                        bool _onlyRender);

    bool needUpdate() const { return m_needUpdate; }

    std::pair<Label*, const Tile*> getLabel(uint32_t _selectionColor) const;

    // Public (not just for LabelManager's own internal use) so unit tests can construct
    // fixed label sets and exercise the deterministic collision-resolution tiebreak
    // directly -- see docs/label-placement-v2-spec.md's "add a unit test asserting...
    // handleOcclusions() produces the same winner regardless of frame-history state,"
    // and tests/unit/labelTests.cpp's priorityComparator determinism test.
    struct LabelEntry {

        LabelEntry(Label* _label, Style* _style, const Tile* _tile, const Marker* _marker,
                   bool _proxy, Range _screenTransform)
            : label(_label),
              style(_style),
              tile(_tile),
              marker(_marker),
              priority(_label->options().priority),
              proxy(_proxy),
              transformRange(_screenTransform) {}

        Label* label;
        Style* style;
        const Tile* tile;
        const Marker* marker;
        float priority;
        bool proxy;

        Range transformRange;
        Range obbsRange;
    };

    static bool priorityComparator(const LabelEntry& _a, const LabelEntry& _b);

protected:

    using AABB = isect2d::AABB<glm::vec2>;
    using OBB = isect2d::OBB<glm::vec2>;
    using CollisionPairs = std::vector<isect2d::ISect2D<glm::vec2>::Pair>;


    void skipTransitions(const Scene& _scene, const std::vector<std::shared_ptr<Tile>>& _tiles,
                         TileManager& _tileManager, float _currentZoom) const;

    void skipTransitions(const std::vector<const Style*>& _styles, Tile& _tile, Tile& _proxy) const;

    void handleOcclusions(const ViewState& _viewState, bool _hideExtraLabels = false);

    bool withinRepeatDistance(Label *_label);

    void updateLabels(const ViewState& _viewState, float _dt, const Scene& _scene,
                      const std::vector<std::shared_ptr<Tile>>& _tiles,
                      const std::vector<std::unique_ptr<Marker>>& _markers,
                      bool _onlyRender);

    void processLabelUpdate(const ViewState& _viewState, const LabelSet* _labelSet, Style* _style,
                            const Tile* _tile, const Marker* _marker, ElevationManager* _elevManager,
                            float _dt, bool _onlyRender, float _viewportMaxElev, bool _haveViewportMaxElev);

    // Lazily computed highest elevation among this frame's resident tiles (see
    // ensureViewportMax()); m_vmaxTiles is non-null only while updateLabels() runs.
    void ensureViewportMax(ElevationManager& _elevManager);
    void flushProfileCounters(bool _onlyRender);
    const std::vector<std::shared_ptr<Tile>>* m_vmaxTiles = nullptr;
    bool m_vmaxComputed = false;
    bool m_vmaxHave = false;
    float m_vmaxValue = 0.f;

    bool m_needUpdate;

    isect2d::ISect2D<glm::vec2> m_isect2d;

    static bool zOrderComparator(const LabelEntry& _a, const LabelEntry& _b);

    std::vector<OBB> m_obbs;
    ScreenTransform::Buffer m_transforms;

    std::vector<LabelEntry> m_labels;
    std::vector<LabelEntry> m_selectionLabels;

    std::unordered_map<size_t, std::vector<Label*>> m_repeatGroups;

    float m_lastZoom;
    // view state for last label update;
    glm::mat4 m_lastViewProj;
    glm::dvec2 m_lastViewPos;

    // docs/label-placement-plan.md round 9 / docs/label-placement-v2-spec.md's anchor-
    // selection revisit (round 15): Label::refineAnchor() samples up to ~45 points per
    // label (5 per candidate anchor x up to 9 anchors); with thousands of peak candidates
    // newly eligible in the same frame (e.g. the frame an elevation mosaic first becomes
    // available over a cluttered view), doing all of them in one frame caused multi-second
    // to (in a Debug build) multi-minute single-frame stalls even after
    // textureShading.cpp's per-mosaic cache (round 8) made each individual sample cheap --
    // the sheer COUNT of one-time refinements due in a single frame was still too much.
    // Round 9 capped this by a fixed LABEL COUNT per frame (kMaxAnchorRefinementsPerFrame)
    // -- but the real cost is wall-clock time, and a fixed count behaves wildly
    // differently between an -O0 Debug build and an -O2 Release build (the same 150-label
    // budget that's imperceptible in Release could still stall Debug badly), needing two
    // separately-tuned constants. Round 15 replaced the count with a measured wall-clock
    // budget (kAnchorRefineBudgetMs, labelManager.cpp) -- tracks actual time spent inside
    // refineAnchor() calls this frame (not overall frame time), reset at the start of
    // every updateLabels() call, checked in processLabelUpdate() before issuing each new
    // call. Once the budget is spent, remaining not-yet-refined labels for this frame keep
    // their current (unrefined/provisional or previously-refined) anchor and are retried
    // next frame, same "retry every frame until success" idiom m_anchorRefined already
    // uses -- this only spreads WHEN the one-time refinement happens across a few more
    // frames, it does not change the final result. The same tuned millisecond value now
    // behaves sanely in both Debug and Release without separate constants.
    double m_anchorRefineTimeUsedMs = 0.0;

    // Round 16 (Sebastian, real trace: LabelsCollect spiking to 1.79s single-frame stalls,
    // zoom 12 over a peak-dense Alps view): refinePriority() had NO budget at all, unlike
    // refineAnchor() above - despite calling into the exact same sampleTextureShading(),
    // whose own single-slot per-mosaic cache (textureShading.cpp round 8) assumes
    // consecutive calls mostly target the same tile's mosaic. That holds within one tile's
    // labels (processLabelUpdate() is called per-tile), but the frame that newly streams in
    // many tiles at once (a big pan/zoom, or first load) forces one full pyramid rebuild
    // (~600K-element decode + kMaxLevels box-filter passes) per tile, for every one of
    // potentially 1000+ peak candidates in a cluttered alpine view, every single frame until
    // each succeeds - exactly the "thousands of candidates newly eligible in the same frame"
    // scenario the comment above already documents refineAnchor needing protection from.
    // Same measured-wall-clock-budget idiom, not a fixed count, for the same
    // Debug-vs-Release reason.
    double m_priorityRefineTimeUsedMs = 0.0;

    // One-slot per-frame memo of ElevationManager::hasTile() for the tile currently being
    // walked by updateLabels() (see processLabelUpdate()); reset each frame.
    // Per-frame profiling accumulators (only filled while a Profiler capture is active), see
    // ProfileAccum in labelManager.cpp.
    enum { kProfUpdate, kProfSetElev, kProfSync, kProfDepth, kProfViewportMax, kProfIterated, kProfCount };
    double m_prof[kProfCount] = {};
    bool m_profOn = false;

    const Tile* m_hasElevTile = nullptr;
    bool m_hasElevValue = false;
};

}

