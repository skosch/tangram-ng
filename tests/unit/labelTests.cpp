#include "catch.hpp"
#include "labels/label.h"
#include "labels/labelCollider.h"
#include "labels/labelManager.h"
#include "labels/screenTransform.h"
#include "labels/textLabel.h"
#include "labels/textLabels.h"
#include "map.h"
#include "style/textStyle.h"
#include "tile/tileID.h"
#include "view/view.h"

#include "glm/mat4x4.hpp"
#include "glm/gtc/matrix_transform.hpp"

#define EPSILON 0.00001

using namespace Tangram;

glm::vec2 screenSize(500.f, 500.f);
TextStyle dummyStyle("textStyle");
TextLabels dummy(dummyStyle);
Label::AABB bounds(0.f, 0.f, 500.f, 500.f);

struct TestTransform {
    ScreenTransform::Buffer buffer;
    Range range;
    ScreenTransform transform;
    TestTransform() : transform(buffer, range) {}
};

TextLabel makeLabel(glm::vec2 _transform, Label::Type _type) {
    Label::Options options;
    options.offset = {0.0f, 0.0f};
    options.anchors.anchor[0] = LabelProperty::Anchor::center;
    options.anchors.count = 1;
    options.showTransition.time = 0.2;
    options.hideTransition.time = 0.2;

    TextRange textRanges;

    return TextLabel({{glm::vec3(_transform, 0)}}, _type, options,
            {}, {0, 0}, dummy, textRanges,
            TextLabelProperty::Align::none);
}

View makeView() {
    View view(256, 256);
    view.setConstrainToWorldBounds(false);
    view.setPosition(0, 0);
    view.setZoom(0);
    view.update();

    return view;
}

TEST_CASE( "Ensure the transition from wait -> sleep when occlusion happens", "[Core][Label]" ) {
    View view = makeView();
    TestTransform t1, t2;

    TextLabel l(makeLabel({screenSize/2.f}, Label::Type::point));

    REQUIRE(l.state() == Label::State::none);
    l.update(glm::ortho(0.f, screenSize.x, screenSize.y, 0.f, -1.f, 1.f), view.state(), &bounds, t1.transform);

    REQUIRE(l.state() != Label::State::sleep);
    REQUIRE(l.state() == Label::State::none);
    REQUIRE(l.canOcclude());

    l.update(glm::ortho(0.f, screenSize.x, screenSize.y, 0.f, -1.f, 1.f), view.state(), &bounds, t2.transform);
    l.occlude(true);
    l.evalState(0);

    REQUIRE(l.state() == Label::State::sleep);
}

TEST_CASE( "Ensure the transition from wait -> visible when no occlusion happens", "[Core][Label]" ) {
    View view = makeView();
    TestTransform t1, t2;

    TextLabel l(makeLabel({screenSize/2.f}, Label::Type::point));

    REQUIRE(l.state() == Label::State::none);

    l.update(glm::ortho(0.f, screenSize.x, screenSize.y, 0.f, -1.f, 1.f), view.state(), &bounds, t1.transform);
    l.occlude(false);
    l.evalState(0);

    REQUIRE(l.state() == Label::State::fading_in);
    REQUIRE(l.canOcclude());

    l.update(glm::ortho(0.f, screenSize.x, screenSize.y, 0.f, -1.f, 1.f), view.state(), &bounds, t2.transform);
    l.evalState(1.f);

    REQUIRE(l.state() == Label::State::visible);
    REQUIRE(l.canOcclude());
}

TEST_CASE( "Ensure the end state after occlusion is leep state", "[Core][Label]" ) {
    View view = makeView();
    TestTransform t1, t2;

    TextLabel l(makeLabel({screenSize/2.f}, Label::Type::point));

    l.update(glm::ortho(0.f, screenSize.x, screenSize.y, 0.f, -1.f, 1.f), view.state(), &bounds, t1.transform);
    l.occlude(false);
    l.evalState(0);

    REQUIRE(l.state() == Label::State::fading_in);
    REQUIRE(l.canOcclude());

    l.update(glm::ortho(0.f, screenSize.x, screenSize.y, 0.f, -1.f, 1.f), view.state(), &bounds, t2.transform);
    l.occlude(true);
    l.evalState(1.f);

    REQUIRE(l.state() == Label::State::sleep);
    REQUIRE(l.canOcclude());
}

TEST_CASE( "Ensure the sleep transition for out of screen labels", "[Core][Label]" ) {
    View view = makeView();
    TestTransform t1, t2;

    TextLabel l(makeLabel({screenSize*4.f}, Label::Type::point));

    REQUIRE(l.state() == Label::State::none);

    l.update(glm::ortho(0.f, screenSize.x, screenSize.y, 0.f, -1.f, 1.f), view.state(), &bounds, t1.transform);

    REQUIRE(l.state() == Label::State::sleep);
    REQUIRE(l.canOcclude());

    l.update(glm::ortho(0.f, screenSize.x * 4.f, screenSize.y * 4.f, 0.f, -1.f, 1.f), view.state(), &bounds, t2.transform);
    l.evalState(0);
    REQUIRE(l.state() != Label::State::none);

    REQUIRE(l.state() == Label::State::fading_in);
    REQUIRE(l.canOcclude());
}

TEST_CASE( "Ensure debug labels are always visible and cannot occlude", "[Core][Label]" ) {
    View view = makeView();

    TextLabel l(makeLabel({screenSize/2.f}, Label::Type::debug));

    REQUIRE(l.state() == Label::State::visible);
    REQUIRE(!l.canOcclude());

    TestTransform t1;
    l.update(glm::ortho(0.f, screenSize.x, screenSize.y, 0.f, -1.f, 1.f), view.state(), &bounds, t1.transform);
    l.evalState(1.f);

    REQUIRE(l.state() == Label::State::visible);
    REQUIRE(!l.canOcclude());
}

TEST_CASE( "Linear interpolation", "[Core][Label][Fade]" ) {
    FadeEffect fadeOut(false, FadeEffect::Interpolation::linear, 1.0);

    REQUIRE(fadeOut.update(0.0) == 1.0);
    REQUIRE(fadeOut.update(0.5) == 0.5);
    REQUIRE(fadeOut.update(0.5) == 0.0);

    fadeOut.update(0.01);

    REQUIRE(fadeOut.isFinished());

    FadeEffect fadeIn(true, FadeEffect::Interpolation::linear, 1.0);

    REQUIRE(fadeIn.update(0.0) == 0.0);
    REQUIRE(fadeIn.update(0.5) == 0.5);
    REQUIRE(fadeIn.update(0.5) == 1.0);

    fadeIn.update(0.01);

    REQUIRE(fadeIn.isFinished());
}

TEST_CASE( "Pow interpolation", "[Core][Label][Fade]" ) {
    FadeEffect fadeOut(false, FadeEffect::Interpolation::pow, 1.0);

    REQUIRE(fadeOut.update(0.0) == 1.0);
    REQUIRE(fadeOut.update(0.5) == 0.75);
    REQUIRE(fadeOut.update(0.5) == 0.0);

    fadeOut.update(0.01);

    REQUIRE(fadeOut.isFinished());

    FadeEffect fadeIn(true, FadeEffect::Interpolation::pow, 1.0);

    REQUIRE(fadeIn.update(0.0) == 0.0);
    REQUIRE(fadeIn.update(0.5) == 0.25);
    REQUIRE(fadeIn.update(0.5) == 1.0);

    fadeIn.update(0.01);

    REQUIRE(fadeIn.isFinished());
}

TEST_CASE( "Sine interpolation", "[Core][Label][Fade]" ) {
    FadeEffect fadeOut(false, FadeEffect::Interpolation::sine, 1.0);

    REQUIRE(std::fabs(fadeOut.update(0.0) - 1.0) < EPSILON);
    REQUIRE(std::fabs(fadeOut.update(1.0) - 0.0) < EPSILON);

    fadeOut.update(0.01);

    REQUIRE(fadeOut.isFinished());

    FadeEffect fadeIn(true, FadeEffect::Interpolation::sine, 1.0);

    REQUIRE(std::fabs(fadeIn.update(0.0) - 0.0) < EPSILON);
    REQUIRE(std::fabs(fadeIn.update(1.0) - 1.0) < EPSILON);

    fadeIn.update(0.01);

    REQUIRE(fadeIn.isFinished());
}

// docs/label-placement-plan.md Task 3: peaks sharing a draw rule get an implicit repeatGroup
// and repeatDistance (one tile width by default, see TextStyleBuilder::applyRule). Two peaks
// within that distance are pre-emptively resolved by LabelCollider::process() at tile-build
// time, using whatever priority is available *then* (provisional/elevation-only for peaks
// without a real OSM prominence tag -- the texture-shading-refined priority only exists later,
// via Label::refinePriority() on the main thread). This locks down that the suppression
// radius really is controlled by Options::repeatDistance (screen pixels, in the normal
// per-tile code path LabelCollider::process() takes -- the tileSize=0 branch of
// filterRepeatGroups, distance measured via Label::screenCenter()), so retuning the peak draw
// rule's repeat_distance in stylus-osm.yaml has the predictable effect the fix relies on.
std::unique_ptr<Label> makeRepeatGroupLabel(glm::vec2 _pos, float _priority, size_t _repeatGroup,
                                            float _repeatDistance) {
    Label::Options options;
    options.offset = {0.0f, 0.0f};
    options.anchors.anchor[0] = LabelProperty::Anchor::center;
    options.anchors.count = 1;
    options.priority = _priority;
    options.repeatGroup = _repeatGroup;
    options.repeatDistance = _repeatDistance;

    TextRange textRanges;

    return std::make_unique<TextLabel>(TextLabel::Coordinates{{_pos, _pos}}, Label::Type::point,
                                       options, TextLabel::VertexAttributes{}, glm::vec2(1.f, 1.f),
                                       dummy, textRanges, TextLabelProperty::Align::none);
}

TEST_CASE( "LabelCollider repeat-group suppression radius tracks repeatDistance", "[Core][LabelCollider]" ) {
    // Two same-repeatGroup labels a fixed distance apart on screen; only repeatDistance
    // varies between sections. TileID(0,0,10) with a real tileSize keeps LabelCollider's
    // internal isect2d grid happy without needing thousands of labels.
    TileID tileID(0, 0, 10);

    SECTION( "closer than repeatDistance: lower-priority label is suppressed" ) {
        LabelCollider collider;
        std::vector<std::unique_ptr<Label>> labels;
        // priority is "lower numeric value wins" (see labelCollider.cpp's priorityComparator).
        labels.push_back(makeRepeatGroupLabel({0.5f, 0.5f}, 0.1f, 1, 200.f));
        labels.push_back(makeRepeatGroupLabel({0.55f, 0.5f}, 0.2f, 1, 200.f));

        collider.addLabels(labels);
        collider.process(tileID, 1.0f, 256.f);

        REQUIRE(labels[0]->state() != Label::State::dead);
        REQUIRE(labels[1]->state() == Label::State::dead);
    }

    SECTION( "farther than repeatDistance: both labels survive" ) {
        LabelCollider collider;
        std::vector<std::unique_ptr<Label>> labels;
        labels.push_back(makeRepeatGroupLabel({0.1f, 0.5f}, 0.1f, 1, 20.f));
        labels.push_back(makeRepeatGroupLabel({0.9f, 0.5f}, 0.2f, 1, 20.f));

        collider.addLabels(labels);
        collider.process(tileID, 1.0f, 256.f);

        REQUIRE(labels[0]->state() != Label::State::dead);
        REQUIRE(labels[1]->state() != Label::State::dead);
    }
}

// docs/label-placement-plan.md Task 4: Label::refineAnchor() steers a peak name label's
// anchor away from both a strong ridge/canyon texture-shading response and nearby vector
// clutter (water/road/trail), main-thread only, once the elevation mosaic is available. The
// sampling itself needs a real ElevationManager/RasterSource (heavy to construct in a unit
// test, same reasoning the Frozen Interface Contract gives for keeping
// sampleTextureShading's pure core separately testable in rasterMosaicTests.cpp) -- so the
// sort *decision* is factored out into Label::sortAnchorIndicesByCost(), tested directly
// here against hand-built cost arrays.
TEST_CASE( "Label::sortAnchorIndicesByCost", "[Core][Label]" ) {
    SECTION( "already-sorted costs come back unchanged" ) {
        std::vector<float> costs = {0.1f, 0.2f, 0.3f, 0.4f};
        std::vector<int> order = Label::sortAnchorIndicesByCost(costs);
        REQUIRE(order == std::vector<int>{0, 1, 2, 3});
    }

    SECTION( "reorders by ascending cost" ) {
        std::vector<float> costs = {0.6f, 0.05f, 0.5f, 0.1f};
        std::vector<int> order = Label::sortAnchorIndicesByCost(costs);
        REQUIRE(order == std::vector<int>{1, 3, 2, 0});
    }

    SECTION( "ties keep their original relative order (stable sort)" ) {
        // Mirrors an all-tied-cost list (e.g. no occupancy grid, flat texture shading):
        // the caller's original order (e.g. the Imhof/Yoeli default) must survive intact.
        std::vector<float> costs = {0.5f, 0.5f, 0.5f, 0.5f};
        std::vector<int> order = Label::sortAnchorIndicesByCost(costs);
        REQUIRE(order == std::vector<int>{0, 1, 2, 3});
    }

    SECTION( "empty cost list is handled safely" ) {
        std::vector<float> costs;
        std::vector<int> order = Label::sortAnchorIndicesByCost(costs);
        REQUIRE(order.empty());
    }

    SECTION( "single candidate is a no-op" ) {
        std::vector<float> costs = {0.42f};
        std::vector<int> order = Label::sortAnchorIndicesByCost(costs);
        REQUIRE(order == std::vector<int>{0});
    }
}

// docs/label-placement-plan.md round 7 (corrected): Label::refinePriority()'s "isolation" term
// compares a peak's own elevation against the current viewport's max elevation (computed once
// per frame by LabelManager::updateLabels() from already-resident tiles, not derivable here
// without a real ElevationManager/RasterSource/Scene), turning that comparison into a score via
// this pure, directly testable function -- same reasoning as sortAnchorIndicesByCost above.
TEST_CASE( "Label::isolationScore", "[Core][Label]" ) {
    SECTION( "peak at or above the regional max scores 1.0" ) {
        REQUIRE(Label::isolationScore(4478.0, 4478.f, 300.f) == 1.f);
        REQUIRE(Label::isolationScore(4478.0, 4000.f, 300.f) == 1.f);
    }

    SECTION( "peak well below the regional max scores 0.0" ) {
        REQUIRE(Label::isolationScore(3000.0, 4478.f, 300.f) == 0.f);
    }

    SECTION( "ramps linearly across the margin" ) {
        // 150m below the max, with a 300m margin -> halfway up the ramp.
        REQUIRE(Label::isolationScore(4328.0, 4478.f, 300.f) == Approx(0.5f));
    }

    SECTION( "non-positive margin degenerates to a hard step at the regional max" ) {
        REQUIRE(Label::isolationScore(4478.0, 4478.f, 0.f) == 1.f);
        REQUIRE(Label::isolationScore(4477.9, 4478.f, 0.f) == 0.f);
        REQUIRE(Label::isolationScore(4477.9, 4478.f, -10.f) == 0.f);
    }
}

// docs/label-placement-v2-spec.md's "Deterministic-but-not-jittery collision resolution":
// the single most important new test this redesign asks for -- the final winner between
// two competing labels must not depend on incidental frame-history (which one happened to
// render, or win a single frame, first during initial tile load), only on a real track
// record. LabelManager::LabelEntry/priorityComparator are public specifically so this test
// can construct them directly, matching LabelCollider's own already-public interface used
// in the repeat-group test above.
std::unique_ptr<Label> makePriorityLabel(glm::vec2 _pos, float _priority) {
    Label::Options options;
    options.offset = {0.0f, 0.0f};
    options.anchors.anchor[0] = LabelProperty::Anchor::center;
    options.anchors.count = 1;
    options.priority = _priority;

    TextRange textRanges;

    return std::make_unique<TextLabel>(TextLabel::Coordinates{{_pos, _pos}}, Label::Type::point,
                                       options, TextLabel::VertexAttributes{}, glm::vec2(1.f, 1.f),
                                       dummy, textRanges, TextLabelProperty::Align::none);
}

TEST_CASE( "LabelManager::priorityComparator is deterministic, not history-dependent", "[Core][LabelManager]" ) {
    View view = makeView();
    auto mvp = glm::ortho(0.f, screenSize.x, screenSize.y, 0.f, -1.f, 1.f);

    auto better = makePriorityLabel({100.f, 100.f}, 14.1f); // lower numeric priority wins
    auto worse = makePriorityLabel({200.f, 200.f}, 14.5f);

    TestTransform tb, tw;
    better->update(mvp, view.state(), &bounds, tb.transform);
    worse->update(mvp, view.state(), &bounds, tw.transform);

    LabelManager::LabelEntry eBetter(better.get(), nullptr, nullptr, nullptr, false, {});
    LabelManager::LabelEntry eWorse(worse.get(), nullptr, nullptr, nullptr, false, {});

    SECTION( "neither label has a track record yet: score decides despite a single lucky frame" ) {
        // Simulate the exact old bug: "worse" wins a single frame (e.g. it happened to
        // render first during load) -- occludedLastFrame() alone used to be sufficient to
        // lock this in, permanently, regardless of score. It shouldn't be anymore:
        // stableFrames() only reaches the threshold after several CONSECUTIVE wins, so a
        // single win must not be enough to override score.
        worse->occlude(false);
        worse->evalState(0);
        worse->update(mvp, view.state(), &bounds, tw.transform);

        better->occlude(true);
        better->evalState(0);
        better->update(mvp, view.state(), &bounds, tb.transform);

        REQUIRE(worse->stableFrames() < 5);
        REQUIRE(better->stableFrames() < 5);
        REQUIRE(LabelManager::priorityComparator(eBetter, eWorse));
        // Re-running with the SAME fixed inputs must give the SAME answer -- no hidden
        // state left over from evaluating the comparator itself.
        REQUIRE(LabelManager::priorityComparator(eBetter, eWorse));
    }

    SECTION( "a label with a real multi-frame track record keeps winning against a technically-better newcomer" ) {
        for (int i = 0; i < 6; i++) {
            worse->occlude(false);
            worse->evalState(0);
            worse->update(mvp, view.state(), &bounds, tw.transform);
        }
        REQUIRE(worse->stableFrames() >= 5);
        REQUIRE_FALSE(LabelManager::priorityComparator(eBetter, eWorse));
    }
}

TEST_CASE("Dominance gate: far peaks need more dominance, near ones little", "[Core][Label]") {
    // ratio 0.08, need = 0.08 * d * d / (d + 10 km)
    // 1 km summit: needs ~1067 m at 20 km, ~926 m at 18 km
    REQUIRE(Label::passesDominanceGate(1000.f, 18000.f, 0.08f, false));
    REQUIRE_FALSE(Label::passesDominanceGate(1000.f, 20000.f, 0.08f, false));
    // a 200 m knoll is fine at 5 km (needs ~133 m) but not at 7 km (~230 m)
    REQUIRE(Label::passesDominanceGate(200.f, 5000.f, 0.08f, false));
    REQUIRE_FALSE(Label::passesDominanceGate(200.f, 7000.f, 0.08f, false));
    // disabled gate always passes; unmeasured dominance never does
    REQUIRE(Label::passesDominanceGate(-1.f, 1e6f, 0.f, false));
    REQUIRE_FALSE(Label::passesDominanceGate(-1.f, 100.f, 0.08f, false));
}

TEST_CASE("Dominance gate: hysteresis keeps a shown peak a little past the threshold", "[Core][Label]") {
    // need ~1067 at 20 km; a shown label survives down to ~907, a hidden one needs 1067
    REQUIRE(Label::passesDominanceGate(950.f, 20000.f, 0.08f, true));
    REQUIRE_FALSE(Label::passesDominanceGate(950.f, 20000.f, 0.08f, false));
    REQUIRE_FALSE(Label::passesDominanceGate(850.f, 20000.f, 0.08f, true));
}
