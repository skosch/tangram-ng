#include "catch.hpp"
#include "labels/label.h"
#include "labels/labelCollider.h"
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
// anchor away from a strong ridge/canyon texture-shading response, main-thread only, once
// the elevation mosaic is available. The sampling itself needs a real ElevationManager/
// RasterSource (heavy to construct in a unit test, same reasoning the Frozen Interface
// Contract gives for keeping sampleTextureShading's pure core separately testable in
// rasterMosaicTests.cpp) -- so the switch-or-not *decision* is factored out into
// Label::pickBetterAnchorIndex(), tested directly here against hand-built cost arrays.
TEST_CASE( "Label::pickBetterAnchorIndex", "[Core][Label]" ) {
    SECTION( "keeps current anchor when it's already the best" ) {
        std::vector<float> costs = {0.1f, 0.5f, 0.5f, 0.5f};
        REQUIRE(Label::pickBetterAnchorIndex(costs, 0, 0.08f) == -1);
    }

    SECTION( "switches to a materially better anchor" ) {
        std::vector<float> costs = {0.6f, 0.05f, 0.5f, 0.5f};
        REQUIRE(Label::pickBetterAnchorIndex(costs, 0, 0.08f) == 1);
    }

    SECTION( "does not switch for a marginal, sub-threshold improvement" ) {
        std::vector<float> costs = {0.30f, 0.25f, 0.5f, 0.5f}; // 0.05 apart, threshold is 0.08
        REQUIRE(Label::pickBetterAnchorIndex(costs, 0, 0.08f) == -1);
    }

    SECTION( "out-of-range current index is handled safely" ) {
        std::vector<float> costs = {0.1f, 0.2f};
        REQUIRE(Label::pickBetterAnchorIndex(costs, -1, 0.08f) == -1);
        REQUIRE(Label::pickBetterAnchorIndex(costs, 5, 0.08f) == -1);
    }

    SECTION( "empty cost list is handled safely" ) {
        std::vector<float> costs;
        REQUIRE(Label::pickBetterAnchorIndex(costs, 0, 0.08f) == -1);
    }
}
