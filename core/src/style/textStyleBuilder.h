#pragma once

#include "labels/textLabel.h"
#include "labels/labelProperty.h"
#include "style/textStyle.h"
#include "text/fontContext.h"
#include "util/mapProjection.h"

#include <memory>
#include <vector>
#include <string>

namespace Tangram {

class AnchorOccupancyGrid;

class TextStyleBuilder : public StyleBuilder {

public:

    // Attributes of the prepared Label
    struct LabelAttributes {
        float width = 0;
        float height = 0;
        size_t quadsStart = 0;
        uint32_t fill = 0;
        uint32_t stroke = 0;
        uint8_t fontScale = 0;
        TextRange textRanges = {};
    };

    TextStyleBuilder(const TextStyle& _style);

    const Style& style() const override { return m_style; }

    bool addFeature(const Feature& _feature, const DrawRule& _rule) override;

    void setup(const Tile& _tile) override;
    void setup(const Marker& _marker, int zoom) override;

    std::unique_ptr<StyledMesh> build() override;

    TextStyle::Parameters applyRule(const DrawRule& _rule, const Properties& _props, bool _iconText) const;

    bool prepareLabel(TextStyle::Parameters& _params, Label::Type _type, LabelAttributes& _attributes);

    // Add label to the mesh using the prepared label data
    Label* addLabel(Label::Type _type, TextLabel::Coordinates _coordinates, const TextStyle::Parameters& _params,
                    const LabelAttributes& _attributes, const DrawRule& _rule);

    void addLineTextLabels(const Feature& _feature, const TextStyle::Parameters& _params,
                           const LabelAttributes& _attributes, const DrawRule& _rule);

    bool addStraightTextLabels(const Line& _feature, float _labelWidth,
                               const std::function<void(glm::vec2,glm::vec2)>& _onAddLabel);

    void addCurvedTextLabels(const Line& _feature, const TextStyle::Parameters& _params,
                             const LabelAttributes& _attributes, const DrawRule& _rule);

    // Phase 5 (docs/label-placement-plan.md): for a sufficiently elongated polygon ring,
    // build a CurvedLabel following its principal axis (reusing addCurvedTextLabels).
    // Returns false if the ring isn't a good curve candidate (too compact, degenerate, or
    // the axis couldn't be cleanly clipped to the boundary) -- caller should fall back to
    // a plain centroid point label in that case.
    bool addCurvedPolygonLabel(const Line& _ring, const TextStyle::Parameters& _params,
                               const LabelAttributes& _attributes, const DrawRule& _rule);

    bool handleBoundaryLabel(const Feature& _feat, const DrawRule& _rule,
                             const TextStyle::Parameters& _params);

    bool checkRule(const DrawRule& _rule) const override;
    std::vector<std::unique_ptr<Label>>* labels() { return &m_labels; }

    void addLayoutItems(LabelCollider& _layout) override;

    // Salience-aware anchor ordering (docs/label-placement-plan.md Phase 3). Returns a
    // copy of `_anchors` re-sorted by ascending "cost" -- how visually disruptive it
    // would be to place a label at each candidate anchor direction, via a
    // vector-proximity term (avoid stamping over nearby water/road/trail geometry, via
    // the tile's AnchorOccupancyGrid). A texture-shading term (avoid strong ridge/canyon
    // response) was scoped for this but found architecturally infeasible here at Phase 6
    // integration: sampleTextureShading() (util/textureShading.h) is main-thread only,
    // while this runs on a TileWorker thread, like the rest of addFeature() -- see
    // anchorCandidateCost()'s comment for the full reasoning and the follow-up this
    // leaves open. The result is handed unchanged to the existing anchor-cycling mechanism
    // (Label::nextAnchor(), LabelManager::handleOcclusions) -- no runtime collision code
    // is touched. `_tileLocalPos` is the label/icon anchor position in the same
    // normalized [0,1] tile-local space Feature geometry is expressed in (i.e. what
    // Label::modelCenter() returns for a point label -- see addFeature() call sites).
    // `_labelSizePixels` is the measured text bounding box (LabelAttributes::width/
    // height), `_iconRadiusPixels` is 0 for a standalone text label or half the linked
    // icon's larger dimension for icon+text labels.
    LabelProperty::Anchors salienceOrderedAnchors(const LabelProperty::Anchors& _anchors,
                                                   glm::vec2 _tileLocalPos,
                                                   glm::vec2 _labelSizePixels,
                                                   float _iconRadiusPixels) const;

protected:

    const TextStyle& m_style;

    // Result: TextLabel container
    std::unique_ptr<TextLabels> m_textLabels;

    // Buffers to hold data for TextLabels until build()
    std::vector<GlyphQuad> m_quads;
    std::bitset<FontContext::max_textures> m_atlasRefs;
    std::vector<std::unique_ptr<Label>> m_labels;

    float m_tileSize = 0;
    float m_tileScale = 0;

    // Shared per-tile occupancy index built once in TileBuilder::build() from all of the
    // tile's raw layers (see labels/anchorOccupancyGrid.h) -- null if the tile had no
    // water/transportation geometry.
    std::shared_ptr<const AnchorOccupancyGrid> m_anchorOccupancyGrid;

    // Per-candidate-direction cost helper for salienceOrderedAnchors().
    float anchorCandidateCost(LabelProperty::Anchor _anchor, glm::vec2 _tileLocalPos,
                              glm::vec2 _labelSizePixels, float _iconRadiusPixels) const;

};

}
