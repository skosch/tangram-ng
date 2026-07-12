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

    bool handleBoundaryLabel(const Feature& _feat, const DrawRule& _rule,
                             const TextStyle::Parameters& _params);

    bool checkRule(const DrawRule& _rule) const override;
    std::vector<std::unique_ptr<Label>>* labels() { return &m_labels; }

    void addLayoutItems(LabelCollider& _layout) override;

    // Salience-aware anchor ordering (docs/label-placement-plan.md Phase 3). Returns a
    // copy of `_anchors` re-sorted by ascending "cost" -- how visually disruptive it
    // would be to place a label at each candidate anchor direction, combining a
    // vector-proximity term (avoid stamping over nearby water/road/trail geometry, via
    // the tile's AnchorOccupancyGrid) and a texture-shading term (avoid strong
    // ridge/canyon response, via the sampleTextureShading() stub -- see
    // util/textureShading.h; contributes 0 when unavailable, which is always true today).
    // The result is handed unchanged to the existing anchor-cycling mechanism
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

    // Captured in setup(const Tile&) for salienceOrderedAnchors(): lets a normalized
    // tile-local position be converted to world ProjectedMeters (tile.getOrigin() +
    // tileLocalPos * tile.getScale(), mirroring Tile::coordToLngLat()) for the
    // texture-shading term's sampleTextureShading() call.
    glm::dvec2 m_tileOrigin{0.0, 0.0};
    double m_tileWorldScale = 0.0;

    // Shared per-tile occupancy index built once in TileBuilder::build() from all of the
    // tile's raw layers (see labels/anchorOccupancyGrid.h) -- null if the tile had no
    // water/transportation geometry.
    std::shared_ptr<const AnchorOccupancyGrid> m_anchorOccupancyGrid;

    // Per-candidate-direction cost helper for salienceOrderedAnchors().
    float anchorCandidateCost(LabelProperty::Anchor _anchor, glm::vec2 _tileLocalPos,
                              glm::vec2 _labelSizePixels, float _iconRadiusPixels) const;

};

}
