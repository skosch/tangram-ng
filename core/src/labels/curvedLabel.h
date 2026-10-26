#pragma once

#include "labels/label.h"
#include "labels/textLabel.h"

#include <glm/glm.hpp>

namespace Tangram {

class TextLabels;
class TextStyle;

class CurvedLabel : public TextLabel {

public:

    struct VertexAttributes {
        uint32_t fill;
        uint32_t stroke;
        uint8_t fontScale;
        uint32_t selectionColor;
    };

    using ModelTransform = std::vector<glm::vec3>;

    CurvedLabel(ModelTransform _modelTransform, Label::Options _options, float _prio,
                TextLabel::VertexAttributes _attrib, glm::vec2 _dim,
                TextLabels& _labels, TextRange _textRanges,
                TextLabelProperty::Align _preferedAlignment, size_t _anchorPoint)

        : TextLabel({{}}, Label::Type::curved, _options, _attrib,
                    _dim, _labels, _textRanges, _preferedAlignment),
          m_modelTransform(std::move(_modelTransform)),
          m_anchorPoint(_anchorPoint),
          m_screenAnchorPoint(_anchorPoint),
          m_prio(_prio) {

        applyAnchor(m_options.anchors[0]);
    }

    bool updateScreenTransform(const glm::mat4& _mvp, const ViewState& _viewState,
                               const AABB* _bounds, ScreenTransform& _transform) override;

    bool setElevation(ElevationManager& elevMgr, glm::dvec2 origin, double scale) override;

    void obbs(ScreenTransform& _transform, OBBBuffer& _obbs) const override;

    void addVerticesToMesh(ScreenTransform& _transform, const glm::vec2& _screenSize) override;

    void applyAnchor(LabelProperty::Anchor _anchor) override;

    void setCandidatePriority(float _p) { m_prio = _p; }

    /// Area labels: the path is a stretch of a shape's spine, longer than the text. The text stays
    /// where it is (anchored to the map) until it would leave the view or no longer fit between
    /// the shores; only then does it move, to the place nearest the middle of the view. `_clearance` is the inscribed
    /// radius at each vertex (model units), `_halfHeight` the clearance the text needs (px).
    /// `_slot` > 0: a spare place beside the best one (the label manager takes the first of a
    /// group that is not blocked by another label).
    void setSlide(const std::vector<glm::vec2>& _path, const std::vector<float>& _clearance, float _halfHeight,
                  float _margin, int _slot);

    float candidatePriority() const override {
        return m_prio;
    }

    glm::vec3 modelCenter() const override {
        return m_modelTransform[m_anchorPoint];
    }

protected:

    ModelTransform m_modelTransform;

    const size_t m_anchorPoint;

    size_t m_screenAnchorPoint;

    float m_prio = 0;

    std::vector<float> m_slideClear;
    std::vector<float> m_slideArc;   // arc length of each vertex in model units
    float m_slideS = -1.f;           // where the text sits (model arc length); kept until it must move
    float m_slideHalfH = 0.f;
    float m_slideMargin = 0.f;
    int m_slideSlot = 0;

    bool slideTransform(const std::vector<glm::vec2>& _screen, glm::vec2 _view, ScreenTransform& _transform,
                        bool _noFlip);
};

}
