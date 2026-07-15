#include "style/pointStyleBuilder.h"

#include "data/properties.h"
#include "data/propertyItem.h"
#include "labels/labelCollider.h"
#include "labels/labelProperty.h"
#include "labels/spriteLabel.h"
#include "log.h"
#include "marker/marker.h"
#include "map.h"
#include "scene/drawRule.h"
#include "scene/spriteAtlas.h"
#include "scene/stops.h"
#include "selection/featureSelection.h"
#include "tile/tile.h"
#include "util/geom.h"
#include "util/lineSampler.h"
#include "view/view.h"

namespace Tangram {


void IconMesh::setTextLabels(std::unique_ptr<StyledMesh> _textLabels) {

    auto* mesh = static_cast<TextLabels*>(_textLabels.get());
    auto& labels = mesh->getLabels();

    typedef std::vector<std::unique_ptr<Label>>::iterator iter_t;
    m_labels.insert(m_labels.end(),
                    std::move_iterator<iter_t>(labels.begin()),
                    std::move_iterator<iter_t>(labels.end()));

    labels.clear();

    textLabels = std::move(_textLabels);
}

void PointStyleBuilder::addLayoutItems(LabelCollider& _layout) {
    _layout.addLabels(m_labels);
    m_textStyleBuilder->addLayoutItems(_layout);
}

std::unique_ptr<StyledMesh> PointStyleBuilder::build() {

    if (m_quads.empty()) { return nullptr; }

    if (Tangram::getDebugFlag(DebugFlags::draw_all_labels)) {
        m_iconMesh->setLabels(m_labels);
    } else {
        size_t sumLabels = 0;

        // Determine number of labels
       for (auto& label : m_labels) {
           if (label->state() != Label::State::dead) { sumLabels +=1; }
        }

       std::vector<std::unique_ptr<Label>> labels;
       labels.reserve(sumLabels);

       for (auto& label : m_labels) {
           if (label->state() != Label::State::dead) {
               labels.push_back(std::move(label));
           }
       }
       m_iconMesh->setLabels(labels);
    }

    std::vector<SpriteQuad> quads(m_quads);
    m_spriteLabels->setQuads(std::move(quads));

    m_quads.clear();
    m_labels.clear();

    m_iconMesh->spriteLabels = std::move(m_spriteLabels);

    if (auto textLabels = m_textStyleBuilder->build()) {
        m_iconMesh->setTextLabels(std::move(textLabels));
    }

    return std::move(m_iconMesh);
}

void PointStyleBuilder::setup(const Tile& _tile) {
    TileID id = _tile.getID();
    m_zoom = id.z;
    m_styleZoom = id.s;

    // < 1.0 when overzooming a tile
    m_tileScale = pow(2, id.s - id.z);

    m_spriteLabels = std::make_unique<SpriteLabels>(m_style);

    m_textStyleBuilder->setup(_tile);
    m_iconMesh = std::make_unique<IconMesh>();
    m_randGen.seed(uint64_t(id.z) << 48 | uint64_t(id.x) << 24 | uint64_t(id.y));
}

void PointStyleBuilder::setup(const Marker& _marker, int zoom) {
    m_zoom = zoom;
    m_styleZoom = zoom;
    m_spriteLabels = std::make_unique<SpriteLabels>(m_style);

    m_textStyleBuilder->setup(_marker, zoom);
    m_iconMesh = std::make_unique<IconMesh>();

    m_texture = _marker.texture();
    m_randGen.seed(_marker.id()*zoom);
}

bool PointStyleBuilder::checkRule(const DrawRule& _rule) const {
    if (m_style.defaultTexture()) {
        return true;
    }
    // require a color or texture atlas/texture to be valid
    uint32_t color;
    if (_rule.get(StyleParamKey::color, color)) {
        return true;
    }

    std::string texture;
    if (_rule.get(StyleParamKey::texture, texture) && !texture.empty()) {
        return true;
    }

    return false;
}

auto PointStyleBuilder::applyRule(const DrawRule& _rule) const -> Parameters {

    Parameters p;

    _rule.get(StyleParamKey::color, p.color);
    float alpha = 1;
    if (_rule.get(StyleParamKey::alpha, alpha)) {
        p.color = Color(p.color).withAlpha(alpha).abgr;
    }

    _rule.get(StyleParamKey::sprite, p.sprite);
    _rule.get(StyleParamKey::offset, p.labelOptions.offset);
    p.labelOptions.offset *= m_style.pixelScale();
    _rule.get(StyleParamKey::buffer, p.labelOptions.buffer);

    std::string repeatGroup;
    StyleParam::Width repeatDistance;

    _rule.get(StyleParamKey::priority, p.labelOptions.priority);
    // docs/label-placement-plan.md Phase 2: opt-in flag (peak draw rule only, currently) --
    // whether addPoint()/addLine()/addPolygon() below should ask LabelManager to refine this
    // label's priority via the CPU texture-shading sampler on the main thread. Read here
    // (feature-independent, just the draw rule's own value); addPoint() etc. additionally
    // suppress it per-feature when a real OSM `prominence` tag is already present, since that's
    // a strictly better signal and the JS priority: function already used it.
    _rule.get(StyleParamKey::priority_texture_shading, p.labelOptions.refinePriorityWithTextureShading);
    _rule.get(StyleParamKey::sprite_default, p.spriteDefault);
    p.dynamicTexture = _rule.get(StyleParamKey::texture, p.texture);

    _rule.get(StyleParamKey::placement, p.placement);
    StyleParam::Width placementSpacing;
    auto placementSpacingParam = _rule.findParameter(StyleParamKey::placement_spacing);
    if (placementSpacingParam.stops) {
        p.placementSpacing = placementSpacingParam.stops->evalFloat(m_styleZoom);
    } else if (_rule.get(StyleParamKey::placement_spacing, placementSpacing)) {
        p.placementSpacing = placementSpacing.value;
    }
    auto placementMinLengthParam = _rule.findParameter(StyleParamKey::placement_min_length_ratio);
    if (placementMinLengthParam.stops) {
        p.placementMinLengthRatio = placementMinLengthParam.stops->evalFloat(m_styleZoom);
    } else {
        _rule.get(StyleParamKey::placement_min_length_ratio, p.placementMinLengthRatio);
    }
    _rule.get(StyleParamKey::tile_edges, p.keepTileEdges);
    _rule.get(StyleParamKey::interactive, p.interactive);
    _rule.get(StyleParamKey::collide, p.labelOptions.collide);
    _rule.get(StyleParamKey::transition_hide_time, p.labelOptions.hideTransition.time);
    _rule.get(StyleParamKey::transition_selected_time, p.labelOptions.selectTransition.time);
    _rule.get(StyleParamKey::transition_show_time, p.labelOptions.showTransition.time);
    _rule.get(StyleParamKey::flat, p.labelOptions.flat);
    _rule.get(StyleParamKey::anchor, p.labelOptions.anchors);

    if (_rule.get(StyleParamKey::repeat_distance, repeatDistance)) {
        p.labelOptions.repeatDistance = repeatDistance.value;
    }

    if (p.labelOptions.repeatDistance > 0.f) {
        size_t repeatGroupHash = 0;
        if (_rule.get(StyleParamKey::repeat_group, repeatGroup)) {
            hash_combine(repeatGroupHash, repeatGroup);
        } else {
            repeatGroupHash = _rule.getParamSetHash();
        }

        p.labelOptions.repeatGroup = repeatGroupHash;
        p.labelOptions.repeatDistance *= m_style.pixelScale();
    }

    if (p.labelOptions.anchors.count == 0) {
        p.labelOptions.anchors.anchor = { {LabelProperty::Anchor::center} };
        p.labelOptions.anchors.count = 1;
    }

    _rule.get(StyleParamKey::angle, p.labelOptions.angle);
    if (std::isnan(p.labelOptions.angle)) {
        p.autoAngle = true;
    }

    auto& strokeWidth = _rule.findParameter(StyleParamKey::outline_width);

    if (_rule.get(StyleParamKey::outline_color, p.outlineColor)) {
        if (strokeWidth.value.is<StyleParam::Width>()) {
            auto& widthParam = strokeWidth.value.get<StyleParam::Width>();
            p.outlineWidth = widthParam.value * m_style.pixelScale();
        }
    }
    float outlineAlpha = 1;
    if (_rule.get(StyleParamKey::outline_alpha, outlineAlpha)) {
        p.outlineColor = Color(p.outlineColor).withAlpha(outlineAlpha).abgr;
    }

    std::hash<Parameters> hash;
    p.labelOptions.paramHash = hash(p);

    if (p.interactive) {
        p.labelOptions.featureId = _rule.selectionColor;
    }

    return p;
}

void PointStyleBuilder::addLabel(const Point& _point, const glm::vec4& _quad, Texture* _texture,
                                 const Parameters& _params, const DrawRule& _rule) {

    uint32_t selectionColor = 0;

    if (_params.interactive) {
        if (_rule.featureSelection) {
            selectionColor = _rule.featureSelection->nextColorIdentifier();
        } else {
            selectionColor = _rule.selectionColor;
        }
    }

    m_labels.push_back(std::make_unique<SpriteLabel>(glm::vec3(_point, 0.f), m_zoom,
                                                     _params.size,
                                                     _params.labelOptions,
                                                     SpriteLabel::VertexAttributes{
                                                             _params.color,
                                                             selectionColor,
                                                             _params.outlineColor,
                                                             _params.antialiasFactor,
                                                             _params.extrudeScale },
                                                     _texture,
                                                     *m_spriteLabels,
                                                     m_quads.size()));

    glm::i16vec2 size = _params.size;

    // Attribute will be normalized - scale to max short;
    glm::vec2 uvTR = glm::vec2{_quad.z, _quad.w} * SpriteVertex::texture_scale;
    glm::vec2 uvBL = glm::vec2{_quad.x, _quad.y} * SpriteVertex::texture_scale;

    float sx = size.x * 0.5f;
    float sy = size.y * 0.5f;

    glm::vec2 v0(-sx, sy);
    glm::vec2 v1(sx, sy);
    glm::vec2 v2(-sx, -sy);
    glm::vec2 v3(sx, -sy);

    if (_params.labelOptions.angle != 0.f) {
        // Rotate the sprite icon quad vertices in clockwise order
        glm::vec2 rotation = rotation2dDeg(-_params.labelOptions.angle);

        v0 = rotate2d(v0, rotation);
        v1 = rotate2d(v1, rotation);
        v2 = rotate2d(v2, rotation);
        v3 = rotate2d(v3, rotation);
    }

    m_quads.push_back({{
        {v0, {uvBL.x, uvTR.y}},
        {v1, {uvTR.x, uvTR.y}},
        {v2, {uvBL.x, uvBL.y}},
        {v3, {uvTR.x, uvBL.y}}}
        });

#ifdef DEBUG
    m_labels.back()->debugTag =  _params.texture + ":" +
        (!_params.sprite.empty() ? _params.sprite : _params.spriteDefault);
#endif
}

bool PointStyleBuilder::getTexture(const Parameters& _params, Texture **_texture) const {

    auto texture = m_style.defaultTexture().get();

    if (m_texture) {
        // If marker texture has been assigned, always use it.
        texture = m_texture;

    } else if (_params.dynamicTexture) {

        if (_params.texture == "") {
            texture = nullptr;
        } else {
            if (!m_style.textures()) {
                return false;
            }

            auto it = m_style.textures()->find(_params.texture);
            if (it == m_style.textures()->end()) {
                return false;
            }
            texture = it->second.get();
        }
    }

    if (texture) {
        *_texture = texture;
    }

    return true;
}

bool PointStyleBuilder::evalSizeParam(const DrawRule& _rule, Parameters& _params, const Texture* _texture) const {
    StyleParam::SizeValue size;
    SpriteNode spriteNode;
    glm::vec2 spriteSize(NAN);

    if (_texture) {
        spriteSize = glm::vec2{_texture->width(), _texture->height()} * _texture->displayScale();

        const auto &atlas = _texture->spriteAtlas();
        if (atlas) {
            if (!atlas->getSpriteNode(_params.sprite, spriteNode) &&
                !atlas->getSpriteNode(_params.spriteDefault, spriteNode)) {
                return false;
            }
            spriteSize = spriteNode.m_size * _texture->displayScale();
        } else if ( !_params.sprite.empty() || !_params.spriteDefault.empty()) {
            // missing sprite atlas for texture but sprite specified in draw rule
            return false;
        }
    }

    auto sizeParam = _rule.findParameter(StyleParamKey::size);

    if (sizeParam.stops) {
        glm::vec2 lower = sizeParam.stops->evalSize(m_styleZoom, spriteSize);
        glm::vec2 higher = sizeParam.stops->evalSize(std::nextafter(m_styleZoom + 1.0f, 0.0f), spriteSize);
        if ((std::isnan(lower.x) || std::isnan(lower.y)) || (std::isnan(higher.x) || std::isnan(higher.y))) {
            // Illegal size values were evaluated.
            return false;
        }
        _params.size = lower;
        _params.extrudeScale = higher.x - lower.x;

    } else if (_rule.get(StyleParamKey::size, size)) {
        glm::vec2 pixelSize = size.getSizePixels(spriteSize);
        if (std::isnan(pixelSize.x) || std::isnan(pixelSize.y)) {
            // Illegal size values were evaluated.
            return false;
        }
        _params.size = pixelSize;
    } else {
        _params.size = spriteSize;
    }

    _params.size *= m_style.pixelScale();

    return true;
}

bool PointStyleBuilder::getUVQuad(Parameters& _params, glm::vec4& _quad, const Texture* _texture) const {

    _quad = glm::vec4(0.0, 1.0, 1.0, 0.0);

    if (_texture) {
        const auto& atlas = _texture->spriteAtlas();
        if (atlas) {
            SpriteNode spriteNode;
            if (!atlas->getSpriteNode(_params.sprite, spriteNode) &&
                !atlas->getSpriteNode(_params.spriteDefault, spriteNode)) {
                return false;
            }
            _quad.x = spriteNode.m_uvBL.x;
            _quad.y = spriteNode.m_uvBL.y;
            _quad.z = spriteNode.m_uvTR.x;
            _quad.w = spriteNode.m_uvTR.y;
        }
    } else {

        float fillEdge = _params.size.x;
        float outlineEdge = 1.f;

        // Adds half the outline width in each direction (* 0.5 * 2.0)
        _params.size += _params.outlineWidth;

        // Fraction at which fill ends
        fillEdge /= _params.size.x;

        if (_params.outlineWidth > 0.f) {
            // Fraction at which outline starts
            outlineEdge = fillEdge - _params.outlineWidth / _params.size.x;
        }

        fillEdge = std::max(0.0001f, fillEdge);
        outlineEdge = std::max(0.0001f, outlineEdge);

        _quad = glm::vec4(-outlineEdge, -fillEdge,
                          outlineEdge, fillEdge);

        _params.antialiasFactor = std::numeric_limits<int16_t>::max() / _params.size.x * 2.0f;
    }

    return true;
}

void PointStyleBuilder::labelPointsPlacing(const Line& _line, const glm::vec4& _uvsQuad, Texture* _texture,
                                           Parameters& params, const DrawRule& _rule) {

    if (_line.size() < 2) { return; }

    auto isOutsideTile = [](const Point& p) {
        float tolerance = 0.0005f;
        float tile_min = 0.0f + tolerance;
        float tile_max = 1.0f - tolerance;
        return ((p.x < tile_min) || (p.x > tile_max) ||
                (p.y < tile_min) || (p.y > tile_max));
    };

    auto angleBetween = [](const Point& p, const Point& q) {
        return RAD_TO_DEG * atan2(q[0] - p[0], q[1] - p[1]);
    };

    float minLineLength = std::max(params.size.x, params.size.y) *
        params.placementMinLengthRatio * m_style.pixelScale() /
        (MapProjection::tileSize() * m_tileScale);

    switch(params.placement) {
        case LabelProperty::Placement::vertex: {
            for (size_t i = 0; i < _line.size() - 1; i++) {
                auto& p = _line[i];
                if (params.keepTileEdges || !isOutsideTile(p)) {
                    auto& q = _line[i+1];
                    if (params.autoAngle) {
                        params.labelOptions.angle = angleBetween(p, q);
                    }
                    addLabel(p, _uvsQuad, _texture, params, _rule);
                    if (i == _line.size() - 2) {
                        // Place label on endpoint
                        addLabel(q, _uvsQuad, _texture, params, _rule);
                    }
                }
            }
            break;
        }
        case LabelProperty::Placement::midpoint:
            for (size_t i = 0; i < _line.size() - 1; i++) {
                auto& p = _line[i];
                auto& q = _line[i+1];
                if ( (params.keepTileEdges || !isOutsideTile(p)) &&
                     (minLineLength == 0.0f || glm::distance(p, q) > minLineLength) ) {
                    if (params.autoAngle) {
                        params.labelOptions.angle = angleBetween(p, q);
                    }
                    glm::vec3 midpoint(0.5f * (p.x + q.x), 0.5f * (p.y + q.y), 0.0f);
                    addLabel(midpoint, _uvsQuad, _texture, params, _rule);
                }
            }
            break;
        case LabelProperty::Placement::spaced: {
            std::uniform_real_distribution<float> randf(0.0, 1.0);
            LineSampler<std::vector<glm::vec3>> sampler;

            sampler.set(_line);

            float lineLength = sampler.sumLength();
            float spacing = params.placementSpacing * m_style.pixelScale() /
                (MapProjection::tileSize() * m_tileScale);

            // for min_length == 0 and line shorter than spacing, place a label w/ probability ~ line length
            float lenratio = lineLength / spacing;
            if (minLineLength == 0 && lenratio < 1 && randf(m_randGen) > lenratio) { break; }
            else if (lineLength <= minLineLength) { break; }

            int numLabels = std::max(std::floor(lenratio), 1.0f);
            float remainderLength = lineLength - (numLabels - 1) * spacing;
            float distance = 0.5f * remainderLength;
            glm::vec2 p, r;
            sampler.advance(distance, p, r);
            do {

                if (sampler.lengthToPrevSegment() < minLineLength*0.5f ||
                    sampler.lengthToNextSegment() < minLineLength*0.5f) {
                    continue;
                }
                if (params.autoAngle) {
                    params.labelOptions.angle = RAD_TO_DEG * atan2(r.x, r.y);
                }

                if (params.keepTileEdges || !isOutsideTile(p)) {
                    addLabel({p.x, p.y}, _uvsQuad, _texture, params, _rule);
                }

            } while (sampler.advance(spacing, p, r));
        }
        break;
        case LabelProperty::Placement::centroid:
            // nothing to be done here.
            break;
    }
}

// docs/label-placement-plan.md Phase 2: a real OSM `prominence` tag (already available on
// `_props` here -- same Properties/parsing path as `feature.ele`, see e.g. the peak draw rule's
// filter in stylus-osm.yaml) is a strictly better salience signal than the CPU texture-shading
// proxy, and the peak draw rule's `priority:` JS function already uses it when present (that
// function runs on the same worker thread as this one, with full Properties access -- no
// threading concern there, only the raster-backed texture-shading term needs the main-thread
// refinement path). Suppress the main-thread refinement request in that case so it doesn't
// later overwrite the better, already-final priority value.
static void suppressTextureShadingIfProminent(PointStyleBuilder::Parameters& p, const Properties& props) {
    if (p.labelOptions.refinePriorityWithTextureShading && props.contains("prominence")) {
        p.labelOptions.refinePriorityWithTextureShading = false;
    }
}

bool PointStyleBuilder::addPoint(const Point& _point, const Properties& _props,
                                 const DrawRule& _rule) {

    Parameters p = applyRule(_rule);
    suppressTextureShadingIfProminent(p, _props);
    glm::vec4 uvsQuad;
    Texture* texture = nullptr;

    if (!getTexture(p, &texture)) {
        return false;
    }
    if (!evalSizeParam(_rule, p, texture)) {
        return false;
    }
    if (!getUVQuad(p, uvsQuad, texture)) {
        return false;
    }

    addLabel(_point, uvsQuad, texture, p, _rule);

    return true;
}

bool PointStyleBuilder::addLine(const Line& _line, const Properties& _props,
                                const DrawRule& _rule) {

    Parameters p = applyRule(_rule);
    suppressTextureShadingIfProminent(p, _props);
    glm::vec4 uvsQuad;
    Texture* texture = nullptr;

    if (!getTexture(p, &texture)) {
        return false;
    }
    if (!evalSizeParam(_rule, p, texture)) {
        return false;
    }
    if (!getUVQuad(p, uvsQuad, texture)) {
        return false;
    }

    labelPointsPlacing(_line, uvsQuad, texture, p, _rule);

    return true;
}

bool PointStyleBuilder::addPolygon(const Polygon& _polygon, const Properties& _props,
                                   const DrawRule& _rule) {

    Parameters p = applyRule(_rule);
    suppressTextureShadingIfProminent(p, _props);
    glm::vec4 uvsQuad;
    Texture* texture = nullptr;

    if (!getTexture(p, &texture)) {
        return false;
    }
    if (!evalSizeParam(_rule, p, texture)) {
        return false;
    }
    if (!getUVQuad(p, uvsQuad, texture)) {
        return false;
    }

    if (p.placement != LabelProperty::centroid) {
        for (auto line : _polygon) {
            labelPointsPlacing(line, uvsQuad, texture, p, _rule);
        }
    } else {
        if (!_polygon.empty()) {
            glm::vec2 c;
            c = centroid(_polygon.front().begin(), _polygon.front().end());
            addLabel(c, uvsQuad, texture, p, _rule);
        }
    }

    return true;
}

bool PointStyleBuilder::addFeature(const Feature& _feat, const DrawRule& _rule) {

    size_t iconsStart = m_labels.size();

    if (!StyleBuilder::addFeature(_feat, _rule)) {
        return false;
    }

    size_t iconsCount = m_labels.size() - iconsStart;

    bool textVisible = true;
    _rule.get(StyleParamKey::text_visible, textVisible);

    if (textVisible && _rule.contains(StyleParamKey::point_text)) {
        if (iconsCount == 0) { return true; }

        auto& textStyleBuilder = static_cast<TextStyleBuilder&>(*m_textStyleBuilder);
        auto& textLabels = *textStyleBuilder.labels();

        TextStyle::Parameters params = textStyleBuilder.applyRule(_rule, _feat.props, true);

        TextStyleBuilder::LabelAttributes attrib;
        size_t primaryLabelsStart = textLabels.size();
        bool havePrimary = textStyleBuilder.prepareLabel(params, Label::Type::point, attrib);
        if (havePrimary) {

            for (size_t i = 0; i < iconsCount; i++) {
                auto pLabel = static_cast<SpriteLabel*>(m_labels[iconsStart + i].get());
                glm::vec2 p = pLabel->modelCenter();

                // Salience-aware anchor ordering (docs/label-placement-plan.md Phase 3):
                // steer the linked text label's anchor fallback order away from nearby
                // water/road/trail geometry (and, once Phase 2 lands, ridge/canyon
                // texture-shading response) instead of always trying the fixed compass
                // order first. Icon radius approximated as half the sprite's larger
                // dimension. The sorted list is handed to the existing anchor-cycling
                // mechanism unchanged -- no collision/occlusion code is touched.
                glm::vec2 iconDim = pLabel->dimension();
                float iconRadiusPx = 0.5f * std::max(iconDim.x, iconDim.y);
                params.labelOptions.anchors = textStyleBuilder.salienceOrderedAnchors(
                    params.labelOptions.anchors, p, glm::vec2(attrib.width, attrib.height), iconRadiusPx);

                textStyleBuilder.addLabel(Label::Type::point, {{p, p}}, params, attrib, _rule);

                bool definePriority = !_rule.contains(StyleParamKey::text_priority);
                // defineCollide introduced in github.com/tangrams/tangram-es/pull/928 resulting in
                //  text.collide style being overwritten; do we even need to check StyleParamKey::collide?
                //bool defineCollide = _rule.contains(StyleParamKey::collide);
                bool defineCollide = _rule.contains(StyleParamKey::collide)
                    && !_rule.contains(StyleParamKey::text_collide);

                // Link labels together
                textLabels.back()->setRelative(*pLabel, definePriority, defineCollide);
            }
        }

        // Secondary linked text label (docs/label-placement-plan.md Task 1): a second,
        // independently-styled label (its own font weight/size via text2_*) linked to the
        // SAME icon via setRelative() -- a sibling of the primary text label, not a child
        // of it. The engine has no per-line/per-glyph-run font override (confirmed: one
        // `alfons::Font` per TextStyle::Parameters, shaped as a single run in
        // FontContext::layoutText) and Label's anchor/offset chain is only ever resolved
        // one level deep (TextLabel::applyAnchor adds m_relative->dimension(), but never
        // walks m_relative->relative()) -- so a true 3-level icon->name->elevation chain
        // would need new chain-walking logic in applyAnchor. Instead: both labels are
        // direct children of the icon (cheap, no applyAnchor changes needed), and the
        // second one is pushed further out along its chosen anchor direction by the
        // primary label's *actual measured* dimension (read back from the already-built
        // primary TextLabel) plus a small gap, so it visually stacks below/above/beside
        // the primary label instead of sitting on top of it. Marked `optional` so that if
        // it can't be placed, only it is dropped -- see LabelCollider::killOccludedLabels,
        // a required (non-optional) child that can't fit kills its *relative* (the icon),
        // which would otherwise take the primary label down with it too.
        // Known limitation: each label's anchor is cycled independently on collision
        // (LabelManager::handleOcclusions), so if the icon+labels are involved in a
        // collision retry the two text labels can end up on different anchors and drift
        // apart. Acceptable for peaks (isolated POIs, collisions are rare); revisit if
        // this mechanism is reused for denser point styles.
        if (havePrimary && _rule.contains(StyleParamKey::text2_source)) {
            TextStyle::Parameters params2 = textStyleBuilder.applySecondaryTextRule(_rule, _feat.props, params);
            TextStyleBuilder::LabelAttributes attrib2;
            if (!params2.text.empty() && textStyleBuilder.prepareLabel(params2, Label::Type::point, attrib2)) {

                size_t primaryLabelsCount = textLabels.size() - primaryLabelsStart;

                for (size_t i = 0; i < iconsCount; i++) {
                    auto pLabel = static_cast<SpriteLabel*>(m_labels[iconsStart + i].get());
                    glm::vec2 p = pLabel->modelCenter();

                    // Pinned to a single fixed anchor (bottom), NOT the primary label's
                    // salience-ordered anchor list: the primary and this label cycle
                    // anchors independently at runtime on collision
                    // (LabelManager::handleOcclusions calls Label::nextAnchor() per-label),
                    // so computing this label's push distance from "whichever anchor the
                    // primary label *currently* has" is fragile -- if salienceOrderedAnchors
                    // (called separately here, with this label's own smaller footprint)
                    // picked a different first anchor than the primary's, or either label
                    // cycles away from anchor 0 independently, the two directions can
                    // disagree and the push computed for one direction ends up applied
                    // while the actual rendered anchor is another, leaving this label
                    // sitting on top of (and losing a collision against) the primary one --
                    // confirmed via a diagnostic build that a fixed anchor + large constant
                    // offset renders fine, whereas the salience-ordered-and-computed version
                    // silently never did. A single fixed anchor sidesteps the mismatch
                    // entirely; the tradeoff is the elevation line doesn't rotate away from
                    // clutter the way the name label does.
                    params2.labelOptions.anchors = LabelProperty::Anchors(LabelProperty::Anchor::bottom);

                    // Push straight down by the primary label's actual measured height
                    // (read back from the already-built primary label for this same icon)
                    // plus a small gap, so it sits just below the name instead of
                    // overlapping it. TextLabel::applyAnchor already offsets this label a
                    // half-extent below the icon, so the push only needs to cover the
                    // primary's full height, not a half/half split -- see the (reverted)
                    // horizontal-anchor version of this comment in git history for the
                    // arithmetic if this needs revisiting.
                    glm::vec2 primaryDim(0);
                    if (i < primaryLabelsCount) {
                        primaryDim = textLabels[primaryLabelsStart + i]->dimension();
                    }
                    const float gapPx = 2.f * m_style.pixelScale();
                    params2.labelOptions.offset = glm::vec2(0.f, primaryDim.y + gapPx);

                    params2.labelOptions.optional = true;

                    textStyleBuilder.addLabel(Label::Type::point, {{p, p}}, params2, attrib2, _rule);

                    bool definePriority = !_rule.contains(StyleParamKey::text_priority);
                    bool defineCollide = _rule.contains(StyleParamKey::collide)
                        && !_rule.contains(StyleParamKey::text_collide);
                    textLabels.back()->setRelative(*pLabel, definePriority, defineCollide);
                }
            }
        }
    }

    return true;
}

}
