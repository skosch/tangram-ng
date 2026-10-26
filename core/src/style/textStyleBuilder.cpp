#include "style/textStyleBuilder.h"

#include "labels/anchorOccupancyGrid.h"
#include "labels/curvedLabel.h"
#include "labels/labelCollider.h"
#include "labels/labelSet.h"
#include "labels/textLabel.h"
#include "labels/textLabels.h"
#include "log.h"
#include "map.h"
#include "marker/marker.h"
#include "selection/featureSelection.h"
#include "scene/drawRule.h"
#include "tile/tile.h"
#include "util/color.h"
#include "util/geom.h"
#include "util/mapProjection.h"
#include "util/labelPath.h"
#include "util/lineSampler.h"
#include "util/textureShading.h"
#include "view/view.h"

#ifdef FONTCONTEXT_STB
#include <codecvt>
#else
#include "unicode/unistr.h"
#include "unicode/schriter.h"
#include "unicode/brkiter.h"
#include "unicode/locid.h"
#endif

#include <glm/gtx/norm.hpp>
#include <algorithm>
#include <cmath>
#include <locale>
#include <mutex>

namespace Tangram {

const static std::string key_name("name");

// True if any vertex of the polygon's outer ring lies on the tile boundary, i.e. the
// polygon is (probably) a fragment of a larger feature clipped to this tile. Fit-to-path
// tracking must not trust such a fragment's size: the neighbouring tile would compute a
// different tracking for the same feature.
static bool touchesTileEdge(const Polygon& _polygon) {
    const float eps = 0.002f;
    for (const auto& p : _polygon.front()) {
        if (p.x < eps || p.x > 1.f - eps || p.y < eps || p.y > 1.f - eps) { return true; }
    }
    return false;
}

TextStyleBuilder::TextStyleBuilder(const TextStyle& _style) : m_style(_style) {}

void TextStyleBuilder::setup(const Tile& _tile){
    m_tileSize = MapProjection::tileSize() * m_style.pixelScale();

    // < 1.0 when overzooming a tile
    m_tileScale = pow(2, _tile.getID().s - _tile.getID().z);
    m_tileSize *= m_tileScale;

    m_anchorOccupancyGrid = _tile.anchorOccupancyGrid();

    m_atlasRefs.reset();

    m_textLabels = std::make_unique<TextLabels>(m_style);
}

void TextStyleBuilder::setup(const Marker& marker, int zoom) {
    float metersPerTile = MapProjection::metersPerTileAtZoom(zoom);

    // In general, a Marker won't cover the same area as a tile, so the effective
    // "tile size" for building a Marker is the size of a tile in pixels multiplied
    // by the ratio of the Marker's extent to the length of a tile side at this zoom.
    m_tileSize = 256 * (marker.extent() / metersPerTile);

    // (Copied from Tile setup function above, purpose unclear)
    m_tileSize *= m_style.pixelScale();

    m_atlasRefs.reset();

    m_textLabels = std::make_unique<TextLabels>(m_style);
}

void TextStyleBuilder::addLayoutItems(LabelCollider& _layout) {
    _layout.addLabels(m_labels);
}

std::unique_ptr<StyledMesh> TextStyleBuilder::build() {

    if (m_quads.empty()) { return nullptr; }

    if (Tangram::getDebugFlag(DebugFlags::draw_all_labels)) {
        m_textLabels->setLabels(m_labels);

        std::vector<GlyphQuad> quads(m_quads);
        m_textLabels->setQuads(std::move(quads), m_atlasRefs);

    } else {

        // TODO this could probably done more elegant

        int quadPos = 0;
        size_t sumQuads = 0;
        size_t sumLabels = 0;
        bool added = false;

        // Determine number of labels and size of final quads vector
        for (auto& label : m_labels) {
            auto* textLabel = static_cast<TextLabel*>(label.get());

            const auto& ranges = textLabel->textRanges();

            bool active = textLabel->state() != Label::State::dead;

            if (ranges.back().end() != quadPos) {
                quadPos = ranges.back().end();
                added = false;
            }

            if (!active) { continue; }

            sumLabels +=1;
            if (!added) {
                added = true;

                for (auto& textRange : ranges) {
                    sumQuads += textRange.length;
                }
            }
        }

        size_t quadEnd = 0;
        size_t quadStart = 0;
        quadPos = 0;

        std::vector<std::unique_ptr<Label>> labels;
        labels.reserve(sumLabels);

        std::vector<GlyphQuad> quads;
        quads.reserve(sumQuads);

        // Add only alive labels
        for (auto& label : m_labels) {
            auto* textLabel = static_cast<TextLabel*>(label.get());

            bool active = textLabel->state() != Label::State::dead;
            if (!active) { continue; }

            auto& ranges = textLabel->textRanges();

            // Add the quads of line-labels only once
            if (ranges.back().end() != quadPos) {
                quadStart = quadEnd;
                quadPos = ranges.back().end();

                for (auto& textRange : ranges) {
                    if (textRange.length > 0) {
                        quadEnd += textRange.length;

                        auto it = m_quads.begin() + textRange.start;
                        quads.insert(quads.end(), it, it + textRange.length);
                    }
                }
            }

            // Update TextRange
            auto start = quadStart;

            for (auto& textRange : ranges) {
                textRange.start = start;
                start += textRange.length;
            }

            labels.push_back(std::move(label));
        }

        m_textLabels->setLabels(labels);
        m_textLabels->setQuads(std::move(quads), m_atlasRefs);
    }

    m_labels.clear();
    m_quads.clear();

    return std::move(m_textLabels);
}

bool getTextSource(const StyleParamKey _key, const DrawRule& _rule, const Properties& _props,
                   std::string& _text) {

    auto& textSource = _rule.findParameter(_key);
    if (textSource.value.is<StyleParam::TextSource>()) {
        for (auto& key : textSource.value.get<StyleParam::TextSource>().keys) {
            _text = _props.getAsString(key);
            if (!_text.empty()) { break; }
        }
    } else if (textSource.value.is<std::string>()) {
        // From function evaluation
        _text = textSource.value.get<std::string>();
    } else if (textSource.value.is<Undefined>()) {
        _text = "";
    } else {
        return false;
    }
    return true;
}

bool TextStyleBuilder::handleBoundaryLabel(const Feature& _feat, const DrawRule& _rule,
                                           const TextStyle::Parameters& _params) {

    if (_feat.geometryType != GeometryType::lines) { return false; }

    LabelAttributes leftAttribs, rightAttribs;
    std::string textLeft = std::move(_params.textLeft);
    std::string textRight = std::move(_params.textRight);

    TextStyle::Parameters rightParams = _params;
    TextStyle::Parameters leftParams = _params;

    // Deactivate offset horizontally for boundary labels
    rightParams.labelOptions.offset.x = 0.0f;
    leftParams.labelOptions.offset.x = 0.0f;

    bool hasLeftLabel = false;
    if (!textLeft.empty()) {
        leftParams.text = std::move(textLeft);
        leftParams.labelOptions.optional = true;
        leftParams.labelOptions.anchors = {LabelProperty::Anchor::top};
        leftParams.labelOptions.buffer = glm::vec2(0);
        hash_combine(leftParams.labelOptions.repeatGroup, leftParams.text);

        hasLeftLabel = prepareLabel(leftParams, Label::Type::line, leftAttribs);
    }

    bool hasRightLabel = false;
    if (!textRight.empty()) {
        rightParams.text = std::move(textRight);
        rightParams.labelOptions.optional = true;
        rightParams.labelOptions.anchors = {LabelProperty::Anchor::bottom};
        rightParams.labelOptions.buffer = glm::vec2(0);
        hash_combine(rightParams.labelOptions.repeatGroup, rightParams.text);

        hasRightLabel = prepareLabel(rightParams, Label::Type::line, rightAttribs);
    }

    if (!hasLeftLabel && !hasRightLabel) {
        return false;
    }

    float labelWidth = std::max(leftAttribs.width, rightAttribs.width);

    auto onAddLabel = [&](glm::vec2 a, glm::vec2 b) {
        Label* left = nullptr;
        Label* right = nullptr;
        if (hasLeftLabel) {
            left = addLabel(Label::Type::line, {{ a, b }}, leftParams, leftAttribs, _rule);
        }
        if (hasRightLabel) {
            right = addLabel(Label::Type::line, {{ a, b }}, rightParams, rightAttribs, _rule);
        }
        if (left && right) {
            left->setRelative(*right, false, false);
            right->setRelative(*left, false, false);
        }
    };

    bool added = false;
    for (auto& line : _feat.lines) {
        added |= addStraightTextLabels(line, labelWidth, onAddLabel);
    }

    return added;
}

bool TextStyleBuilder::addFeature(const Feature& _feat, const DrawRule& _rule) {

    if (!checkRule(_rule)) { return false; }

    TextStyle::Parameters params = applyRule(_rule, _feat.props, false);
    if (!params.font) { return false; }

    Label::Type labelType;
    if (_feat.geometryType == GeometryType::lines) {
        labelType = Label::Type::line;
        params.wordWrap = false;
    } else {
        labelType = Label::Type::point;
    }
    // Keep start position of new quads
    size_t quadsStart = m_quads.size();
    size_t numLabels = m_labels.size();

    if (!params.textLeft.empty() || !params.textRight.empty()) {
        if (!handleBoundaryLabel(_feat, _rule, params)) { return false; }

    } else {
        LabelAttributes attrib;
        if (!prepareLabel(params, labelType, attrib)) { return false; }

        if (_feat.geometryType == GeometryType::points) {
            for (auto& point : _feat.points) {
                auto p = glm::vec2(point);
                // Standalone text label (no linked icon) -- salience-aware anchor
                // ordering (docs/label-placement-plan.md Phase 3), zero icon radius.
                params.labelOptions.anchors = salienceOrderedAnchors(
                    params.labelOptions.anchors, p, glm::vec2(attrib.width, attrib.height), 0.f);
                addLabel(Label::Type::point, {{ p }}, params, attrib, _rule);
            }

        } else if (_feat.geometryType == GeometryType::polygons) {
            const auto& polygons = _feat.polygons;
            for (const auto& polygon : polygons) {
                if (polygon.empty() || polygon.front().empty()) { continue; }

                // Phase 5 (docs/label-placement-plan.md): elongated polygons (lakes,
                // parks, forests) get a curved label along their principal axis instead
                // of a bare centroid point. Only attempt this for simple polygons (a
                // single ring, i.e. no holes) with no line-offset/complex-shaping text,
                // matching the preconditions addLineTextLabels already uses for curved
                // line labels -- anything else falls back to the point label below.
                if (params.letterSpacingFit > 0.f && !params.hasComplexShaping) {
                    // Area label: lake / park / glacier name that must stay inside the shape.
                    addAreaLabel(polygon, params, _rule);
                    continue;
                }
                bool curved = false;
                if (!params.hasComplexShaping && params.labelOptions.offset == glm::vec2(0)) {
                    curved = addCurvedPolygonLabel(polygon, params, attrib, _rule);
                }

                if (!curved) {
                    glm::vec2 c;
                    c = centroid(polygon.front().begin(), polygon.front().end());

                    // A compact polygon (round lake, small park) can still be tracked out to
                    // fill its width: fit the label to the horizontal chord through the
                    // centroid.
                    TextStyle::Parameters pointParams = params;
                    LabelAttributes pointAttrib = attrib;
                    if (params.letterSpacingFit > 0.f && !touchesTileEdge(polygon)) {
                        glm::vec2 s, e;
                        if (clipLineToPolygon(c, glm::vec2(1.f, 0.f), polygon.front().begin(),
                                              polygon.front().end(), s, e)) {
                            LabelAttributes tracked;
                            TextStyle::Parameters trackedParams = params;
                            trackedParams.wordWrap = false;
                            if (prepareTrackedLabel(trackedParams, Label::Type::point, tracked,
                                                    glm::length(e - s) * m_tileSize)) {
                                pointParams = trackedParams;
                                pointAttrib = tracked;
                            }
                        }
                    }

                    pointParams.labelOptions.anchors = salienceOrderedAnchors(
                        pointParams.labelOptions.anchors, c,
                        glm::vec2(pointAttrib.width, pointAttrib.height), 0.f);
                    addLabel(Label::Type::point, {{ c }}, pointParams, pointAttrib, _rule);
                }
            }

        } else if (_feat.geometryType == GeometryType::lines) {
            addLineTextLabels(_feat, params, attrib, _rule);
        }
    }

    if (numLabels == m_labels.size()) {
        // Drop quads when no label was added
        m_quads.resize(quadsStart);
    }
    return true;
}

#define TANGRAM_NEW_STRAIGHT_LABELS

bool TextStyleBuilder::addStraightTextLabels(const Line& _line, float _labelWidth,
                                             const std::function<void(glm::vec2,glm::vec2)>& _onAddLabel) {

    // Size of pixel in tile coordinates
    float pixelSize = 1.0f/m_tileSize;

    // Minimal length of line needed for the label
    float minLength = _labelWidth * pixelSize;

    // Allow labels to appear later than tile's min-zoom
    minLength *= 0.6f;

    //float tolerance = pow(pixelScale * 2, 2);
    float tolerance = powf(pixelSize * 1.5f, 2);
    float sqDirLimit = powf(1.99f, 2);

    for (size_t i = 0; i < _line.size() - 1; i++) {
        glm::vec2 p0 = glm::vec2(_line[i]);
        glm::vec2 p1 = glm::vec2(_line[i+1]);

        float segmentLength = glm::length(p0 - p1);

        glm::vec2 dir0 = (p0 - p1) / segmentLength;
        glm::vec2 dir1 = dir0;
        glm::vec2 dir2;

        int merged = 0;

        size_t j = i + 2;
        for (; j < _line.size(); j++) {
            glm::vec2 p2 = glm::vec2(_line[j]);

            segmentLength = glm::length(p1 - p2);
            dir2 = (p1 - p2) / segmentLength;

            glm::vec2 pp = glm::vec2(_line[j-1]);
            float d = pointSegmentDistanceSq(pp, p0, p2);
            if (d > tolerance) { break; }

            if ((glm::length2(dir1 + dir2) < sqDirLimit) ||
                (glm::length2(dir0 + dir2) < sqDirLimit)) {
                break;
            }

            merged++;

            p1 = p2;
            dir1 = dir2;
        }

        // place labels at segment-subdivisions
        int run = 1;
        if (merged) { segmentLength = glm::length(p0 - p1); }

        while (segmentLength > minLength && run <= 4) {
            glm::vec2 a = p0;
            glm::vec2 b = glm::vec2(p1 - p0) / float(run);

            for (int r = 0; r < run; r++) {
#ifdef TANGRAM_NEW_STRAIGHT_LABELS
                // no labels outside tile
                const float tol = 2*minLength;
                glm::vec2 z = a+b;
                if((a.x < -tol || a.x > 1+tol || a.y < -tol || a.y > 1+tol) &&
                   (z.x < -tol || z.x > 1+tol || z.y < -tol || z.y > 1+tol)) { continue; }
#endif
                _onAddLabel(a, a+b);

                a += b;
            }
            run *= 2;
            segmentLength /= 2.0f;
        }

        if (i == 0 && j == _line.size()) {
            // Simple straight line
            return true;
        }

        // Skip merged segments in outer loop
        i += merged;

    }
    return false;
}

//#define TANGRAM_NEW_CURVED_LABELS

void TextStyleBuilder::addCurvedTextLabels(const Line& _line, const TextStyle::Parameters& _params,
                                           const LabelAttributes& _attributes, const DrawRule& _rule,
                                           bool _centered) {

    // Size of pixel in tile coordinates
    const float pixelSize = 1.0f/m_tileSize;
    // length of line needed for the label
    const float labelLength = _attributes.width * pixelSize;
    // Allow labels to appear later than tile's min-zoom
    const float minLength = labelLength * 0.6f;

    // Chord length for minimal ~120 degree inner angles (squared)
    // sin(60)*2
    const float sqDirLimit = powf(1.7f, 2);
    // Range to check for angle changes
    const float sampleWindow = pixelSize * 50;

    // Minimal ~10 degree counts as change of direction
    // cross(dir1,dir2) < sin(10)
    const float flipTolerance = 0.17f;

    LineSampler<std::vector<glm::vec3>> sampler;

    sampler.set(_line);

    if (sampler.sumLength() < minLength) { return; }

    struct LineRange {
        size_t start, end;
        int flips;
        float sumAngle;
    };

    std::vector<LineRange> ranges;
    size_t npoints = sampler.numPoints();  // this can be less than _line.size() if _line has duplicate points

    if (_centered && npoints >= 2) {
        // One candidate spanning the whole (pre-smoothed, pre-trimmed) path, centered on its
        // midpoint: the spine of a polygon. No window sliding, no direction-change splits
        // (the spine is already a calm curve; updateScreenTransform still rejects folds).
        float sumAngle = 0.f;
        for (size_t k = 1; k + 1 < npoints; k++) {
            sumAngle += std::abs(perpDotProduct(sampler.segmentDirection(k - 1), sampler.segmentDirection(k)));
        }
        ranges.push_back(LineRange{0, npoints, 0, sumAngle});
    }

    for (size_t i = 0; !_centered && i < npoints-1; i++) {
#ifdef TANGRAM_NEW_CURVED_LABELS
        // only process labels starting in this tile
        const auto& p0 = sampler.point(i);
        const float tol = 0.0005f;
        if(p0.x < -tol || p0.x > 1+tol || p0.y < -tol || p0.y > 1+tol) continue;  // !_params.keepTileEdges
#endif
        int flips = 0;
        float lastAngle = 0;
        float sumAngle = 0;
        size_t lastBreak = 0;

        glm::vec2 dir1 = sampler.segmentDirection(i);

        for (size_t j = i + 1; j < npoints-1; j++) {
            glm::vec2 dir2 = sampler.segmentDirection(j);
            bool splitLine = false;

            if (glm::length2(dir1 + dir2) < sqDirLimit) {
                // Split if the angle between current and next segment is
                // not whithin 120 < a < 240 degree
                splitLine = true;
            } else {
                // Take cross product of direction (unit-) vectors of the current
                // and next segment. The magnitude of the cross product is the sine
                // angle between dir1 and dir2.
                float angle = perpDotProduct(dir1, dir2);

                if (std::abs(angle) > flipTolerance) {
                    if (lastAngle > 0) {
                        if (angle < 0) { flips++; }
                    } else if (lastAngle < 0) {
                        if (angle > 0) { flips++; }
                    }
                    lastAngle = angle;
                }

                // Limit number of direction changes (Avoid squiggly labels)
                if (flips > 2) {
                    splitLine = true;
                } else {
                    sumAngle += std::abs(angle);
                }
            }

            if (!splitLine) {
                // Go back within window to check for hard direction changes
                for (int k = j - 1; k >= int(i); k--){
                    if (glm::length2(sampler.segmentDirection(k) + dir2) < sqDirLimit) {
                        splitLine = true;
                    }
                    if (sampler.point(k).z < sampler.point(j).z - sampleWindow) {
                        break;
                    }
                }
            }

#ifdef TANGRAM_NEW_CURVED_LABELS
            float length = sampler.point(j).z - sampler.point(i).z;
            if (length > labelLength || (length > minLength && splitLine)) {
                // label candidate spacing to avoid excessive number of candidates; not related to repeat_distance!
                const float labelPeriod = 30 * pixelSize;
                int seg = int(sampler.point(i).z/labelPeriod);
                int segprev = ranges.empty() ? seg+1 : int(sampler.point(ranges.back().start).z/labelPeriod);
                if (seg == segprev) {
                    if (ranges.back().sumAngle <= sumAngle)
                      break;  // discard current placement
                    ranges.pop_back();  // discard last placement
                }
                ranges.push_back(LineRange{i, j+1, flips, sumAngle});
                break;
            } else if (splitLine) {
                break;
            } else {
                dir1 = dir2;
            }
#else
            if (splitLine) {
                float length = sampler.point(j).z - sampler.point(i).z;
                if (length > minLength) {
                    ranges.push_back(LineRange{i, j+1, flips, sumAngle});
                }

                lastBreak = j;
                break;

            } else {
                dir1 = dir2;
            }
#endif
        }
#ifndef TANGRAM_NEW_CURVED_LABELS
        // Add segment from 'i' unless line got split.
        if (lastBreak == 0) {
            float length = sampler.sumLength() - sampler.point(i).z;
            if (length > minLength) {
                ranges.push_back(LineRange{i, npoints, flips, sumAngle});
            }
        }
#endif
    }

    for (auto& range : ranges) {
        glm::vec2 center;
        glm::vec2 rotation;
        float startLen = sampler.point(range.start).z;
        float length = (sampler.point(range.end-1).z - startLen);
        float mid = startLen + length * 0.5f;

        sampler.sample(mid, center, rotation);
        size_t offset = sampler.curSegment();

        std::vector<glm::vec3> l;
        l.reserve(range.end - range.start + 1);

        for (size_t j = range.start; j < range.end; j++) {
            auto p = sampler.point(j);
            l.emplace_back(p.x, p.y, 0.f);

            if (j == offset) {
                l.push_back(glm::vec3(center, 0.f));
            }
        }
        size_t anchor = offset - range.start + 1;

        // NB: Just some heuristic to prefer longer and less curvy parts..
        float prio = (1.f + range.sumAngle) / length;

        uint32_t selectionColor = 0;
        if (_params.interactive) {
            // markers have no tile FeatureSelection: use the colour their rule was given
            selectionColor = _rule.featureSelection ? _rule.featureSelection->nextColorIdentifier()
                                                    : _rule.selectionColor;
        }

        m_labels.emplace_back(new CurvedLabel(l, _params.labelOptions, prio,
                                               {_attributes.fill,
                                                       _attributes.stroke,
                                                       _attributes.fontScale,
                                                       selectionColor},
                                               {_attributes.width, _attributes.height},
                                               *m_textLabels, _attributes.textRanges,
                                               TextLabelProperty::Align::center,
                                               anchor));
#ifdef DEBUG
        m_labels.back()->debugTag = _params.text;
#endif
    }
}

// Elongation ratio (major/minor axis, see util/geom.h PolygonAxis) above which a polygon
// label is curved along its spine instead of sitting at a bare centroid point.
// Tuned by eye; per docs/label-placement-plan.md Phase 5 this needs human visual tuning.
static constexpr float kPolygonLabelElongationThreshold = 1.5f;

bool TextStyleBuilder::prepareTrackedLabel(TextStyle::Parameters& _params, Label::Type _type,
                                           LabelAttributes& _attributes, float _pathLengthPx) {
    const float base = _params.letterSpacing;
    if (_params.letterSpacingFit <= 0.f || _pathLengthPx <= 0.f) {
        return prepareLabel(_params, _type, _attributes);
    }

    // Width is affine in the tracking: w(ls) = w(base) + gaps * (ls - base) * fontSize.
    // Measure the slope with a second probe instead of counting clusters by hand, so it
    // is exact for whatever the shaper decided (ligatures, combining marks, line breaks).
    const size_t quadsStart = m_quads.size();
    constexpr float kProbe = 0.2f;

    LabelAttributes a0, a1;
    _params.letterSpacing = base;
    if (!prepareLabel(_params, _type, a0)) { return false; }
    _params.letterSpacing = base + kProbe;
    if (!prepareLabel(_params, _type, a1)) { return false; }
    m_quads.resize(quadsStart);  // drop both probes' quads

    float gaps = (a1.width - a0.width) / (kProbe * _params.fontSize);
    float ls = base;
    if (gaps > 0.5f) {
        float target = _pathLengthPx * _params.letterSpacingFit;
        float extra = (target - a0.width) / (gaps * _params.fontSize);
        ls = std::min(base + std::max(extra, 0.f), std::max(_params.letterSpacingMax, base));
    }
    _params.letterSpacing = ls;
    return prepareLabel(_params, _type, _attributes);
}

// ---------------------------------------------------------------------------------------
// Area label planner (docs/label-typography.md)
// ---------------------------------------------------------------------------------------

// A tile's polygon is only a fragment of the real feature. Edges that lie ON the tile boundary
// are clip edges, not shoreline: the feature goes on in the neighbouring tile. Treat them as
// open by pushing them outwards a little, so a name wider than one tile can still be placed
// (the neighbouring tiles' fragments compete for the same name via a repeat group).
static std::vector<glm::vec2> extendOpenTileEdges(const std::vector<glm::vec2>& ring, float maxDepth) {
    const float eps = 0.003f;
    std::vector<glm::vec2> out;
    const size_t n = ring.size();
    for (size_t i = 0; i < n; i++) {
        glm::vec2 a = ring[i], b = ring[(i + 1) % n];
        out.push_back(a);
        glm::vec2 nrm(0.f);
        if (a.x < eps && b.x < eps) { nrm = {-1.f, 0.f}; }
        else if (a.x > 1.f - eps && b.x > 1.f - eps) { nrm = {1.f, 0.f}; }
        else if (a.y < eps && b.y < eps) { nrm = {0.f, -1.f}; }
        else if (a.y > 1.f - eps && b.y > 1.f - eps) { nrm = {0.f, 1.f}; }
        float len = glm::length(b - a);
        if ((nrm.x != 0.f || nrm.y != 0.f) && len > 1e-4f) {
            // How far the feature may be assumed to continue: a thin lake crossing the tile
            // edge continues a little (and may bend away), a park spanning most of the edge
            // continues a lot.
            float depth = std::min(maxDepth, 0.35f * len);
            out.push_back(a + nrm * depth);
            out.push_back(b + nrm * depth);
        }
    }
    return out;
}

static size_t utf8Length(const std::string& s) {
    size_t n = 0;
    for (unsigned char c : s) { if ((c & 0xC0) != 0x80) { n++; } }
    return n;
}

// Line breaking for area labels. Text is cut into tokens at spaces and at '\n', which the scene
// uses as a SOFT HYPHEN ("Theodul\ngletscher"): unbroken it reads "Theodulgletscher", broken it
// reads "Theodul-" / "gletscher". Nothing else is ever split.
struct BreakToken { std::string text; bool soft = false; };  // `soft`: soft hyphen follows

static std::vector<BreakToken> tokenizeForBreaks(const std::string& text) {
    std::vector<BreakToken> toks;
    std::string cur;
    for (char c : text) {
        if (c == ' ' || c == '\n') {
            if (!cur.empty()) { toks.push_back({cur, c == '\n'}); cur.clear(); }
            else if (c == '\n' && !toks.empty()) { toks.back().soft = true; }
        } else {
            cur.push_back(c);
        }
    }
    if (!cur.empty()) { toks.push_back({cur, false}); }
    return toks;
}

// Text for lines made of tokens [cuts[i-1]+1 .. cuts[i]]; `cuts` are indices of the last token of
// each line but the final one. Also returns the length of the longest line in characters.
static std::string joinLines(const std::vector<BreakToken>& toks, const std::vector<size_t>& cuts,
                             size_t& longest) {
    std::string out;
    size_t lineLen = 0;
    longest = 0;
    size_t cutIdx = 0;
    for (size_t i = 0; i < toks.size(); i++) {
        out += toks[i].text;
        lineLen += utf8Length(toks[i].text);
        if (i + 1 == toks.size()) { break; }
        if (cutIdx < cuts.size() && cuts[cutIdx] == i) {
            if (toks[i].soft) { out += "-"; lineLen += 1; }
            out += "\n";
            longest = std::max(longest, lineLen);
            lineLen = 0;
            cutIdx++;
        } else if (!toks[i].soft) {
            out += " ";
            lineLen += 1;
        }
    }
    longest = std::max(longest, lineLen);
    return out;
}

// Best balanced split of the tokens into exactly `lines` lines (smallest longest line, soft
// hyphen breaks slightly disfavoured). Returns false if there are too few tokens.
static bool bestLineBreaks(const std::vector<BreakToken>& toks, int lines, std::string& textOut) {
    const size_t n = toks.size();
    if (lines < 1 || n < size_t(lines)) { return false; }
    std::vector<size_t> best, cuts;
    float bestCost = 1e30f;
    auto consider = [&](const std::vector<size_t>& c) {
        size_t longest;
        joinLines(toks, c, longest);
        float cost = float(longest);
        for (size_t k : c) { if (toks[k].soft) { cost += 2.5f; } }
        if (cost < bestCost) { bestCost = cost; best = c; }
    };
    if (lines == 1) { consider({}); }
    else if (lines == 2) { for (size_t a = 0; a + 1 < n; a++) { consider({a}); } }
    else { for (size_t a = 0; a + 2 < n; a++) { for (size_t b = a + 1; b + 1 < n; b++) { consider({a, b}); } } }
    if (bestCost > 1e29f) { return false; }
    size_t longest;
    textOut = joinLines(toks, best, longest);
    return true;
}

bool TextStyleBuilder::addAreaLabel(const Polygon& _polygon, const TextStyle::Parameters& _params,
                                    const DrawRule& _rule) {
    if (_polygon.empty() || _polygon.front().size() < 3) { return true; }

    const float ps = m_tileSize;               // tile units -> px
    const float pixelSize = 1.f / ps;
    const float pixelScale = m_style.pixelScale();

    // Rings (outer ring with its open tile edges pushed out), in tile units and in px.
    std::vector<std::vector<glm::vec2>> ringsTile, ringsPx, realPx;
    const float kExtend = 0.5f;  // max tile units pushed out past an open clip edge
    for (size_t r = 0; r < _polygon.size(); r++) {
        std::vector<glm::vec2> ring;
        ring.reserve(_polygon[r].size());
        for (const auto& pt : _polygon[r]) { ring.emplace_back(pt.x, pt.y); }
        if (ring.size() < 3) { continue; }
        {
            std::vector<glm::vec2> rpx(ring.size());
            for (size_t i = 0; i < ring.size(); i++) { rpx[i] = ring[i] * ps; }
            realPx.push_back(simplifyPolyline(rpx, 0.35f * pixelScale));
            if (realPx.back().size() < 3) { realPx.pop_back(); }
        }
        if (r == 0) { ring = extendOpenTileEdges(ring, kExtend); }
        ringsTile.push_back(ring);
        std::vector<glm::vec2> px(ring.size());
        for (size_t i = 0; i < ring.size(); i++) { px[i] = ring[i] * ps; }
        ringsPx.push_back(simplifyPolyline(px, 0.35f * pixelScale));
        if (ringsPx.back().size() < 3) { ringsPx.pop_back(); ringsTile.pop_back(); }
    }
    if (ringsPx.empty() || realPx.empty()) { return true; }

    InsideMask mask;       // polygon + its open tile edges pushed out (search space)
    if (!mask.build(ringsPx)) { return true; }

    TextStyle::Parameters base = _params;
    base.labelOptions.anchors = {LabelProperty::Anchor::center};
    base.labelOptions.offset = glm::vec2(0.f);
    base.pathOffset = 0.f;

    const float fs = base.fontSize;
    const float lsMin = std::max(base.letterSpacing, 0.f);
    const float lsMax = std::max(base.letterSpacingMax, lsMin);
    const float fill = Tangram::clamp(base.letterSpacingFit, 0.3f, 0.95f);
    const float spMin = std::max(base.lineSpacing, 0.15f * fs);
    const float spMax = std::max(3.f * fs, spMin);
    const float margin = (2.f + 0.5f * base.strokeWidth) * pixelScale;
    constexpr float kProbe = 0.2f;

    // Measure the text as it would be laid out (quads are dropped again).
    auto measure = [&](TextStyle::Parameters p, float ls, float sp, Label::Type type,
                       LabelAttributes& a) {
        p.letterSpacing = ls;
        p.lineSpacing = sp;
        const size_t q = m_quads.size();
        a = LabelAttributes{};
        bool ok = prepareLabel(p, type, a);
        m_quads.resize(q);
        return ok;
    };

    {   // apply the text transform once, so word splitting sees the final string
        LabelAttributes tmp;
        TextStyle::Parameters p = base;
        p.wordWrap = false;
        if (!measure(p, lsMin, spMin, Label::Type::point, tmp)) { return true; }
        base.text = p.text;
    }

    const std::vector<BreakToken> toks = tokenizeForBreaks(base.text);
    if (toks.empty()) { return true; }

    struct Choice {
        bool valid = false;
        TextStyle::Parameters params;
        float ls = 0.f, sp = 0.f, w = 0.f, h = 0.f, cx = 0.f, cy = 0.f, avail = 0.f, score = 0.f;
        int lines = 1;
    };

    // Vertical room at x-centre `cx` for a box `w` wide, growing symmetrically about `cy`.
    auto verticalRoom = [&](float cx, float cy, float w, float h0, float hCap) {
        float lo = h0, hi = std::max(hCap, h0);
        if (!mask.rectInside({cx - w * 0.5f - margin, cy - hi * 0.5f - margin},
                             {cx + w * 0.5f + margin, cy + hi * 0.5f + margin})) {
            for (int it = 0; it < 8; it++) {
                float mid = 0.5f * (lo + hi);
                if (mask.rectInside({cx - w * 0.5f - margin, cy - mid * 0.5f - margin},
                                    {cx + w * 0.5f + margin, cy + mid * 0.5f + margin})) { lo = mid; }
                else { hi = mid; }
            }
            return lo;
        }
        return hi;
    };

    auto shapeMatch = [](float boxAspect, float regionAspect) {
        float d = std::abs(std::log(std::max(boxAspect, 0.05f) / std::max(regionAspect, 0.05f)));
        return Tangram::clamp(1.f - 0.35f * d, 0.25f, 1.f);
    };

    // Other map content under a candidate box (ferry routes, roads, shorelines), sampled from
    // the tile's coarse occupancy grid: a label should not be struck through by a ferry line.
    std::function<float(glm::vec2, glm::vec2)> busy = [&](glm::vec2 lo, glm::vec2 hi) {
        float cost = 0.f;
        if (!m_anchorOccupancyGrid) { return cost; }
        {
            int hits = 0, total = 0;
            for (int iy = 0; iy < 3; iy++) {
                for (int ix = 0; ix < 8; ix++) {
                    glm::vec2 q = lo + (hi - lo) * glm::vec2((ix + 0.5f) / 8.f, (iy + 0.5f) / 3.f);
                    total++;
                    if (m_anchorOccupancyGrid->sample(q * pixelSize) > 0.3f) { hits++; }
                }
            }
            cost += float(hits) / float(total);
        }
        return cost;
    };

    // --- horizontal blocks, 1..3 lines ---------------------------------------------------
    Choice horizontal;
    float room1 = 0.f, avail1 = 0.f;
    bool tall = true;  // until a single line is shown to sit comfortably
    const int maxLines = int(std::min<size_t>(3, toks.size()));
    for (int k = 1; k <= maxLines; k++) {
        TextStyle::Parameters p = base;
        if (!bestLineBreaks(toks, k, p.text)) { continue; }
        if (k == 1) {
            p.wordWrap = false;
        } else {
            p.wordWrap = true;                 // only the explicit line breaks apply
            p.maxLines = k;
            p.maxLineWidth = 1u << 20;
            p.align = TextLabelProperty::Align::center;
        }
        if (k > 1 && !tall && horizontal.valid) { break; }

        LabelAttributes a0, a1;
        if (!measure(p, lsMin, spMin, Label::Type::point, a0)) { continue; }
        if (!measure(p, lsMin + kProbe, spMin, Label::Type::point, a1)) { continue; }
        const float slopeW = (a1.width - a0.width) / kProbe;  // px per em of tracking
        const float w0 = a0.width, h0 = a0.height;
        const float wTrackMax = w0 + slopeW * (lsMax - lsMin);

        InsideMask::Band band;
        if (!mask.widestBand(h0 + 2.f * margin, w0 + 2.f * margin,
                             (wTrackMax + 2.f * margin) / fill, band, busy, fill)) { continue; }
        const float avail = band.width - 2.f * margin;

        float w = Tangram::clamp(avail * fill, w0, std::max(wTrackMax, w0));
        float ls = slopeW > 1.f ? lsMin + (w - w0) / slopeW : lsMin;
        ls = Tangram::clamp(ls, lsMin, lsMax);
        w = w0 + slopeW * (ls - lsMin);

        const float hCap = k == 1 ? 12.f * h0 : h0 + (k - 1) * (spMax - spMin);
        const float room = verticalRoom(band.cx, band.cy, w, h0, hCap);
        if (k == 1) {
            room1 = room; avail1 = avail;
            tall = room >= 0.7f * avail;
        }

        float h = h0, sp = spMin;
        if (k > 1) {
            h = Tangram::clamp(room * 0.8f, h0, hCap);
            sp = spMin + (h - h0) / float(k - 1);
        }
        if (k > 1 && !tall) { continue; }

        Choice c;
        c.valid = true; c.params = p; c.ls = ls; c.sp = sp; c.w = w; c.h = h;
        c.cx = band.cx; c.cy = band.cy; c.avail = avail; c.lines = k;
        c.score = shapeMatch(w / h, avail / std::max(room, h0)) * std::pow(0.85f, float(k - 1));
        if (!horizontal.valid || c.score > horizontal.score) { horizontal = c; }
    }

    // --- one line along the spine ---------------------------------------------------------
    Choice path;
    std::vector<glm::vec2> pathTile;
    float pathTurning = 0.f;
    PolygonAxis axis = polygonPrincipalAxis(ringsTile.front().begin(), ringsTile.front().end());
    if (axis.elongation >= 1.8f) {
        TextStyle::Parameters p = base;
        p.wordWrap = false;
        if (!bestLineBreaks(toks, 1, p.text)) { p.text = base.text; }
        LabelAttributes a0, a1;
        if (measure(p, lsMin, spMin, Label::Type::line, a0) &&
            measure(p, lsMin + kProbe, spMin, Label::Type::line, a1)) {
            const float slopeW = (a1.width - a0.width) / kProbe;
            const float w0 = a0.width, h0 = a0.height;
            const float wTrackMax = w0 + slopeW * (lsMax - lsMin);
            const float sigma = (p.pathSmoothing > 0.f ? p.pathSmoothing : 14.f * pixelScale) * pixelSize;

            PolygonCenterline spine;
            if (polygonCenterline(ringsTile, axis.direction, (h0 + 2.f * margin) * pixelSize,
                                  (w0 + 2.f * margin) * pixelSize, sigma, spine)) {
                std::vector<glm::vec2> sp;
                for (const auto& q : spine.points) { sp.push_back(q * ps); }
                const float step = std::max(2.f * pixelScale, mask.cell());
                sp = resamplePolyline(sp, step);
                const size_t n = sp.size();

                // Local clearance on both sides of the spine, and the longest run where the
                // whole text strip fits.
                const float half = h0 * 0.5f + margin;
                std::vector<float> wid(n, 0.f);
                std::vector<char> clear(n, 0);
                for (size_t i = 0; i < n; i++) {
                    glm::vec2 t = sp[std::min(i + 1, n - 1)] - sp[i > 0 ? i - 1 : 0];
                    float tl = glm::length(t);
                    if (tl < 1e-6f) { continue; }
                    t /= tl;
                    glm::vec2 nrm(-t.y, t.x);
                    float d1 = mask.rayLength(sp[i], nrm, 4.f * h0);
                    float d2 = mask.rayLength(sp[i], -nrm, 4.f * h0);
                    wid[i] = d1 + d2;
                    clear[i] = (d1 >= half && d2 >= half && mask.pointInside(sp[i])) ? 1 : 0;
                }
                size_t bestS = 0, bestE = 0;
                for (size_t i = 0; i < n;) {
                    if (!clear[i]) { i++; continue; }
                    size_t s = i;
                    while (i < n && clear[i]) { i++; }
                    if (i - s > bestE - bestS) { bestS = s; bestE = i; }
                }
                const float runLen = float(bestE - bestS > 0 ? bestE - bestS - 1 : 0) * step;
                if (runLen >= w0 + 2.f * margin) {
                    const float avail = runLen - 2.f * margin;
                    float w = Tangram::clamp(avail * fill, w0, std::max(wTrackMax, w0));
                    float ls = slopeW > 1.f ? Tangram::clamp(lsMin + (w - w0) / slopeW, lsMin, lsMax) : lsMin;
                    w = w0 + slopeW * (ls - lsMin);

                    // Slide the window to the thickest part of the run.
                    const size_t wn = std::min(bestE - bestS, size_t(std::ceil((w + 2.f * margin) / step)) + 1);
                    std::vector<float> pre(n + 1, 0.f);
                    for (size_t i = 0; i < n; i++) { pre[i + 1] = pre[i] + std::min(wid[i], 4.f * h0); }
                    size_t bestStart = bestS;
                    float bestMean = -1.f;
                    for (size_t s0 = bestS; s0 + wn <= bestE; s0++) {
                        float mean = (pre[s0 + wn] - pre[s0]) / float(wn);
                        float centre = 0.5f * (bestS + bestE) - 0.5f * (s0 + s0 + wn);
                        mean -= 0.01f * std::abs(centre) * step;  // prefer the middle on ties
                        if (mean > bestMean) { bestMean = mean; bestStart = s0; }
                    }
                    float meanWid = std::max((pre[bestStart + wn] - pre[bestStart]) / float(wn), h0);

                    size_t pad = std::min<size_t>(2, bestStart > bestS ? bestStart - bestS : 0);
                    size_t padE = std::min<size_t>(2, bestE > bestStart + wn ? bestE - (bestStart + wn) : 0);
                    for (size_t i = bestStart - pad; i < bestStart + wn + padE; i++) {
                        pathTile.push_back(sp[i] * pixelSize);
                    }

                    // Reject strongly bent paths: angle between the first and last thirds.
                    float turning = 0.f;
                    if (pathTile.size() >= 6) {
                        size_t m = pathTile.size();
                        glm::vec2 d1 = pathTile[m / 3] - pathTile[0];
                        glm::vec2 d2 = pathTile[m - 1] - pathTile[2 * m / 3];
                        float l1 = glm::length(d1), l2 = glm::length(d2);
                        if (l1 > 1e-9f && l2 > 1e-9f) {
                            turning = std::acos(Tangram::clamp(glm::dot(d1, d2) / (l1 * l2), -1.f, 1.f));
                        }
                    }
                    pathTurning = turning;
                    path.valid = pathTile.size() >= 2 && turning < 0.9f;
                    path.params = p; path.ls = ls; path.w = w; path.h = h0; path.avail = avail;
                    path.score = shapeMatch(w / h0, avail / meanWid);
                }
            }
        }
    }

    // Lakes, glaciers: follow the shape of the water / ice (a tongue, a cirque) unless there is
    // plenty of room around the name -- then plain horizontal type reads best. Parks
    // (smooth == 0) are set as a line-broken block and bent only if no block fits.
    const bool bendFirst = base.pathSmoothing > 0.f;
    const bool roomy = horizontal.valid && mask.insideArea() >= 0.5f * ps * ps &&
                       horizontal.avail >= 1.5f * horizontal.w;
    bool usePath = path.valid && (!horizontal.valid || (bendFirst && !roomy));

    const Choice& pick = usePath ? path : horizontal;
    if (!pick.valid) { return true; }  // no room: no label

    // --- emit ----------------------------------------------------------------------------
    TextStyle::Parameters params = pick.params;
    params.letterSpacing = pick.ls;
    params.lineSpacing = usePath ? params.lineSpacing : pick.sp;

    // One label per feature: all fragments of this name compete, the roomiest wins.
    hash_combine(params.labelOptions.repeatGroup, params.text);
    params.labelOptions.repeatDistance = std::max(params.labelOptions.repeatDistance, 1200.f * pixelScale);
    params.labelOptions.priority += 0.03f * (1.f - std::min(1.f, pick.avail / (3.f * pick.w)));

    LabelAttributes attrib;
    if (usePath) {
        if (!prepareLabel(params, Label::Type::line, attrib)) { return true; }
        const size_t before = m_labels.size();
        addCurvedTextLabels(pathTile, params, attrib, _rule, true);
        (void)before;
        return true;
    }

    const size_t quadsStart = m_quads.size();
    if (!prepareLabel(params, Label::Type::point, attrib)) { return true; }
    // Verify with the real layout: never trespass.
    glm::vec2 lo(pick.cx - attrib.width * 0.5f - margin, pick.cy - attrib.height * 0.5f - margin);
    glm::vec2 hi(pick.cx + attrib.width * 0.5f + margin, pick.cy + attrib.height * 0.5f + margin);
    if (!mask.rectInside(lo, hi)) {
        m_quads.resize(quadsStart);
        return true;
    }

    addLabel(Label::Type::point, {{ glm::vec2(pick.cx, pick.cy) * pixelSize }}, params, attrib, _rule);
    return true;
}

bool TextStyleBuilder::addCurvedPolygonLabel(const Polygon& _polygon, const TextStyle::Parameters& _params,
                                             const LabelAttributes& _attributes, const DrawRule& _rule) {
    if (_polygon.empty() || _polygon.front().size() < 3) { return false; }
    const Line& ring = _polygon.front();

    PolygonAxis axis = polygonPrincipalAxis(ring.begin(), ring.end());
    if (axis.elongation < kPolygonLabelElongationThreshold) { return false; }

    const float pixelSize = 1.0f / m_tileSize;

    // A path label is a single line of text, whatever the point-label wrapping said.
    TextStyle::Parameters params = _params;
    params.wordWrap = false;
    LabelAttributes attrib;
    const size_t quadsStart = m_quads.size();
    if (!prepareLabel(params, Label::Type::line, attrib)) { return false; }

    // Where does a text strip of this height physically fit? Same 0.6 length heuristic as
    // addCurvedTextLabels: the label may be a little longer than the spine at the tile's
    // own zoom, since it is shown over [z, z+1).
    const float minWidth = (attrib.height * 1.1f + 2.f) * pixelSize;
    const float minLength = attrib.width * pixelSize * 0.6f;
    const float sigma = (params.pathSmoothing > 0.f ? params.pathSmoothing : 14.f * m_style.pixelScale()) * pixelSize;

    PolygonCenterline spine;
    if (!polygonCenterline(_polygon, axis.direction, minWidth, minLength, sigma, spine)) {
        m_quads.resize(quadsStart);
        return false;
    }

    // Keep the glyphs off the shoreline/boundary at both tips.
    const float margin = std::min(6.f * pixelSize, 0.1f * spine.length);
    std::vector<glm::vec2> path = trimPolyline(spine.points, margin, margin);
    if (path.size() < 2) { m_quads.resize(quadsStart); return false; }

    // Fit-to-spine tracking (lakes, parks, glaciers: "L A C   L É M A N" filling the lake).
    if (params.letterSpacingFit > 0.f && !touchesTileEdge(_polygon)) {
        m_quads.resize(quadsStart);
        attrib = LabelAttributes{};
        if (!prepareTrackedLabel(params, Label::Type::line, attrib,
                                 polylineLength(path) * m_tileSize)) {
            return false;
        }
    }

    size_t labelsBefore = m_labels.size();
    addCurvedTextLabels(path, params, attrib, _rule, true);
    if (m_labels.size() == labelsBefore) { return false; }
    return true;
}

void TextStyleBuilder::addLineTextLabels(const Feature& _feat, const TextStyle::Parameters& _params,
                                         const LabelAttributes& _attributes, const DrawRule& _rule) {

    auto straightLabelCb = [&](glm::vec2 a, glm::vec2 b) {
        addLabel(Label::Type::line, {{ a, b }}, _params, _attributes, _rule);
    };

    const float pixelSize = 1.0f / m_tileSize;

    for (const auto& rawLine : _feat.lines) {

        // Typographic smoothing (docs/label-typography.md): follow the large-scale shape of
        // a ridge / valley / river, not its digitizing noise.
        Line smoothed;
        if (_params.pathSmoothing > 0.f && rawLine.size() > 2) {
            smoothed = smoothPolyline(rawLine, 3.f * pixelSize, _params.pathSmoothing * pixelSize);
        }
        Line shifted;
        if (_params.pathOffset != 0.f && (smoothed.empty() ? rawLine : smoothed).size() >= 2) {
            // Set the name beside the line, not on it: shift the path perpendicular to itself,
            // to the left of the reading direction (i.e. above it for left-to-right text).
            shifted = smoothed.empty() ? rawLine : smoothed;
            if (shifted.back().x < shifted.front().x) { std::reverse(shifted.begin(), shifted.end()); }
            Line src = shifted;
            const float d = _params.pathOffset * pixelSize;
            for (size_t i = 0; i < src.size(); i++) {
                glm::vec2 t = glm::vec2(src[std::min(i + 1, src.size() - 1)]) -
                              glm::vec2(src[i > 0 ? i - 1 : 0]);
                float tl = glm::length(t);
                if (tl < 1e-9f) { continue; }
                t /= tl;
                shifted[i].x = src[i].x - t.y * d;
                shifted[i].y = src[i].y + t.x * d;
            }
        }
        const Line& line = !shifted.empty() ? shifted : (smoothed.empty() ? rawLine : smoothed);

        if (!addStraightTextLabels(line, _attributes.width, straightLabelCb) &&
            line.size() > 2 && !_params.hasComplexShaping &&
            // TODO: support line offset for curved labels
            _params.labelOptions.offset == glm::vec2(0)) {
            addCurvedTextLabels(line, _params, _attributes, _rule);
        }
    }
}

bool TextStyleBuilder::checkRule(const DrawRule& _rule) const {
    if (_rule.hasParameterSet(StyleParamKey::text_font_family) ||
        _rule.hasParameterSet(StyleParamKey::text_font_fill) ||
        _rule.hasParameterSet(StyleParamKey::text_font_size) ||
        _rule.hasParameterSet(StyleParamKey::text_font_stroke_color) ||
        _rule.hasParameterSet(StyleParamKey::text_font_stroke_width) ||
        _rule.hasParameterSet(StyleParamKey::text_font_style) ||
        _rule.hasParameterSet(StyleParamKey::text_font_weight)) {
        return true;
    }
    return false;
}

TextStyle::Parameters TextStyleBuilder::applyRule(const DrawRule& _rule,
                                                  const Properties& _props,
                                                  bool _iconText) const {

    const static std::string defaultWeight("400");
    const static std::string defaultStyle("regular");
    const static std::string defaultFamily("default");

    TextStyle::Parameters p;

    bool hasLeft = getTextSource(StyleParamKey::text_source_left, _rule, _props, p.textLeft);
    bool hasRight = getTextSource(StyleParamKey::text_source_right, _rule, _props, p.textRight);

    if (!hasLeft && !hasRight) {
        if (!getTextSource(StyleParamKey::text_source, _rule, _props, p.text)) {
            // Use default key
            p.text = _props.getString(key_name);
        }
    }

    if (p.text.empty() && p.textLeft.empty() && p.textRight.empty()) {
        return p;
    }

    auto fontFamily = _rule.get<std::string>(StyleParamKey::text_font_family);
    fontFamily = (!fontFamily) ? &defaultFamily : fontFamily;

    auto fontWeight = _rule.get<std::string>(StyleParamKey::text_font_weight);
    fontWeight = (!fontWeight) ? &defaultWeight : fontWeight;

    auto fontStyle = _rule.get<std::string>(StyleParamKey::text_font_style);
    fontStyle = (!fontStyle) ? &defaultStyle : fontStyle;

    _rule.get(StyleParamKey::text_font_size, p.fontSize);
    p.fontSize *= m_style.pixelScale();

    p.font = m_style.context()->getFont(*fontFamily, *fontStyle, *fontWeight, p.fontSize);
    if (!p.font) {
        LOGW("Missing font for %s / %s / %s / %d", fontFamily->c_str(), fontStyle->c_str(), fontWeight->c_str(), p.fontSize);
        return p;
    }

    // Typography (docs/label-typography.md): tracking in em, optional fit-to-path tracking,
    // path smoothing radius (px).
    _rule.get(StyleParamKey::text_font_letter_spacing, p.letterSpacing);
    _rule.get(StyleParamKey::text_font_letter_spacing_fit, p.letterSpacingFit);
    _rule.get(StyleParamKey::text_font_letter_spacing_max, p.letterSpacingMax);
    _rule.get(StyleParamKey::text_smooth, p.pathSmoothing);
    p.pathSmoothing *= m_style.pixelScale();
    _rule.get(StyleParamKey::text_path_offset, p.pathOffset);
    p.pathOffset *= m_style.pixelScale();
    _rule.get(StyleParamKey::text_font_fill, p.fill);
    float alpha = 1;
    if (_rule.get(StyleParamKey::text_font_alpha, alpha)) {
        p.fill = Color(p.fill).withAlpha(alpha).abgr;
    }

    _rule.get(StyleParamKey::text_font_stroke_color, p.strokeColor);
    float strokeAlpha = 1;
    if (_rule.get(StyleParamKey::text_font_stroke_alpha, strokeAlpha)) {
        p.strokeColor = Color(p.strokeColor).withAlpha(strokeAlpha).abgr;
    }

    _rule.get(StyleParamKey::text_font_stroke_width, p.strokeWidth);
    p.strokeWidth *= m_style.pixelScale();

    _rule.get(StyleParamKey::transition_hide_time, p.labelOptions.hideTransition.time);
    _rule.get(StyleParamKey::transition_selected_time, p.labelOptions.selectTransition.time);
    _rule.get(StyleParamKey::transition_show_time, p.labelOptions.showTransition.time);

    _rule.get(StyleParamKey::text_max_lines, p.maxLines);

    size_t repeatGroupHash = 0;
    std::string repeatGroup;
    StyleParam::Width repeatDistance;
    glm::vec2 defaultBuffer = glm::vec2(p.fontSize * 0.25f);

    if (_iconText) {
        _rule.get(StyleParamKey::text_priority, p.labelOptions.priority);
        _rule.get(StyleParamKey::text_collide, p.labelOptions.collide);
        if (!_rule.get(StyleParamKey::text_interactive, p.interactive)) {
            _rule.get(StyleParamKey::interactive, p.interactive);
        }
        _rule.get(StyleParamKey::text_offset, p.labelOptions.offset);
        p.labelOptions.offset *= m_style.pixelScale();

        _rule.get(StyleParamKey::text_anchor, p.labelOptions.anchors);
        if (p.labelOptions.anchors.count == 0) {
            // docs/label-placement-plan.md Task 4 follow-up: default fallback order for
            // icon-linked text with no explicit text_anchor, changed from the old fixed
            // 4-direction {bottom, top, right, left} to the classic 8-position cartographic
            // point-label priority ranking (Yoeli, P. (1972), "The Logic of Automated Map
            // Lettering," The Cartographic Journal 9(2):99-108 -- the numeric ranking
            // commonly (if loosely) attributed to Imhof's descriptive principles;
            // Christensen/Marks/Shieber, "An Empirical Study of Algorithms for
            // Point-Feature Label Placement," ACM ToG 14(3), 1995, adopt the same ranking
            // as their placement-quality objective function): upper-right first (Latin-
            // script reading order and ascender-heavy letterforms make a label above/right
            // of its symbol read as unambiguously "attached" to it), then upper-left,
            // lower-left, lower-right, then the four axis-centered sides (right, top, left,
            // bottom last). This is only the STARTING order -- both
            // TextStyleBuilder::salienceOrderedAnchors() (tile-build time, vector-proximity
            // cost) and Label::refineAnchor() (main-thread, texture-shading + vector cost)
            // re-sort it per-label; this default only matters as the base/tie-break order
            // before that happens, or when neither refinement runs (e.g. no terrain data).
            p.labelOptions.anchors.anchor = { {LabelProperty::Anchor::top_right, LabelProperty::Anchor::top_left,
                                               LabelProperty::Anchor::bottom_left, LabelProperty::Anchor::bottom_right,
                                               LabelProperty::Anchor::right, LabelProperty::Anchor::top,
                                               LabelProperty::Anchor::left, LabelProperty::Anchor::bottom} };
            p.labelOptions.anchors.count = 8;
        }

        // docs/label-placement-plan.md Task 4: opt-in flag (peak name text only, currently)
        // for Label::refineAnchor()'s main-thread ridge-avoidance pass.
        _rule.get(StyleParamKey::text_anchor_texture_shading, p.labelOptions.refineAnchorWithTextureShading);

        // docs/label-placement-plan.md user feedback 2026-07-15 ("labels too far from
        // peaks"): optional icon<->label gap scale, default unchanged (1.0) unless set.
        _rule.get(StyleParamKey::text_anchor_gap_scale, p.labelOptions.anchorGapScale);

        // child text's repeat group params
        if (_rule.get(StyleParamKey::text_repeat_distance, repeatDistance)) {
            p.labelOptions.repeatDistance = repeatDistance.value;
        } else {
            p.labelOptions.repeatDistance = MapProjection::tileSize();
        }

        if (p.labelOptions.repeatDistance > 0.f) {
            if (_rule.get(StyleParamKey::text_repeat_group, repeatGroup)) {
                hash_combine(repeatGroupHash, repeatGroup);
            } else if (_rule.get(StyleParamKey::repeat_group, repeatGroup)) { //inherit from parent point
                hash_combine(repeatGroupHash, repeatGroup);
            } else {
                repeatGroupHash = _rule.getParamSetHash();
            }
        }

        _rule.get(StyleParamKey::text_transition_hide_time, p.labelOptions.hideTransition.time);
        _rule.get(StyleParamKey::text_transition_selected_time, p.labelOptions.selectTransition.time);
        _rule.get(StyleParamKey::text_transition_show_time, p.labelOptions.showTransition.time);

        if (!_rule.get(StyleParamKey::text_buffer, p.labelOptions.buffer)) {
            p.labelOptions.buffer = defaultBuffer;
        }
    } else {
        _rule.get(StyleParamKey::priority, p.labelOptions.priority);
        _rule.get(StyleParamKey::collide, p.labelOptions.collide);
        _rule.get(StyleParamKey::interactive, p.interactive);
        _rule.get(StyleParamKey::offset, p.labelOptions.offset);
        p.labelOptions.offset *= m_style.pixelScale();

        _rule.get(StyleParamKey::anchor, p.labelOptions.anchors);
        if (p.labelOptions.anchors.count == 0) {
            p.labelOptions.anchors = {LabelProperty::Anchor::center};
        }

        if (_rule.get(StyleParamKey::repeat_distance, repeatDistance)) {
            p.labelOptions.repeatDistance = repeatDistance.value;
        } else {
            p.labelOptions.repeatDistance = MapProjection::tileSize();
        }

        if (p.labelOptions.repeatDistance > 0.f) {
            if (_rule.get(StyleParamKey::repeat_group, repeatGroup)) {
                hash_combine(repeatGroupHash, repeatGroup);
            } else {
                repeatGroupHash = _rule.getParamSetHash();
            }
        }

        if (!_rule.get(StyleParamKey::buffer, p.labelOptions.buffer)) {
            p.labelOptions.buffer = defaultBuffer;
        }
    }

    if (p.labelOptions.repeatDistance > 0.f) {
        hash_combine(repeatGroupHash, p.text);
        p.labelOptions.repeatGroup = repeatGroupHash;
        p.labelOptions.repeatDistance *= m_style.pixelScale();
    }

    _rule.get(StyleParamKey::text_wrap, p.maxLineWidth);

    if (auto* transform = _rule.get<std::string>(StyleParamKey::text_transform)) {
        TextLabelProperty::transform(*transform, p.transform);
    }

    if (auto* align = _rule.get<std::string>(StyleParamKey::text_align)) {
        bool res = TextLabelProperty::align(*align, p.align);
        if (!res && p.labelOptions.anchors.count > 0) {
            p.align = TextLabelProperty::alignFromAnchor(p.labelOptions.anchors[0]);
        }
    }

    _rule.get(StyleParamKey::text_optional, p.labelOptions.optional);

    std::hash<TextStyle::Parameters> hash;
    p.labelOptions.paramHash = hash(p);

    p.lineSpacing = 2 * m_style.pixelScale();

    if (p.interactive) {
        p.labelOptions.featureId = _rule.selectionColor;
    }

    return p;
}

TextStyle::Parameters TextStyleBuilder::applySecondaryTextRule(const DrawRule& _rule,
                                                                const Properties& _props,
                                                                const TextStyle::Parameters& _primary) const {
    const static std::string defaultWeight("400");
    const static std::string defaultStyle("regular");
    const static std::string defaultFamily("default");

    // Start from a copy of the primary label's Parameters: fill, stroke, buffer, anchors,
    // wrap/align/optional, and (via setRelative() at the call site) priority/collide
    // inheritance all carry over unchanged -- only text and font are independent.
    TextStyle::Parameters p = _primary;
    p.text.clear();
    p.textLeft.clear();
    p.textRight.clear();

    if (!getTextSource(StyleParamKey::text2_source, _rule, _props, p.text) || p.text.empty()) {
        p.text.clear();
        return p;
    }

    auto fontFamily = _rule.get<std::string>(StyleParamKey::text2_font_family);
    fontFamily = (!fontFamily) ? &defaultFamily : fontFamily;

    auto fontWeight = _rule.get<std::string>(StyleParamKey::text2_font_weight);
    fontWeight = (!fontWeight) ? &defaultWeight : fontWeight;

    float fontSize;
    if (_rule.get(StyleParamKey::text2_font_size, fontSize)) {
        p.fontSize = fontSize * m_style.pixelScale();
    } // else: keep _primary's (already pixelScale'd) fontSize as a sane default

    p.font = m_style.context()->getFont(*fontFamily, defaultStyle, *fontWeight, p.fontSize);
    if (!p.font) {
        LOGW("Missing font for %s / %s / %s / %d", fontFamily->c_str(), defaultStyle.c_str(),
             fontWeight->c_str(), p.fontSize);
        p.text.clear();
        return p;
    }

    // Tracking is a per-label-weight property: the secondary label (an elevation figure,
    // say) must not inherit the primary name's letter-spacing.
    p.letterSpacing = 0.f;
    p.letterSpacingFit = 0.f;
    p.pathSmoothing = 0.f;
    p.pathOffset = 0.f;
    _rule.get(StyleParamKey::text2_font_letter_spacing, p.letterSpacing);
    _rule.get(StyleParamKey::text2_font_fill, p.fill);  // else: the primary label's fill

    std::hash<TextStyle::Parameters> hash;
    p.labelOptions.paramHash = hash(p);

    return p;
}

#ifdef FONTCONTEXT_STB

static std::wstring to_wstring(std::string const& s) {
  std::wstring_convert<std::codecvt_utf8<wchar_t> > conv;
  return conv.from_bytes(s);
}

static std::string to_string(std::wstring const& s) {
  std::wstring_convert<std::codecvt_utf8<wchar_t> > conv;
  return conv.to_bytes(s);
}

void applyTextTransform(const TextStyle::Parameters& _params, std::string& _string) {
  static std::locale userlocale("");

  if(_params.transform == TextLabelProperty::Transform::none)
    return;
  auto _wstring = to_wstring(_string);
  switch (_params.transform) {
  case TextLabelProperty::Transform::capitalize: {
      bool was_space = true;
      for (auto& c : _wstring) {
          if(was_space)
              c = std::toupper(c, userlocale);
          was_space = std::isspace(c, userlocale);
      }
      break;
  }
  case TextLabelProperty::Transform::lowercase:
      for (auto& c : _wstring)
        c = std::tolower(c, userlocale);
      break;
  case TextLabelProperty::Transform::uppercase:
      for (auto& c : _wstring)
        c = std::toupper(c, userlocale);
      break;
  default:
      break;
  }
  _string = to_string(_wstring);
}

#else

void applyTextTransform(const TextStyle::Parameters& _params,
                        icu::UnicodeString& _string) {

    icu::Locale loc("en");

    switch (_params.transform) {
    case TextLabelProperty::Transform::capitalize: {
        UErrorCode status{U_ZERO_ERROR};
        auto *wordIterator = icu::BreakIterator::createWordInstance(loc, status);

        if (U_SUCCESS(status)) { _string.toTitle(wordIterator); }

        delete wordIterator;
        break;
    }
    case TextLabelProperty::Transform::lowercase:
        _string.toLower(loc);
        break;
    case TextLabelProperty::Transform::uppercase:
        _string.toUpper(loc);
        break;
    default:
        break;
    }
}

bool isComplexShapingScript(const icu::UnicodeString& _text) {

    // Taken from:
    // https://github.com/tangrams/tangram/blob/labels-rebase/src/styles/text/canvas_text.js#L538-L553
    // See also http://r12a.github.io/scripts/featurelist/

    icu::StringCharacterIterator iterator(_text);
    for (UChar c = iterator.first(); c != icu::CharacterIterator::DONE; c = iterator.next()) {
        if (c >= u'\u0600' && c <= u'\u18AF') {
            if ((c <= u'\u06FF') ||                   // Arabic:     "\u0600-\u06FF"
                (c >= u'\u1800' && c <= u'\u18AF')) { // Mongolian:  "\u1800-\u18AF"
                return true;
            }
        }
    }
    return false;
}
#endif

bool TextStyleBuilder::prepareLabel(TextStyle::Parameters& _params, Label::Type _type,
                                    LabelAttributes& _attributes) {

    if (_params.text.empty() || _params.fontSize <= 0.f) {
        // Nothing to render!
        return false;
    }

#ifdef FONTCONTEXT_STB
    auto& text = _params.text;
    applyTextTransform(_params, text);
    _params.fontScale = 1.f;
#else
    auto text = icu::UnicodeString::fromUTF8(_params.text);

    applyTextTransform(_params, text);

    if (_type == Label::Type::line) {
        _params.hasComplexShaping = isComplexShapingScript(text);
    }

    // Scale factor by which the texture glyphs are scaled to match fontSize
    _params.fontScale = _params.fontSize / _params.font->size();
#endif

    // Stroke width is normalized by the distance of the SDF spread, then
    // scaled to 255 and packed into the "alpha" channel of stroke.
    // Maximal strokeWidth is 3px, attribute is normalized to 0-1 range.

    auto ctx = m_style.context();

    uint32_t strokeAttrib = std::max(_params.strokeWidth / ctx->maxStrokeWidth() * 255.f, 0.f);
    if (strokeAttrib > 255) {
        LOGN("stroke_width too large: %f / %f", _params.strokeWidth, strokeAttrib/255.f);
        strokeAttrib = 255;
    }
    _attributes.stroke = (_params.strokeColor & 0x00ffffff) + (strokeAttrib << 24);
    _attributes.fill = _params.fill;
    _attributes.fontScale = std::min(int(_params.fontScale * 64.f), 255);
    _attributes.quadsStart = m_quads.size();
    _attributes.textRanges = TextRange{};

    glm::vec2 bbox(0);
    if (ctx->layoutText(_params, text, m_quads, m_atlasRefs, bbox, _attributes.textRanges)) {

        int start = _attributes.quadsStart;
        for (auto& range : _attributes.textRanges) {
            assert(range.start == start);
            assert(range.length >= 0);
            start += range.length;
        }
        _attributes.width = bbox.x;
        _attributes.height = bbox.y;
        return true;
    }

    return false;
}

// anchorCandidateCost() intentionally scores only vector-feature proximity (water/
// road/trail), not texture-shading/ridge avoidance, per a Phase 6 integration finding:
// Phase 2 (util/textureShading.h) established that RasterSource::m_textures/m_mosaics
// access is main-thread only, and this function -- like the rest of TextStyleBuilder/
// PointStyleBuilder::addFeature() -- runs on a TileWorker thread (tileWorker.cpp), so
// sampleTextureShading() cannot be called here at all, not even with a real
// RasterSource. The same constraint forced Phase 2's own priority refinement into a
// main-thread LabelManager pass (Label::refinePriority()); doing the equivalent for
// anchor ordering would mean re-ordering/re-colliding labels after tile build, a real
// redesign out of scope for this integration -- left as a documented future extension,
// not attempted here.

float TextStyleBuilder::anchorCandidateCost(LabelProperty::Anchor _anchor, glm::vec2 _tileLocalPos,
                                            glm::vec2 _labelSizePixels, float _iconRadiusPixels) const {

    glm::vec2 dir = LabelProperty::anchorDirection(_anchor);

    // Candidate footprint: the label's measured bounding box, offset from the anchor
    // position by icon_radius + half_label_extent in the anchor's direction (Frozen
    // Interface Contract, docs/label-placement-plan.md). half_label_extent is taken as
    // the half-diagonal of the label box so the offset is a sensible "clear of both the
    // icon and the label" distance in every one of the 8 compass directions, including
    // diagonals.
    glm::vec2 halfExtentPx = _labelSizePixels * 0.5f;
    float offsetPx = _iconRadiusPixels + glm::length(halfExtentPx);
    glm::vec2 footprintCenterPx = dir * offsetPx;

    // Convert pixel-space offsets to the normalized tile-local [0,1] space Feature
    // geometry (and m_anchorOccupancyGrid) is expressed in. m_tileSize is "pixels per
    // full tile width at this tile's build zoom" (see setup()), matching the same
    // px/tile-unit conversion already used elsewhere in this file (e.g.
    // addStraightTextLabels()'s `pixelSize = 1.0f/m_tileSize`).
    glm::vec2 footprintCenterTile = footprintCenterPx / m_tileSize;
    glm::vec2 halfExtentTile = halfExtentPx / m_tileSize;

    // Sample a handful of points spread across the candidate footprint (not just its
    // center), per the plan doc's "evaluated over the small candidate footprint... not
    // just the center".
    constexpr int kNumSamples = 5;
    const glm::vec2 sampleOffsets[kNumSamples] = {
        glm::vec2(0.f, 0.f),
        glm::vec2(-1.f, -1.f), glm::vec2(1.f, -1.f),
        glm::vec2(-1.f, 1.f), glm::vec2(1.f, 1.f),
    };

    float proximityCost = 0.f;

    for (const auto& sampleOffset : sampleOffsets) {
        glm::vec2 samplePosTile = _tileLocalPos + footprintCenterTile + sampleOffset * halfExtentTile;

        if (m_anchorOccupancyGrid) {
            proximityCost += m_anchorOccupancyGrid->sample(samplePosTile);
        }
    }

    return proximityCost / float(kNumSamples);
}

LabelProperty::Anchors TextStyleBuilder::salienceOrderedAnchors(const LabelProperty::Anchors& _anchors,
                                                                 glm::vec2 _tileLocalPos,
                                                                 glm::vec2 _labelSizePixels,
                                                                 float _iconRadiusPixels) const {
    if (_anchors.count <= 1) {
        // Nothing to reorder.
        return _anchors;
    }
    if (!m_anchorOccupancyGrid) {
        // Both cost terms unavailable (vector-proximity grid empty/absent for this
        // tile, texture-shading always unavailable pre-Phase-6) -- every candidate
        // costs 0, so a stable sort is a no-op. Skip the work.
        return _anchors;
    }

    struct ScoredAnchor {
        LabelProperty::Anchor anchor;
        float cost;
    };
    std::array<ScoredAnchor, LabelProperty::max_anchors> scored;
    int n = _anchors.count;
    for (int i = 0; i < n; i++) {
        LabelProperty::Anchor anchor = _anchors[i];
        float cost = (anchor == LabelProperty::Anchor::center) ? 0.f
            : anchorCandidateCost(anchor, _tileLocalPos, _labelSizePixels, _iconRadiusPixels);
        scored[i] = { anchor, cost };
    }

    std::stable_sort(scored.begin(), scored.begin() + n,
                      [](const ScoredAnchor& a, const ScoredAnchor& b) { return a.cost < b.cost; });

    LabelProperty::Anchors result;
    result.count = n;
    for (int i = 0; i < n; i++) {
        result.anchor[i] = scored[i].anchor;
    }
    return result;
}

Label* TextStyleBuilder::addLabel(Label::Type _type, TextLabel::Coordinates _coordinates,
                                const TextStyle::Parameters& _params, const LabelAttributes& _attributes,
                                const DrawRule& _rule) {

    uint32_t selectionColor = 0;

    if (_params.interactive) {
        //selectionColor = _rule.featureSelection->nextColorIdentifier();
        selectionColor = _rule.featureSelection ?  // copied from PointStyleBuilder::addLabel()
              _rule.featureSelection->nextColorIdentifier() : _rule.selectionColor;
    }

    m_labels.emplace_back(new TextLabel(_coordinates, _type, _params.labelOptions,
                                        {_attributes.fill,
                                         _attributes.stroke,
                                         _attributes.fontScale,
                                         selectionColor},
                                        {_attributes.width, _attributes.height},
                                        *m_textLabels, _attributes.textRanges,
                                        _params.align));
#ifdef DEBUG
    m_labels.back()->debugTag = _params.text;
#endif

    return m_labels.back().get();
}

}
