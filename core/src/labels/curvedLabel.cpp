#include "labels/curvedLabel.h"

#include "debug/profiler.h"
#include "gl/dynamicQuadMesh.h"
#include "labels/obbBuffer.h"
#include "labels/screenTransform.h"
#include "log.h"
#include "style/textStyle.h"
#include "text/fontContext.h"
#include "textLabels.h"
#include "util/geom.h"
#include "util/lineSampler.h"
#include "util/elevationManager.h"
#include "view/view.h"

#include <glm/gtx/norm.hpp>

#include <algorithm>

namespace Tangram {


void CurvedLabel::applyAnchor(LabelProperty::Anchor _anchor) {

    using namespace TextLabelProperty;

    if (m_preferedAlignment == Align::none) {
        Align newAlignment = alignFromAnchor(_anchor);
        m_textRangeIndex = int(newAlignment);
    } else {
        m_textRangeIndex = int(m_preferedAlignment);
    }

    if (m_textRanges[m_textRangeIndex].length == 0) {
        m_textRangeIndex = 0;
    }

    glm::vec2 offset = m_dim;
    if (isChild()) { offset += m_relative->dimension() * m_options.anchorGapScale; }

    m_anchor = LabelProperty::anchorDirection(_anchor) * offset * 0.5f;
}

bool CurvedLabel::setElevation(ElevationManager& elevMgr, glm::dvec2 origin, double scale) {
    bool ok = true;
    for (glm::vec3& p : m_modelTransform) {
        p.z = elevMgr.m_terrainScale * elevMgr.getElevation(origin + glm::dvec2(p)*scale, ok)/scale;
        if(!ok) break;
    }
    return ok;
}

namespace {

// Decides whether the text fits on the projected path (the sampler holds it, `anchor` is the
// vertex the text is centred on) and which way up it reads; `_anchorOut` gets the anchor's
// index in the possibly reversed path. Clears the path when the text does not fit.
bool layoutText(LineSampler<ScreenTransform>& sampler, ScreenTransform& _transform, size_t anchor,
                float width, bool noFlip, size_t& _anchorOut) {
    _anchorOut = anchor;
    // Chord length for minimal ~120 degree inner angles (squared)
    // sin(60)*2
    const float sqDirLimit = powf(1.7f, 2);
    // Range to check for angle changes
    const float sampleWindow = 20;

    float start = sampler.point(anchor).z - width * 0.5f;

    if (start < 0 || start + width > sampler.sumLength()) {
        sampler.clearPoints();
        return false;
    }

    // Sets current segment to the segment on which the text starts.
    glm::vec2 startPos, r;
    sampler.sample(start, startPos, r);
    size_t startSegment = sampler.curSegment();

    glm::vec2 dir = sampler.segmentDirection(startSegment);

    // Cannot reverse the direction when some glyphs must be
    // placed in forward direction and vice versa.
        const float flipTolerance = noFlip ? FLT_MAX : sin(DEG_TO_RAD * 45);
    bool mustForward = dir.x > flipTolerance;
    bool mustReverse = dir.x < -flipTolerance;

    for (size_t i = startSegment + 1; i < _transform.size(); i++) {
        float currLength = sampler.point(i).z;
        dir = sampler.segmentDirection(i);

        if (dir.x > flipTolerance) {
            mustForward = true;
        }
        if (dir.x < -flipTolerance) {
            mustReverse = true;
        }

        if (mustForward && mustReverse) {
            sampler.clearPoints();
            return false;
        }

        // Go back within window to check for hard direction changes
        for (int k = i - 1; k >= int(startSegment); k--) {
            if (glm::length2(sampler.segmentDirection(k) + dir) < sqDirLimit) {
                sampler.clearPoints();
                return false;
            }
            if (sampler.point(k).z < sampler.point(i).z - sampleWindow) {
                break;
            }
        }
        if (currLength > start + width) {
            break;
        }
    }

    if (!mustReverse && !noFlip) {
        // TODO use better heuristic to decide flipping
        glm::vec2 endPos;
        sampler.sample(start + width, endPos, r);
        if (startPos.x > endPos.x) {
            mustReverse = true;
        }
    }

    if (mustReverse) {
        sampler.reversePoints();
        _anchorOut = _transform.size() - anchor - 1;
    }

    return true;
}


}  // namespace

bool CurvedLabel::updateScreenTransform(const glm::mat4& _mvp, const ViewState& _viewState,
                                        const AABB* _bounds, ScreenTransform& _transform) {

    glm::vec2 min(-m_dim.y);
    glm::vec2 max(_viewState.viewportSize + m_dim.y);

    bool clipped = false;
    bool inside = false;

    const bool slide = !m_slideClear.empty();
    LineSampler<ScreenTransform> sampler { _transform };
    std::vector<glm::vec2> screen;  // sliding labels: the whole path, projected
    if (slide) { screen.reserve(m_modelTransform.size()); }

    for (auto& p : m_modelTransform) {
        glm::vec4 sp = worldToScreenSpace(_mvp, glm::vec4(p, 1.0),
                                          _viewState.viewportSize, clipped);

        if (clipped || sp.z > 1.0f) { sampler.clearPoints(); return false; }

        if (slide) { screen.emplace_back(sp.x, sp.y); } else { sampler.add(glm::vec2(sp)); }

        if (!inside) {
            if ((sp.x > min.x && sp.x < max.x &&
                 sp.y > min.y && sp.y < max.y)) {
                inside = true;
            }
        }
    }

    const bool noFlip = std::isnan(m_options.angle);  // angle: auto to fix orientation of text

    if (slide) {
        if (!inside) { return false; }
        return slideTransform(screen, _viewState.viewportSize, _transform, noFlip);
    }

    if (!inside || sampler.sumLength() < m_dim.x) {
        sampler.clearPoints();
        return false;
    }

    // Set center for repeatGroup distance calculations
    m_screenCenter = worldToScreenSpace(_mvp, glm::vec4(m_modelTransform[m_anchorPoint], 1.0),
                                        _viewState.viewportSize, clipped);
    return layoutText(sampler, _transform, m_anchorPoint, m_dim.x, noFlip, m_screenAnchorPoint);
}

// An area label: the path is a stretch of the shape's spine that is wide enough for the text,
// longer than the text. The text keeps its place on the path (so it does not crawl while the map
// is panned) as long as it lies on screen and fits between the shores; when it cannot, it moves
// to the position nearest the middle of the view. Nothing is handed over between labels, so a
// 1px pan cannot make it blink.
void CurvedLabel::setSlide(const std::vector<glm::vec2>& _path, const std::vector<float>& _clearance,
                           float _halfHeight, float _margin, int _slot) {
    m_slideSlot = _slot;
    // The model path has one more vertex than the path it was made from (the anchor, inserted
    // on a segment): carry the clearance over by arc length.
    std::vector<float> arc(_path.size(), 0.f);
    for (size_t i = 1; i < _path.size(); i++) { arc[i] = arc[i - 1] + glm::distance(_path[i], _path[i - 1]); }
    m_slideClear.assign(m_modelTransform.size(), 0.f);
    float s = 0.f;
    size_t k = 1;
    for (size_t i = 0; i < m_modelTransform.size(); i++) {
        if (i > 0) { s += glm::distance(glm::vec2(m_modelTransform[i]), glm::vec2(m_modelTransform[i - 1])); }
        while (k + 1 < _path.size() && arc[k] < s) { k++; }
        const float t = (s - arc[k - 1]) / std::max(arc[k] - arc[k - 1], 1e-9f);
        m_slideClear[i] = glm::mix(_clearance[k - 1], _clearance[k], std::min(std::max(t, 0.f), 1.f));
    }
    m_slideArc.assign(m_modelTransform.size(), 0.f);
    for (size_t i = 1; i < m_modelTransform.size(); i++) {
        m_slideArc[i] = m_slideArc[i - 1] + glm::distance(glm::vec2(m_modelTransform[i]), glm::vec2(m_modelTransform[i - 1]));
    }
    m_slideS = -1.f;
    m_slideHalfH = _halfHeight;
    m_slideMargin = _margin;
}

bool CurvedLabel::slideTransform(const std::vector<glm::vec2>& _p, glm::vec2 _view,
                                 ScreenTransform& _transform, bool _noFlip) {
    PROFILE_SCOPE("areaLabels:slide");
    const size_t n = _p.size();
    if (n < 2) { return false; }

    // arc length, and the inscribed radius in px (model units -> px per segment)
    std::vector<float> arc(n, 0.f), clr(n, 0.f), ratio(n, 0.f);
    for (size_t i = 1; i < n; i++) {
        arc[i] = arc[i - 1] + glm::distance(_p[i], _p[i - 1]);
        const float model = glm::distance(glm::vec2(m_modelTransform[i]), glm::vec2(m_modelTransform[i - 1]));
        const float r = model > 0.f ? (arc[i] - arc[i - 1]) / model : 0.f;
        ratio[i - 1] = ratio[i - 1] > 0.f ? 0.5f * (ratio[i - 1] + r) : r;
        ratio[i] = r;
    }
    for (size_t i = 0; i < n; i++) { clr[i] = m_slideClear[i] * ratio[i]; }
    const float length = arc.back();
    const float W = m_dim.x;
    if (length < W) { return false; }

    auto pointAt = [&](float s) {
        size_t i = std::upper_bound(arc.begin(), arc.end(), s) - arc.begin();
        i = std::min(std::max<size_t>(i, 1), n - 1);
        const float t = (s - arc[i - 1]) / std::max(arc[i] - arc[i - 1], 1e-6f);
        return glm::mix(_p[i - 1], _p[i], std::min(std::max(t, 0.f), 1.f));
    };

    // stretches where the text's height fits between the shores
    struct Run { float a, b; };
    std::vector<Run> runs;
    {
        float open = -1.f;
        auto cross = [&](size_t i) {
            const float t = (m_slideHalfH - clr[i - 1]) / (clr[i] - clr[i - 1]);
            return arc[i - 1] + t * (arc[i] - arc[i - 1]);
        };
        auto close = [&](float end) {
            const float a = open + m_slideMargin, b = end - m_slideMargin;
            if (b - a >= W) { runs.push_back({ a, b }); }
            open = -1.f;
        };
        for (size_t i = 0; i < n; i++) {
            const bool ok = clr[i] >= m_slideHalfH;
            if (ok && open < 0.f) { open = (i == 0) ? 0.f : cross(i); }
            if (!ok && open >= 0.f) { close(cross(i)); }
        }
        if (open >= 0.f) { close(length); }
    }
    if (runs.empty()) { return false; }

    // How much of the text is on screen with its middle at arc `c`: nine places along its length.
    const glm::vec2 mid = 0.5f * _view;
    auto inView = [&](glm::vec2 q) { return q.x >= 0.f && q.y >= 0.f && q.x <= _view.x && q.y <= _view.y; };
    auto visible = [&](float c) {
        int on = 0;
        for (int k = -4; k <= 4; k++) { on += inView(pointAt(c + k * W / 8.f)); }
        return on;
    };
    constexpr int kKeep = 5;   // a place is kept while at least half of the text is on screen
    constexpr int kChoose = 7; // a new place has nearly all of it

    auto lerpAt = [&](const std::vector<float>& from, const std::vector<float>& to, float v) {
        size_t i = std::upper_bound(from.begin(), from.end(), v) - from.begin();
        i = std::min(std::max<size_t>(i, 1), from.size() - 1);
        const float t = (v - from[i - 1]) / std::max(from[i] - from[i - 1], 1e-9f);
        return glm::mix(to[i - 1], to[i], std::min(std::max(t, 0.f), 1.f));
    };

    // The best place on a run is the middle of its visible part -- so at the end of a spine the
    // text sits between the end and the edge of the view, not pushed against the end to be at
    // the centre of the screen. (The runs already stop an em short of the shores and the ends.)
    struct Cand { float c, score, target, lo, hi; };
    std::vector<Cand> cands;
    for (const Run& r : runs) {
        float va = -1.f, vb = -1.f;
        for (float s = r.a;; s += 4.f) {
            s = std::min(s, r.b);
            if (inView(pointAt(s))) { if (va < 0.f) { va = s; } vb = s; }
            if (s >= r.b) { break; }
        }
        if (va < 0.f) { continue; }
        const float lo = r.a + 0.5f * W, hi = r.b - 0.5f * W;
        // spare slots sit to either side of the best place
        static const float kSlotOffset[] = { 0.f, 0.5f, -0.5f, 1.f, -1.f, 1.5f, -1.5f };
        const float mean = 0.5f * (va + vb);
        const float target = m_slideSlot == 0 ? std::min(std::max(mean, lo), hi)
                                              : mean + kSlotOffset[m_slideSlot % 7] * W;
        for (float c = lo;; c += 6.f) {
            c = std::min(c, hi);
            const int on = visible(c);
            const float off = std::fabs(c - target);
            if (on >= kKeep && (m_slideSlot == 0 || off <= 0.25f * W)) {
                cands.push_back({ c, off + (on < kChoose ? 1e6f : 0.f), target, lo, hi });
            }
            if (c >= hi) { break; }
        }
    }
    if (cands.empty()) { return false; }
    std::sort(cands.begin(), cands.end(), [](const Cand& x, const Cand& y) { return x.score < y.score; });

    // the place it already has, if it still works
    if (m_slideS >= 0.f && m_slideArc.size() == n) {
        const float c = lerpAt(m_slideArc, arc, m_slideS);
        for (const Run& r : runs) {
            if (c >= r.a + 0.5f * W - 1.f && c <= r.b - 0.5f * W + 1.f && visible(c) >= kKeep) {
                cands.insert(cands.begin(), { c, -1.f, c, c, c });
                break;
            }
        }
    }

    // best position first; if the text cannot be laid there (it folds back on itself) the next
    // one a good way off
    std::vector<float> tried;
    for (const Cand& cand : cands) {
        bool near = false;
        for (float t : tried) { near |= std::fabs(t - cand.c) < 0.25f * W; }
        if (near) { continue; }
        if (tried.size() >= 6) { break; }
        tried.push_back(cand.c);

        float c = cand.c;
        if (cand.score >= 0.f && tried.size() <= 2) {  // refine the coarse step
            float best = std::fabs(cand.c - cand.target);
            for (float d = -5.f; d <= 5.f; d += 1.f) {
                const float q = cand.c + d;
                if (q < cand.lo || q > cand.hi || visible(q) < kChoose) { continue; }
                if (std::fabs(q - cand.target) < best) { best = std::fabs(q - cand.target); c = q; }
            }
        }

        const float s0 = std::max(0.f, c - 0.5f * W - 0.05f * W), s1 = std::min(length, c + 0.5f * W + 0.05f * W);
        LineSampler<ScreenTransform> sampler { _transform };
        sampler.add(pointAt(s0));
        size_t i = 0;
        while (i < n && arc[i] <= s0) { i++; }
        for (; i < n && arc[i] < c; i++) { sampler.add(_p[i]); }
        sampler.add(pointAt(c));
        const size_t anchor = _transform.size() - 1;
        for (; i < n && arc[i] <= c; i++) {}
        for (; i < n && arc[i] < s1; i++) { sampler.add(_p[i]); }
        sampler.add(pointAt(s1));

        if (_transform.size() < 2) { sampler.clearPoints(); continue; }
        m_screenCenter = glm::vec4(pointAt(c), 0.f, 1.f);
        if (layoutText(sampler, _transform, anchor, W, _noFlip, m_screenAnchorPoint)) {
            m_slideS = lerpAt(arc, m_slideArc, c);
            return true;
        }
    }
    return false;
}

void CurvedLabel::obbs(ScreenTransform& _transform, OBBBuffer& _obbs) const {

    glm::vec2 dim = m_dim - m_options.buffer;

    if (m_occludedLastFrame) { dim += Label::activation_distance_threshold; }

    // TODO: Remove - Only for testing
    if (state() == State::dead) { dim -= 4; }

    float width = dim.x;
    LineSampler<ScreenTransform> sampler { _transform };

    auto center = sampler.point(m_screenAnchorPoint).z;
    //auto center = sampler.sumLength() * 0.5;

    auto start = center - width * 0.5f;

    glm::vec2 p1, p2, rotation;
    sampler.sample(start, p1, rotation);

    float prevLength = start;

    for (size_t i = sampler.curSegment()+1; i < _transform.size(); i++) {

        float currLength = sampler.point(i).z;
        float segmentLength = currLength - prevLength;

        if (start + width > currLength) {
            p2 = glm::vec2(sampler.point(i));

            rotation = sampler.segmentDirection(i-1);
            _obbs.append({(p1 + p2) * 0.5f, rotation, segmentLength, dim.y});
            prevLength = currLength;
            p1 = p2;

        } else {

            segmentLength = (start + width) - prevLength;
            sampler.sample(start + width, p2, rotation);
            _obbs.append({(p1 + p2) * 0.5f, rotation, segmentLength, dim.y});
            break;
        }
    }
}

void CurvedLabel::addVerticesToMesh(ScreenTransform& _transform, const glm::vec2& _screenSize) {
    if (!visibleState()) { return; }

    TextVertex::State state {
        m_fontAttrib.selectionColor,
        m_fontAttrib.fill,
        m_fontAttrib.stroke,
        uint16_t(m_alpha * TextVertex::alpha_scale),
        uint16_t(m_fontAttrib.fontScale),
    };

    auto it = m_textLabels.quads.begin() + m_textRanges[m_textRangeIndex].start;
    auto end = it + m_textRanges[m_textRangeIndex].length;
    auto& style = m_textLabels.style;

    auto& meshes = style.getMeshes();

    glm::vec2 rotation;
    LineSampler<ScreenTransform> sampler { _transform };

    float width = m_dim.x;

    if (sampler.sumLength() < width) { return; }

    float center = sampler.point(m_screenAnchorPoint).z;

    std::array<glm::i16vec2, 4> vertexPosition;

    glm::i16vec2 min(-m_dim.y * TextVertex::position_scale);
    glm::i16vec2 max((_screenSize + m_dim.y) * TextVertex::position_scale);

    for (; it != end; ++it) {
        auto quad = *it;

        glm::vec2 origin = {(quad.quad[0].pos.x + quad.quad[2].pos.x) * 0.5f, 0 };
        glm::vec2 point, p1, p2, r1, r2;

        auto xStart = quad.quad[0].pos.x * TextVertex::position_inv_scale;
        auto xEnd = quad.quad[2].pos.x * TextVertex::position_inv_scale;

        float px = origin.x * TextVertex::position_inv_scale;
        if (!sampler.sample(center + px, point, rotation)) {
            continue;
        }

        bool ok1 = sampler.sample(center + xStart, p1, r1);
        bool ok2 = sampler.sample(center + xEnd, p2, r2);
        if (ok1 && ok2) {
            if (r1 == r2) {
                rotation = r1;
            } else {
                rotation = glm::normalize(p2 - p1);
            }
            //point = (p1 + p2) * 0.5f;
            point = point * 0.5f + (p1 + p2) * 0.25f;
        }

        glm::i16vec2 p(point * TextVertex::position_scale);

        rotation = {rotation.x, -rotation.y};

        bool visible = false;

        for (int i = 0; i < 4; i++) {

            vertexPosition[i] = p + glm::i16vec2{rotate2d(glm::vec2(quad.quad[i].pos) - origin, rotation)};

            if (!visible &&
                vertexPosition[i].x > min.x &&
                vertexPosition[i].x < max.x &&
                vertexPosition[i].y > min.y &&
                vertexPosition[i].y < max.y) {
                visible = true;
            }
        }

        if (!visible) { continue; }

        auto* quadVertices = meshes[it->atlas]->pushQuad();

        for (int i = 0; i < 4; i++) {
            TextVertex& v = quadVertices[i];
            v.pos = vertexPosition[i];
            v.uv = quad.quad[i].uv;
            v.state = state;
        }
    }
}

}
