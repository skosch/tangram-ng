#include "debug/profiler.h"
#include "style/areaLabelStyle.h"

#include "data/propertyItem.h"
#include "labels/areaRegistry.h"
#include "labels/curvedLabel.h"
#include "labels/textLabels.h"
#include "log.h"
#include "marker/marker.h"
#include "scene/drawRule.h"
#include "scene/scene.h"
#include "style/textStyleBuilder.h"
#include "tile/tile.h"
#include "util/hash.h"
#include "util/labelPath.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <sstream>

namespace Tangram {

namespace {

// Where in a label's life the tracking is planned: a marker is built at integer zoom z and shown
// over [z, z+1), i.e. at 1x..2x the scale it is built at. The text must fit at 1x or the label
// would be hidden for the first half of every zoom level; it only gets looser from there.
constexpr float kPlanScale = 1.0f;
// The text is planned a little narrower than the longest run, so it still fits when the view is a
// hair below the zoom the tracking was planned for.
constexpr float kWindowSlack = 1.08f;
constexpr size_t kMaxRuns = 4;           // paths per part: the text slides along each
constexpr float kVertexSpacing = 14.f;    // px: a label is "on screen" if one of its vertices is

std::string kindOf(const Properties& _props) {
    if (_props.contains("water")) { return "water"; }
    std::string natural;
    if (_props.getString("natural", natural) && natural == "glacier") { return "glacier"; }
    std::string v;
    if (_props.getString("boundary", v) && v == "national_park") { return "park"; }
    if (_props.getString("leisure", v) && v == "nature_reserve") { return "park"; }
    if (_props.getString("protect_class", v) &&
        (v == "1" || v == "1a" || v == "1b" || v == "2" || v == "3" || v == "4" || v == "5")) { return "park"; }
    return "area";
}

}  // namespace

class AreaLabelStyleBuilder : public TextStyleBuilder {

public:
    AreaLabelStyleBuilder(const AreaLabelStyle& _style) : TextStyleBuilder(_style), m_areaStyle(_style) {}

    void setup(const Tile& _tile) override {
        m_marker = false;
        m_frags.clear();
        const TileID id = _tile.getID();
        m_tileKey = AreaRegistry::tileKey(_tile.sourceID(), id.x, id.y, id.z);
        m_origin = _tile.getOrigin();
        m_scale = _tile.getScale();
        m_registry = m_areaStyle.registry();
        // The tile's own styling zoom changes the zoom this is built at, not what is in it: once
        // a data tile has been seen there is nothing new to deposit.
        m_skip = !m_registry || m_registry->hasTile(m_tileKey);
    }

    void setup(const Marker& _marker, int _zoom) override {
        m_marker = true;
        m_extent = _marker.extent();
        m_zoom = _zoom;
        m_registry = m_areaStyle.registry();
        TextStyleBuilder::setup(_marker, _zoom);
    }

    bool addFeature(const Feature& _feat, const DrawRule& _rule) override {
        if (m_marker) { return addSpineLabel(_feat, _rule); }
        if (m_skip || _feat.geometryType != GeometryType::polygons) { return false; }

        std::string name;
        if (!_feat.props.getString("name", name) || name.empty()) { return false; }

        auto frag = std::make_shared<AreaFragment>();
        frag->tileKey = m_tileKey;
        frag->kind = kindOf(_feat.props);
        frag->name = name;
        frag->props = _feat.props;
        bool first = true;
        for (const auto& polygon : _feat.polygons) {
            AreaPolygon poly;
            for (const auto& ring : polygon) {
                if (ring.size() < 3) { continue; }
                AreaRing r;
                r.reserve(ring.size());
                for (const auto& p : ring) {
                    glm::dvec2 w = glm::dvec2(p.x, p.y) * m_scale + m_origin;
                    r.push_back(w);
                    if (first) { frag->lo = frag->hi = w; first = false; }
                    frag->lo = glm::min(frag->lo, w);
                    frag->hi = glm::max(frag->hi, w);
                }
                poly.push_back(std::move(r));
            }
            if (!poly.empty()) { frag->polys.push_back(std::move(poly)); }
        }
        if (frag->polys.empty()) { return false; }
        m_frags.push_back(std::move(frag));
        return true;
    }

    std::unique_ptr<StyledMesh> build() override {
        if (m_marker) { return TextStyleBuilder::build(); }
        if (!m_skip && m_registry) { m_registry->setTile(m_tileKey, std::move(m_frags)); }
        m_frags.clear();
        return nullptr;
    }

private:
    // One name for a part. The manager supplies the part's paths laid end to end (the spine, then
    // horizontal chords; a park has only chords), local marker units. Each stretch where the text
    // fits becomes one SLIDING label (CurvedLabel::setSlide): it picks its own place along the
    // stretch every frame. The stretches of a part are one repeat group with an unlimited repeat
    // distance, so only the best one shows. Props: `clr` inscribed radius (m) per vertex, `brk`
    // "<first vertex><s|c> ..." (s = spine, c = chord), `grp` identity of the part, `zb` the
    // (fractional) zoom the tracking is planned for.
    bool addSpineLabel(const Feature& _feat, const DrawRule& _rule) {
        PROFILE_SCOPE("areaLabels:build");
        TextStyle::Parameters params = applyRule(_rule, _feat.props, false);
        if (!params.font || _feat.geometryType != GeometryType::lines || _feat.lines.empty()) { LOGD("addSpine fail: font/geom"); return false; }
        params.wordWrap = false;

        const Line& line = _feat.lines.front();
        std::vector<float> clr;
        {
            std::istringstream in(_feat.props.getString("clr"));
            float v;
            while (in >> v) { clr.push_back(v); }
        }
        if (line.size() < 2 || clr.size() != line.size() || m_extent <= 0.f) { LOGD("addSpine fail: sizes %zu %zu extent %f", line.size(), clr.size(), m_extent); return false; }

        struct Seg {
            std::vector<glm::vec2> p;
            std::vector<float> s, c;  // arc length and clearance, px
            bool spine = false;
            float e0 = 0.f, e1 = 0.f;  // px of wide-enough spine beyond the ends (the path is clipped to the view)
        };
        std::vector<Seg> segs;
        float zoomFactor = 1.f;
        {
            float zb = 0.f;
            std::istringstream zin(_feat.props.getString("zb"));
            if (zin >> zb) { zoomFactor = std::exp2(zb - float(m_zoom)); }
        }
        const float pxPerUnit = m_tileSize * zoomFactor;          // marker-local unit -> px at the planned zoom
        const float pxPerMeter = pxPerUnit / m_extent;
        {
            std::vector<std::pair<size_t, bool>> starts;
            std::vector<float> exts;
            {
                std::istringstream ein(_feat.props.getString("ext"));
                float v;
                while (ein >> v) { exts.push_back(v * pxPerMeter); }
            }
            std::istringstream in(_feat.props.getString("brk"));
            size_t idx;
            char type;
            while (in >> idx >> type) { starts.emplace_back(idx, type == 's'); }
            if (starts.empty()) { starts.emplace_back(0, true); }
            for (size_t k = 0; k < starts.size(); k++) {
                const size_t a = starts[k].first, b = (k + 1 < starts.size()) ? starts[k + 1].first : line.size();
                if (b <= a + 1 || b > line.size()) { continue; }
                Seg sg;
                sg.spine = starts[k].second;
                if (2 * k + 1 < exts.size()) { sg.e0 = exts[2 * k]; sg.e1 = exts[2 * k + 1]; }
                for (size_t i = a; i < b; i++) {
                    sg.p.push_back(glm::vec2(line[i]));
                    sg.s.push_back(i == a ? 0.f : sg.s.back() + glm::length(sg.p.back() - sg.p[sg.p.size() - 2]) * pxPerUnit);
                    sg.c.push_back(clr[i] * pxPerMeter);
                }
                segs.push_back(std::move(sg));
            }
        }
        if (segs.empty()) { LOGD("addSpine fail: no segs (line %zu, brk '%s')", line.size(), _feat.props.getString("brk").c_str()); return false; }

        const size_t quadsStart = m_quads.size();
        LabelAttributes base;
        struct Run { size_t seg; float a, b; };
        std::vector<Run> runs;
        float runMax = 0.f, trackMax = 0.f;  // longest run on this path / including the spine beyond a clipped end
        float halfH = 0.f, margin = 0.f;
        // Imhof set his lettering to the size of the thing it names: where the shores are too close
        // for the type at its full size, take it down (to 70%) before giving up on the label.
        const float fullSize = params.fontSize;
        for (float scale = 1.f;; scale -= 0.05f) {
            params.fontSize = fullSize * scale;
            if (!prepareLabel(params, Label::Type::line, base)) { LOGD("addSpine fail: prepareLabel"); return false; }
            m_quads.resize(quadsStart);

            // Runs where a strip of the text's height fits between the shores, per path.
            halfH = 0.5f * base.height * 1.2f + 2.f * m_style.pixelScale();
            // an em (and the tracking) of clear spine before the shores and the ends
            margin = params.fontSize * (1.f + params.letterSpacing);
            runs.clear();
            runMax = trackMax = 0.f;
            for (size_t k = 0; k < segs.size(); k++) {
                const Seg& sg = segs[k];
                const size_t n = sg.p.size();
                auto cross = [&](size_t i) {  // arc where clearance crosses halfH on segment i-1 -> i
                    float t = (halfH - sg.c[i - 1]) / (sg.c[i] - sg.c[i - 1]);
                    return sg.s[i - 1] + t * (sg.s[i] - sg.s[i - 1]);
                };
                float runA = -1.f;
                auto close = [&](float runB) {
                    const float a = runA + margin, b = runB - margin;
                    if (b - a > 0.f) {
                        runs.push_back({ k, a, b });
                        runMax = std::max(runMax, b - a);
                        const float extra = (runA <= 1e-3f ? sg.e0 : 0.f) + (runB >= sg.s[n - 1] - 1e-3f ? sg.e1 : 0.f);
                        trackMax = std::max(trackMax, b - a + extra);
                    }
                    runA = -1.f;
                };
                for (size_t i = 0; i < n; i++) {
                    const bool ok = sg.c[i] >= halfH;
                    if (ok && runA < 0.f) { runA = (i == 0) ? sg.s[0] : cross(i); }
                    if (!ok && runA >= 0.f) { close(cross(i)); }
                }
                if (runA >= 0.f) { close(sg.s[n - 1]); }
            }
            if (runMax >= base.width * kWindowSlack) { break; }
            if (scale <= 0.7f + 1e-3f) {  // no room at this zoom: no label (a bigger zoom will do)
                LOGD("AreaLabel '%s': no room (run %.0f px, text %.0f px, need half-height %.1f)", params.text.c_str(), runMax, base.width, halfH);
                return true;
            }
        }

        // One tracking for every window: fill the longest run as it is at the planned zoom.
        if (params.letterSpacingFit > 0.f) {
            float pathPx = std::min(std::min(trackMax, 0.97f * trackMax / params.letterSpacingFit), runMax) / kWindowSlack;
            if (!prepareTrackedLabel(params, Label::Type::line, base, pathPx)) { return false; }
            m_quads.resize(quadsStart);
        }
        const float W = base.width;

        auto at = [&](const Seg& sg, float arc) {
            size_t i = 1;
            while (i < sg.p.size() - 1 && sg.s[i] < arc) { i++; }
            float t = (arc - sg.s[i - 1]) / std::max(sg.s[i] - sg.s[i - 1], 1e-6f);
            return std::pair<glm::vec2, float>(glm::mix(sg.p[i - 1], sg.p[i], std::min(std::max(t, 0.f), 1.f)),
                                               glm::mix(sg.c[i - 1], sg.c[i], std::min(std::max(t, 0.f), 1.f)));
        };

        // the runs the text fits in, spine first, then longest; a few are enough
        std::vector<Run> use;
        for (const Run& r : runs) { if (r.b - r.a >= W) { use.push_back(r); } }
        std::stable_sort(use.begin(), use.end(), [&](const Run& x, const Run& y) {
            if (segs[x.seg].spine != segs[y.seg].spine) { return segs[x.seg].spine; }
            return x.b - x.a > y.b - y.a;
        });
        if (use.size() > kMaxRuns) { use.resize(kMaxRuns); }

        // a name is shown once per part, wherever on screen: one repeat group, infinite distance
        hash_combine(params.labelOptions.repeatGroup, params.text);
        hash_combine(params.labelOptions.repeatGroup, _feat.props.getString("grp"));
        params.labelOptions.repeatDistance = 20000.f;

        LabelAttributes attrib;
        if (!prepareLabel(params, Label::Type::line, attrib)) { return false; }
        const size_t before = m_labels.size();
        const float vstep = kVertexSpacing * m_style.pixelScale();
        for (size_t rank = 0; rank < use.size(); rank++) {
            const Run& r = use[rank];
            const Seg& sg = segs[r.seg];
            // the run, and a little beyond it (the label measures its own runs again)
            const float a = std::max(0.f, r.a - margin - 1.f), b = std::min(sg.s.back(), r.b + margin + 1.f);
            // dense vertices: whether a label is on screen is decided from them
            Line path;
            std::vector<float> clearUnits;
            std::vector<glm::vec2> pathXY;
            const int steps = std::max(1, int(std::ceil((b - a) / vstep)));
            for (int i = 0; i <= steps; i++) {
                const auto v = at(sg, a + (b - a) * float(i) / float(steps));
                path.push_back(v.first);
                pathXY.push_back(v.first);
                clearUnits.push_back(v.second / pxPerUnit);
            }
            // the best place first, then spare places to either side: the label manager shows
            // the first one that no other label is in the way of
            const int slots = sg.spine ? 7 : 3;
            for (int slot = 0; slot < slots; slot++) {
                const size_t had = m_labels.size();
                addCurvedTextLabels(path, params, attrib, _rule, true);
                for (size_t i = had; i < m_labels.size(); i++) {
                    auto* cl = static_cast<CurvedLabel*>(m_labels[i].get());
                    cl->setCandidatePriority(float(rank * 8 + slot));
                    cl->setSlide(pathXY, clearUnits, halfH, margin, slot);
                }
            }
        }
        LOGD("AreaLabel '%s': %zu paths (run %.0f px, text %.0f px, tracking %.2f em)", params.text.c_str(),
             m_labels.size() - before, runMax, attrib.width, params.letterSpacing);
        if (m_labels.size() == before) { m_quads.resize(quadsStart); }
        return true;
    }

    const AreaLabelStyle& m_areaStyle;
    std::shared_ptr<AreaRegistry> m_registry;
    bool m_marker = false;
    bool m_skip = false;
    uint64_t m_tileKey = 0;
    glm::dvec2 m_origin{0.0};
    double m_scale = 1.0;
    float m_extent = 0.f;
    int m_zoom = 0;
    std::vector<AreaFragmentPtr> m_frags;
};

void AreaLabelStyle::build(const Scene& _scene) {
    m_registry = _scene.areaRegistry();
    Style::build(_scene);
}

std::unique_ptr<StyleBuilder> AreaLabelStyle::createBuilder() const {
    return std::make_unique<AreaLabelStyleBuilder>(*this);
}

}  // namespace Tangram
