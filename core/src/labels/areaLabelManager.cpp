#include "labels/areaLabelManager.h"

#include "data/propertyItem.h"
#include "debug/profiler.h"
#include <cstdlib>
#include "log.h"
#include "marker/markerManager.h"
#include "scene/scene.h"
#include "tile/tile.h"
#include "tile/tileManager.h"
#include "util/hash.h"
#include "util/mapProjection.h"
#include "view/view.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>

namespace Tangram {

namespace {

constexpr double kAnalyzeBudgetMs = 4.0;   // per frame
constexpr float kKeepSeconds = 10.f;       // markers outlive their tiles' visibility by this much
constexpr float kMinPartShare = 0.15f;     // a part needs this share of the largest part's area

uint64_t labelKey(uint64_t _cluster, size_t _part, int _salt) {
    size_t h = 0;
    hash_combine(h, _cluster);
    hash_combine(h, _part);
    hash_combine(h, _salt);
    return h;
}

}  // namespace

void AreaLabelManager::assignIdentities() {
    PROFILE_SCOPE("areaLabels:identities");
    std::unordered_set<uint64_t> taken;
    for (AreaCluster& c : m_clusters) {
        auto& list = m_idents[c.kind + '\x1f' + c.name];
        Ident* best = nullptr;
        double bestOverlap = -1.0;
        for (Ident& i : list) {
            if (taken.count(i.id)) { continue; }
            glm::dvec2 lo = glm::max(i.lo, c.lo), hi = glm::min(i.hi, c.hi);
            if (hi.x < lo.x || hi.y < lo.y) { continue; }
            double o = (hi.x - lo.x + 1.0) * (hi.y - lo.y + 1.0);
            if (o > bestOverlap) { bestOverlap = o; best = &i; }
        }
        if (best) {
            c.id = best->id;
            best->lo = c.lo;
            best->hi = c.hi;
        } else {
            uint64_t id = c.id;
            while (taken.count(id)) { id++; }
            list.push_back({ id, c.kind, c.name, c.lo, c.hi });
            m_numIdents++;
            c.id = id;
        }
        taken.insert(c.id);
    }
    if (m_numIdents > 4000) { m_idents.clear(); m_numIdents = 0; }
}

// Each window of a label gets a priority from where it sits in the view: nearer the middle is
// better, and only windows in the inner part of the screen are eligible at all (so a window
// hands over to the next one before it hits the edge, and the label follows the view along the
// spine). The label manager's own sort then picks among the eligible windows, collisions included.
AreaLabelManager::AreaLabelManager(Scene& _scene, std::shared_ptr<AreaRegistry> _registry)
    : m_scene(_scene), m_registry(std::move(_registry)) {}

bool AreaLabelManager::update(const View& _view, float _dt, TileManager& _tiles, MarkerManager& _markers) {
    PROFILE_SCOPE("areaLabels:update");
    if (!m_scene.isReady()) { return false; }
    // A/B switch for perf / flicker hunting: ASCEND_NO_AREA_LABELS=1 plans no world-space area labels
    static const bool disabled = getenv("ASCEND_NO_AREA_LABELS") && atoi(getenv("ASCEND_NO_AREA_LABELS")) != 0;
    if (disabled) { return false; }
    m_time += _dt;

    if (m_debug < 0) {
        const auto& global = m_scene.config()["global"];
        m_debug = (global && global["debug_area_parts"] && global["debug_area_parts"].as<bool>(false)) ? 1 : 0;
    }

    // What is on screen: the data tiles in view (a tile's key does not depend on its styling zoom).
    uint64_t visHash = 0;
    for (const auto& tile : _tiles.getVisibleTiles()) {
        const TileID id = tile->getID();
        visHash += AreaRegistry::tileKey(tile->sourceID(), id.x, id.y, id.z) * 0x9e3779b97f4a7c15ull;
    }
    const uint64_t version = m_registry->version();
    const int zoom = _view.getIntegerZoom();
    const glm::dmat2 bounds = _view.getBoundsRect();
    const glm::dvec2 vc = 0.5 * (bounds[0] + bounds[1]);
    const glm::dvec2 vsize = bounds[1] - bounds[0];

    // A label whose zoom plan or cut-out path no longer suits the view is rebuilt at once, not at the
    // next periodic pass: after zooming out the text planned for the old zoom does not fit the
    // shorter spine, and until it is rebuilt the label is simply gone.
    bool stale = false;
    {
        const float zf = _view.getZoom();
        for (const auto& kv : m_entries) {
            const Entry& e = kv.second;
            if (e.marker == 0 || e.zb <= 0.f) { continue; }
            stale |= zf >= e.zb + 0.25f || zf < e.zb - 0.03f ||
                     (e.clipped && (std::fabs(vc.x - e.buildCentre.x) > 0.5 * vsize.x ||
                                    std::fabs(vc.y - e.buildCentre.y) > 0.5 * vsize.y || vsize.x > 1.5 * e.buildSize.x));
        }
    }
    const bool moved = std::fabs(vc.x - m_lastCenter.x) > 0.25 * vsize.x || std::fabs(vc.y - m_lastCenter.y) > 0.25 * vsize.y;
    const bool setChanged = version != m_lastVersion || visHash != m_lastVisible || zoom != m_lastZoom;
    const bool changed = setChanged || m_time - m_lastClean >= 2.f || moved || stale;
    if (!changed && !m_pending) { return false; }
    if (moved) { m_lastCenter = vc; }

    bool markersChanged = false;
    std::unordered_map<uint64_t, bool> wanted;
    std::unordered_set<uint64_t> processed;   // clusters whose parts were all decided in this pass

    if (changed || m_pending) {
        if (setChanged) {
            PROFILE_SCOPE("areaLabels:clusters");
            m_lastVersion = version;
            m_lastVisible = visHash;
            m_lastZoom = zoom;

            std::unordered_set<uint64_t> visible;
            for (const auto& tile : _tiles.getVisibleTiles()) {
                const TileID id = tile->getID();
                visible.insert(AreaRegistry::tileKey(tile->sourceID(), id.x, id.y, id.z));
            }
            m_clusters = m_registry->clusters(visible);
            m_numClusters = m_clusters.size();
            assignIdentities();
        }
        m_pending = false;

        using Clock = std::chrono::steady_clock;
        const auto t0 = Clock::now();
        // px per projected meter at the top of this zoom level: a part that cannot hold even a
        // few pixels of text there never will
        const double pxPerMeterTop = 256.0 * std::exp2(double(zoom + 1)) / MapProjection::EARTH_CIRCUMFERENCE_METERS;

        for (const AreaCluster& cluster : m_clusters) {
            const bool park = cluster.kind == "park";
            if (cluster.kind != "water" && cluster.kind != "glacier" && !park) { continue; }  // no marker style for it
            std::string layer = cluster.kind;
            if (park) {
                std::string b;
                const bool national = cluster.frags.front()->props.getString("boundary", b) && b == "national_park";
                if (zoom < (national ? 8 : 10)) { continue; }
                if (national) { layer = "park_national"; }
            }
            // too small to hold a name at the top of this zoom level: don't even analyse
            if (glm::length(cluster.hi - cluster.lo) * pxPerMeterTop < 60.0) { continue; }
            auto it = m_analysis.find(cluster.id);
            if (it == m_analysis.end() || it->second.signature != cluster.signature) {
                if (std::chrono::duration<double, std::milli>(Clock::now() - t0).count() > kAnalyzeBudgetMs) {
                    m_pending = true;  // finish next frame
                    continue;
                }
                const auto ta = Clock::now();
                std::vector<AreaPolygon> polys;
                for (const auto& f : cluster.frags) { polys.insert(polys.end(), f->polys.begin(), f->polys.end()); }
                Analysis a;
                a.signature = cluster.signature;
                AreaShapeParams sp;
                sp.alwaysHorizontal = park;
                sp.chordRows = park ? 40 : 16;
                {
                    PROFILE_SCOPE("areaLabels:analyze");
                    if (!analyzeArea(polys, sp, a.shape)) { a.shape.parts.clear(); }
                }
                m_analysis[cluster.id] = std::move(a);
                it = m_analysis.find(cluster.id);
                LOGD("AreaLabels: analysed '%s' (%zu fragments, %d cells, %zu parts) in %.2f ms", cluster.name.c_str(),
                     cluster.frags.size(), it->second.shape.cells, it->second.shape.parts.size(),
                     std::chrono::duration<double, std::milli>(Clock::now() - ta).count());
            }
            const AreaShape& shape = it->second.shape;
            processed.insert(cluster.id);
            if (shape.parts.empty()) { continue; }

            const float largest = shape.parts.front().area;
            for (size_t k = 0; k < shape.parts.size(); k++) {
                const AreaPart& part = shape.parts[k];
                if (part.area < kMinPartShare * largest) { continue; }
                if (part.maxRadius * pxPerMeterTop < 5.0) { continue; }

                // identity of the label: cluster + part
                const uint64_t key = labelKey(cluster.id, k, 0);
                size_t h = 0;
                hash_combine(h, cluster.signature);
                wanted[key] = true;
                if (m_debug == 1 && zoom >= 9) {
                    wanted[labelKey(cluster.id, k, 1)] = true;
                    auto d = m_entries.find(labelKey(cluster.id, k, 1));
                    if (d != m_entries.end()) { d->second.lastSeen = m_time; }
                }

                Entry& e = m_entries[key];
                e.lastSeen = m_time;
                e.cluster = cluster.id;
                e.kind = cluster.kind;
                e.name = cluster.name;
                e.lo = cluster.lo;
                e.hi = cluster.hi;
                const bool fresh = e.marker == 0 || e.hash != h;

                // The zoom the tracking is planned for moves in quarter steps (with a little
                // hysteresis downwards): finer than the integer zoom markers are built at, so
                // the text widens in small steps instead of one big jump per zoom level.
                const float zf = _view.getZoom();
                float zb = e.zb;
                if (e.marker == 0 || zf >= zb + 0.25f || zf < zb - 0.03f) { zb = std::floor(zf * 4.f) / 4.f; }
                // A part much bigger than the screen only gets windows near the view; once the view
                // has travelled well away from where they were laid out, lay out new ones.
                const bool recentre = e.clipped && (std::fabs(vc.x - e.buildCentre.x) > 0.5 * vsize.x ||
                                                    std::fabs(vc.y - e.buildCentre.y) > 0.5 * vsize.y ||
                                                    vsize.x > 1.5 * e.buildSize.x);
                if (!fresh && zb == e.zb && !recentre) { continue; }

                // Paths for the label laid end to end: the spine, then
                // horizontal chords (a park has no spine, only chords), cut to the neighbourhood
                // of the view.
                const glm::dvec2 rlo = vc - 1.25 * vsize, rhi = vc + 1.25 * vsize;
                auto inR = [&](glm::dvec2 w) { return w.x >= rlo.x && w.x <= rhi.x && w.y >= rlo.y && w.y <= rhi.y; };
                std::vector<LngLat> pts;
                std::string clr, brk, ext;
                bool clipped = false;
                char buf[32];
                // how far the spine goes on past each end of a clipped piece while it stays wide
                // enough for text (a conservative 11 px at the planned zoom): the tracking is fitted to
                // the whole stretch, not to the piece that happens to be near the view
                const double pxPerMeterZb = 256.0 * std::exp2(double(zb)) / MapProjection::EARTH_CIRCUMFERENCE_METERS;
                const float minClear = float(11.0 / pxPerMeterZb);
                auto add = [&](const std::vector<glm::vec2>& path0, const std::vector<float>& cl0, char type) {
                    // vertices at most a fifth of the view apart: whether a stretch is near the view is
                    // decided from them, and a coarse spine can cross the view between two of its own
                    std::vector<glm::vec2> path;
                    std::vector<float> cl;
                    const float maxStep = float(0.2 * std::min(vsize.x, vsize.y));
                    for (size_t q = 0; q < path0.size(); q++) {
                        path.push_back(path0[q]);
                        cl.push_back(cl0[q]);
                        if (q + 1 == path0.size()) { break; }
                        const int parts = std::max(1, int(std::ceil(glm::distance(path0[q], path0[q + 1]) / maxStep)));
                        for (int t = 1; t < parts; t++) {
                            const float f = float(t) / float(parts);
                            path.push_back(glm::mix(path0[q], path0[q + 1], f));
                            cl.push_back(glm::mix(cl0[q], cl0[q + 1], f));
                        }
                    }
                    size_t i = 0;
                    while (i < path.size()) {
                        while (i < path.size() && !inR(shape.origin + glm::dvec2(path[i]))) { i++; clipped = true; }
                        size_t j = i;
                        while (j < path.size() && inR(shape.origin + glm::dvec2(path[j]))) { j++; }
                        if (j - i >= 2) {
                            snprintf(buf, sizeof(buf), "%zu%c ", pts.size(), type);
                            brk += buf;
                            float e0 = 0.f, e1 = 0.f;
                            for (size_t q = i; q > 0 && cl[q - 1] >= minClear; q--) { e0 += glm::distance(path[q], path[q - 1]); }
                            for (size_t q = j - 1; q + 1 < path.size() && cl[q + 1] >= minClear; q++) { e1 += glm::distance(path[q], path[q + 1]); }
                            snprintf(buf, sizeof(buf), "%.0f %.0f ", e0, e1);
                            ext += buf;
                            for (size_t q = i; q < j; q++) {
                                const glm::dvec2 w = shape.origin + glm::dvec2(path[q]);
                                pts.push_back(MapProjection::projectedMetersToLngLat(w));
                                snprintf(buf, sizeof(buf), "%.1f ", cl[q]);
                                clr += buf;
                            }
                        }
                        if (j < path.size()) { clipped = true; }
                        i = j;
                    }
                };
                if (!park || part.chords.empty()) { add(part.spine, part.clearance, 's'); }
                for (const AreaChord& ch : part.chords) { add(ch.pts, ch.clearance, 'c'); }
                if (pts.size() < 2) { continue; }

                if (e.marker == 0) { e.marker = _markers.add(); }
                e.hash = h;
                e.zb = zb;
                e.clipped = clipped;
                e.buildCentre = vc;
                e.buildSize = vsize;
                markersChanged = true;

                Properties props = cluster.frags.front()->props;
                props.set("clr", clr);
                props.set("brk", brk);
                props.set("ext", ext);
                props.set("grp", std::to_string(key));
                snprintf(buf, sizeof(buf), "%.2f", zb);
                props.set("zb", std::string(buf));
                _markers.setPolyline(e.marker, pts.data(), int(pts.size()));
                _markers.setProperties(e.marker, std::move(props));
                _markers.setStylingFromPath(e.marker, ("layers.area_label_" + layer + ".draw.text").c_str());

                if (m_debug == 1 && fresh && zoom >= 9) {
                    pts.clear();
                    for (const auto& sp : part.spine) {
                        pts.push_back(MapProjection::projectedMetersToLngLat(shape.origin + glm::dvec2(sp)));
                    }
                    static const char* colors[] = { "#ff00ff", "#00e5ff", "#ffea00", "#76ff03", "#ff6e40" };
                    Entry& d = m_entries[labelKey(cluster.id, k, 1)];
                    d.lastSeen = m_time;
                    d.cluster = cluster.id; d.kind = cluster.kind; d.name = cluster.name; d.lo = cluster.lo; d.hi = cluster.hi;
                    wanted[labelKey(cluster.id, k, 1)] = true;
                    if (d.marker == 0) { d.marker = _markers.add(); }
                    _markers.setPolyline(d.marker, pts.data(), int(pts.size()));
                    _markers.setStylingFromString(d.marker,
                        (std::string("{ style: unlit-lines, color: '") + colors[k % 5] +
                         "', width: 3px, order: 950 }").c_str());
                }
            }
            if (m_debug == 1) {
                for (size_t k = 0; k < shape.necks.size(); k++) {
                    const uint64_t key = labelKey(cluster.id, 1000 + k, 2);
                    wanted[key] = true;
                    Entry& e = m_entries[key];
                    e.lastSeen = m_time;
                    e.cluster = cluster.id; e.kind = cluster.kind; e.name = cluster.name; e.lo = cluster.lo; e.hi = cluster.hi;
                    if (e.marker != 0) { continue; }
                    e.marker = _markers.add();
                    markersChanged = true;
                    std::vector<LngLat> ring;
                    const float r = std::max(shape.necks[k].radius, shape.cell);
                    for (int i = 0; i <= 16; i++) {
                        double a = 2.0 * M_PI * i / 16;
                        glm::dvec2 w = shape.origin + glm::dvec2(shape.necks[k].pos) + glm::dvec2(std::cos(a), std::sin(a)) * double(r);
                        ring.push_back(MapProjection::projectedMetersToLngLat(w));
                    }
                    _markers.setPolyline(e.marker, ring.data(), int(ring.size()));
                    _markers.setStylingFromString(e.marker,
                        "{ style: unlit-lines, color: '#ff1744', width: 3px, order: 950 }");
                }
            }
        }
        if (m_analysis.size() > 600) {
            for (auto it = m_analysis.begin(); it != m_analysis.end();) {
                bool live = std::any_of(m_clusters.begin(), m_clusters.end(),
                                        [&](const AreaCluster& c) { return c.id == it->first; });
                it = live ? std::next(it) : m_analysis.erase(it);
            }
        }
    }

    if (markersChanged) {
        LOGN("AreaLabels: t=%.1fs %zu clusters in view, %zu analysed, %zu markers, zoom %d%s", m_time, m_clusters.size(),
             m_analysis.size(), m_entries.size(), zoom, m_pending ? " (more pending)" : "");
    }

    // drop labels that have been out of sight for a while (or whose feature is gone)
    if (m_time - m_lastClean >= 2.f || (changed && !m_pending)) {
        m_lastClean = m_time;
        for (auto it = m_entries.begin(); it != m_entries.end();) {
            // an old label of a feature that has a current one (its cluster was re-identified, or
            // it left the screen and came back under a new id) must not stay up beside it
            bool superseded = false;
            if (!wanted.count(it->first) && !it->second.name.empty()) {
                bool alive = false;
                for (const AreaCluster& c : m_clusters) { alive |= c.id == it->second.cluster; }
                for (const AreaCluster& c : m_clusters) {
                    if (!alive && c.kind == it->second.kind && c.name == it->second.name && c.lo.x <= it->second.hi.x &&
                        it->second.lo.x <= c.hi.x && c.lo.y <= it->second.hi.y && it->second.lo.y <= c.hi.y) {
                        superseded = true;
                        break;
                    }
                }
            }
            // a part that the latest analysis of its cluster no longer has (the lake grew, two parts
            // became one) goes at once: it must not linger beside its replacement
            // (debug spines below zoom 9 cannot be drawn: the marker build fails and is retried every frame)
            const bool orphan = !wanted.count(it->first) &&
                                (processed.count(it->second.cluster) > 0 || (m_debug == 1 && zoom < 9 && it->second.zb <= 0.f));
            if (!wanted.count(it->first) && (orphan || superseded || m_time - it->second.lastSeen > kKeepSeconds)) {
                if (it->second.marker != 0) { _markers.remove(it->second.marker); }
                it = m_entries.erase(it);
                markersChanged = true;
            } else {
                ++it;
            }
        }
    }
    return markersChanged || m_pending;
}

}  // namespace Tangram
