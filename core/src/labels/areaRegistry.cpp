#include "debug/profiler.h"
#include "labels/areaRegistry.h"

#include "util/hash.h"

#include <algorithm>
#include <map>
#include <set>

namespace Tangram {

bool AreaRegistry::hasTile(uint64_t _tileKey) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_tiles.find(_tileKey) != m_tiles.end();
}

void AreaRegistry::addFragment(const AreaFragmentPtr& _f) const {
    Group& g = m_groups[{ _f->kind, _f->name }];
    g.frags.push_back(_f);
    g.dirty = true;
}

void AreaRegistry::removeFragment(const AreaFragmentPtr& _f) const {
    auto git = m_groups.find({ _f->kind, _f->name });
    m_fragLoc.erase(_f.get());
    if (git == m_groups.end()) { return; }
    auto& fr = git->second.frags;
    fr.erase(std::remove(fr.begin(), fr.end(), _f), fr.end());
    if (fr.empty()) {
        m_groups.erase(git);
    } else {
        git->second.dirty = true;
    }
}

void AreaRegistry::setTile(uint64_t _tileKey, std::vector<AreaFragmentPtr>&& _frags) {
    std::lock_guard<std::mutex> lock(m_mutex);
    auto it = m_tiles.find(_tileKey);
    if (it == m_tiles.end()) {
        m_order.push_back(_tileKey);
        for (const auto& f : _frags) { addFragment(f); }
        m_tiles.emplace(_tileKey, std::move(_frags));
        while (m_order.size() > kMaxTiles) {
            auto old = m_tiles.find(m_order.front());
            if (old != m_tiles.end()) {
                for (const auto& f : old->second) { removeFragment(f); }
                m_tiles.erase(old);
            }
            m_order.pop_front();
        }
    } else {
        for (const auto& f : it->second) { removeFragment(f); }
        for (const auto& f : _frags) { addFragment(f); }
        it->second = std::move(_frags);
    }
    m_version++;
}

// Link the group's fragments whose boxes touch into clusters.
void AreaRegistry::stitch(const GroupKey& _key, Group& g) const {
    PROFILE_SCOPE("areaLabels:stitch");
    constexpr double eps = 5.0;  // m; fragments of neighbouring tiles share an edge
    auto& frags = g.frags;
    const size_t n = frags.size();
    g.clusters.clear();
    std::vector<int> parent(n);
    for (size_t i = 0; i < n; i++) { parent[i] = int(i); }
    auto find = [&](int a) { while (parent[a] != a) { parent[a] = parent[parent[a]]; a = parent[a]; } return a; };
    // sweep over x: only boxes whose x-ranges overlap can touch
    std::vector<int> byX(n);
    for (size_t i = 0; i < n; i++) { byX[i] = int(i); }
    std::sort(byX.begin(), byX.end(), [&](int a, int b) { return frags[a]->lo.x < frags[b]->lo.x; });
    for (size_t ii = 0; ii < n; ii++) {
        const auto& a = *frags[byX[ii]];
        for (size_t jj = ii + 1; jj < n; jj++) {
            const auto& b = *frags[byX[jj]];
            if (b.lo.x - eps > a.hi.x) { break; }
            if (a.lo.y - eps <= b.hi.y && b.lo.y - eps <= a.hi.y) {
                parent[find(byX[ii])] = find(byX[jj]);
            }
        }
    }
    std::map<int, AreaCluster> byRoot;
    for (size_t i = 0; i < n; i++) {
        AreaCluster& c = byRoot[find(int(i))];
        if (c.frags.empty()) { c.kind = _key.first; c.name = _key.second; c.lo = frags[i]->lo; c.hi = frags[i]->hi; }
        c.lo = glm::min(c.lo, frags[i]->lo);
        c.hi = glm::max(c.hi, frags[i]->hi);
        c.frags.push_back(frags[i]);
    }
    for (auto& kv : byRoot) {
        AreaCluster& c = kv.second;
        uint64_t minTile = ~uint64_t(0);
        size_t sig = 0;
        std::sort(c.frags.begin(), c.frags.end(), [](const AreaFragmentPtr& a, const AreaFragmentPtr& b) {
            return a->tileKey < b->tileKey;
        });
        for (const auto& f : c.frags) {
            minTile = std::min(minTile, f->tileKey);
            hash_combine(sig, reinterpret_cast<uintptr_t>(f.get()));
        }
        size_t id = 0;
        hash_combine(id, c.kind);
        hash_combine(id, c.name);
        hash_combine(id, minTile);
        c.id = id;
        c.signature = sig;
        const int idx = int(g.clusters.size());
        for (const auto& f : c.frags) { m_fragLoc[f.get()] = { &g, idx }; }
        g.clusters.push_back(std::move(c));
    }
    g.dirty = false;
}

std::vector<AreaCluster> AreaRegistry::clusters(const std::unordered_set<uint64_t>& _visible) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    for (auto& kv : m_groups) {
        if (kv.second.dirty) { stitch(kv.first, kv.second); }
    }
    std::set<std::pair<const Group*, int>> picked;
    for (uint64_t key : _visible) {
        auto it = m_tiles.find(key);
        if (it == m_tiles.end()) { continue; }
        for (const auto& f : it->second) {
            auto loc = m_fragLoc.find(f.get());
            if (loc != m_fragLoc.end()) { picked.insert({ loc->second.first, loc->second.second }); }
        }
    }
    std::vector<AreaCluster> out;
    out.reserve(picked.size());
    for (const auto& p : picked) { out.push_back(p.first->clusters[p.second]); }
    return out;
}

}  // namespace Tangram
