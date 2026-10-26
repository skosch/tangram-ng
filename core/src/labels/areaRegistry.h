#pragma once

#include "data/properties.h"
#include "util/areaShape.h"

#include <atomic>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace Tangram {

// One named area feature's polygons from ONE data tile, in world (projected) meters. Deposited
// by AreaLabelStyle's tile builders (worker threads); stitched into whole features by
// AreaRegistry::clusters() and analysed by AreaLabelManager (docs/label-typography.md,
// "Round 6: world-space area labels").
struct AreaFragment {
    uint64_t tileKey = 0;
    std::string kind;           // "water", "glacier", "area" -> which marker style labels it
    std::string name;
    Properties props;           // the first feature's properties, for the label's JS styling
    std::vector<AreaPolygon> polys;
    glm::dvec2 lo{0.0}, hi{0.0};
};
using AreaFragmentPtr = std::shared_ptr<const AreaFragment>;

// All fragments that belong together: same kind and name, touching bounding boxes. A lake that
// crosses five z11 tiles is five fragments and one cluster.
struct AreaCluster {
    uint64_t id = 0;            // stable while the cluster's lowest tile does not change
    uint64_t signature = 0;     // changes whenever the fragment set changes
    std::string kind, name;
    std::vector<AreaFragmentPtr> frags;
    glm::dvec2 lo{0.0}, hi{0.0};
};

class AreaRegistry {
public:
    static uint64_t tileKey(int32_t _source, int32_t _x, int32_t _y, int32_t _z) {
        return (uint64_t(uint8_t(_z)) << 56) ^ (uint64_t(uint32_t(_x) & 0xfffffff) << 28) ^
               (uint64_t(uint32_t(_y) & 0xfffffff)) ^ (uint64_t(uint8_t(_source)) << 48);
    }

    // --- worker threads -------------------------------------------------------------------
    bool hasTile(uint64_t _tileKey) const;
    /// Replace everything deposited for this tile (an empty list marks "seen, nothing here").
    void setTile(uint64_t _tileKey, std::vector<AreaFragmentPtr>&& _frags);

    // --- main thread ----------------------------------------------------------------------
    uint64_t version() const { return m_version.load(); }
    /// Clusters that have at least one fragment from a tile in `_visible`.
    std::vector<AreaCluster> clusters(const std::unordered_set<uint64_t>& _visible) const;

private:
    static constexpr size_t kMaxTiles = 3000;

    // Fragments of one (kind, name), and the clusters they stitch into. Rebuilt only when a tile
    // adds or removes one of its fragments, so a frame that merely looks at other tiles costs
    // nothing here, and a tile arriving costs one name's worth of stitching, not the registry's.
    struct Group {
        std::vector<AreaFragmentPtr> frags;
        std::vector<AreaCluster> clusters;
        bool dirty = true;
    };
    using GroupKey = std::pair<std::string, std::string>;
    void addFragment(const AreaFragmentPtr& _f) const;       // m_mutex held
    void removeFragment(const AreaFragmentPtr& _f) const;    // m_mutex held
    void stitch(const GroupKey& _key, Group& _g) const;      // m_mutex held

    mutable std::mutex m_mutex;
    std::unordered_map<uint64_t, std::vector<AreaFragmentPtr>> m_tiles;
    std::deque<uint64_t> m_order;
    std::atomic<uint64_t> m_version{1};

    mutable std::map<GroupKey, Group> m_groups;
    mutable std::unordered_map<const AreaFragment*, std::pair<Group*, int>> m_fragLoc;  // fragment -> its cluster
};

}  // namespace Tangram
