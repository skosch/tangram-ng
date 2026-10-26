#pragma once

#include "labels/areaRegistry.h"
#include "util/areaShape.h"
#include "util/types.h"

#include <memory>
#include <unordered_map>

namespace Tangram {

class MarkerManager;
class Scene;
class TileManager;
class View;

// World-space planner for area labels (docs/label-typography.md, "Round 6"). Stitches the tile
// fragments the registry holds into whole features, finds each feature's parts / spines
// (util/areaShape.h) -- once per feature, cached until its fragment set changes -- and keeps one
// marker per labelled part. The markers are ordinary text-on-a-line markers: AreaLabelStyle lays
// the name out on the part's spine for the zoom they are built at, the label manager then does
// its usual per-frame projection / fit / collision work, which is how a label hides in 3D when
// the foreshortened spine can no longer hold it.
class AreaLabelManager {
public:
    AreaLabelManager(Scene& _scene, std::shared_ptr<AreaRegistry> _registry);

    /// Main thread, once per frame, before MarkerManager::update. True when markers changed.
    bool update(const View& _view, float _dt, TileManager& _tiles, MarkerManager& _markers);

    // Stats for the debug overlay / profiling
    /// More clusters wait for analysis: keep the frames coming.
    bool pending() const { return m_pending; }
    size_t numClusters() const { return m_numClusters; }
    size_t numMarkers() const { return m_entries.size(); }

private:
    void assignIdentities();
    struct Analysis {
        uint64_t signature = 0;
        AreaShape shape;
    };
    struct Entry {
        MarkerID marker = 0;
        uint64_t hash = 0;
        float lastSeen = 0.f;
        uint64_t cluster = 0;
        std::string kind, name;
        glm::dvec2 lo{0.0}, hi{0.0};
        float zb = 0.f;             // zoom the label's tracking was planned for
        bool clipped = false;       // paths cover only the neighbourhood of the view
        glm::dvec2 buildCentre{0.0};
        glm::dvec2 buildSize{0.0};  // size of the view the paths were cut for
    };
    // Cluster ids must survive the cluster growing (tiles arriving): matched by name + overlap.
    struct Ident {
        uint64_t id = 0;
        std::string kind, name;
        glm::dvec2 lo{0.0}, hi{0.0};
    };

    Scene& m_scene;
    std::shared_ptr<AreaRegistry> m_registry;

    std::unordered_map<uint64_t, Analysis> m_analysis;  // by cluster id
    std::unordered_map<uint64_t, Entry> m_entries;      // by label key
    std::vector<AreaCluster> m_clusters;
    std::unordered_map<std::string, std::vector<Ident>> m_idents;  // by kind + name
    size_t m_numIdents = 0;

    uint64_t m_lastVersion = 0, m_lastVisible = 0;
    int m_lastZoom = -1;
    glm::dvec2 m_lastCenter{0.0};
    bool m_pending = false;
    float m_time = 0.f, m_lastClean = 0.f;
    size_t m_numClusters = 0;
    int m_debug = -1;  // -1 unknown, 0 off, 1 on (global.debug_area_parts)
};

}  // namespace Tangram
