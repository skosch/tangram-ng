#pragma once

#include "data/tileData.h"
#include "data/tileSource.h"
#include "tile/tile.h"
#include "tile/tileID.h"
#include "tile/tileTask.h"
#include "tile/tileWorker.h"

#include <map>
#include <memory>
#include <mutex>
#include <tuple>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Tangram {

class Platform;
class TileSource;
class RasterSource;
class TileCache;
class View;
struct ViewState;

/* Singleton container of <TileSet>s
 *
 * TileManager is a singleton that maintains a set of Tiles based on the current
 * view into the map
 */
class TileManager {

    const static size_t DEFAULT_CACHE_SIZE = 32*1024*1024; // 32 MB

public:

    TileManager(Platform& platform, TileTaskQueue& _tileWorker, std::weak_ptr<ScenePrana> _prana);

    virtual ~TileManager();

    /* Sets the tile TileSources */
    void setTileSources(const std::vector<std::shared_ptr<TileSource>>& _sources);

    /* Updates visible tile set and load missing tiles */
    bool updateTileSets(const View& _view);

    void clearTileSets(bool clearSourceCaches = false);

    void clearTileSet(int32_t _sourceId);

    /* Returns the set of currently visible tiles */
    const auto& getVisibleTiles() const { return m_tiles; }

    int numLoadingTiles() const { return m_tilesInProgress; }
    int numTotalTiles() const;

    std::shared_ptr<TileSource> getTileSource(int32_t _sourceId);

    void addClientTileSource(std::shared_ptr<TileSource> _source);

    bool removeClientTileSource(int32_t _sourceId);

    const std::unique_ptr<TileCache>& getTileCache() const { return m_tileCache; }

    /* @_cacheSize: Set size of in-memory tile cache in bytes.
     * This cache holds recently used <Tile>s that are ready for rendering.
     */
    void setCacheSize(size_t _cacheSize);

protected:

    enum class ProxyID : uint8_t;
    struct TileEntry;

    struct TileSet {
        TileSet(std::shared_ptr<TileSource> _source);
        ~TileSet();
        //void cancelTasks();

        std::shared_ptr<TileSource> source;

        std::set<TileID> visibleTiles;
        std::map<TileID, TileEntry> tiles;

        int64_t sourceGeneration = 0;

        // R5 (tile-pipeline-perf-plan.md): count and steady_clock time (seconds) of the most
        // recent failure for a neighbor-prefetch TileID that was canceled/errored before
        // completing (see the erase branch in updateTileSet()) - checked by
        // enqueueNeighborPrefetch() to avoid re-enqueueing a permanently-failing neighbor
        // (e.g. a 404 at the data edge, or offline) every single frame forever.
        std::map<TileID, std::pair<int, double>> prefetchFailures;

        // R5: last observed view center/zoom and when it last differed from the previous
        // update's - used to gate ring prefetches on the view having actually settled (they
        // only matter once the center tiles are up anyway), so a fast pan doesn't spend fetch
        // slots on ring tiles the user is about to leave behind. Per-TileSet (not shared)
        // since each TileSet's updateTileSet() call is independent and any could be first.
        double lastViewCenterX = 0.0, lastViewCenterY = 0.0;
        float lastViewZoom = -1.f;
        double viewSettledSince = 0.0;

        TileSet(const TileSet&) = delete;
        TileSet(TileSet&&) = default;
        TileSet& operator=(const TileSet&) = delete;
        TileSet& operator=(TileSet&&) = default;
    };

    void updateTileSet(TileSet& tileSet, const ViewState& _view);

    void enqueueTask(TileSet& _tileSet, const TileID& _tileID, const ViewState& _view);

    // Phase 1 (docs/texture-shading-plan.md): for TileSets whose source has
    // RasterSource::m_prefetchNeighbors set, proactively enqueue low-priority
    // background fetches of each visible tile's 8 same-zoom neighbor tiles. No-op for
    // any other source. See TileEntry::m_prefetchOnly for how these are kept from
    // affecting rendering/proxy logic for real (visible) tiles.
    void enqueueNeighborPrefetch(TileSet& _tileSet, const ViewState& _view);

    void loadTiles();

    // add new visible tile to TileSet
    bool addTile(TileSet& _tileSet, const TileID& _tileID);

    // Revisit a tile's raster attachment(s) and swap in a better-resolution texture if one has
    // become available since the tile was built (see definition for details): the shared 3Wx3W
    // texture-shading mosaic for a tile built as prefetch-only/proxy, or the opportunistic
    // overzoom elevation composite for a draped vector tile (e.g. `osm` polygons) whose attached
    // elevation raster was baked in at build time before its finer descendant tiles were cached.
    // `_liveId` is the tile's CURRENT visible TileID -- its `s` is the LIVE styling zoom, which
    // must drive the composite target zoom. The s frozen into raster.tileID at build time must
    // NOT be used for that: a tile built (or cache-restored) at a different view zoom would
    // keep targeting the OLD zoom forever (observed live as stable pairs of cached tile
    // variants flipping with pan, each keeping its own stale-zoom drape and its own set of
    // "white patch" artifacts).
    void upgradeAttachedRasters(TileSet& _tileSet, Tile& _tile, const TileID& _liveId);

    // check cache for proxy for new tile
    void updateProxyTiles(TileSet& _tileSet, const TileID& _tileID, TileEntry& _tile);

    TileSet* findTileSet(int64_t sourceId);

    int32_t m_tilesInProgress = 0;

    std::vector<TileSet> m_tileSets;
    std::vector<TileSet> m_auxTileSets;

    /* Current tiles ready for rendering */
    std::vector<std::shared_ptr<Tile>> m_tiles;

    std::unique_ptr<TileCache> m_tileCache;
    size_t m_maxCacheLimit = DEFAULT_CACHE_SIZE;

    TileTaskQueue& m_workers;

    std::weak_ptr<ScenePrana> m_scenePrana;

    bool m_tileSetChanged = false;

    /* Callback for TileSource:
     * Passes TileTask back with data for further processing by <TileWorker>s
     */
    TileTaskCb m_dataCallback;

    /* Temporary list of tiles that need to be loaded */
    struct TileLoadTask { double dist; TileSet* tileSet; TileID tileID; };
    std::vector<TileLoadTask> m_loadTasks;

    // Perf (2026-07 profiling): upgradeAttachedRasters() (Case 2, draped vector tiles) looks up
    // the elevation RasterSource's own TileSet and builds normalized `rendered`/`ready` id sets
    // from its ENTIRE visibleTiles/tiles to compute per-cell displayed-zoom caps - see that
    // function's doc comment for why these sets must exist at all. Both the elevSet lookup and
    // the sets built from it are identical for every raster attachment on every tile processed
    // within one updateTileSets() pass (elevSet's own visibleTiles/tiles don't change again once
    // that pass has reached this tileSet), yet the un-cached code was redoing this full rebuild
    // once per (visible tile x raster attachment) - up to ~140 times per update for the same
    // handful of RasterSources. Cache keyed by RasterSource*, populated lazily on first use each
    // pass and cleared at the top of updateTileSets() below.
    std::unordered_map<RasterSource*, std::pair<std::set<TileID>, std::set<TileID>>> m_upgradeRasterCache;

};

}
