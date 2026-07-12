#include "tile/tileManager.h"

#include "debug/profiler.h"
#include "data/tileSource.h"
#include "data/rasterSource.h"
#include "map.h"
#include "platform.h"
#include "tile/tile.h"
#include "tile/tileCache.h"
#include "util/mapProjection.h"
#include "view/view.h"

#include "glm/gtx/norm.hpp"

#include <algorithm>
#include <bitset>
#include <chrono>

namespace Tangram {

#define MAX_TILE_SETS 64

namespace {
double nowSeconds() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
// R5 (tile-pipeline-perf-plan.md): a neighbor whose prefetch has failed this many times within
// kPrefetchFailureBackoffSeconds is skipped rather than re-enqueued every single frame forever
// (e.g. a 404 at the data edge, or offline) - see TileSet::prefetchFailures.
const int kPrefetchFailureBackoffCount = 2;
const double kPrefetchFailureBackoffSeconds = 60.0;
// R5: don't bother enqueueing ring prefetches while the view is still actively moving - they
// only matter once the center tiles are up anyway, and a fast pan would otherwise spend fetch
// slots on ring tiles the user is about to leave behind.
const double kPrefetchSettleDelaySeconds = 0.2;
}

struct TileManager::TileEntry {

    TileEntry(std::shared_ptr<Tile>& _tile) : tile(_tile) {}

    ~TileEntry() { clearTask(); }

    std::shared_ptr<Tile> tile;
    std::shared_ptr<TileTask> task;

    /* A Counter for number of tiles this tile acts a proxy for */
    int32_t m_proxyCounter = 0;

    // set if tile has failed raster subtasks
    int32_t numMissingRasters = -1;

    // is tile in TileSet.visibleTiles?
    bool m_visible = false;

    // Phase 1 (docs/texture-shading-plan.md): true for entries created purely to
    // prefetch a same-zoom neighbor of some visible elevation tile (see
    // TileManager::enqueueNeighborPrefetch). Such entries must never be treated as
    // visible or as a real proxy - they exist only to keep their TileTask alive long
    // enough to finish loading and get cached, so must be kept out of m_tiles (i.e.
    // never rendered) and out of label/mesh-driving logic despite living in the same
    // TileSet::tiles map as real tiles.
    bool m_prefetchOnly = false;

    bool isInProgress() {
        return bool(task) && !task->isCanceled();
    }

    bool isCanceled() {
        return bool(task) && task->isCanceled();
    }

    bool needsLoading() {
        if (bool(tile)) { return false; }
        if (!task) { return true; }
        if (task->isCanceled()) { return false; }
        if (task->needsLoading()) { return true; }

        for (auto& subtask : task->subTasks()) {
            if (subtask->needsLoading() && !subtask->isCanceled()) { return true; }
        }
        return false;
    }

    // Complete task only when
    // - task still exists
    // - task has a tile ready
    // - tile has all rasters set
    bool completeTileTask() {
        if (bool(task) && task->isReady()) {

            for (auto& subtask : task->subTasks()) {
                if (!subtask->isReady() && !subtask->isCanceled()) { return false; }
            }

            task->complete();
            --task->shareCount;
            for (auto& subtask : task->subTasks()) {
              --subtask->shareCount;
            }
            tile = task->getTile();
            task.reset();
            numMissingRasters = -1;  // tile for ClientDataSource can be replaced w/o new TileEntry

            return true;
        }
        return false;
    }

    void clearTask() {
        if (!task) { return; }
        for (auto& subtask : task->subTasks()) {
            if (--subtask->shareCount <= 0 && !subtask->isCanceled()) {
                subtask->cancel();
                subtask->source()->cancelLoadingTile(*subtask);
            }
        }
        task->subTasks().clear();
        if (--task->shareCount <= 0 && !task->isCanceled()) {
            task->cancel();
            task->source()->cancelLoadingTile(*task);
        }
        task.reset();
    }

    // Is tile in TileSet.visibleTiles?
    bool isVisible() const {
        return m_visible;
    }

    void setVisible(bool _visible) {
        m_visible = _visible;
        // a tile that becomes genuinely visible is no longer just a prefetch stub
        if (_visible) { m_prefetchOnly = false; }
    }
};

TileManager::TileSet::TileSet(std::shared_ptr<TileSource> _source) : source(_source) {}

TileManager::TileSet::~TileSet() {}  //cancelTasks();

TileManager::TileManager(Platform& platform, TileTaskQueue& _tileWorker, std::weak_ptr<ScenePrana> _prana) :
    m_workers(_tileWorker), m_scenePrana(_prana) {

    m_tileCache = std::unique_ptr<TileCache>(new TileCache(DEFAULT_CACHE_SIZE));

    // Callback to pass task from Download-Thread to Worker-Queue
    m_dataCallback = TileTaskCb{[&](std::shared_ptr<TileTask> task) {

        if (task->isReady()) {
             platform.requestRender();

        } else if (task->hasData()) {
            m_workers.enqueue(task);

        } else {
            task->cancel();
        }
    }};
}

TileManager::~TileManager() {}  //m_tileSets.clear();

void TileManager::setTileSources(const std::vector<std::shared_ptr<TileSource>>& _sources) {

    m_tileCache->clear();
    assert(m_tileSets.empty() && m_auxTileSets.empty() && "setTileSources() should only be called once!");
    for (const auto& source : _sources) {
        if (source->generateGeometry()) {
            m_tileSets.emplace_back(source);
        } else {
            m_auxTileSets.emplace_back(source);
        }
    }
    while (m_tileSets.size() > MAX_TILE_SETS) { m_tileSets.pop_back(); }
}

std::shared_ptr<TileSource> TileManager::getTileSource(int32_t _sourceId) {
    auto it = std::find_if(m_tileSets.begin(), m_tileSets.end(),
         [&](auto& ts) { return ts.source->id() == _sourceId; });

    if (it != m_tileSets.end()) {
        return it->source;
    }
    return nullptr;
}

void TileManager::addClientTileSource(std::shared_ptr<TileSource> _tileSource) {
    auto it = std::find_if(m_tileSets.begin(), m_tileSets.end(),
        [&](auto& ts) { return ts.source->id() == _tileSource->id(); });

    if (it == m_tileSets.end()) {
        m_tileSets.emplace_back(_tileSource);
    }
}

bool TileManager::removeClientTileSource(int32_t _sourceId) {

    auto it = std::find_if(m_tileSets.begin(), m_tileSets.end(),
                           [&](auto& ts) { return ts.source->id() == _sourceId; });

    if (it != m_tileSets.end()) {
        m_tileSets.erase(it);
        return true;
    }
    return false;
}

void TileManager::clearTileSets(bool clearSourceCaches) {

    for (auto& tileSet : m_tileSets) {
        tileSet.tiles.clear();  //tileSet.cancelTasks();

        if (clearSourceCaches) {
            tileSet.source->clearData();
        }
    }

    m_tileCache->clear();
}

void TileManager::clearTileSet(int32_t _sourceId) {
    for (auto& tileSet : m_tileSets) {
        if (tileSet.source->id() != _sourceId) { continue; }

        tileSet.tiles.clear();  //tileSet.cancelTasks();
    }

    m_tileCache->clear();
    m_tileSetChanged = true;
}

bool TileManager::updateTileSets(const View& _view) {

    m_tiles.clear();
    m_tilesInProgress = 0;
    m_tileSetChanged = false;
    // Perf: drop last pass's elevSet lookup/rendered/ready cache (see m_upgradeRasterCache's doc
    // comment in tileManager.h) - it is only valid for the duration of a single updateTileSets()
    // pass, since it's built from tile-set contents that this very pass will go on to mutate.
    m_upgradeRasterCache.clear();

    // UX diagnosis (2026-07-11): see m_areaDiagRefTile's doc comment in tileManager.h. Pick a
    // fresh reference tile from the camera's current look-at target the moment each capture
    // starts (edge-detected via the capturing-state transition), then log that SAME tile's true
    // continuous on-screen area every frame for the rest of the capture, regardless of whether
    // it ends up actually visible/subdivided this frame - we want the raw geometry, not the
    // discretized LOD decision.
    bool capturingNow = Profiler::isCapturing();
    if (capturingNow && !m_areaDiagCapturing) {
        auto pos = _view.getPosition();
        m_areaDiagRefTile = MapProjection::projectedMetersTile(glm::dvec2(pos.x, pos.y), 16);
    }
    m_areaDiagCapturing = capturingNow;
    if (capturingNow && m_areaDiagRefTile != NOT_A_TILE) {
        Profiler::counter("areaDiagRefTileArea", double(_view.getTileScreenArea(m_areaDiagRefTile)));
    }

    if (!getDebugFlag(DebugFlags::freeze_tiles)) {

        float maxEdge = 2 * _view.pixelScale() * float(MapProjection::tileSize());
        float maxArea = maxEdge*maxEdge;

        using TileSetMask = std::bitset<MAX_TILE_SETS>;
        // enable recursion by passing lambda ref to itself; auto type creates a generic (i.e. templated) lambda
        auto getVisibleTiles = [&](auto&& self, TileID tileId, TileSetMask active){
            // if pitch == 0, this will only return 0 or FLT_MAX
            float area = _view.getTileScreenArea(tileId);
            if (area <= 0) { return; }  // offscreen

            TileSetMask nextActive = active;
            for (size_t ii = 0; ii < m_tileSets.size(); ++ii) {
                if (!active[ii]) { continue; }
                auto& tileSet = m_tileSets[ii];
                int zoomBias = tileSet.source->zoomBias();
                // Per-source resolution-retention bias (see TileSource::lodAreaBias's doc
                // comment): >1.0 shrinks this source's own effective area threshold, so the
                // quadtree keeps subdividing to a finer zoom over a wider range of on-screen
                // area than the shared `maxArea` alone would - i.e. it keeps the finest already-
                // cached tile data on screen longer while zooming out (and, symmetrically,
                // reaches finer resolution slightly sooner while zooming in). 1.0 (default)
                // reproduces today's exact behavior. Applied here (not baked into the shared
                // maxArea above) because it must vary per source - e.g. landcover (`osm`) and
                // hillshading/texture-shading (`elevation`) are tuned independently via GUI
                // sliders (MapsApp).
                float lodAreaBias = tileSet.source->lodAreaBias();
                float effMaxArea = maxArea / (lodAreaBias * lodAreaBias);
                // substantial redesign needed for something like this to work:
                //for (auto& rs : tileSet.source->rasterSources()) { maxZoom = std::max(maxZoom, rs->maxZoom()); }
                int maxZoom = std::min(tileSet.source->maxZoom(), _view.getIntegerZoom() - zoomBias);
                bool stoppedByMaxZoom = tileId.z >= maxZoom;
                if (stoppedByMaxZoom || area < effMaxArea*std::exp2(2*float(zoomBias))) {
                    // Perf/UX diagnosis (2026-07-11): lodAreaBias (the "resolution retention"
                    // sliders) only ever loosens the AREA half of this OR - it can never push a
                    // tile past `maxZoom`, which is a single value for the WHOLE view
                    // (_view.getIntegerZoom(), not per-tile/per-distance). If distant tiles are
                    // stopping here because of maxZoom rather than the area threshold, no amount
                    // of lodAreaBias will ever change their resolution - these counters make that
                    // distinguishable instead of guessed at.
                    if (Profiler::isCapturing()) {
                        const char* srcName = tileSet.source->name().c_str();
                        if (stoppedByMaxZoom) {
                            Profiler::counterAdd(Profiler::intern(std::string("stopByMaxZoom:") + srcName), 1);
                            Profiler::counter(Profiler::intern(std::string("stopZoomByMaxZoom:") + srcName), double(tileId.z));
                        } else {
                            Profiler::counterAdd(Profiler::intern(std::string("stopByArea:") + srcName), 1);
                            Profiler::counter(Profiler::intern(std::string("stopZoomByArea:") + srcName), double(tileId.z));
                        }
                    }
                    TileID visId = tileId;
                    // Ensure that s = z + bias (larger s OK if overzoomed) so that proxy tiles can be found
                    // - otherwise, we get frames where tiles disappear due to no proxy for new tile
                    if (visId.z < tileSet.source->maxZoom()) {
                        visId.s = visId.z + zoomBias;
                    } else {
                        // Perf/UX (2026-07-11): overzoomStepExponent (default 2.0, i.e. today's
                        // exact behavior) controls how many doublings of on-screen area must pass
                        // before `s` - and so e.g. the elevation mosaic's composite resolution -
                        // steps up by one whole level; raising it spaces those same-size jumps
                        // further apart in distance/zoom, see ZoomOptions::overzoomStepExponent.
                        float stepExponent = tileSet.source->overzoomStepExponent();
                        int s = tileId.z + std::max(0, int(std::ceil(std::log2(area/effMaxArea)/stepExponent)));
                        visId.s = std::max(std::min(s, _view.getIntegerZoom()), visId.z + zoomBias);
                    }
                    tileSet.visibleTiles.insert(visId);
                    nextActive.reset(ii);
                }
            }
            // subdivide if any active tile sets remaining
            if (nextActive.any()) {
                for (int i = 0; i < 4; i++) {
                    self(self, tileId.getChild(i, 100), nextActive);
                }
            }
        };

        for (auto& tileSet : m_tileSets) {
            tileSet.visibleTiles.clear();
        }

        TileSetMask allActive = (1 << m_tileSets.size()) - 1;
        getVisibleTiles(getVisibleTiles, TileID(0,0,0), allActive);
    }

    // R4 (tile-pipeline-perf-plan.md): grant every raster source a fresh main-thread stitch
    // budget for this update cycle BEFORE any tileSet is processed below - a draped vector
    // tileSet (e.g. `osm`) can trigger stitches on the elevation RasterSource just as much as
    // the elevation tileSet's own completions do, and both must see this cycle's reset rather
    // than whichever tileSet happens to iterate first getting a stale/zero budget.
    const int kStitchBudgetPerUpdate = 2;
    // Perf (2026-07 profiling): same reasoning as kStitchBudgetPerUpdate above, but for
    // flushDirtyMosaics()'s GPU re-upload (see RasterSource::m_uploadBudget) - reset alongside
    // the stitch budget, before any tileSet (and so any flushDirtyMosaics() call) runs this cycle.
    const int kUploadBudgetPerUpdate = 12;
    for (auto& tileSet : m_tileSets) {
        if (tileSet.source->isRaster()) {
            auto* rasterSource = static_cast<RasterSource*>(tileSet.source.get());
            rasterSource->resetStitchBudget(kStitchBudgetPerUpdate);
            rasterSource->resetUploadBudget(kUploadBudgetPerUpdate);
        }
    }

    for (auto& tileSet : m_tileSets) {
        // check if tile set is active for zoom (zoom might be below min_zoom)
        if (tileSet.source->isActiveForZoom(_view.getZoom()) && tileSet.source->isVisible()) {
            updateTileSet(tileSet, _view.state());
        }
        // R2 (tile-pipeline-perf-plan.md): flush any patch-driven re-uploads deferred by
        // RasterSource::patchNeighborMosaics() once per update cycle rather than immediately,
        // regardless of this source's own active/visible state above (a mosaic can still be
        // live and rendered via other tiles' attachments even if this pass skipped it).
        if (tileSet.source->isRaster()) {
            static_cast<RasterSource*>(tileSet.source.get())->flushDirtyMosaics();
        }
    }

    for (auto& tileSet : m_auxTileSets) {
        auto it = tileSet.tiles.begin();
        while (it != tileSet.tiles.end()) {
            if (!it->second.task || it->second.task->isReady() || it->second.task->isCanceled()) {
                it->second.task.reset();  // avoid call to clearTask() in ~TileEntry()
                it = tileSet.tiles.erase(it);
            } else {
                ++it;
            }
        }
    }

    loadTiles();

    // no longer need to sort or dedup m_tiles since it is populated in order from TileSet.tiles (std::map)

    // grow tile cache if needed - goal is to cache about 1 screen worth of tiles
    if (m_tileCache->cacheSizeLimit() < m_maxCacheLimit) {
        size_t memused = 0;
        for (const auto& tile : m_tiles) { memused += tile->getMemoryUsage(); }
        if (memused > 1.5*m_tileCache->cacheSizeLimit()) {
            m_tileCache->limitCacheSize(std::min(m_maxCacheLimit, memused));
        }
    }

    if (Profiler::isCapturing()) {
        size_t proxyTiles = 0;
        for (const auto& tile : m_tiles) { if (tile->isProxy()) { ++proxyTiles; } }
        Profiler::counter("visibleTiles", double(m_tiles.size()));
        Profiler::counter("proxyTiles", double(proxyTiles));
        Profiler::counter("pendingTiles", double(m_tilesInProgress));
        Profiler::counter("tileCacheEntries", double(m_tileCache->getNumEntries()));
        Profiler::counter("tileCacheKB", double(m_tileCache->getMemoryUsage()) / 1024.0);
    }

    return m_tileSetChanged;
}

void TileManager::updateTileSet(TileSet& _tileSet, const ViewState& _view) {

    PROFILE_SCOPE(Profiler::intern("updateTileSet:" + _tileSet.source->name()));
    //FrameInfo::scope _trace("updateTileSet " + _tileSet.source->name());

    if (_tileSet.sourceGeneration != _tileSet.source->generation()) {
        _tileSet.sourceGeneration = _tileSet.source->generation();
    }

    auto& tiles = _tileSet.tiles;
    const auto& visibleTiles = _tileSet.visibleTiles;

    // Phase 1 (docs/texture-shading-plan.md): for the elevation source only, make sure
    // each visible tile's 8 same-zoom neighbors are also being fetched in the
    // background. Must run before the diff loop below so freshly-added prefetch
    // entries are picked up by the same-frame proxy/cleanup passes.
    enqueueNeighborPrefetch(_tileSet, _view);

    // Pending proxy tiles too far from current zoom level will be canceled
    int maxProxyZ = 0, minProxyZ = 0, maxVisS = 0;
    if (!visibleTiles.empty()) {
        int zmax = visibleTiles.begin()->z, zmin = visibleTiles.rbegin()->z;
        maxProxyZ = zmin != zmax ? zmax + 2 : int(glm::round(_view.zoom + 1.f));
        minProxyZ = zmin != zmax ? zmin - 3 : int(glm::round(_view.zoom - 2.f));
        maxVisS = visibleTiles.begin()->s;
    }

    // Loop over visibleTiles and add any needed tiles to tileSet
    auto curTilesIt = tiles.begin();
    auto visTilesIt = visibleTiles.begin();

    auto generation = _tileSet.source->generation();

    while (visTilesIt != visibleTiles.end() || curTilesIt != tiles.end()) {

        auto& visTileId = visTilesIt == visibleTiles.end() ? NOT_A_TILE : *visTilesIt;
        auto& curTileId = curTilesIt == tiles.end() ? NOT_A_TILE : curTilesIt->first;

        if (visTileId == curTileId) {
            // tiles in both sets match
            assert(visTilesIt != visibleTiles.end() &&
                   curTilesIt != tiles.end());

            auto& entry = curTilesIt->second;
            entry.setVisible(true);

            // The task's proxy flag was set true on previous frames while this entry was
            // prefetch-only (or a proxy) and is only refreshed further down; clear it BEFORE
            // completing the task so RasterTileTask::addRaster() doesn't skip the elevation
            // mosaic for a tile that is in fact becoming visible right now.
            if (entry.task) { entry.task->setProxyState(false); }

            if (entry.completeTileTask()) {
                m_tileSetChanged = true;
            }

            // A tile completed on an earlier frame while prefetch-only/proxy still holds the
            // plain (mip-less) elevation texture; swap in the shared mosaic. Also covers a
            // draped vector tile (e.g. `osm`) whose attached elevation raster was baked in at
            // build time before finer descendant tiles were cached -- see definition.
            // visTileId carries the LIVE styling zoom (see the _liveId doc in tileManager.h).
            if (entry.tile) { upgradeAttachedRasters(_tileSet, *entry.tile, visTileId); }

            if (entry.needsLoading()) {
                // Not yet available - enqueue for loading
                if (!entry.task) {
                    entry.task = _tileSet.source->createTask(visTileId);
                }
                enqueueTask(_tileSet, visTileId, _view);
            }

            // NB: Special handling to update tiles from ClientDataSource.
            // Can be removed once ClientDataSource is immutable
            if (entry.tile) {
                auto sourceGeneration = entry.tile->sourceGeneration();
                if ((sourceGeneration < generation) && !entry.isInProgress()) {
                    // Tile needs update - enqueue for loading
                    entry.task = _tileSet.source->createTask(visTileId);
                    enqueueTask(_tileSet, visTileId, _view);
                }
            } else if (entry.isCanceled()) {
                auto sourceGeneration = entry.task->sourceGeneration();
                if (sourceGeneration < generation) {
                    // Tile needs update - enqueue for loading
                    entry.task = _tileSet.source->createTask(visTileId);
                    enqueueTask(_tileSet, visTileId, _view);
                }
            }

            if (entry.isInProgress()) {
                m_tilesInProgress++;
            }

            ++curTilesIt;
            ++visTilesIt;

        } else if (curTileId > visTileId) {
            // tileSet is missing an element present in visibleTiles
            // NB: if (curTileId == NOT_A_TILE) it is always > visTileId
            //     and if curTileId > visTileId, then visTileId cannot be
            //     NOT_A_TILE. (for the current implementation of > operator)
            assert(visTilesIt != visibleTiles.end());

            if (!addTile(_tileSet, visTileId)) {
                // Not in cache - enqueue for loading
                enqueueTask(_tileSet, visTileId, _view);
                m_tilesInProgress++;
            }

            ++visTilesIt;

        } else {
            // tileSet has a tile not present in visibleTiles
            assert(curTilesIt != tiles.end());

            auto& entry = curTilesIt->second;
            if (entry.completeTileTask()) {
                m_tileSetChanged = true;
            }
            entry.setVisible(false);
            ++curTilesIt;
        }
    }

    int minCurS = tiles.rbegin()->first.s; //, maxCurS = tiles.begin()->first.s;
    auto zoomBias = _tileSet.source->zoomBias();
    // find proxy tiles
    for (curTilesIt = tiles.begin(); curTilesIt != tiles.end(); ++curTilesIt) {
        auto& tileId = curTilesIt->first;
        auto& entry = curTilesIt->second;
        if (!entry.isVisible()) {
            // handle child proxy (i.e. look for visible parents w/o tile)
            for (auto id = tileId.getParent(zoomBias); id.s >= minCurS; id = id.getParent(zoomBias)) {
                auto it = tiles.find(id);
                if (it == tiles.end()) { continue; }
                // visible tile w/ tile (so no proxy needed) or a better proxy found before visible tile?
                if (it->second.tile) { break; }
                // found visible tile (w/o tile) to proxy for?
                if (it->second.isVisible()) { entry.m_proxyCounter++; break; }
            }
        } else if (!entry.tile) {
            // visible tile w/o tile - look for parents which can be proxy
            for (auto id = tileId.getParent(zoomBias); id.s >= minCurS; id = id.getParent(zoomBias)) {
                auto it = tiles.find(id);
                if (it != tiles.end()) {
                    it->second.m_proxyCounter++;
                    if (it->second.tile) { break; }
                }
            }
        }
    }

    // add ready tiles to m_tiles and remove tiles not in visibleTiles and not being used as proxy
    for (curTilesIt = tiles.begin(); curTilesIt != tiles.end();) {
        auto& tileId = curTilesIt->first;
        auto& entry = curTilesIt->second;

#ifdef TANGRAM_DEBUG_TILESETS //0 && LOG_LEVEL >= 3
        size_t rasterLoading = 0, rasterDone = 0;
        if (entry.task) {
            for (auto &raster : entry.task->subTasks()) {
                if (raster->isReady()) { rasterDone++; }
                else { rasterLoading++; }
            }
        }
        LOGD("%s > %s - ready:%d proxy:%d/%d loading:%d rDone:%d rLoading:%d canceled:%d",
             _tileSet.source->name().c_str(), tileId.toString().c_str(), bool(entry.tile),
             entry.getProxyCounter(), entry.m_proxies, entry.task && !entry.task->isReady(),
             rasterDone, rasterLoading, entry.task && entry.task->isCanceled());
#endif

        bool canLoad = entry.isInProgress() && (tileId.z < maxProxyZ && tileId.z > minProxyZ);
        // Phase 1 (docs/texture-shading-plan.md): keep a neighbor-prefetch-only entry
        // alive while its fetch is still in flight, without treating it as visible or
        // as a real proxy. Note entry.tile and entry.isInProgress() are mutually
        // exclusive (completeTileTask() resets task exactly when tile is assigned), so
        // this can never cause a prefetch-only tile to reach the `if (entry.tile)`
        // rendering branch below - once its tile is ready this condition is false and
        // it falls through to the cache/erase branch instead.
        bool keepPrefetching = entry.m_prefetchOnly && entry.isInProgress();
        if (entry.isVisible() || keepPrefetching || (entry.m_proxyCounter > 0 && (entry.tile || canLoad))) {
            if (entry.tile) {
                entry.tile->setProxyDepth(entry.m_proxyCounter > 0 ? std::max(maxVisS - tileId.s, 1) : 0);
                m_tiles.push_back(entry.tile);
                // check to see if a replacement is now available for missing raster
                if (entry.numMissingRasters != 0) {
                    // to persist numMissingRasters for cached tiles, we'd need to store in Tile itself, so
                    //  instead we use a initial value of -1 to force check for new TileEntry, whether Tile
                    //  is new or from cache
                    auto& srcs = _tileSet.source->rasterSources();
                    auto& rasters = entry.tile->rasters();
                    size_t offset = rasters.size() - srcs.size();
                    entry.numMissingRasters = 0;
                    for (size_t ii = 0; ii < srcs.size(); ++ii) {
                        Raster& raster = rasters[ii+offset];
                        // The zoom this raster SHOULD be attached at -- same arithmetic
                        // RasterSource::addRasterTask (rasterSource.cpp) uses for the subtask ID.
                        int32_t zoomDiff = srcs[ii]->zoomBias() - _tileSet.source->zoomBias();
                        TileID intended = (zoomDiff > 0)
                                ? tileId.zoomBiasAdjusted(zoomDiff).withMaxSourceZoom(srcs[ii]->maxZoom())
                                : tileId.withMaxSourceZoom(srcs[ii]->maxZoom());
                        bool missing = raster.texture == srcs[ii]->emptyTexture();
                        // A raster left at a coarser PARENT by an earlier pass of this loop must
                        // keep being treated as missing so it HEALS once the intended texture
                        // lands in the source's shared cache (typically fetched meanwhile by the
                        // raster source's own tileset, e.g. terrain-ground, or by a neighbor).
                        // Before, pinning a parent zeroed numMissingRasters and this block never
                        // ran again: the drape then permanently sampled coarser elevation than
                        // the terrain rendered beneath it (upgradeAttachedRasters can only
                        // rebuild composites AT the pinned parent's own footprint, capped at
                        // parent z + kOverzoomCompositeMaxLevels, which for deep fallbacks never
                        // reaches the terrain's real zoom) -- observed as persistent unshaded
                        // "white patch" triangles born while panning that never went away.
                        if (!missing && raster.tileID.z >= intended.z) { continue; }
                        if (auto exact = srcs[ii]->getTexture(TileID(intended.x, intended.y, intended.z))) {
                            raster.tileID = TileID(intended.x, intended.y, intended.z, tileId.s);
                            raster.texture = exact;
                            LOGD("Healed '%s' subtask raster '%s' %s with exact texture", _tileSet.source->name().c_str(),
                                 srcs[ii]->name().c_str(), tileId.toString().c_str());
                            continue;  // healed: intended-zoom texture attached
                        }
                        // still awaiting the intended texture: keep this entry revisited
                        ++entry.numMissingRasters;
                        if (!missing) { continue; }  // already showing a parent; keep it until healed
                        TileID id(intended.x, intended.y, intended.z);
                        do {
                            id = id.getParent();
                            auto proxy = srcs[ii]->getTexture(id);
                            if (proxy) {
                                raster.tileID = TileID(id.x, id.y, id.z, tileId.s);
                                raster.texture = proxy;
                                LOGD("Found proxy %s for missing '%s' subtask raster '%s' %s", id.toString().c_str(),
                                     _tileSet.source->name().c_str(), srcs[ii]->name().c_str(), tileId.toString().c_str());
                                break;  // numMissingRasters stays counted: still awaiting exact
                            }
                        } while (id.z > 13 || (id.z > 0 && id.z + 2 >= tileId.z));
                    }
                }
            } else if (entry.isInProgress()) {
                auto& task = entry.task;
                // Update tile distance to map center for load priority.
                auto tileCenter = MapProjection::tileCenter(tileId);
                double scaleDiv = exp2(tileId.z - _view.zoom);
                if (scaleDiv < 1) { scaleDiv = 0.1/scaleDiv; } // prefer parent tiles
                task->setPriority(glm::length2(tileCenter - _view.center) * scaleDiv);
                // keep prefetch-only fetches deprioritized in the TileWorker queue
                // (same mechanism used for real proxy tiles) every frame, since this
                // would otherwise reset it back to false each update
                task->setProxyState(entry.m_proxyCounter > 0 || entry.m_prefetchOnly);
            }
            entry.m_proxyCounter = 0;  // reset for next update
            ++curTilesIt;
        } else {
            // Remove entry and move tile (if present) to cache
            if (entry.tile) {
                if (entry.m_prefetchOnly) {
                    LOGD("Neighbor prefetch: %s finished loading, texture now cached (source '%s')",
                         tileId.toString().c_str(), _tileSet.source->name().c_str());
                }
                m_tileCache->put(_tileSet.source->id(), entry.tile);
            } else if (entry.m_prefetchOnly) {
                // R5 (tile-pipeline-perf-plan.md): reached here without a tile means the
                // prefetch-only fetch was canceled/errored (isInProgress() went false while
                // entry.tile never got set - the mutually-exclusive relationship noted above)
                // rather than completed successfully. Record it so enqueueNeighborPrefetch()
                // can back off instead of re-enqueueing this neighbor every frame forever.
                auto& fail = _tileSet.prefetchFailures[tileId];
                fail.first++;
                fail.second = nowSeconds();
            }
            // Remove tile from set - this will call clearTask() and thus cancelLoadingTile() as appropriate
            curTilesIt = tiles.erase(curTilesIt);
        }
    }
}

// Phase 1 (docs/texture-shading-plan.md "Neighbor tile prefetching"): for a RasterSource
// with m_prefetchNeighbors set (only the elevation source used by ElevationManager, see
// scene.cpp), proactively enqueue background fetches for each visible tile's 8 same-zoom
// neighbor tiles (N/S/E/W/NE/NW/SE/SW), wrapping in x at the antimeridian (2^z) and
// skipping out-of-range y at the poles - TileID has no existing helper for this, so it is
// computed here. This is a no-op for every other source.
//
// Neighbor entries are inserted into the *same* TileSet::tiles map used for real tiles
// (reusing the existing task-creation path, RasterSource::createTask(), so caching/
// dedup/subtask logic all behave exactly as for a real tile) but are never added to
// TileSet::visibleTiles, are marked TileEntry::m_prefetchOnly, and are deprioritized in
// the TileWorker queue via TileTask::setProxyState(true) (the same mechanism used for
// real proxy tiles) - see the retention/priority logic in updateTileSet() above for how
// this keeps them from ever being rendered or affecting label/mesh building for real
// tiles, while still letting their task run to completion and populate
// RasterSource::m_textures. Once loaded they are cached via TileManager's normal
// TileCache so the texture stays resident like any other recently-used tile.
//
// R5 (tile-pipeline-perf-plan.md): a neighbor fetch that keeps failing (e.g. 404 at a data
// edge, or offline) is backed off via TileSet::prefetchFailures instead of being retried every
// single frame forever, and the ring is only prefetched once the view has settled (it only
// matters once the center tiles are up anyway, so a fast pan doesn't burn fetch slots on ring
// tiles about to be left behind) - see kPrefetchFailureBackoff{Count,Seconds} and
// kPrefetchSettleDelaySeconds above.
void TileManager::enqueueNeighborPrefetch(TileSet& _tileSet, const ViewState& _view) {

    auto* rasterSource = dynamic_cast<RasterSource*>(_tileSet.source.get());
    if (!rasterSource || !rasterSource->m_prefetchNeighbors) { return; }

    double now = nowSeconds();
    bool viewMoved = _view.center.x != _tileSet.lastViewCenterX ||
                      _view.center.y != _tileSet.lastViewCenterY ||
                      _view.zoom != _tileSet.lastViewZoom;
    if (viewMoved) {
        _tileSet.lastViewCenterX = _view.center.x;
        _tileSet.lastViewCenterY = _view.center.y;
        _tileSet.lastViewZoom = _view.zoom;
        _tileSet.viewSettledSince = now;
        return;  // still moving - ring prefetch can wait until the view settles
    }
    if (now - _tileSet.viewSettledSince < kPrefetchSettleDelaySeconds) { return; }

    static const int neighborDx[8] = { -1,  0,  1, -1, 1, -1, 0, 1 };
    static const int neighborDy[8] = { -1, -1, -1,  0, 0,  1, 1, 1 };

    auto& tiles = _tileSet.tiles;
    auto& visibleTiles = _tileSet.visibleTiles;

    for (const TileID& visId : visibleTiles) {

        int maxCoord = 1 << visId.z;

        for (int ii = 0; ii < 8; ++ii) {
            int ny = visId.y + neighborDy[ii];
            if (ny < 0 || ny >= maxCoord) { continue; }  // no tile past the poles - don't wrap y
            int nx = ((visId.x + neighborDx[ii]) % maxCoord + maxCoord) % maxCoord;  // wrap x at antimeridian

            TileID neighborId(nx, ny, visId.z, visId.s);

            if (visibleTiles.count(neighborId)) { continue; }  // already being fetched as a real visible tile
            if (tiles.count(neighborId)) { continue; }  // already tracked (in-flight prefetch or stale entry)
            if (rasterSource->getTexture(neighborId)) {
                // Already cached -- nothing to fetch. NOT logged (was LOGD before): a TileEntry
                // can't be planted here to remember this and skip the recheck next time, because
                // any entry that isn't visible/proxying/in-progress gets erased by this same
                // update's tile-cleanup pass below before the frame even ends (confirmed by
                // reading that pass's keep condition) -- so this exact check necessarily re-runs
                // every call to this function, which is EVERY FRAME (updateTileSets() is called
                // unconditionally each frame, confirmed live via this line's log spam persisting
                // long after the camera stopped moving -- see is_animating in map.cpp, driven by
                // label/marker animation state, not just camera easing/fling). The recheck itself
                // is cheap (one weak-ref texture-cache lookup) and therefore harmless to repeat;
                // the LOGD call was the actual cost (unbounded log volume), so drop it rather than
                // engineer memoization the cleanup pass would just undo.
                continue;
            }
            auto failIt = _tileSet.prefetchFailures.find(neighborId);
            if (failIt != _tileSet.prefetchFailures.end()) {
                if (failIt->second.first >= kPrefetchFailureBackoffCount &&
                    now - failIt->second.second < kPrefetchFailureBackoffSeconds) {
                    continue;  // recently failed repeatedly - back off rather than retry every frame
                }
                _tileSet.prefetchFailures.erase(failIt);  // backoff window passed - allow a fresh try
            }

            std::shared_ptr<Tile> noTile;
            auto entryit = tiles.emplace(neighborId, noTile);
            TileEntry& entry = entryit.first->second;
            entry.task = _tileSet.source->createTask(neighborId);
            entry.m_prefetchOnly = true;

            enqueueTask(_tileSet, neighborId, _view);

            LOGD("Neighbor prefetch: enqueued %s (neighbor of visible %s, source '%s')",
                 neighborId.toString().c_str(), visId.toString().c_str(), rasterSource->name().c_str());
        }
    }
}

void TileManager::enqueueTask(TileSet& _tileSet, const TileID& _tileID,
                              const ViewState& _view) {

    // Keep the items sorted by distance
    auto tileCenter = MapProjection::tileCenter(_tileID);
    double distance = glm::length2(tileCenter - _view.center);

    auto it = std::upper_bound(m_loadTasks.begin(), m_loadTasks.end(), distance,
        [](const double& d, const TileLoadTask& other){ return d < other.dist; });

    m_loadTasks.insert(it, {distance, &_tileSet, _tileID});
}

TileManager::TileSet* TileManager::findTileSet(int64_t sourceId) {
    for (auto& ts : m_tileSets) {
        if (ts.source->id() == sourceId) { return &ts; }
    }
    for (auto& ts : m_auxTileSets) {
        if (ts.source->id() == sourceId) { return &ts; }
    }
    return nullptr;
}

void TileManager::loadTiles() {

    if (m_loadTasks.empty()) { return; }

    for (auto& loadTask : m_loadTasks) {
        TileSet* tileSet = loadTask.tileSet;
        auto tileIt = tileSet->tiles.find(loadTask.tileID);
        auto tileTask = tileIt->second.task;

        for (auto& subtask : tileTask->subTasks()) {
            // needsLoading() will be false if, e.g., texture was already cached by RasterSource
            if (!subtask->needsLoading()) { ++subtask->shareCount; continue; }
            TileSet* ts = findTileSet(subtask->sourceId());
            if (!ts) { continue; }  // should never happen

            auto it = ts->tiles.find(subtask->tileId());
            if (it != ts->tiles.end()) {
                if (it->second.task && !it->second.task->isReady() && !it->second.task->isCanceled()) {
                    subtask = it->second.task;
                }
            } else if (!ts->source->generateGeometry()) {
                // add to aux tile set - this will be the master task for any subsequent duplicates
                std::shared_ptr<Tile> dummy;
                auto res = ts->tiles.emplace(subtask->tileId(), dummy);
                if (res.second) {
                    res.first->second.task = subtask;
                }
            }
            // shareCount > 1 prevents tile cancelation (shareCount decremented by cancel and complete)
            ++subtask->shareCount;
        }
        ++tileTask->shareCount;

        tileTask->setScenePrana(m_scenePrana);
        tileSet->source->loadTileData(tileTask, m_dataCallback);

        LOGTO("Load Tile: %s %s", tileSet->source->name().c_str(), loadTask.tileID.toString().c_str());
    }

    m_loadTasks.clear();
}

// A tile built while it was prefetch-only (or a lower-zoom proxy) holds the plain per-tile
// elevation texture (RasterTileTask::addRaster() skips mosaic stitching for isProxy() tasks;
// the plain texture has no mip chain, so texture shading comes out exactly neutral for the
// tile, and the ELEVATION_MOSAIC UV remap mis-samples it). When such a tile is (re)used as a
// visible tile, upgrade its raster to the shared 3Wx3W mosaic. Pointer equality with the
// RasterSource texture cache identifies a plain texture (a raster holding the plain texture
// keeps exactly that cache entry alive); a tile already holding a mosaic fails the pointer
// test and is left unchanged, so this is cheap to call per update.
//
// Second, distinct case (docs/terrain-illumination-plan.md "attachment lifecycle" finding):
// a draped VECTOR tile (e.g. `osm` polygons, not a RasterSource's own tile) gets its elevation
// raster(s) attached exactly ONCE, as a side effect of RasterTileTask::addRaster() running as a
// subtask of the vector TileTask's own build -- whatever raster (plain / partial-or-complete
// overzoom composite) happens to be cache-available AT THAT INSTANT is baked into
// Tile::rasters() and never revisited afterward, even as the elevation source keeps fetching
// finer descendant tiles in the background. This is why the "white patches" correlate strongly
// with PANNING rather than static load time: a polygon tile that enters view (and gets built)
// races its own elevation prefetch, and whichever races ahead wins PERMANENTLY for that tile's
// lifetime -- a tile built early with a coarse/absent composite stays coarse even seconds later
// once every descendant it needed is sitting in cache. Fix: also revisit each attached raster
// (Tile::rasters()[i] <-> the owning TileSource's rasterSources()[i], same index order as
// TileSource::addRasterTasks/addRasterTask populate them) the same way, per raster source's own
// contract (texture-shading mosaic vs. opportunistic overzoom composite).
void TileManager::upgradeAttachedRasters(TileSet& _tileSet, Tile& _tile, const TileID& _liveId) {
    PROFILE_SCOPE("UpgradeAttachedRasters");
    if (_tileSet.source->isRaster()) {
        // Case 1 (pre-existing): _tileSet.source IS the elevation RasterSource, and this tile
        // is its own raster tile (terrain-ground/hillshade) -- single self-referencing entry.
        auto* rs = static_cast<RasterSource*>(_tileSet.source.get());
        if (!rs->m_buildElevationMosaic || _tile.rasters().empty()) { return; }
        auto& raster = _tile.rasters().front();
        // 3-arg TileID (s = z) to match the texture cache's normalized keys
        TileID texId(raster.tileID.x, raster.tileID.y, raster.tileID.z);
        if (raster.texture && raster.texture == rs->getTexture(texId)) {
            if (auto mosaic = rs->buildElevationMosaic(raster.tileID, raster.texture)) {
                raster.texture = mosaic;
            }
        }
        return;
    }

    // Case 2 (new): _tileSet.source is a vector (or other non-raster) source with one or more
    // raster attachments (global.elevation_sources etc.) -- e.g. `osm` polygons draped with
    // elevation. Zip Tile::rasters() with the owning source's rasterSources(): the LAST
    // srcs.size() entries of rasters() correspond index-for-index to rasterSources() (both
    // populated in the same iteration order by TileSource::addRasterTasks ->
    // RasterSource::addRasterTask, completed in that order by TileTask::complete()'s m_subTasks
    // loop; the offset skips any leading own-raster entries -- same convention as the
    // missing-raster proxy pass in updateTileSet() above).
    //
    // Skip tiles currently serving as proxies: they are on their way out (or their replacement
    // is on its way in), so re-stitching THEIR composites is wasted work that piles onto
    // exactly the frames that are already busiest (pan/zoom churn). They get upgraded the
    // moment they are visible again.
    if (_tile.isProxy()) { return; }
    auto& sources = _tileSet.source->rasterSources();
    auto& rasters = _tile.rasters();
    if (sources.empty() || rasters.size() < sources.size()) { return; }
    size_t offset = rasters.size() - sources.size();

    // Each vector tile's OWN styling zoom `s` is computed independently from ITS OWN on-screen
    // area (updateTileSets(), the `visId.s` formula) -- nothing constrains it to agree with its
    // neighbors', so two geographically-adjacent tiles routinely land on different `s` (e.g. one
    // just inside the z15 threshold, its neighbor just inside z16 -- observed live via the
    // diagnostic zoom-stripe overlay: green tiles directly bordering blue ones). Each tile's
    // composite/lattice is sized to its OWN target zoom only, so two neighbors with different `s`
    // have DIFFERENT NODE COUNTS along their shared edge (e.g. 128 vs 256) -- a classic
    // non-conforming-mesh "T-junction crack": the coarser side draws a straight line between its
    // sparser edge nodes exactly where the finer side follows real terrain detail, worst in
    // narrow high-curvature features (a cliff's vertical grooves) and, being purely a geometric
    // lattice-density mismatch rather than a data-freshness issue, NOT something that heals with
    // time (observed live: persists well after the tile system settles idle).
    // Fix: use the FINEST `s` among this tile and its 8 same-zoom neighbors (if visible) as the
    // effective target for EVERYTHING below, so a tile bordering a finer neighbor gets its own
    // lattice bumped to match (node-aligning the shared edge exactly). This is safe precisely
    // because of the existing per-cell fallback machinery: cells in the tile's OWN interior that
    // don't actually have finer real data available yet simply triangle-PL-upsample from
    // whatever coarser real ancestor is cached (see fillLatticeUpsampledCell/buildOverzoomElevat
    // ionMosaic), exactly reproducing that coarser surface there -- so "borrowing" resolution
    // near a finer neighbor never fabricates fake detail, it only ensures the BORDER itself lines
    // up. A single neighbor-max pass (not iterated to a fixed point) is sufficient given zt only
    // spans 3 levels here (z14-z16): it directly resolves the reported z15/z16 boundary case,
    // and any residual z14/z15 mismatch is far smaller in absolute terms (a 2x, not 4x, density
    // gap) and anyway subject to the same fallback safety net.
    int32_t effectiveS = _liveId.s;
    {
        static const int nbDx[8] = { -1,  0,  1, -1, 1, -1, 0, 1 };
        static const int nbDy[8] = { -1, -1, -1,  0, 0,  1, 1, 1 };
        int32_t maxCoord = 1 << _liveId.z;
        int32_t nbx[8], nby[8];
        int nbCount = 0;
        for (int ii = 0; ii < 8; ii++) {
            int32_t ny = _liveId.y + nbDy[ii];
            if (ny < 0 || ny >= maxCoord) { continue; }  // no tile past the poles
            nbx[nbCount] = ((_liveId.x + nbDx[ii]) % maxCoord + maxCoord) % maxCoord;  // wrap x
            nby[nbCount] = ny;
            nbCount++;
        }
        // TileID::operator< sorts primarily by s (descending) -- see tileID.h -- so a same-(x,y,z)
        // entry at a DIFFERENT s than _liveId.s can't be located via set::find/lower_bound (which
        // need an exact s to match operator==/ordering too); a linear scan is the simplest correct
        // way to find "this neighbor coordinate, whatever its s". visibleTiles is bounded by the
        // current viewport's tile count (tens, not thousands) and this runs once per visible
        // primary tile per update, not per frame independent of tile count.
        for (const TileID& t : _tileSet.visibleTiles) {
            if (t.z != _liveId.z) { continue; }
            for (int k = 0; k < nbCount; k++) {
                if (t.x == nbx[k] && t.y == nby[k]) { effectiveS = std::max(effectiveS, int32_t(t.s)); break; }
            }
        }
    }

    for (size_t i = 0; i < sources.size(); i++) {
        RasterSource* rs = sources[i];
        Raster& raster = rasters[i + offset];
        if (!rs || !raster.texture) { continue; }

        // Target everything below at the LIVE, NEIGHBOR-AWARE styling zoom (effectiveS above),
        // not the s frozen into raster.tileID when the attachment was first built: the composite
        // target zoom must track the CURRENT view (both finer AND coarser than at build time), or
        // a cached tile re-entering view keeps draping at its build-time zoom forever -- see the
        // doc comment on this function in tileManager.h. raster.tileID's x/y/z are kept as-is
        // (they name the attachment's actual footprint, including the parent-fallback case).
        TileID livePrimary(raster.tileID.x, raster.tileID.y, raster.tileID.z,
                           std::max<int32_t>(effectiveS, raster.tileID.z));

        // Not overzoomed enough for a composite to apply at all (the ordinary, non-draped-vector
        // case) -- nothing for this function to do. Cheap: skips straight past the common case
        // without ever touching m_textures.
        if (rs->overzoomTargetZoom(livePrimary) <= int(livePrimary.z)) { continue; }

        // 3-arg TileID (s = z) to match the texture cache's normalized keys. raster.tileID may
        // be a lower-zoom PARENT of the tile's own id (the missing-raster proxy pass in
        // updateTileSet() substitutes one when the exact fetch failed) -- everything below
        // handles that transparently, since all lookups key off raster.tileID itself.
        TileID texId(raster.tileID.x, raster.tileID.y, raster.tileID.z);
        auto plainTex = rs->getTexture(texId);

        ElevationMosaicInfo* info = raster.texture->userData
                ? static_cast<ElevationMosaicInfo*>(raster.texture->userData.get()) : nullptr;
        bool holdsPlain = plainTex && raster.texture == plainTex;
        // Textures that are neither the plain texture nor carry mosaic/composite info (the
        // empty texture, parent-proxy fill-ins whose own cache entry expired) are left alone.
        if (!holdsPlain && !info) { continue; }

        // The plain texture is the required base for (re)building composites (stitch dimension
        // reference + ultimate per-cell fallback); reach it through the attached info's
        // plainTexture when the weak cache entry has already expired.
        std::shared_ptr<Texture> plainBase = plainTex;
        if (!plainBase && info) { plainBase = info->plainTexture; }
        if (!plainBase) { continue; }

        // Per-cell displayed-terrain zoom caps (see buildOverzoomElevationMosaic's doc comment
        // in rasterSource.h): the composite must reproduce the surface the elevation tileset
        // is RENDERING under this tile right now, not the finest data that happens to be
        // cached -- panning fills the cache with fine tiles while the terrain's screen-area
        // LOD keeps displaying coarser ones over the far parts of a footprint, and a
        // "finest cached" composite then disagrees with the rendered terrain by the full
        // inter-zoom DEM delta, permanently (2026-07 live diagnosis: the persistent unshaded
        // "white patch" land-cover mode occurred exclusively on composite-attached tiles).
        // Cap each composite cell at the zoom of the traversal-target tile covering it; cells
        // not covered by any target tile (offscreen parts of the footprint) stay uncapped.
        int zt = rs->overzoomTargetZoom(livePrimary);
        int levels = zt - int(livePrimary.z);
        int N = 1 << levels;
        std::vector<uint8_t> cellCaps;

        // Perf (2026-07 profiling): the elevSet lookup and the rendered/ready sets built from it
        // below are identical for every raster attachment on every tile processed against this
        // `rs` within one updateTileSets() pass (elevSet's own visibleTiles/tiles are constant
        // for the rest of the pass once it's reached this call - see m_upgradeRasterCache's doc
        // comment in tileManager.h), so compute them at most once per pass and reuse afterward.
        auto cacheIt = m_upgradeRasterCache.find(rs);
        if (cacheIt == m_upgradeRasterCache.end()) {
            TileSet* elevSet = nullptr;
            for (auto& ts : m_tileSets) {
                if (ts.source.get() == static_cast<TileSource*>(rs)) { elevSet = &ts; break; }
            }
            if (!elevSet) {
                for (auto& ts : m_auxTileSets) {
                    if (ts.source.get() == static_cast<TileSource*>(rs)) { elevSet = &ts; break; }
                }
            }
            if (!elevSet) {
                // Caps CANNOT be computed -- the composite would silently fall back to
                // "finest cached" and disagree with the rendered terrain (the persistent
                // white-patch mechanism). This is not expected to ever happen (the raster source
                // is shared by name with its own displayed tileset); scream once if it does.
                static bool warned = false;
                if (!warned) {
                    warned = true;
                    LOGW("upgradeAttachedRasters: no tileset found for raster source '%s' -- "
                         "displayed-zoom caps disabled, drape may mismatch terrain!", rs->name().c_str());
                }
            }
            std::set<TileID> rendered;  // normalized traversal-target ids
            std::set<TileID> ready;  // normalized ids of entries with a built (ready) Tile
            if (elevSet) {
                // Cap against the tileset's CURRENT TRAVERSAL RESULT (visibleTiles) -- the
                // steady-state display target -- NOT the per-frame rendered set (visible entries
                // plus whatever proxies happen to be up this instant): folding ALL live proxies
                // in caused per-frame re-stitch churn from proxy-*selection* noise ("everything
                // is suddenly slow" regression). visibleTiles changes only when the view
                // actually changes.
                //
                // BUT the target's own data readiness still needs checking (2026-07 live
                // diagnosis: patches on tiles whose diagnostic stripe color -- the drape's own
                // lattice zoom -- never changed, yet appeared/disappeared while panning back and
                // forth over already-visited terrain): a target tile whose GPU texture was
                // evicted under memory pressure and is being re-fetched from the local mbtiles
                // cache is NOT actually what's on screen while that reload is in flight -- the
                // renderer falls back to drawing whatever coarser ANCESTOR is still resident, the
                // same fallback search updateTileSet() itself performs above. Capping against the
                // not-yet-ready target instead of that actually-displayed ancestor reproduces the
                // exact mismatch this whole mechanism exists to prevent. Unlike whole-scale live
                // proxy selection, checking one specific tile's readiness is a monotonic fact
                // (changes only on load/evict events, not every frame), so this doesn't
                // reintroduce the churn regression.
                // BOTH sets below are normalized (x,y,z) triples, deliberately dropping `s`: TileSet::
                // tiles (and visibleTiles) are keyed by the RAW traversal TileID, whose `s` climbs
                // above `z` exactly when THAT tile itself is displayed overzoomed (same formula as
                // everywhere else) -- at extreme view zoom this now applies to the ELEVATION
                // tileset's OWN z16 tiles too (previously only the draped VECTOR tile was ever
                // overzoomed enough for this to matter). A readiness lookup keyed with a
                // freshly-constructed 3-arg TileID(x,y,z) (s forced = z) therefore silently misses
                // any entry actually stored under s > z -- confirmed live: mottled, contour-scale,
                // contiguous salmon patches at close/grazing zoom on already-at-cap (z16) tiles,
                // NOT healed by settling, because the lookup was deterministically wrong, not stale.
                // Building explicit normalized sets up front (one pass over visibleTiles/tiles) lets
                // every subsequent lookup below compare like-for-like instead of needing to guess a
                // real `s` for each candidate (x,y,z).
                for (const TileID& vid : elevSet->visibleTiles) {
                    rendered.insert(TileID(vid.x, vid.y, vid.z));
                }
                for (auto& kv : elevSet->tiles) {
                    if (kv.second.tile) { ready.insert(TileID(kv.first.x, kv.first.y, kv.first.z)); }
                }
            }
            cacheIt = m_upgradeRasterCache.emplace(
                    rs, std::make_pair(std::move(rendered), std::move(ready))).first;
        }
        const std::set<TileID>& rendered = cacheIt->second.first;
        const std::set<TileID>& ready = cacheIt->second.second;
        if (!rendered.empty()) {
            cellCaps.assign(size_t(N) * size_t(N), uint8_t(zt));
            int32_t baseX = int32_t(livePrimary.x) << levels;
            int32_t baseY = int32_t(livePrimary.y) << levels;
            for (int row = 0; row < N; row++) {
                for (int col = 0; col < N; col++) {
                    int32_t cx = baseX + col, cy = baseY + row;
                    int zTarget = -1;
                    for (int za = zt; za >= 0; za--) {
                        int dl = zt - za;
                        if (rendered.count(TileID(cx >> dl, cy >> dl, za))) {
                            zTarget = za;
                            break;
                        }
                    }
                    if (zTarget < 0) { continue; }  // no target covers this cell; leave at zt
                    int dl = zt - zTarget;
                    int32_t txAtTarget = cx >> dl, tyAtTarget = cy >> dl;
                    bool isReady = ready.count(TileID(txAtTarget, tyAtTarget, zTarget)) != 0;
                    int chosen = zTarget;
                    if (!isReady) {
                        // Target not loaded (yet) -- walk to the nearest ready ANCESTOR,
                        // mirroring what the renderer is actually showing as a proxy right
                        // now. If nothing is ready even down to the vector tile's own
                        // zoom, fall back to that zoom (equivalent to the pre-composite
                        // plain-texture baseline for this cell).
                        chosen = int(livePrimary.z);
                        for (int za2 = zTarget - 1; za2 >= int(livePrimary.z); za2--) {
                            int dl2 = zTarget - za2;
                            if (ready.count(TileID(txAtTarget >> dl2, tyAtTarget >> dl2, za2))) {
                                chosen = za2;
                                break;
                            }
                        }
                    }
                    cellCaps[size_t(row) * size_t(N) + size_t(col)] =
                            uint8_t(std::max(chosen, int(livePrimary.z)));
                }
            }
        }
        const uint8_t* caps = cellCaps.empty() ? nullptr : cellCaps.data();

        // Always call through with livePrimary (LIVE s -> live target zoom):
        // buildElevationMosaic()/buildOverzoomElevationMosaic() each do their own cheap
        // signature check internally and return the existing object unchanged (no re-stitch)
        // when nothing has actually changed -- see their doc comments. Only swap the attachment
        // when that yields a genuinely different object, which covers plain -> composite,
        // composite coverage upgrades as finer descendant tiles keep streaming in, LIVE
        // target-zoom changes (s flips in either direction, including cache-restored tiles
        // built at another zoom), AND displayed-terrain zoom changes under the footprint, all
        // uniformly.
        if (rs->m_buildElevationMosaic) {
            auto mosaic = rs->buildElevationMosaic(livePrimary, plainBase, caps);
            if (mosaic && mosaic != raster.texture) {
                raster.texture = mosaic;
            }
        } else {
            auto composite = rs->buildOverzoomElevationMosaic(livePrimary, plainBase, caps);
            if (composite && composite != raster.texture) {
                raster.texture = composite;
            }
        }
    }
}

bool TileManager::addTile(TileSet& _tileSet, const TileID& _tileID) {

    auto tile = m_tileCache->get(_tileSet.source->id(), _tileID);

    if (tile) {
        if (tile->sourceGeneration() == _tileSet.source->generation()) {
            // Reset tile on potential internal dynamic data set
            tile->resetState();

            // _tileID is the CURRENT visible id -- its live s may differ from the s this cached
            // tile was built at (see the _liveId doc in tileManager.h).
            upgradeAttachedRasters(_tileSet, *tile, _tileID);
        } else {
            // Clear stale tile data
            tile.reset();
        }
    }

    // Add TileEntry to TileSet
    auto entryit = _tileSet.tiles.emplace(_tileID, tile);
    TileEntry& entry = entryit.first->second;

    if (!tile) {
        // check cache for proxy (proxy already in TileSet will be found by updateTileSet())
        updateProxyTiles(_tileSet, _tileID, entry);

        entry.task = _tileSet.source->createTask(_tileID);
    }
    entry.setVisible(true);

    return bool(tile);
}

void TileManager::updateProxyTiles(TileSet& _tileSet, const TileID& _tileID, TileEntry& _tile) {
    // should we prefer child over parent as proxy?

    auto zoomBias = _tileSet.source->zoomBias();
    auto maxZoom = _tileSet.source->maxZoom();

    auto addProxy = [&](TileID id){
        auto tile = m_tileCache->get(_tileSet.source->id(), id);
        if (tile) { _tileSet.tiles.emplace(id, tile); }
        return bool(tile);
    };

    TileID parentId = _tileID.getParent(zoomBias);
    if (addProxy(parentId)) { return; }
    if (addProxy(parentId.getParent())) { return; }
    for(int ii = 0; ii < 4; ++ii) {
        addProxy(_tileID.getChild(ii, maxZoom));
    }
}

void TileManager::setCacheSize(size_t _cacheSize) {
    m_maxCacheLimit = _cacheSize;
    if (m_tileCache->cacheSizeLimit() > _cacheSize) {
        m_tileCache->limitCacheSize(_cacheSize);
    }
}

int TileManager::numTotalTiles() const {
    int tot = 0;
    for (const auto& tileSet : m_tileSets) { tot += tileSet.visibleTiles.size(); }
    return tot;
}

}
