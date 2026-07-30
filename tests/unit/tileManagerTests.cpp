#include "catch.hpp"

#include "data/rasterSource.h"
#include "data/tileSource.h"
#include "gl/hardware.h"
#include "gl/texture.h"
#include "mockPlatform.h"
#include "tile/tileManager.h"
#include "tile/tileWorker.h"
#include "util/mapProjection.h"
#include "util/fastmap.h"
#include "view/view.h"

#include <deque>

using namespace Tangram;

namespace {
// See rasterMosaicTests.cpp's identical helper: outside a real GL context,
// Hardware::loadCapabilities() never runs, so force the NPOT support flag every real target
// (GL3+/GLES3) always has in practice.
struct ForceNPOTSupport2 {
    ForceNPOTSupport2() { Hardware::supportsTextureNPOT = true; }
} forceNPOTSupport2;

TextureOptions rasterHealTestOptions() {
    TextureOptions options;
    options.pixelFormat = PixelFormat::ALPHA;  // 1 byte/pixel -- simplest to reason about
    return options;
}

std::unique_ptr<Texture> makeConstTextureForCache(int W, uint8_t value) {
    auto tex = std::make_unique<Texture>(rasterHealTestOptions());
    std::vector<GLubyte> data(size_t(W) * size_t(W), value);
    tex->setPixelData(W, W, 1, data.data(), data.size());
    return tex;
}

// RasterSource::cacheTexture() is protected (friended only to RasterTileTask/TileSource/
// FrameInfo). This test needs to drive it directly to simulate a finer descendant elevation
// tile landing in the cache moments after a vector tile's own (coarser) attachment was built --
// same idiom as TestTileManager below, which does the same for TileManager's own protected
// members.
struct TestRasterSource : public RasterSource {
    TestRasterSource()
        : RasterSource("elevation", nullptr, rasterHealTestOptions(), TileSource::ZoomOptions(-1, -1, 16, 0)) {}
    using RasterSource::cacheTexture;
};

uint8_t compositePixel(const Texture& composite, int row, int col) {
    return composite.bufferData()[size_t(row) * size_t(composite.width()) + size_t(col)];
}
} // namespace

ViewState viewState { glm::vec2(0), 1, 0, 1.f, glm::vec2(0), 256.f };

struct TestTileWorker : TileTaskQueue {
    int processedCount = 0;
    bool pendingTiles = false;

    std::deque<std::shared_ptr<TileTask>> tasks;

    void enqueue(std::shared_ptr<TileTask> task) override{
        tasks.push_back(std::move(task));
    }

    bool checkProcessedTiles() {
        if (pendingTiles) {
            pendingTiles = false;
            return true;
        }
        return false;
    }

    void processTask() {
        while (!tasks.empty()) {
            auto task = tasks.front();
            tasks.pop_front();
            if (task->isCanceled()) {
                continue;
            }

            task->setTile(std::make_unique<Tile>(task->tileId(),
                                                 task->source()->id(),
                                                 task->source()->generation()));

            pendingTiles = true;
            processedCount++;
            break;
        }
    }
    void processTask(int position) {

        auto task = tasks[position];
        tasks.erase(tasks.begin() + position);

        task->setTile(std::make_unique<Tile>(task->tileId(),
                                             task->source()->id(),
                                             task->source()->generation()));

        pendingTiles = true;
        processedCount++;
    }

    void dropTask() {
        if (!tasks.empty()) {
            auto task = tasks.front();
            tasks.pop_front();
            task->cancel();
        }
    }
};

struct TestTileSource : TileSource {
    class Task : public TileTask {
    public:
        bool gotData = false;

        Task(TileID& _tileId, TileSource* _source)
            : TileTask(_tileId, _source) {}

        bool hasData() const override { return gotData; }
    };

    int tileTaskCount = 0;

    TestTileSource() : TileSource("test", nullptr) {
        m_generateGeometry = true;
    }

    void loadTileData(std::shared_ptr<TileTask> _task, TileTaskCb _cb) override {
        tileTaskCount++;
        static_cast<Task*>(_task.get())->gotData = true;
        _task->startedLoading();
        _cb.func(std::move(_task));
    }

    void cancelLoadingTile(TileTask& _tile) override {}

    std::shared_ptr<TileData> parse(const TileTask& _task) const override {
        return nullptr;
    }

    void clearData() override {}

    std::shared_ptr<TileTask> createTask(TileID _tileId) override {
        return std::make_shared<Task>(_tileId, this);
    }
};

class TestTileManager : public TileManager {
public:
    //using Base = TileManager;
    //using Base::Base;
    TestTileManager(Platform& platform, TileTaskQueue& _tileWorker) : TileManager(platform, _tileWorker, {}) {}

    void updateTiles(const ViewState& _view, std::set<TileID> _visibleTiles) {
        // Mimic TileManager::updateTileSets(View& _view)
        m_tiles.clear();
        m_tilesInProgress = 0;
        m_tileSetChanged = false;

        TileSet& tileSet = m_tileSets[0];

        tileSet.visibleTiles = _visibleTiles;

        TileManager::updateTileSet(tileSet, _view);

        loadTiles();

        // Make m_tiles an unique list of tiles for rendering sorted from
        // high to low zoom-levels.
        std::sort(m_tiles.begin(), m_tiles.end(), [](auto& a, auto& b){
                return a->getID() < b->getID(); });

        // Remove duplicates: Proxy tiles could have been added more than once
        m_tiles.erase(std::unique(m_tiles.begin(), m_tiles.end()), m_tiles.end());

    }

    // Exercise upgradeAttachedRasters() directly for a hand-built Tile, bypassing the full
    // updateTileSet() pipeline -- lets a test drive TileManager's protected raster-repair pass
    // in isolation for a known tile / proxy-state combination.
    void testUpgradeAttachedRasters(std::shared_ptr<TileSource> vectorSource, Tile& tile, const TileID& liveId) {
        TileSet ts(vectorSource);
        upgradeAttachedRasters(ts, tile, liveId);
    }
};

TEST_CASE( "Use proxy Tile - Dont remove proxy if it is now visible", "[TileManager][updateTileSets]" ) {
    TestTileWorker worker;
    MockPlatform platform;
    TestTileManager tileManager(platform, worker);

    auto source = std::make_shared<TestTileSource>();
    std::vector<std::shared_ptr<TileSource>> sources = { source };
    tileManager.setTileSources(sources);

    /// Start loading tile 0/0/0
    std::set<TileID> visibleTiles_1 = {TileID{0,0,0}};
    tileManager.updateTiles(viewState, visibleTiles_1);

    REQUIRE(tileManager.getVisibleTiles().size() == 0);
    REQUIRE(source->tileTaskCount == 1);
    REQUIRE(worker.processedCount == 0);

    /// Start loading tile 0/0/1 - uses 0/0/0 as proxy
    std::set<TileID> visibleTiles_2 = {TileID{0,0,1}};

    tileManager.updateTiles(viewState, visibleTiles_2);

    REQUIRE(tileManager.getVisibleTiles().size() == 0);
    REQUIRE(source->tileTaskCount == 2);
    REQUIRE(worker.processedCount == 0);

    /// Process tile task 0/0/1
    worker.processTask(1);

    /// Go back to tile 0/0/0 - uses 0/0/1 as proxy
    tileManager.updateTiles(viewState, visibleTiles_1);

    REQUIRE(tileManager.getVisibleTiles().size() == 1);
    REQUIRE(source->tileTaskCount == 2);
    REQUIRE(worker.processedCount == 1);
    REQUIRE(tileManager.getVisibleTiles()[0]->isProxy() == true);
    REQUIRE(tileManager.getVisibleTiles()[0]->getID() == TileID(0,0,1));

    // Process tile task 0/0/0
    worker.processTask(0);
    tileManager.updateTiles(viewState, visibleTiles_1);
    REQUIRE(tileManager.getVisibleTiles().size() == 1);
    REQUIRE(tileManager.getVisibleTiles()[0]->isProxy() == false);
    REQUIRE(tileManager.getVisibleTiles()[0]->getID() == TileID(0,0,0));
}

TEST_CASE( "Mock TileWorker Initialization", "[TileManager][Constructor]" ) {

    TestTileWorker worker;
    MockPlatform platform;
    TileManager tileManager(platform, worker, {});
}

TEST_CASE( "Real TileWorker Initialization", "[TileManager][Constructor]" ) {
    MockPlatform platform;
    TileWorker worker(platform, 1);
    TileManager tileManager(platform, worker, {});
}

TEST_CASE( "Load visible Tile", "[TileManager][updateTileSets]" ) {
    TestTileWorker worker;
    MockPlatform platform;
    TestTileManager tileManager(platform, worker);

    auto source = std::make_shared<TestTileSource>();
    std::vector<std::shared_ptr<TileSource>> sources = { source };
    tileManager.setTileSources(sources);

    std::set<TileID> visibleTiles = {TileID{0,0,0}};
    tileManager.updateTiles(viewState, visibleTiles);
    worker.processTask();

    REQUIRE(tileManager.getVisibleTiles().size() == 0);
    REQUIRE(source->tileTaskCount == 1);
    REQUIRE(worker.processedCount == 1);

    tileManager.updateTiles(viewState, visibleTiles);

    REQUIRE(tileManager.getVisibleTiles().size() == 1);
    REQUIRE(source->tileTaskCount == 1);
    REQUIRE(worker.processedCount == 1);

}


TEST_CASE( "Use proxy Tile", "[TileManager][updateTileSets]" ) {
    TestTileWorker worker;
    MockPlatform platform;
    TestTileManager tileManager(platform, worker);

    auto source = std::make_shared<TestTileSource>();
    std::vector<std::shared_ptr<TileSource>> sources = { source };
    tileManager.setTileSources(sources);

    std::set<TileID> visibleTiles = {TileID{0,0,0}};
    tileManager.updateTiles(viewState, visibleTiles);
    worker.processTask();

    REQUIRE(tileManager.getVisibleTiles().size() == 0);
    REQUIRE(source->tileTaskCount == 1);
    REQUIRE(worker.processedCount == 1);

    std::set<TileID> visibleTiles2 = {TileID{0,0,1}};
    tileManager.updateTiles(viewState, visibleTiles2);
    worker.processTask();

    REQUIRE(tileManager.getVisibleTiles().size() == 1);
    REQUIRE(tileManager.getVisibleTiles()[0]->isProxy() == true);
    REQUIRE(tileManager.getVisibleTiles()[0]->getID() == TileID(0,0,0));
    REQUIRE(source->tileTaskCount == 2);
    REQUIRE(worker.processedCount == 2);

    tileManager.updateTiles(viewState, visibleTiles2);
    worker.processTask();

    REQUIRE(tileManager.getVisibleTiles().size() == 1);
    REQUIRE(tileManager.getVisibleTiles()[0]->isProxy() == false);
    REQUIRE(tileManager.getVisibleTiles()[0]->getID() == TileID(0,0,1));
    REQUIRE(source->tileTaskCount == 2);
    REQUIRE(worker.processedCount == 2);

}


TEST_CASE( "Use proxy Tile - circular proxies", "[TileManager][updateTileSets]" ) {
    TestTileWorker worker;
    MockPlatform platform;
    TestTileManager tileManager(platform, worker);

    auto source = std::make_shared<TestTileSource>();
    std::vector<std::shared_ptr<TileSource>> sources = { source };
    tileManager.setTileSources(sources);

    /// Start loading tile 0/0/0
    std::set<TileID> visibleTiles_1 = {TileID{0,0,0}};
    tileManager.updateTiles(viewState, visibleTiles_1);

    REQUIRE(tileManager.getVisibleTiles().size() == 0);
    REQUIRE(source->tileTaskCount == 1);
    REQUIRE(worker.processedCount == 0);

    /// Start loading tile 0/0/1 - add 0/0/0 as proxy
    std::set<TileID> visibleTiles_2 = {TileID{0,0,1}};
    tileManager.updateTiles(viewState, visibleTiles_2);

    REQUIRE(tileManager.getVisibleTiles().size() == 0);
    REQUIRE(source->tileTaskCount == 2);
    REQUIRE(worker.processedCount == 0);

    /// Go back to tile 0/0/0
    /// NB: does not add 0/0/1 as proxy, since no newTiles were loaded
    tileManager.updateTiles(viewState, visibleTiles_1);

    REQUIRE(tileManager.getVisibleTiles().size() == 0);
    REQUIRE(source->tileTaskCount == 2);
    REQUIRE(worker.processedCount == 0);

    REQUIRE(worker.tasks.size() == 2);
    // tile 0/0/0 still loading
    REQUIRE(worker.tasks[0]->isCanceled() == false);
    // tile 0/0/1 canceled ... no longer true w/ new proxy tile logic (and proxy cycle no longer possible)
    //REQUIRE(worker.tasks[1]->isCanceled() == true);

    worker.processTask();
    tileManager.updateTiles(viewState, visibleTiles_1);

    REQUIRE(tileManager.getVisibleTiles().size() == 1);
    REQUIRE(tileManager.getVisibleTiles()[0]->isProxy() == false);
    REQUIRE(tileManager.getVisibleTiles()[0]->getID() == TileID(0,0,0));

}

// --- Reproduction: does a PROXY vector tile's overzoom-elevation composite ever get upgraded as
// finer descendant elevation data becomes available, the way an ordinary (non-proxy) tile's
// does? ---
//
// This is the mechanism behind the live "landcover floating over the valley, fine on the
// mountain" report: the Zoom-labels debug overlay showed one stale (coarser-zoom) proxy tile
// sitting amid three already-updated finer neighbors. TileManager::upgradeAttachedRasters
// (tileManager.cpp:945) is the pass meant to swap in a better elevation composite as finer data
// streams in, but it early-returns for any tile still serving as a proxy
// (tileManager.cpp:976, "Skip tiles currently serving as proxies... They get upgraded the
// moment they are visible again") -- and RasterTileTask::addRaster() (rasterSource.cpp:617)
// also only does a cache-only lookup (never an active stitch) for a proxy at build time. If
// nothing else independently registers the composite in the meantime, a tile that entered view
// as a proxy can stay attached to its coarse plain elevation texture for as long as it remains
// a proxy, even once the exact finer descendant tiles it needs are sitting in cache.
TEST_CASE("upgradeAttachedRasters - a proxy tile's elevation composite is skipped even once "
          "finer descendant data is cached, while an ordinary tile's is upgraded",
          "[TileManager][RasterMosaic]") {
    const int W = 4;
    TestTileWorker worker;
    MockPlatform platform;
    TestTileManager tileManager(platform, worker);

    auto elevation = std::make_shared<TestRasterSource>();
    elevation->m_keepTextureData = true;  // required by buildOverzoomElevationMosaic

    auto vectorSource = std::make_shared<TestTileSource>();
    vectorSource->addRasterSource(elevation);

    // A vector tile whose own native data zoom is 10, but is being DISPLAYED at zoom 12 (a
    // typical overzoom situation: its own source has no z12 data, but the elevation source
    // does, up to maxZoom 16) -- zt = min(16, 12) capped at z+kOverzoomCompositeMaxLevels(2) =
    // 12, so this genuinely is overzoomed and upgradeAttachedRasters' overzoom-only gate
    // (tileManager.cpp:1051) does not skip it.
    TileID liveId(50, 50, 10, 12);

    auto ownTex = elevation->cacheTexture(TileID(50, 50, 10), makeConstTextureForCache(W, 100));

    auto runScenario = [&](bool isProxy) {
        Tile tile(TileID(50, 50, 10), vectorSource->id(), vectorSource->generation());
        tile.setProxyDepth(isProxy ? 1 : 0);
        // Simulates the state right after RasterTileTask::addRaster() ran with nothing finer
        // than the plain texture available yet -- the coarse fallback every overzoomed tile
        // starts from.
        tile.rasters().emplace_back(TileID(50, 50, 10), ownTex);

        // Baseline pass: still nothing finer cached anywhere.
        tileManager.testUpgradeAttachedRasters(vectorSource, tile, liveId);
        auto beforeTex = tile.rasters()[0].texture;

        // A finer z12 descendant covering this tile's (row=0, col=0) grid cell finishes loading
        // moments later -- e.g. because terrain-ground's own traversal needed it, exactly the
        // ordinary case during panning. m_textures only holds a WEAK ref (mirroring production:
        // a Tile elsewhere is what keeps a real tile's texture alive), so the returned
        // shared_ptr must be kept alive here or it's evicted before the next lookup.
        auto descendantTex = elevation->cacheTexture(TileID(200, 200, 12), makeConstTextureForCache(W, 222));

        tileManager.testUpgradeAttachedRasters(vectorSource, tile, liveId);
        return std::make_pair(beforeTex, tile.rasters()[0].texture);
    };

    SECTION("ordinary (non-proxy) tile: composite is upgraded to reflect the new real cell") {
        auto [beforeTex, afterTex] = runScenario(false);
        REQUIRE(afterTex != nullptr);
        REQUIRE(afterTex != ownTex);
        // The upgrade must actually have happened (not just returned the same stale object).
        REQUIRE(afterTex != beforeTex);
        // Grid cell (row=0 [north], col=0 [west]) of the composite must now hold the REAL
        // descendant's value, not an upsampled placeholder derived from the coarse own texture
        // (100). Composite buffer rows run south->north (see stitchElevationOverzoomMosaic's
        // doc comment and the "Overzoom mosaic - dimensions and real children land in the
        // correct grid cell" test), so grid row=0 (north) lands in the TOP pixel band
        // [(N-1)*W, N*W) -- here N=4, W=4, so rows [12, 16).
        for (int r = 0; r < W; r++) {
            for (int c = 0; c < W; c++) {
                REQUIRE(compositePixel(*afterTex, 3 * W + r, c) == 222);
            }
        }
    }

    SECTION("proxy tile: composite is never attempted, stays on the coarse plain texture") {
        auto [beforeTex, afterTex] = runScenario(true);
        REQUIRE(beforeTex == ownTex);
        // Even after the exact finer descendant it needs is cached, a proxy tile's attachment
        // is untouched: upgradeAttachedRasters returns immediately at isProxy(), never calling
        // buildOverzoomElevationMosaic/buildElevationMosaic at all.
        REQUIRE(afterTex == ownTex);
    }
}
