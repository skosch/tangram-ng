#pragma once

#include "data/tileSource.h"
#include "gl/texture.h"
#include "tile/tileTask.h"
#include "tile/tileHash.h"
#include "util/mapProjection.h"

#include <array>
#include <functional>
#include <map>
#include <mutex>

namespace Tangram {

class RasterTileTask;

// Combine a tile's own WxW texture with (up to) 8 neighbor textures (same zoom: N, S, E, W,
// NE, NW, SE, SW) into a single padded 3Wp x 3Wp "mosaic" texture (Wp = W - overlap, see
// below), so hillshading/texture-shading shaders can read real neighbor data across tile
// edges instead of clamping at the tile boundary (see docs/texture-shading-plan.md,
// "Frozen Interface Contract" + "Phase 2").
//
// Exact cell layout (this is the part other phases depend on staying stable):
//   `neighbors` is a 3x3 grid flattened row-major in TILE offsets: index (dy+1)*3 + (dx+1)
//   for dx, dy in {-1, 0, 1} (dy = -1: north neighbor, tile y - 1). Index 4 (dx=0, dy=0) is
//   ignored -- `own` is always used for the center cell. Each cell is that tile's buffer
//   cropped to its first Wp rows/columns, where Wp = W - (W % 2): node-registered tiles
//   (odd W, e.g. 257) share their outermost row/col with the adjacent tile, and dropping
//   that duplicate line keeps the mosaic an evenly-spaced global sample grid AND keeps
//   neighboring tiles' mosaic mip grids aligned (mosaic width 3*Wp halves evenly through
//   the whole mip chain) so the high-pass texture shading is seam-free across tiles -- see
//   the layout notes in rasterSource.cpp. `own` occupies the center third [Wp, 2Wp) x
//   [Wp, 2Wp); each neighbor cell is at column range [(dx+1)*Wp, (dx+2)*Wp) and row range
//   [(1-dy)*Wp, (2-dy)*Wp): buffer rows run south -> north (row 0 = tile's south edge,
//   matching how rasters are sampled - see ElevationManager::elevationLerp), i.e. OPPOSITE
//   to tile y, so the north neighbor (dy = -1) occupies the mosaic's top rows [2Wp, 3Wp).
//
// A null entry in `neighbors` (tile not cached yet) is synthesized via mirror-extrapolation
// of `own`'s data about the shared tile edge: flipped vertically for a N/S neighbor,
// horizontally for an E/W neighbor, and both ways for a diagonal neighbor. This is a cheap,
// byte-only (no pixel decode) fallback -- it only ever feeds the coarsest mip levels of the
// shader-side pyramid, so exactness doesn't matter, it works unmodified for both
// terrarium-RGB and float elevation encodings since it never interprets pixel values, and
// it is transient: RasterSource::patchNeighborMosaics() overwrites it with real data in
// place when the neighbor tile arrives.
//
// Returns nullptr if `own` has no CPU-side buffer data (RasterSource::m_keepTextureData must
// be true for the source for this to be usable) or its width/height aren't equal and > 0.
//
// Exposed at namespace scope (rather than as a RasterSource member) so it can be unit-tested
// directly with hand-built Texture objects -- see tests/unit/rasterMosaicTests.cpp.
std::shared_ptr<Texture> stitchElevationMosaic(const std::shared_ptr<Texture>& own,
                                                const std::array<std::shared_ptr<Texture>, 9>& neighbors);

// One cell of the N x N grid stitchElevationOverzoomMosaic() assembles (see its doc comment
// below). `texture` is whatever REAL cached tile actually covers this cell's world position --
// not necessarily at the composite's own target zoom zt: a single vector tile's footprint can
// straddle a zoom boundary in the terrain tileset itself (near sub-tiles rendered at z16, far
// ones only at z15 under a tilted camera -- the terrain tileset picks a display zoom per elevation
// tile from live on-screen area, which shrinks with distance, so it varies WITHIN one coarser
// vector tile's footprint; see docs/terrain-illumination-plan.md "intra-footprint zoom variation"),
// so building the whole composite from one uniform achieved level can never match everywhere. Each
// cell instead names the ACTUAL tile (`texture`, at whatever zoom it was cached at) that would be
// rendered under it, and `blockSize` > 1 signals that this cell is one of `blockSize`^2 grid cells
// all sharing that same coarser tile (`blockColOriginWest`/`blockRowOriginNorth` locate that
// tile's own footprint in GRID units, i.e. TILE-offset units at zt, matching `col`/`row`'s own
// convention in stitchElevationOverzoomMosaic's doc comment: col west -> east, row north -> south)
// -- stitchElevationOverzoomMosaic() then fills that cell not by nearest-sampling the coarser
// tile's raw pixels but by evaluating the SAME fixed 64-cell anti-diagonal piecewise-linear
// surface the terrain mesh itself renders for that tile (RasterStyle::build), at this cell's exact
// sub-position within it -- see fillLatticeUpsampledCell in rasterSource.cpp. Because a finer
// lattice with the same anti-diagonal split nests exactly inside a coarser one (every fine
// triangle lies within one coarse triangle), this reproduces the coarser tile's terrain-mesh
// surface EXACTLY, not merely approximately -- unlike the old uniform-level nearest-neighbor
// fallback this replaces, which was flat (blocky) and could disagree with the terrain mesh by a
// full cell's worth of relief. `blockSize == 1` (the ordinary case: real data exists AT zt for
// this cell) means direct copy instead -- see stitchElevationOverzoomMosaic's dispatch.
struct OverzoomCell {
    std::shared_ptr<Texture> texture;
    int32_t blockColOriginWest = 0;
    int32_t blockRowOriginNorth = 0;
    int32_t blockSize = 1;
};

// Opportunistic overzoom-compensation composite (docs/terrain-illumination-plan.md, "raster-zoom
// mismatch" section): when an elevation raster is attached to a primary vector tile task that is
// itself overzoomed relative to its own source (TileID s > z, e.g. an `osm` vector tile whose
// source caps at z14 being displayed at z16), RasterSource::addRasterTask attaches the elevation
// raster at the PRIMARY task's coarse data zoom (z), never at the elevation source's own finer
// native zoom nor at the true display zoom (s) -- so draped polygon geometry samples genuinely
// coarser elevation data than the co-located `terrain-ground` mesh, which always fetches its own
// tiles natively. This produces a real (not just resolution/tessellation) vertical mismatch
// between the two nominally-coincident surfaces, which parallax-amplifies into visible pale
// "white patch" gaps at grazing camera angles on rugged terrain.
//
// This function builds a single composite texture covering the SAME [0,1] UV footprint as `own`
// (the primary tile's plain coarse texture), stitched from an N x N grid of cells at a finer
// target zoom zt (N = 2^(zt - primary.z)) -- see OverzoomCell below for how each cell describes
// its own data source, which need not be uniform across the grid. Because the composite keeps
// the exact same UV footprint as `own`, callers can register the result at the PRIMARY task's
// own (coarse) TileID unchanged -- see Style::setupTileShaderUniforms's tileID.z vs
// raster.tileID.z crop logic (style.cpp) -- with no shader or crop-math changes needed.
//
// Node-registration crop (Wp = W - overlap, see copyMosaicCell/copyOverzoomGridCell in
// rasterSource.cpp) applies per-cell same as stitchElevationMosaic(), EXCEPT the composite's own
// outer boundary (row == 0, the north-most band; col == N-1, the east-most band) is NOT cropped
// in that dimension, since no further cell exists beyond it to supply the shared edge that
// cropping would otherwise drop -- fixing a real, shipped bug (docs/terrain-illumination-plan.md
// "missing far-edge line") where the composite came out one texel short of the tile's true
// north/east edge, so the terrain lattice's UV=1 row/column (getTerrainMeshElevation,
// elevation.yaml) sampled the wrong sample there, streaking along the north/east tile border
// even given a fully-populated N x N grid. The composite is therefore
// (N - 1) * Wp + own->width() texels wide/tall, not simply N * Wp.
//
// Returns nullptr if `own` has no CPU-side buffer data, isn't square, `cells.size()` != N * N,
// or N <= 1.
std::shared_ptr<Texture> stitchElevationOverzoomMosaic(const std::shared_ptr<Texture>& own,
                                                        const std::vector<OverzoomCell>& cells,
                                                        int N);

// Attached via Texture::userData to every 3Wp neighbor-ring mosaic (m_mosaics, built by
// RasterSource::buildElevationMosaic) AND to every overzoom composite (m_overzoomMosaics, built
// by RasterSource::buildOverzoomElevationMosaic). `ownTexture == nullptr` distinguishes the two:
// null marks the texture as being a composite itself.
struct ElevationMosaicInfo {
    // Mosaics only (null on a composite's own info): the center-cell texture actually used
    // (plain per-tile texture or an overzoom composite), kept alive for as long as the mosaic
    // lives: m_textures only holds weak refs and Tiles reference the mosaic instead of the
    // original, so without this the original expires immediately and neighbors stitched later
    // would permanently fall back to mirror-extrapolation for this tile's cell (-> texture
    // shading tile seams).
    std::shared_ptr<Texture> ownTexture;
    // The tile's PLAIN per-tile texture, kept alive even when the center cell is a composite
    // (in which case ownTexture is the composite and nothing else would pin the plain texture
    // in m_textures): styling-zoom churn (see compositeZoom below) later needs it as the base
    // for rebuilding a composite at a DIFFERENT level -- stitchElevationOverzoomMosaic requires
    // the plain texture as its dimension reference and fallback-fill source. Equal to ownTexture
    // when the center IS the plain texture; also set on a composite's own info.
    std::shared_ptr<Texture> plainTexture;
    // dimensionless RMS-slope statistic of the tile at a mid-pyramid-band scale, input for
    // texture shading auto-contrast (see RasterSource::aggregateRugosity)
    float rugosity = 0.f;
    // Target zoom (zt = RasterSource::overzoomTargetZoom(primary)) this texture's grid is built
    // at -- NOT a claim that every cell actually holds zt-zoom data (see OverzoomCell: cells
    // without real zt data are triangle-PL-upsampled from whatever coarser real tile covers
    // them, exactly reproducing that tile's own terrain-mesh surface -- see
    // fillLatticeUpsampledCell). This is now purely a bookkeeping/logging field: the shader no
    // longer needs to be told a lattice zoom at all (getTerrainMeshElevation in elevation.yaml
    // derives its lattice cell count directly from the bound texture's own pixel dimensions --
    // rasterPixelSize -- so CPU-side stitching decisions can never disagree with what the shader
    // assumes; see the "single source of truth" note on RasterSource::buildOverzoomElevationMosaic).
    int compositeZoom = -1;
    // Cheap per-cell "did any new real data arrive" fingerprint (one bit per zt-grid cell, see
    // RasterSource::buildOverzoomElevationMosaic): drives the coverage-driven rebuild-vs-reuse
    // decision without needing to re-stitch (or even re-walk the ancestor chain) on every check.
    // Composites only; -1 (all bits unset is a valid signature, so use a separate flag) isn't
    // needed since 0 already means "nothing real cached yet, all cells fell back to `own`" and a
    // freshly-built composite is always registered once, so "never built" is simply "not in the
    // registry" rather than a sentinel value here.
    uint64_t coverageSignature = 0;
    // Mosaics only: bit ((dy+1)*3+(dx+1)) set means that neighbor cell holds REAL (not
    // mirror-extrapolated) data as of the last stitch/patch; bit 4 (dx=dy=0, "self") is unused.
    // All 8 neighbor bits set == kAllRealNeighborsMask means the ring is complete. Drives the
    // R2 patch-upload debounce below (tile-pipeline-perf-plan.md).
    uint16_t neighborMask = 0;
    // Mosaics only: strong refs to each same-zoom neighbor's texture that this mosaic's cells
    // were actually stitched from (real data only -- a mirror-extrapolated cell has no entry
    // here), indexed the same way as neighborMask ((dy+1)*3+(dx+1); index 4 unused). m_textures
    // only holds WEAK refs, so without this a neighbor tile that's still referenced by nothing
    // else (e.g. it isn't a Tile's own visible raster right now, just data this mosaic
    // borrowed) can be evicted the instant it's cached -- then the next time THIS mosaic
    // rebuilds (styling-zoom churn, cache eviction of the center tile, etc.) it finds that
    // neighbor gone and silently downgrades that cell to mirror-extrapolation, reintroducing a
    // seam that wasn't there a moment ago (2026-07 UX report: "edge artifacts... even at
    // zenith... fine most of the time" -- confirmed via mosaicNeighborRegression* counters
    // firing repeatedly under real cache pressure). Mirrors the exact reasoning already applied
    // to ownTexture/plainTexture above, extended to the other 8 textures a mosaic depends on.
    std::array<std::shared_ptr<Texture>, 9> neighborTextures;
    // Mosaics only: steady_clock seconds this mosaic was first marked dirty by
    // RasterSource::patchNeighborMosaics() since its last GPU-upload flush; 0 means "not
    // currently dirty". A patched mosaic's `resize()` (full re-upload + mip regen on next
    // bind) is deferred until RasterSource::flushDirtyMosaics() decides the ring is complete
    // or this has gone stale too long, so several neighbor arrivals in quick succession
    // collapse into one re-upload instead of one each (tile-pipeline-perf-plan.md R2).
    double firstDirtyTime = 0.0;
};

// All 8 neighbor bits set (see ElevationMosaicInfo::neighborMask), i.e. every neighbor cell
// holds real data and none is still mirror-extrapolated.
constexpr uint16_t kAllRealNeighborsMask = 0x1EF;

class RasterSource : public TileSource {

    using Cache = std::map<TileID, std::weak_ptr<Texture>>;
    std::shared_ptr<Cache> m_textures;

    // Live mosaic registry (elevation source only, i.e. m_buildElevationMosaic set): one
    // shared mosaic texture per TileID, kept alive by the Tile(s) whose rasters() reference
    // it. Serves two purposes: (1) dedup - every Tile referencing the same elevation tile
    // shares one stitched mosaic instead of each addRaster() call stitching its own copy;
    // (2) live patching - when a tile texture later lands in m_textures (e.g. a Phase 1
    // neighbor prefetch completing), patchNeighborMosaics() copies its pixels over the
    // mirror-extrapolated placeholder cell in each registered neighboring mosaic, in place,
    // so mirrored fallback data is only ever transient. Main-thread only (same threading
    // rule as m_textures, see RasterTileTask::addRaster()).
    std::map<TileID, std::weak_ptr<Texture>> m_mosaics;

    // Live registry for the opportunistic overzoom-compensation composite (see
    // stitchElevationOverzoomMosaic() above and buildOverzoomElevationMosaic() below): ONE entry
    // per (x, y, z, zt) -- zt = overzoomTargetZoom(primary), itself already stable across the `s`
    // churn of ordinary panning (see its own doc comment), so a tile has at most one live entry
    // at a time (a stale different-zt entry from before a styling-zoom flip is simply replaced,
    // not kept alongside). Deliberately separate from m_mosaics: that registry's entries are
    // 3Wp x 3Wp neighbor-ring mosaics for the unrelated texture-shading feature and are patched
    // in place by patchNeighborMosaics() under a different layout contract -- mixing the two
    // would corrupt both.
    //
    // Unlike an earlier version of this registry, an entry here is NOT immutable once built: the
    // composite is always built at the FULL target resolution zt immediately (every zt-grid cell
    // that lacks real zt data is triangle-PL-upsampled from whatever coarser real ancestor tile
    // is cached -- see OverzoomCell/fillLatticeUpsampledCell), so there is no more "achieved a
    // coarser level, wait for the finer one" state to track; what CAN still change over time is
    // how many of its cells hold real (vs. upsampled) data as finer descendant tiles keep
    // streaming in. buildOverzoomElevationMosaic() re-derives a cheap per-cell "real data
    // present" bitset (ElevationMosaicInfo::coverageSignature) on every call and only pays for a
    // full re-stitch when that signature has actually changed, replacing the registry entry with
    // a freshly-built Texture object (never mutated in place -- simpler and safe to share with
    // anything already holding the previous object).
    //
    // Holds STRONG references (unlike m_mosaics/m_textures's weak-ref pattern), deliberately:
    // the primary vector tile a composite is attached to gets rebuilt routinely during ordinary
    // panning (TileManager treats a TileID with a different `s` -- recomputed every frame from
    // live on-screen area -- as a structurally different visible-tile entry, so `s` churn alone
    // spawns a fresh TileTask even though the underlying tile is physically the same). Each
    // rebuild drops the old Tile's raster reference and, with it, a weakly-held composite -- a
    // weak-ref cache here would therefore re-walk the ancestor chain from scratch (against
    // whatever fraction of descendants happens to be cached at that exact instant) on essentially
    // every rebuild; a strong cache means a composite's coverage only ever improves (or, on a
    // zt change, gets rebuilt once for the new target) instead of randomly re-deriving. Bounded
    // by kMaxOverzoomMosaicEntries (rasterSource.cpp) to avoid unbounded growth over a long
    // panning session -- see pruneOverzoomMosaicsIfNeeded().
    std::map<TileID, std::shared_ptr<Texture>> m_overzoomMosaics;

    // Simple bound for m_overzoomMosaics: if adding a new entry would exceed
    // kMaxOverzoomMosaicEntries (rasterSource.cpp), clear the whole registry first. A full clear
    // (rather than proper LRU eviction) is a deliberate simplification -- this only fires after
    // a long session has visited many distinct overzoomed tiles, is rare compared to the
    // per-pan-step churn this cache exists to eliminate, and self-heals immediately (any tile
    // still on screen just rebuilds its composite on next rebuild, same as first-ever visit).
    void pruneOverzoomMosaicsIfNeeded();

    TextureOptions m_texOptions;

    std::shared_ptr<Texture> m_emptyTexture;

    // R4 (tile-pipeline-perf-plan.md): remaining number of real mosaic/composite stitches
    // RasterTileTask::addRaster() may perform this update cycle before falling back to the
    // plain per-tile texture (same fallback proxy tiles already use). Reset once per
    // TileManager::updateTileSets() via resetStitchBudget() below, so a burst of tile
    // completions in one frame can't stack up an unbounded amount of main-thread stitch work;
    // TileManager::upgradeAttachedRasters (which already re-checks every update) picks up the
    // deferred stitch on a later, less busy frame.
    int m_stitchBudget = 0;

    // Consume one unit of this update cycle's stitch budget; false means the caller should
    // skip stitching for now and reuse whatever's already registered instead.
    bool tryConsumeStitchBudget() {
        if (m_stitchBudget <= 0) { return false; }
        --m_stitchBudget;
        return true;
    }

    // Perf (2026-07 profiling): remaining number of mosaic GPU re-uploads flushDirtyMosaics()
    // may perform this update cycle. Mirrors m_stitchBudget above but bounds Texture::resize()
    // calls (the actual GPU upload) rather than CPU-side stitching: a fast pan/zoom burst can
    // make many mosaics ring-complete-or-stale-dirty in the same update, and uploading all of
    // them in one frame was observed to spike a single frame's texture upload by up to ~500MB.
    // Reset once per TileManager::updateTileSets() via resetUploadBudget() below.
    int m_uploadBudget = 0;

    // Consume one unit of this update cycle's upload budget; false means the caller should
    // leave the mosaic dirty and retry the upload on a later, less busy update.
    bool tryConsumeUploadBudget() {
        if (m_uploadBudget <= 0) { return false; }
        --m_uploadBudget;
        return true;
    }

    friend class RasterTileTask;
    friend class TileSource;
    friend class FrameInfo;
protected:
    std::shared_ptr<TileData> m_tileData;

    std::shared_ptr<TileData> parse(const TileTask& _task) const override;

    std::shared_ptr<RasterTileTask> createRasterTask(TileID _tileId, bool subTask);

    void addRasterTask(TileTask& _tileTask);

    std::unique_ptr<Texture> createTexture(TileID _tile, const std::vector<char>& _rawTileData);

    std::shared_ptr<Texture> cacheTexture(const TileID& _tileId, std::unique_ptr<Texture> _texture);

    // Copy `_texture` (the freshly-cached tile texture for `_tileId`) over the corresponding
    // cell of each still-alive neighboring mosaic in m_mosaics, replacing whatever was there
    // (typically a mirror-extrapolated placeholder), and flag those mosaics for GPU
    // re-upload (+ mipmap regeneration) on next bind. Called from cacheTexture() whenever a
    // new texture lands in m_textures and m_buildElevationMosaic is set.
    void patchNeighborMosaics(const TileID& _tileId, const std::shared_ptr<Texture>& _texture);

public:
    // Area-mean of ElevationMosaicInfo::rugosity over all live mosaics - i.e. over roughly the
    // visible tiles plus the prefetched neighbor ring, a window a few times the viewport, so
    // panning changes it only gradually. Input for texture shading auto-contrast (see
    // MapsApp::mapUpdate). Returns -1 when no mosaics are live. Main thread only, like all
    // other m_mosaics access.
    float aggregateRugosity();

    // Flush pending patch-driven re-uploads (see ElevationMosaicInfo::firstDirtyTime): call
    // Texture::resize() on every live mosaic whose neighbor ring is complete or whose oldest
    // unflushed patch is older than the debounce window, then clear its dirty state. Meant to
    // be called once per TileManager update cycle (not per patch, not per frame necessarily),
    // so several neighbor arrivals in quick succession cost one GPU re-upload instead of one
    // each (tile-pipeline-perf-plan.md R2). Main thread only, like all other m_mosaics access.
    void flushDirtyMosaics();

    // R4 (tile-pipeline-perf-plan.md): called once per TileManager::updateTileSets() to grant
    // this source a fresh main-thread stitch budget for the update cycle about to run (see
    // m_stitchBudget above).
    void resetStitchBudget(int n) { m_stitchBudget = n; }

    // Perf (2026-07 profiling): called once per TileManager::updateTileSets() to grant this
    // source a fresh mosaic GPU-upload budget for the update cycle about to run (see
    // m_uploadBudget above).
    void resetUploadBudget(int n) { m_uploadBudget = n; }

    // Get-or-build the shared 3W x 3W mosaic texture for `_tile`: an already-registered live
    // mosaic is returned as-is (its cells are kept current by patchNeighborMosaics());
    // otherwise one is stitched from `_ownTexture` plus whatever of its 8 same-zoom neighbors
    // are already in `m_textures`, with missing neighbors mirror-extrapolated, and registered
    // in m_mosaics for later in-place patching. See stitchElevationMosaic() above for the
    // exact layout. Only ever called when m_buildElevationMosaic is set (elevation source
    // only), on the main thread. `_cellZoomCaps` is forwarded to the internal
    // buildOverzoomElevationMosaic() call that produces the center cell (same contract).
    std::shared_ptr<Texture> buildElevationMosaic(TileID _tile, const std::shared_ptr<Texture>& _ownTexture,
                                                  const uint8_t* _cellZoomCaps = nullptr);

    // Cheap reuse-only lookup (single map find + weak_ptr lock, no stitching): returns an
    // already-registered live mosaic for `_tile` if one exists, else nullptr. Safe to call for
    // proxy/prefetch-only tiles, which must never trigger the expensive full stitch in
    // buildElevationMosaic() above (see RasterTileTask::addRaster()).
    std::shared_ptr<Texture> getExistingMosaic(TileID _tile);

    // Get-or-build the opportunistic overzoom-compensation composite for `_primary` (see
    // stitchElevationOverzoomMosaic() above for the "why"). No-op (returns nullptr) unless
    // `_primary` is actually overzoomed relative to both this source's own zoomOptions.maxZoom
    // and `_primary.s` (i.e. there is real finer data to opportunistically pull in).
    //
    // Always builds directly AT the full target resolution zt = overzoomTargetZoom(_primary) --
    // there is no more "achieved level" to fall back through. Per zt-grid cell (independently!),
    // the cell's TARGET zoom data is used if cached; otherwise the finest real ancestor down to
    // _primary.z+1 that IS cached is triangle-PL-upsampled to fill it (see OverzoomCell /
    // fillLatticeUpsampledCell), falling back to `_ownTexture` itself (always available) only
    // when no ancestor at all is cached yet -- this is what lets one composite correctly track a
    // vector tile footprint whose underlying terrain tileset renders a MIX of zooms across it
    // (near sub-tiles at zt, far ones one or more levels coarser under a tilted camera), instead
    // of requiring one uniform "achieved" level for the whole grid the way this used to. Cheap
    // per-cell signature comparison (ElevationMosaicInfo::coverageSignature) against the
    // currently-registered entry decides whether a re-stitch is actually needed -- calling this
    // every update (as TileManager::upgradeAttachedRasters does) is therefore cheap in the
    // steady state, same as the old zoom-equality check it replaces.
    //
    // `_cellZoomCaps` (optional; N*N = 4^(zt - _primary.z) row-major entries, cells[row][col]
    // covering tile (baseX+col, baseY+row, zt)): per-cell MAX data zoom, clamped to
    // [_primary.z, zt]. This is how the caller (TileManager::upgradeAttachedRasters, which
    // unlike this class can see which elevation tiles the terrain tileset is CURRENTLY
    // RENDERING) makes the composite reproduce the DISPLAYED terrain surface instead of the
    // finest CACHED data. The distinction is the root cause of the persistent "white patch"
    // drape mismatch (2026-07 live diagnosis: persistent patches occurred exclusively on
    // composite-attached tiles): panning fills m_textures with fine tiles, but the terrain's
    // screen-area LOD keeps DISPLAYING coarser tiles over the far parts of a footprint -- a
    // composite stitched from "finest cached" then disagrees with the rendered terrain by the
    // full inter-zoom DEM delta, permanently, since ever-better coverage is exactly what the
    // old signature converged to. Caps are folded into the signature, so a display-zoom flip
    // in either direction triggers exactly one re-stitch. Null = uncapped (old behavior:
    // build-time attachment, no tileset context).
    //
    // Cache-only (never triggers a new fetch; docs/terrain-illumination-plan.md); callers must
    // keep falling back to `_ownTexture` when this returns nullptr (only possible when
    // `_ownTexture` itself is unavailable). Requires m_keepTextureData. Registered in
    // m_overzoomMosaics keyed by (x, y, z, zt) -- a single entry per tile, replaced (not
    // multiplied) as zt or coverage changes. Main thread only, like all other
    // m_textures/m_mosaics access.
    std::shared_ptr<Texture> buildOverzoomElevationMosaic(TileID _primary, const std::shared_ptr<Texture>& _ownTexture,
                                                          const uint8_t* _cellZoomCaps = nullptr);

    // Cheap reuse-only lookup (single map find, no stitching) for m_overzoomMosaics, mirroring
    // getExistingMosaic() above -- safe for proxy/prefetch-only tasks, which must never trigger
    // the (mildly) more expensive full build above. Looks up exactly the (x, y, z, zt) key
    // buildOverzoomElevationMosaic would use -- no more per-level fallback search, since a
    // registered entry is always already built at the full target resolution.
    std::shared_ptr<Texture> getExistingOverzoomMosaic(TileID _primary);

    // Target zoom buildOverzoomElevationMosaic() would stitch `_primary` up to, or `_primary.z`
    // itself if `_primary` isn't overzoomed enough to be worth compensating for (callers must
    // treat that as "not applicable"). Capped at kOverzoomCompositeMaxLevels zoom levels above
    // `_primary.z` to bound composite size regardless of how overzoomed the display is. Public
    // because TileManager::upgradeAttachedRasters compares it against an attached raster's
    // ElevationMosaicInfo::compositeZoom to decide whether an attachment is still final.
    //
    // Deliberately depends only on `_primary.z`/`_primary.s` and this source's own
    // zoomOptions.maxZoom -- both m_overzoomMosaics' cache key and every lookup MUST go through
    // this (never key on the raw `_primary` TileID directly): `_primary.s` is not a stable
    // per-tile property, it's continuously recomputed by TileManager from live on-screen tile
    // area (`updateTileSets()`'s `visId.s = tileId.z + ceil(log2(area/maxArea)/2)`) and ticks up
    // and down by small amounts on essentially every pan/zoom step, even when the actual target
    // resolution this composite would stitch to (`zt`, which saturates once
    // kOverzoomCompositeMaxLevels is hit -- e.g. s=16 and s=17 both cap to the same zt=16 for a
    // z=14 primary) hasn't changed at all. Keying on raw `_primary` was a real, shipped bug:
    // confirmed via live debug logs showing the exact same physical tile rebuilt multiple times
    // in quick succession during ordinary panning, each time from whatever fraction of
    // descendants happened to be cached at that instant (present counts differing between
    // rebuilds of the "same" tile) -- producing visible, tile-sized flicker of the composite's
    // completeness as the user panned, distinct from (and in addition to) the underlying
    // opportunistic/partial-coverage limitation. Keying by (x, y, z, zt) instead makes the cache
    // entry -- and thus whether a given on-screen tile shows a stitched composite or the plain
    // fallback -- depend only on the physical tile and its actual target resolution, stable
    // across the `s` churn of ordinary panning.
    int overzoomTargetZoom(TileID _primary) const;

protected:

public:

    bool m_keepTextureData = false;

// Phase 1 (docs/texture-shading-plan.md "Frozen Interface Contract"): when true,
    // TileManager::enqueueNeighborPrefetch() will proactively enqueue low-priority
    // background fetches for a visible tile's 8 same-zoom neighbor tiles, so that a
    // later mosaic-stitching step has real (not extrapolated) neighbor data available
    // more often. Only ever set for the elevation raster source (see scene.cpp, near
    // where m_keepTextureData is set). Must not affect any other raster source.
    bool m_prefetchNeighbors = false;

    // When true, the texture attached to Tile::rasters() for this source is replaced with a
    // padded 3W x 3W neighbor mosaic (see stitchElevationMosaic()), uploaded with hardware
    // mipmaps. Set only for the elevation RasterSource (scene.cpp, alongside
    // m_keepTextureData, which this depends on -- the stitcher needs CPU-side buffer data for
    // both the tile and its cached neighbors). Must never be set for other raster sources
    // (worldcover, strava heatmaps, contour tiles, etc.) -- see
    // docs/texture-shading-plan.md Frozen Interface Contract.
    bool m_buildElevationMosaic = false;

    RasterSource(const std::string& _name, std::unique_ptr<DataSource> _sources,
                 TextureOptions _options, TileSource::ZoomOptions _zoomOptions = {});

    // TODO Is this always PNG or can it also be JPEG?
    const char* mimeType() const override { return "image/png"; };

    std::shared_ptr<Texture> getTexture(TileID _tile);

    Raster getRaster(ProjectedMeters _meters);

    std::shared_ptr<TileTask> createTask(TileID _tile) override;

    bool isRaster() const override { return true; }

    void generateGeometry(bool _generateGeometry) override;

    std::shared_ptr<Texture> emptyTexture() { return m_emptyTexture; }

};

}
