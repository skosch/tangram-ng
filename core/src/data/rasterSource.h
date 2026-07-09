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
// (the primary tile's plain coarse texture), stitched from an N x N grid of `children` -- same-
// size descendant tiles at a finer target zoom (N = 2^levels) -- laid out row-major in TILE
// coordinate order: children[row * N + col], col = west -> east (u ascending), row = north ->
// south (tile y ascending). A missing or dimension-mismatched cell (nullptr entry, or a texture
// that isn't square / doesn't match `own`'s width or pixel format) is filled by nearest-neighbor
// upsampling the corresponding sub-rectangle of `own` -- never a hole, just locally-coarser detail
// where finer data isn't available. Because the composite keeps the exact same UV footprint as
// `own`, callers can register the result at the PRIMARY task's own (coarse) TileID unchanged --
// see Style::setupTileShaderUniforms's tileID.z vs raster.tileID.z crop logic (style.cpp) -- with
// no shader or crop-math changes needed.
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
// Deliberately takes "whatever textures are available" (nulls allowed) as input rather than an
// active-fetch guarantee, so it can be reused unchanged by a future fuller fix that actively
// fetches the N children via subtasks instead of only opportunistically finding them already
// cached -- see RasterSource::buildOverzoomElevationMosaic's cache-only lookup for the current,
// deliberately scoped-down, first pass (that caller additionally requires ALL N*N children to be
// real, non-fallback data before ever using a composite it builds here -- see its doc comment --
// but this free function itself still accepts and fills partial input, since it is also
// exercised directly with partial input by unit tests, and staying maximally permissive here
// keeps it reusable for the active-fetch fuller fix too).
//
// Returns nullptr if `own` has no CPU-side buffer data, isn't square, `children.size()` !=
// N * N, or N <= 1.
std::shared_ptr<Texture> stitchElevationOverzoomMosaic(const std::shared_ptr<Texture>& own,
                                                        const std::vector<std::shared_ptr<Texture>>& children,
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
    // Zoom level of the elevation DATA this texture carries at lattice-node scale: the tile's
    // own data zoom for a plain-centered mosaic, or the (single, uniform) zoom of a composite's
    // cells. The drape is only exact where this EQUALS the zoom of the terrain tiles actually
    // rendered underneath (min(styling zoom, source maxZoom); the two tilesets compute the same
    // styling zoom for the same geography -- the on-screen-area formula telescopes across zoom
    // levels). FINER data is just as wrong as coarser: the z15 DEM is a downsample of z16 and
    // the two differ by meters on cliffs, so a z16-composite drape over z15 terrain tiles
    // mismatches just like a z14-value drape over z15 terrain does. All upgrade/downgrade
    // decisions (buildElevationMosaic's early-return, TileManager::upgradeAttachedRasters)
    // therefore compare this for EQUALITY with the live overzoomTargetZoom(), never with >=,
    // and a styling-zoom flip near the camera (a few meters of panning moves a tile across the
    // s=15/16 on-screen-area threshold) triggers a rebuild in BOTH directions -- without that,
    // a tile keeps draping wrong-zoom values against terrain that flipped along with it
    // (observed live as white patches born from meter-scale pans that never healed).
    int compositeZoom = -1;
};

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
    // stitchElevationOverzoomMosaic() above and buildOverzoomElevationMosaic() below): one
    // shared composite per (x, y, z, compositeZoom) -- a tile can have SEVERAL entries at
    // different composite zooms (e.g. a z15 one and a z16 one) as its styling zoom flips near
    // the camera; each is complete and immutable once registered, and the builder picks the one
    // matching the live target zoom (see ElevationMosaicInfo::compositeZoom's doc comment for
    // why EQUALITY with the terrain's zoom, not "finest available", is the correctness
    // criterion). Deliberately separate from m_mosaics: that registry's entries are 3Wp x 3Wp
    // neighbor-ring mosaics for the unrelated texture-shading feature and are patched in place
    // by patchNeighborMosaics() under a different layout contract -- mixing the two would
    // corrupt both.
    //
    // Holds STRONG references (unlike m_mosaics/m_textures's weak-ref pattern), deliberately:
    // the primary vector tile a composite is attached to gets rebuilt routinely during ordinary
    // panning (TileManager treats a TileID with a different `s` -- recomputed every frame from
    // live on-screen area -- as a structurally different visible-tile entry, so `s` churn alone
    // spawns a fresh TileTask even though the underlying tile is physically the same). Each
    // rebuild drops the old Tile's raster reference and, with it, a weakly-held composite --
    // so a weak-ref cache here would rebuild from scratch (against whatever fraction of
    // descendants happens to be cached at that exact instant) on essentially every rebuild,
    // which is worse than merely the documented "opportunistic/partial" limitation: it produces
    // visible tile-sized flicker as completeness randomly differs between rebuilds of what is,
    // physically, the same tile (confirmed via live debug logs showing the identical cache key
    // rebuilt seconds apart with a different descendant count each time). A strong cache means
    // once a composite at a given level is resolved (buildOverzoomElevationMosaic requires ALL
    // N*N descendants at that level cached before ever registering one here -- see its doc
    // comment), it stays resolved and stable for the rest of the session, independent of how
    // many times the primary tile itself gets rebuilt. Bounded by kMaxOverzoomMosaicEntries
    // (rasterSource.cpp) to avoid unbounded growth over a long panning session -- see
    // pruneOverzoomMosaicsIfNeeded().
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

    // Get-or-build the shared 3W x 3W mosaic texture for `_tile`: an already-registered live
    // mosaic is returned as-is (its cells are kept current by patchNeighborMosaics());
    // otherwise one is stitched from `_ownTexture` plus whatever of its 8 same-zoom neighbors
    // are already in `m_textures`, with missing neighbors mirror-extrapolated, and registered
    // in m_mosaics for later in-place patching. See stitchElevationMosaic() above for the
    // exact layout. Only ever called when m_buildElevationMosaic is set (elevation source
    // only), on the main thread.
    std::shared_ptr<Texture> buildElevationMosaic(TileID _tile, const std::shared_ptr<Texture>& _ownTexture);

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
    // Level selection is EXACT-target-zoom-first with fallback one level at a time down to
    // z+1: the returned composite is built at the highest level ztry in (z, overzoomTargetZoom]
    // for which either a composite is already registered or ALL 2^(ztry-z) squared descendant
    // tiles are resident in `m_textures` -- partial grids are never stitched (fallback-filled
    // cells would freeze load-order-dependent data into the permanent registry). The returned
    // texture's userData (ElevationMosaicInfo, ownTexture == null) reports the achieved level in
    // compositeZoom; callers needing exactness must compare it to overzoomTargetZoom() -- a
    // fallback-level composite (e.g. z15 for an s=16 tile whose full z16 grid never loads
    // because the far part of the footprint only ever renders z15 terrain) exactly matches the
    // terrain wherever the terrain also fell back, and is far closer than the plain texture
    // everywhere else.
    //
    // Cache-only (never triggers a new fetch; docs/terrain-illumination-plan.md); callers must
    // keep falling back to `_ownTexture` when this returns nullptr. Requires m_keepTextureData.
    // Registered in m_overzoomMosaics keyed by (x, y, z, achieved zoom). Main thread only, like
    // all other m_textures/m_mosaics access.
    std::shared_ptr<Texture> buildOverzoomElevationMosaic(TileID _primary, const std::shared_ptr<Texture>& _ownTexture);

    // Cheap reuse-only lookup (map finds only, no stitching) for m_overzoomMosaics, mirroring
    // getExistingMosaic() above -- safe for proxy/prefetch-only tasks, which must never trigger
    // the (mildly) more expensive full build above. Same exact-target-first, fall-back-one-level
    // selection as buildOverzoomElevationMosaic, restricted to already-registered entries.
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
