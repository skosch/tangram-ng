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
// NE, NW, SE, SW) into a single padded 3W x 3W "mosaic" texture, so hillshading/texture-
// shading shaders can read real neighbor data across tile edges instead of clamping at the
// tile boundary (see docs/texture-shading-plan.md, "Frozen Interface Contract" + "Phase 2").
//
// Exact cell layout (this is the part other phases depend on staying stable):
//   `neighbors` is a 3x3 grid flattened row-major: index (dy+1)*3 + (dx+1) for dx, dy in
//   {-1, 0, 1}. Index 4 (dx=0, dy=0) is ignored -- `own` is always used for the center cell.
//   The resulting mosaic buffer places `own` in pixel range [W, 2W) x [W, 2W) (the center
//   third), and each neighbor cell (dx, dy) at pixel range [(dx+1)*W, (dx+2)*W) x
//   [(dy+1)*W, (dy+2)*W) -- the natural grid arrangement, using the same row/column
//   handedness as the raw per-tile buffers themselves (no flipping needed, since TileID x/y
//   offsets and buffer row/col offsets increase in the same direction for every tile of a
//   given source).
//
// A null entry in `neighbors` (tile not cached yet) is synthesized via mirror-extrapolation
// of `own`'s data: flipped vertically for a N/S neighbor, horizontally for an E/W neighbor,
// and both ways for a diagonal neighbor. This is a cheap, byte-only (no pixel decode)
// fallback -- it only ever feeds the coarsest mip levels of the shader-side pyramid, so
// exactness doesn't matter, and it works unmodified for both terrarium-RGB and float
// elevation encodings since it never interprets pixel values.
//
// Returns nullptr if `own` has no CPU-side buffer data (RasterSource::m_keepTextureData must
// be true for the source for this to be usable) or its width/height aren't equal and > 0.
//
// Exposed at namespace scope (rather than as a RasterSource member) so it can be unit-tested
// directly with hand-built Texture objects -- see tests/unit/rasterMosaicTests.cpp.
std::shared_ptr<Texture> stitchElevationMosaic(const std::shared_ptr<Texture>& own,
                                                const std::array<std::shared_ptr<Texture>, 9>& neighbors);

class RasterSource : public TileSource {

    using Cache = std::map<TileID, std::weak_ptr<Texture>>;
    std::shared_ptr<Cache> m_textures;

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

    // Build the 3W x 3W mosaic texture for `_tile` out of `_ownTexture` plus whatever of its
    // 8 same-zoom neighbors are already in `m_textures` (looked up via getTexture()); missing
    // neighbors are mirror-extrapolated. See stitchElevationMosaic() above for the exact
    // layout. Only ever called when m_buildElevationMosaic is set (elevation source only).
    std::shared_ptr<Texture> buildElevationMosaic(TileID _tile, const std::shared_ptr<Texture>& _ownTexture);

public:

    bool m_keepTextureData = false;

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
