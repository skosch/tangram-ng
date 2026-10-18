#pragma once

#include "tile/tileID.h"
#include "util/mapProjection.h"

#include <functional>

namespace Tangram {

class RasterSource;
class Texture;

// CPU port of the GPU texture-shading formula (docs/texture-shading-plan.md's band-weighted
// Laplacian pyramid + Brown's contrast curve, as actually implemented by hillshade.yaml's
// textureShading() GLSL function), for use as a peak-label salience signal from CPU code --
// see docs/label-placement-plan.md "Frozen Interface Contract" and Phase 2 for the full
// writeup of why this exists, what it deliberately does and does not replicate from the live
// shader, and the threading investigation that shaped this file's public API.
//
// Both functions are main-thread only (see sampleTextureShading's own comment below for why) --
// callers must not use this from TextStyleBuilder/PointStyleBuilder::addFeature() or any other
// code that runs on a TileWorker thread (tileWorker.cpp).

// Pure, RasterSource-independent core: evaluate the texture-shading formula directly against
// an already-stitched 3Wp x 3Wp neighbor-ring mosaic buffer (see the cell-layout doc comment on
// stitchElevationMosaic(), rasterSource.h) at `worldPos`, which must fall within `tileId`'s own
// footprint (the mosaic's center cell). Exposed separately from sampleTextureShading() below so
// it can be unit-tested directly with a hand-built Texture, the same way stitchElevationMosaic()
// itself is (tests/unit/rasterMosaicTests.cpp) -- no RasterSource/TileSource machinery required.
//
// Returns 0.5f (the formula's neutral/flat value) if `mosaic` isn't a valid, populated, square
// buffer, or if `worldPos` doesn't actually fall inside `tileId`'s footprint.
float sampleTextureShadingAtMosaic(const Texture& mosaic, TileID tileId, ProjectedMeters worldPos);

// Returns the same [0,1]-ish contrast-curved "shade" value the GPU shader would compute at
// `worldPos`, using the elevation source's already-stitched, already-registered neighbor-ring
// mosaic (RasterSource::getExistingMosaic() -- a cheap, reuse-only lookup, never triggers a new
// stitch) at that position's tile, evaluated at the mosaic's own native (level 0) resolution.
//
// Unlike the live GPU shader (hillshade.yaml's textureShading()), this does NOT track continuous
// view zoom (its fractional base LOD `b`), the zoom-dependent alpha_min blend, or the live
// auto-contrast multiplier: a peak-label priority ranking needs one stable number per peak, not
// a value that changes as the camera moves. See textureShading.cpp for the fixed alpha/contrast
// defaults used instead (matching the shader's current un-auto-adjusted defaults).
//
// `ok` is set false (return value is then meaningless) when the elevation mosaic for this
// position isn't currently resident -- e.g. a flat/non-terrain map style, texture shading
// toggled off, or the tile hasn't stitched a mosaic yet. Callers must fall back to another
// priority signal in that case (see docs/label-placement-plan.md Phase 2's "??" fallback chain).
//
// THREADING: main-thread only, like every other RasterSource::m_textures/m_mosaics access.
// RasterSource::getTexture()/getExistingMosaic() are documented not thread-safe -- safe only
// because RasterTileTask::addRaster() (which builds/patches mosaics) never runs on a
// TileWorker thread (see rasterSource.cpp). This function was investigated for use directly
// from TextStyleBuilder/PointStyleBuilder::addFeature() (which DO run on TileWorker threads,
// per tileWorker.cpp's TileWorker::run()) and found unsafe for that use -- see
// docs/label-placement-plan.md Phase 2 for the investigation and the main-thread integration
// (LabelManager per-frame update) it led to instead.
float sampleTextureShading(RasterSource& elevationSource, ProjectedMeters worldPos, bool& ok);

// --- Peak dominance ---------------------------------------------------------------------
//
// Distance-aware label density needs a per-peak "how much does this stand out" number that
// scales with distance the way apparent size does. Topographic isolation does exactly that:
// the radius within which the peak is the highest point. A knoll that is merely a shoulder of
// its neighbour has an isolation of a few hundred metres and is worth a label only when you
// are standing next to it; the Matterhorn's is ~9 km and is worth one from 90 km away.
//
// Pure function over an elevation callback so it is unit-testable with a synthetic DEM and
// independent of RasterSource/ElevationManager.
//
//  elevAt(pos, elev): fill `elev` (metres) and return true, or return false where no DEM is
//                     available at `pos` (projected meters).
//  groundPerProjected: ground meters per projected (Web Mercator) meter at `pos`, i.e.
//                      cos(latitude). Radii below are ground meters.
//  maxRadius:         ground meters; returned radius saturates here.
struct PeakDominance {
    bool valid = false;      // false: no DEM at the peak position itself, retry later
    bool complete = true;    // false: many ring samples were unavailable (neighbour tiles not
                             //  resident yet), the radius may be an over-estimate
    float radiusMeters = 0.f;
    double peakElev = 0.0;   // reference height used (DEM max in a ~60 m neighbourhood)
};

using PeakElevationFn = std::function<bool(ProjectedMeters pos, double& elev)>;

PeakDominance peakDominanceRadius(const PeakElevationFn& elevAt, ProjectedMeters pos,
                                  double groundPerProjected, float maxRadius = 12000.f);

}
