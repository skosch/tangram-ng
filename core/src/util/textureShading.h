#pragma once

#include "util/mapProjection.h"

namespace Tangram {

class RasterSource;

// CPU texture-shading sampler -- shared primitive frozen by
// docs/texture-shading-plan.md's "Frozen Interface Contract" (extended by
// docs/label-placement-plan.md for Phase 3's use). Returns the same [0,1]-ish
// contrast-curved "shade" value the hillshade GPU shader would compute at this world
// position (0.5 == neutral/flat, deviation from 0.5 == ridge/canyon response), using
// CPU-side elevation-mosaic buffer reads instead of textureLod. `ok` is set to false if
// the elevation mosaic isn't resident/available here, in which case the returned value
// must be ignored by the caller.
//
// STUB NOTICE (docs/label-placement-plan.md Phase 3): this file is a placeholder. The
// real implementation -- a literal CPU port of the shader formula in
// docs/texture-shading-plan.md (band weight pow(2,-k*alpha), contrast curve
// az/(2*sqrt(1+az*az))+0.5, software box-filter octave pyramid over the 3Wx3W mosaic
// buffer) -- is Phase 2's deliverable, developed concurrently in a separate worktree. It
// is NOT implemented here. This stub always reports the signal unavailable (`ok =
// false`) so that Phase 3's anchor-cost code, which is written against this exact
// signature, degrades gracefully to its vector-proximity term alone. Do not build on top
// of this stub's return value -- it is a placeholder, not an approximation.
//
// Deviation from the Frozen Interface Contract's literal text: the contract specifies
// `const RasterSource&` (a live reference). Phase 3's TextStyleBuilder/PointStyleBuilder
// do not currently have an elevation RasterSource plumbed in anywhere (that plumbing is
// Phase 2/6's responsibility, including the threading-safety question the contract flags
// as its "single biggest open risk"), so this stub takes a nullable `const RasterSource*`
// instead -- the contract's own text permits adjusting types "to preserve this shape"
// when adapting to what's actually available. Phase 3's only caller always passes
// `nullptr`. Phase 6 integration should decide whether to keep the pointer (simplest,
// since Phase 3's call site already tolerates "no source yet") or restore the reference
// once a real source is threaded through -- either way, update this comment and the
// Frozen Interface Contract section of docs/label-placement-plan.md to match whatever is
// decided.
float sampleTextureShading(const RasterSource* elevationSource, ProjectedMeters worldPos, bool& ok);

} // namespace Tangram
