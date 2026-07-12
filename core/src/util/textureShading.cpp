#include "util/textureShading.h"

namespace Tangram {

// See textureShading.h for the full stub notice. This worktree (Phase 3, salience-aware
// anchor ordering) never has a real elevation source to sample from, so this
// unconditionally reports the signal as unavailable. Phase 6 integration replaces this
// file's body with Phase 2's real CPU port of the texture-shading formula.
float sampleTextureShading(const RasterSource* /*elevationSource*/, ProjectedMeters /*worldPos*/, bool& ok) {
    ok = false;
    return 0.5f; // neutral midpoint; must be ignored by callers since ok == false
}

} // namespace Tangram
