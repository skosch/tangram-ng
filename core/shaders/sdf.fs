#pragma tangram: extensions

#ifdef GL_ES
precision highp float;
#define LOWP lowp
#else
#define LOWP
#endif

#pragma tangram: defines

uniform sampler2D u_tex;
uniform vec3 u_map_position;
uniform vec4 u_tile_origin;
uniform vec2 u_resolution;
uniform float u_time;
uniform float u_meters_per_pixel;
uniform float u_device_pixel_ratio;
uniform float u_max_stroke_width;
uniform LOWP int u_pass;
// Background-adaptive halo (docs/label-placement-plan.md Phase 4): snapshot of the default
// framebuffer's color contents, blitted just before text styles draw (Scene::render) and
// bound here by TextStyle::onBeginDrawFrame. Sampled per-fragment in the stroke pass below
// so halo visibility reacts to what's actually underneath, not a per-label flag.
uniform sampler2D u_background_tex;
// Halo fade-out thresholds, set unconditionally in TextStyle::onBeginDrawFrame - see
// TextStyle::UniformBlock::uHaloLuminanceDark's comment for why these are hardcoded here
// rather than injected via the "#pragma tangram: uniforms" block below (this file is
// shared by several distinct TextStyle-derived Style objects, not just one).
uniform float u_halo_luminance_dark;
uniform float u_halo_luminance_light;

#pragma tangram: uniforms

varying vec4 v_color;
varying vec2 v_texcoords;
varying float v_sdf_threshold;
varying float v_alpha;
varying float v_sdf_pixel;

#pragma tangram: global

// TEMP DIAGNOSTIC (docs/label-halo-stroke-alignment-investigation.md): visualizes fill-vs-
// stroke SDF alignment for investigating the reported peak-label halo "shadow above/below
// the font" artifact. 0 = fully original behavior, bit-for-bit unchanged (this is the only
// line that needs editing to enable it; nothing else in this file's normal code path is
// touched). Set to 1 and rebuild to enable. Safe/reversible: a single manually-edited
// compile-time constant, no new uniforms, no YAML/C++ changes, no effect on the default
// (0) build.
#define TANGRAM_DEBUG_HALO_ALIGN 0

// TEMP DIAGNOSTIC (round 2, same doc): live report from Sebastian ruled out a positional
//  mismatch (TANGRAM_DEBUG_HALO_ALIGN's red/cyan test showed a concentric ring, not an
//  offset one) but described the ring's own COVERAGE varying wildly and anisotropically
//  with sub-pixel pan (flooding in on one side, nearly vanishing on another, direction
//  flipping with a 1px pan). Leading suspect: the stroke-only `pow(signed_distance,
//  1/(2*signed_distance))` transform just below is numerically explosive for small
//  signed_distance (the exponent approaches infinity as signed_distance -> 0, which is
//  exactly the regime the stroke pass samples in, since it deliberately reads further
//  from the glyph's core outline than the fill pass does) - an ordinary sub-pixel shift
//  in the bilinear-filtered SDF texture read is enough of a change in the "base" to swing
//  a pow() with a huge exponent wildly. Set to 1 and rebuild to bypass ONLY this transform
//  (falls back to the same plain smoothstep(signed_distance) the fill pass already uses),
//  keeping everything else (background-adaptive halo, real colors) intact, to see the
//  actual visual result rather than a red/cyan debug view. ANSWERED: bypassing pow() made no
//  difference (corruption persisted) - back to 0 (real pow() path) for round 3 below.
#define TANGRAM_DEBUG_SKIP_STROKE_POW 0

// TEMP DIAGNOSTIC (round 3, same doc): rounds 1-2 ruled out a positional mismatch and the
//  stroke-only pow() transform. Since fill (u_pass==0) stays rock solid while stroke (u_pass==1)
//  corrupts, and both read the exact same texture at the exact same v_texcoords, the remaining
//  question is whether the raw SDF TEXTURE DATA itself is unstable at the stroke's sampling
//  depth, or whether the data is fine and something downstream (e.g. adjacent glyphs' stroke
//  quads - expanded past their tight fill bbox by fonsSetBlur's quad padding - overlapping and
//  alpha-double-blending) is the real source. Set to 1 to recolor the final pixel by the raw
//  signed_distance sample, tagged by pass (stroke -> red channel, fill -> green channel), while
//  leaving the REAL alpha computation (threshold, pow, background halo, v_alpha) completely
//  untouched - unlike round 3's first attempt (which forced alpha=1.0 and, since fill/stroke
//  share the exact same quad geometry and fill draws strictly after stroke, made fill always
//  fully overdraw stroke everywhere - explaining why NO red was ever visible, only a green-to-
//  black gradient where neighboring glyphs' quads overlap in draw order). Preserving the real
//  alpha lets stroke actually show through wherever production rendering would show it.
#define TANGRAM_DEBUG_RAW_SDF 0

void main(void) {

    vec4 color = v_color;

    float signed_distance = texture2D(u_tex, v_texcoords).r;

    // - At the glyph outline alpha is 0.5
    //
    // - The sdf-radius is 3.0px, i.e. within 3px distance
    //   from the outline alpha is in the range (0.5 -> 0.0)
    //
    // - 0.5 pixel threshold (to both sides of the outline)
    //   plus 0.25 for a bit of smoothness
    //
    //   ==> (0.5 / 3.0) * (0.5 + 0.25) == 0.1245
    //   This value is added to sdf_threshold to antialias
    //   the outline within one pixel for the *unscaled* glyph.
    //
    // - sdf_scale == fontScale / glyphScale:
    //   When the glyph is scaled down, 's' must be increased
    //   (used to interpolate 1px of the scaled glyph around v_sdf_threshold)

    float add_smooth = 0.25;
    float filter_width = (v_sdf_pixel * (0.5 + add_smooth));

    float start = max(v_sdf_threshold - filter_width, 0.0);
    float end = v_sdf_threshold + filter_width;

    float alpha;

    if (u_pass == 0) {
        alpha = smoothstep(start, end, signed_distance);
    } else {
#if TANGRAM_DEBUG_SKIP_STROKE_POW
        float smooth_signed_distance = signed_distance;
#else
        // smooth the signed distance for outlines
        float signed_distance_1_over_2 = 1.0 / (2.0 * signed_distance);
        float smooth_signed_distance = pow(signed_distance, signed_distance_1_over_2);
#endif

        alpha = smoothstep(start, end, smooth_signed_distance);

        // Background-adaptive halo (docs/label-placement-plan.md Phase 4): fade the halo
        // out over light backgrounds (e.g. snow, pale fill) and keep it fully visible over
        // dark ones (forest, water, shadow), decided per-fragment from the actual composited
        // pixel underneath - not a per-label flag, so a single label's halo can visibly vary
        // in strength across its own extent (e.g. straddling a shadow edge). u_halo_luminance_dark
        // / u_halo_luminance_light are gui_variables-tunable (stylus-osm.yaml, same pattern as
        // hillshade's u_texture_shading_* - see docs/label-placement-plan.md Phase 4) rather
        // than hardcoded, since the exact falloff is a human visual-tuning call.
        vec3 bg = texture2D(u_background_tex, gl_FragCoord.xy / u_resolution).rgb;
        float bg_luminance = dot(bg, vec3(0.299, 0.587, 0.114));
        float halo_visibility = 1.0 - smoothstep(u_halo_luminance_dark, u_halo_luminance_light, bg_luminance);
        alpha *= halo_visibility;
    }

#if TANGRAM_DEBUG_HALO_ALIGN
    // TEMP DIAGNOSTIC: bypass the background-halo multiply, the label fade (v_alpha), and
    // the #pragma color/filter blocks entirely, so the only thing left on screen is each
    // pass's own alpha test result at full opacity - fill in solid red, stroke in solid
    // cyan. Take a frozen screenshot over a settled (non-fading, non-loading) peak label
    // and zoom in: if the cyan ring's inner boundary doesn't hug the red glyph's edge
    // symmetrically top-to-bottom (e.g. a visible gap on one side, overlap on the other),
    // that is a real, directly-visible geometric mismatch. If instead the cyan ring looks
    // concentric with the red glyph but merely uneven/jagged in thickness top vs bottom,
    // that supports this doc's leading hypothesis (SDF peripheral-sampling asymmetry made
    // visible by the ring's thinness) over an actual position bug.
    if (u_pass == 0) {
        gl_FragColor = vec4(1.0, 0.0, 0.0, alpha);
    } else {
        gl_FragColor = vec4(0.0, 1.0, 1.0, alpha);
    }
    return;
#endif

#if TANGRAM_DEBUG_RAW_SDF
    color = (u_pass == 0) ? vec4(0.0, signed_distance, 0.0, 1.0) : vec4(signed_distance, 0.0, 0.0, 1.0);
#endif

    color.a *= v_alpha * alpha;

    #pragma tangram: color
    #pragma tangram: filter

    gl_FragColor = color;
}
