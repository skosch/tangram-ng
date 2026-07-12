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

#pragma tangram: uniforms

varying vec4 v_color;
varying vec2 v_texcoords;
varying float v_sdf_threshold;
varying float v_alpha;
varying float v_sdf_pixel;

#pragma tangram: global

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
        // smooth the signed distance for outlines
        float signed_distance_1_over_2 = 1.0 / (2.0 * signed_distance);
        float smooth_signed_distance = pow(signed_distance, signed_distance_1_over_2);

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

    color.a *= v_alpha * alpha;

    #pragma tangram: color
    #pragma tangram: filter

    gl_FragColor = color;
}
