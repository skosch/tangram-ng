#pragma once

#include "gl/dynamicQuadMesh.h"
#include "style/style.h"
#include "labels/labelProperty.h"
#include "labels/textLabel.h"
#include "util/hash.h"

#include <memory>
#include <vector>
#include <string>

#ifdef FONTCONTEXT_STB
namespace Tangram {
    using FontHandle = size_t;
}
#else
namespace alfons { class Font; }
namespace Tangram {
    using FontHandle = std::shared_ptr<alfons::Font>;
}
#endif

namespace Tangram {

class FontContext;
struct Properties;

class TextStyle : public Style {

public:

    constexpr static uint32_t DEFAULT_TEXT_WRAP_LENGTH = 15;

    struct Parameters {
        FontHandle font;
        std::string text = "";
        std::string textLeft = "";
        std::string textRight = "";
        bool interactive = false;
        uint32_t fill = 0xff000000;
        uint32_t strokeColor = 0xffffffff;
        float strokeWidth = 0.0f;
        float fontSize = 12.0f;
        Label::Options labelOptions;
        bool wordWrap = true;
        uint32_t maxLines = 0;
        uint32_t maxLineWidth = DEFAULT_TEXT_WRAP_LENGTH;

        TextLabelProperty::Transform transform = TextLabelProperty::Transform::none;
        TextLabelProperty::Align align = TextLabelProperty::Align::none;

        float fontScale = 1;
        float lineSpacing = 0;

        // Typographic tracking, in em (0.1 == 10% of the font size added between every
        // pair of adjacent characters). docs/label-typography.md.
        float letterSpacing = 0.f;
        // Optional "fit" tracking for labels that follow a path/area: if > 0 the label is
        // tracked out so that it spans this fraction of the path length (never tighter
        // than `letterSpacing`, never wider than `letterSpacingMax` em).
        float letterSpacingFit = 0.f;
        float letterSpacingMax = 0.6f;
        // Gaussian smoothing radius (px) applied to line/area label paths, so labels follow
        // the large-scale shape of a ridge, valley or river and not its digitizing jitter.
        float pathSmoothing = 0.f;
        // Perpendicular shift (px, positive = towards the top of the screen at rotation 0)
        // applied to a line label's path: sets a river name beside the river instead of on it.
        float pathOffset = 0.f;

        bool hasComplexShaping = false;
    };

    auto& context() const { return m_context; }

protected:

    bool m_sdf;

    FontContext* m_context;

    struct UniformBlock {
        UniformLocation uTexScaleFactor{"u_uv_scale_factor"};
        UniformLocation uTex{"u_tex"};
        UniformLocation uOrtho{"u_ortho"};
        UniformLocation uPass{"u_pass"};
        UniformLocation uMaxStrokeWidth{"u_max_stroke_width"};
        // Background-adaptive halo (docs/label-placement-plan.md Phase 4): sampler for the
        // per-frame default-framebuffer color snapshot (RenderState::backgroundTexture),
        // read by sdf.fs's stroke pass to fade the halo out over light backgrounds. Only
        // bound/used on m_mainUniforms - the selection pass doesn't draw halos.
        UniformLocation uBackgroundTex{"u_background_tex"};
        // Halo fade-out thresholds (docs/label-placement-plan.md Phase 4) - background
        // luminance at/below uHaloLuminanceDark keeps the halo at full strength, at/above
        // uHaloLuminanceLight the halo is fully invisible. Set as plain uniforms (not via
        // the YAML "shaders: uniforms:" gui_variables mechanism the rest of this feature
        // otherwise uses) because sdf.fs is shared by multiple distinct TextStyle-derived
        // Style objects beyond the "text" built-in (contour-labels, and one implicitly
        // created per point style that has a linked text child, e.g. "points"/"poi-points"/
        // "track-markers"/"loc-points") - the YAML mechanism only configures one Style
        // instance per name, so per-style declarations don't reliably reach all of them and
        // several ended up failing to compile ("undeclared") in Phase 6 integration testing.
        // Hardcoding both the declaration (sdf.fs) and the value (onBeginDrawFrame) is what
        // guarantees every TextStyle-derived instance compiles, at the cost of losing live
        // GUI-slider tuning for now - see onBeginDrawFrame for the values.
        UniformLocation uHaloLuminanceDark{"u_halo_luminance_dark"};
        UniformLocation uHaloLuminanceLight{"u_halo_luminance_light"};
    } m_mainUniforms, m_selectionUniforms;

    mutable std::vector<std::unique_ptr<DynamicQuadMesh<TextVertex>>> m_meshes;

public:

    TextStyle(std::string _name, bool _sdf = false, Blending _blendMode = Blending::overlay,
              GLenum _drawMode = GL_TRIANGLES, bool _selection = true);

    void constructVertexLayout() override;
    void constructShaderProgram() override;

    /* Create the LabelMeshes associated with FontContext GlyphTexture<s>
     * No GL involved, called from Tangram::update()
     */
    virtual void onBeginUpdate() override;

    /* Upload the buffers of the text batches
     * Upload the texture atlases
     */
    virtual void onBeginFrame(RenderState& rs) override;

    /* Performs the actual drawing of the meshes in two passes
     * - First pass if signed distance field is on, draw outlines
     * - Second pass, draw the inner glyph pixels
     */
    virtual void onBeginDrawFrame(RenderState& rs, const View& _view) override;

    virtual void onBeginDrawSelectionFrame(RenderState& rs, const View& _view) override;

    virtual bool draw(RenderState& rs, const Tile& _tile) override { return false; }

    virtual bool draw(RenderState& rs, const Marker& _marker) override { return false; }

    std::unique_ptr<StyleBuilder> createBuilder() const override;

    DynamicQuadMesh<TextVertex>& getMesh(size_t id) const;

    auto& getMeshes() const { return m_meshes; }

    virtual size_t dynamicMeshSize() const override;

    virtual ~TextStyle() override;

    void setFontContext(FontContext& _fontContext) {
        m_context = &_fontContext;
    }

private:

    const std::string& applyTextSource(const Parameters& _parameters, const Properties& _props) const;

};

}

namespace std {
    template <>
    struct hash<Tangram::TextStyle::Parameters> {
        size_t operator() (const Tangram::TextStyle::Parameters& p) const {
            std::hash<Tangram::Label::Options> optionsHash;
            std::size_t seed = 0;
            // TODO
            //hash_combine(seed, p.fontId);
            hash_combine(seed, p.text);
            hash_combine(seed, p.fill);
            hash_combine(seed, p.strokeColor);
            hash_combine(seed, p.strokeWidth);
            hash_combine(seed, p.fontSize);
            hash_combine(seed, p.letterSpacing);
            hash_combine(seed, p.wordWrap);
            hash_combine(seed, p.maxLineWidth);
            hash_combine(seed, int(p.transform));
            hash_combine(seed, int(p.align));
            hash_combine(seed, optionsHash(p.labelOptions));
            return seed;
        }
    };
}
