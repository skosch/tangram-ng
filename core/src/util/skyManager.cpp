#include "util/skyManager.h"
#include "gl/mesh.h"
#include "gl/renderState.h"
#include "gl/shaderProgram.h"
#include "view/view.h"

// probably should move these to separate files in core/shaders

static const char* sky_vs = R"RAW_GLSL(
//#pragma tangram: extensions

#ifdef GL_ES
precision mediump float;
#endif

//#pragma tangram: defines
//#pragma tangram: uniforms

attribute vec2 a_position;

varying vec4 v_position;

//#pragma tangram: global

void main() {
    // full-screen quad: below the horizon row it paints the horizon color, so the wedges left
    //  between a bent (sagging) terrain edge and a straight horizon line are never background
    vec4 pos = vec4(a_position, 0., 1.);
    v_position = pos;
    gl_Position = pos;
}
)RAW_GLSL";

static const char* sky_fs = R"RAW_GLSL(
//#pragma tangram: extensions

#ifdef GL_ES
precision highp float;
#endif

#pragma tangram: defines

uniform vec4 u_horizon_color;
uniform vec4 u_zenith_color;
uniform float u_horizon_y;     // NDC y of the horizon row

#pragma tangram: uniforms

varying vec4 v_position;

//#pragma tangram: material
//#pragma tangram: lighting
//#pragma tangram: raster
#pragma tangram: global

void main(void) {

    #pragma tangram: setup

    #pragma tangram: color

    // 0 at the horizon row .. 1 at the top of the screen; the exponent < 1 reaches the deep zenith blue
    //  quickly (a Berann sky is dark overhead and pale only in a thin band at the horizon)
    float t = clamp((v_position.y - u_horizon_y) / max(1.0 - u_horizon_y, 1e-3), 0.0, 1.0);
    t = pow(t, 0.6);
    vec4 color = mix(u_horizon_color, u_zenith_color, t);

    #pragma tangram: filter

    gl_FragColor = color;
}
)RAW_GLSL";

namespace Tangram {

struct SkyVertex {
    SkyVertex(float x, float y) : position(x, y) {}
    glm::vec2 position;
};


SkyManager::SkyManager() : m_shaderSource(std::make_unique<ShaderSource>()) {}

void SkyManager::setupUniforms(RenderState& rs, const View& _view)
{
    m_shaderProgram->setUniformf(rs, m_uniforms.uTime, rs.frameTime());
    m_shaderProgram->setUniformf(rs, m_uniforms.uDevicePixelRatio, _view.pixelScale());
    m_shaderProgram->setUniformf(rs, m_uniforms.uResolution, _view.getWidth(), _view.getHeight());

    // probably want to use something like StyleUniform for these
    m_shaderProgram->setUniformf(rs, m_uniforms.uHorizonColor, _view.skyHorizonColor());
    m_shaderProgram->setUniformf(rs, m_uniforms.uZenithColor, _view.skyZenithColor());
}

void SkyManager::buildProgram()
{
    m_vertexLayout = std::shared_ptr<VertexLayout>(new VertexLayout({
        {"a_position", 2, GL_FLOAT, false, 0},
    }));

    m_shaderSource->setSourceStrings(sky_fs, sky_vs);

    // add defines ...
    //m_shaderSource->addSourceBlock("defines", "#define TANGRAM_SKY\n", false);

    std::string vertSrc = m_shaderSource->buildVertexSource();
    std::string fragSrc = m_shaderSource->buildFragmentSource();

    m_shaderProgram = std::make_unique<ShaderProgram>(vertSrc, fragSrc, m_vertexLayout.get());
    m_shaderProgram->setDescription("SkyManager");

    m_shaderSource.reset();
}

void SkyManager::buildMesh()  //float x0, float y0, float x1, float y1)
{
    MeshData<SkyVertex> meshData({ 0, 1, 2, 3 }, {{-1, -1}, {1, -1}, {-1, 1}, {1, 1}});
    //{{-1, -1}, {1, -1}, {-1, 1}, {1, 1}}); //{{x0, y0}, {x1, y0}, {x0, y1}, {x1, y1}});

    // Create mesh from vertices.
    auto mesh = std::make_unique<Mesh<SkyVertex>>(m_vertexLayout, GL_TRIANGLE_STRIP);
    mesh->compile(meshData);
    m_mesh = std::move(mesh);
    //m_meshData.clear();
}


void SkyManager::draw(RenderState& rs, View& _view)
{
    float horizon = _view.bentHorizonScreenPosition()/_view.getHeight();
    if(horizon < 0 || horizon > 1) { return; }
    if (!m_shaderProgram) { buildProgram(); }
    if (!m_mesh) { buildMesh(); }

    setupUniforms(rs, _view);
    // u_horizon_y was the quad's shift, now the horizon row in NDC (same 1 - 2*y expression)
    m_shaderProgram->setUniformf(rs, m_uniforms.uHorizonY, 1 - 2*horizon);

    rs.blending(GL_FALSE);
    //rs.blendingFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    rs.depthTest(GL_FALSE);
    rs.depthMask(GL_FALSE);

    m_mesh->draw(rs, *m_shaderProgram);
}

}
