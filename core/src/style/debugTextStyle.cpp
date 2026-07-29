#include "style/debugTextStyle.h"

#include "labels/textLabels.h"
#include "map.h"
#include "scene/drawRule.h"
#include "style/textStyleBuilder.h"
#include "tile/tile.h"

namespace Tangram {

class DebugTextStyleBuilder : public TextStyleBuilder {

public:

    DebugTextStyleBuilder(const TextStyle& _style)
        : TextStyleBuilder(_style) {}

    void setup(const Tile& _tile) override;
    void setup(const Marker& _marker, int zoom) override;

    std::unique_ptr<StyledMesh> build() override;

private:
    std::string m_tileID;
    int m_tileZ = 0;

};

void DebugTextStyleBuilder::setup(const Tile& _tile) {
    // zoom_labels (big data-zoom number, e.g. to spot LOD-selection jumps under tilt) takes
    // precedence over tile_bounds' detailed x/y/z/s string when both are on - see build() below.
    if (!Tangram::getDebugFlag(Tangram::DebugFlags::tile_bounds) &&
        !Tangram::getDebugFlag(Tangram::DebugFlags::zoom_labels)) {
        return;
    }

    m_tileID = _tile.getID().toString();
    m_tileZ = _tile.getID().z;

    TextStyleBuilder::setup(_tile);
}

void DebugTextStyleBuilder::setup(const Marker& _marker, int zoom) {
    if (!Tangram::getDebugFlag(Tangram::DebugFlags::tile_bounds)) {
        return;
    }

    m_tileID = "I AM A MARKER";

    TextStyleBuilder::setup(_marker, zoom);
}

std::unique_ptr<StyledMesh> DebugTextStyleBuilder::build() {
    bool zoomLabels = Tangram::getDebugFlag(Tangram::DebugFlags::zoom_labels);
    if (!zoomLabels && !Tangram::getDebugFlag(Tangram::DebugFlags::tile_bounds)) {
        return nullptr;
    }

    TextStyle::Parameters params;
    params.wordWrap = false;
    if (zoomLabels) {
        // Big, high-contrast data-zoom number (just TileID.z, not the full x/y/z/s string) meant
        // to be legible over hillshade/terrain at any tilt, to spot LOD-selection resolution
        // jumps by eye - see DebugFlags::zoom_labels' doc comment (map.h).
        params.text = std::to_string(m_tileZ);
        params.fontSize = 80.f;
        params.fill = 0xff00ffff;    // opaque yellow (ABGR)
        params.strokeColor = 0xff000000;  // opaque black outline
        params.strokeWidth = 4.f;
        params.font = m_style.context()->getFont("sans-serif", "regular", "700", 64 * m_style.pixelScale());
    } else {
        params.text = m_tileID;
        params.fontSize = 30.f;
        params.font = m_style.context()->getFont("sans-serif", "regular", "400", 32 * m_style.pixelScale());
    }

    TextStyleBuilder::LabelAttributes attrib;
    if (!prepareLabel(params, Label::Type::debug, attrib)) {
        return nullptr;
    }

    DrawRule rule({"", 0, {}}, "", 0);
    addLabel(Label::Type::debug, {{ glm::vec3(0.5f, 0.5f, 0.f) }}, params, attrib, rule);

    m_textLabels->setLabels(m_labels);

    std::vector<GlyphQuad> quads(m_quads);
    m_textLabels->setQuads(std::move(quads), m_atlasRefs);

    m_quads.clear();
    m_atlasRefs.reset();
    m_labels.clear();

    return std::move(m_textLabels);
}

std::unique_ptr<StyleBuilder> DebugTextStyle::createBuilder() const {
    return std::make_unique<DebugTextStyleBuilder>(*this);
}

}
