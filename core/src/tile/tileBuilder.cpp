#include "tile/tileBuilder.h"

#include "debug/profiler.h"
#include "data/properties.h"
#include "data/propertyItem.h"
#include "data/tileSource.h"
#include "gl/mesh.h"
#include "labels/anchorOccupancyGrid.h"
#include "log.h"
#include "scene/dataLayer.h"
#include "scene/scene.h"
#include "selection/featureSelection.h"
#include "tile/tile.h"
#include "util/mapProjection.h"
#include "view/view.h"

namespace Tangram {

TileBuilder::TileBuilder(const Scene& _scene)
    : m_scene(_scene),
      m_styleContext(std::make_unique<StyleContext>()) {
}

TileBuilder::TileBuilder(const Scene& _scene, StyleContext* _styleContext)
    : m_scene(_scene),
      m_styleContext(std::unique_ptr<StyleContext>(_styleContext)) {
}

void TileBuilder::init() {
    m_styleContext->initFunctions(m_scene);

    // Initialize StyleBuilders
    for (const auto& style : m_scene.styles()) {
        if (auto builder = style->createBuilder()) {
            m_styleBuilder[style->getName()] = std::move(builder);
        }
    }
}

StyleBuilder* TileBuilder::getStyleBuilder(const std::string& _name) {
    auto it = m_styleBuilder.find(_name);
    if (it == m_styleBuilder.end()) { return nullptr; }

    return it->second.get();
}

void TileBuilder::applyStyling(const Feature& _feature, const SceneLayer& _layer) {

    // If no rules matched the feature, return immediately
    if (!m_ruleSet.match(_feature, _layer, *m_styleContext)) { return; }

    uint32_t selectionColor = 0;
    bool added = false;

    // For each matched rule, find the style to be used and
    // build the feature with the rule's parameters
    for (auto& rule : m_ruleSet.matchedRules()) {

        StyleBuilder* builder = getStyleBuilder(rule.getStyleName());

        if (!builder) {
            LOGN("Invalid style %s", rule.getStyleName().c_str());
            continue;
        }

        // Apply default draw rules defined for this style
        builder->style().applyDefaultDrawRules(rule);

        if (!m_ruleSet.evaluateRuleForContext(rule, *m_styleContext)) {
            continue;
        }

        bool interactive = false;
        if (rule.get(StyleParamKey::interactive, interactive) && interactive) {
            if (selectionColor == 0) {
                selectionColor = m_scene.featureSelection()->nextColorIdentifier();
            }
            rule.selectionColor = selectionColor;
            rule.featureSelection = m_scene.featureSelection().get();
        } else {
            rule.selectionColor = 0;
        }

        // build outline explicitly with outline style
        const auto& outlineStyleName = rule.findParameter(StyleParamKey::outline_style);
        if (outlineStyleName) {
            auto& styleName = outlineStyleName.value.get<std::string>();
            auto* outlineStyle = getStyleBuilder(styleName);
            if (!outlineStyle) {
                LOGN("Invalid style %s", styleName.c_str());
            } else {
                rule.isOutlineOnly = true;
                outlineStyle->addFeature(_feature, rule);
                rule.isOutlineOnly = false;
            }
        }

        // build feature with style
        added |= builder->addFeature(_feature, rule);
    }

    if (added && (selectionColor != 0)) {
        m_selectionFeatures[selectionColor] = std::make_shared<Properties>(_feature.props);
    }
}

void TileBuilder::build(Tile& tile, const TileData& _tileData, const TileSource& _source) {

    m_selectionFeatures.clear();

    tile.initGeometry(int(m_scene.styles().size()));

    m_styleContext->setTileID(tile.getID());
    // update globals in JS context if changed ... should be doing atomic cmp xchg
    if(globalsGeneration < m_scene.globalsGeneration) {
      globalsGeneration = m_scene.globalsGeneration;
      m_styleContext->setSceneGlobals(m_scene.config()["global"]);
    }

    // Coarse per-tile index of water/road/trail geometry from *all* of this call's raw
    // layers, built once before any individual style's builder runs -- a single style's
    // addFeature() only ever sees the feature(s) of the layer(s) its own draw rules
    // matched, not the tile's other layers, so this shared pre-pass is the only way for
    // (e.g.) peak-label anchor placement to react to nearby water/road geometry.
    // See docs/label-placement-plan.md Phase 3 and labels/anchorOccupancyGrid.h.
    // Only overwrite an existing grid when this call actually found relevant geometry,
    // in case build() is ever invoked multiple times for the same Tile across different
    // sources that don't all carry water/transportation layers.
    if (auto grid = AnchorOccupancyGrid::build(_tileData)) {
        tile.setAnchorOccupancyGrid(std::move(grid));
    }

    for (auto& builder : m_styleBuilder) {
        if (builder.second) { builder.second->setup(tile); }
    }

    {
        PROFILE_SCOPE(Profiler::intern("applyStyling:" + _source.name()));
        for (const auto& datalayer : m_scene.layers()) {

            if (datalayer.source() != _source.name() || !datalayer.enabled()) { continue; }

            for (const auto& collection : _tileData.layers) {

                if (!collection.name.empty()) {
                    const auto& dlc = datalayer.collections();
                    bool layerContainsCollection =
                        std::find(dlc.begin(), dlc.end(), collection.name) != dlc.end();

                    if (!layerContainsCollection) { continue; }
                }

                for (const auto& feat : collection.features) {
                    applyStyling(feat, datalayer);
                }
            }
        }
    }

    PROFILE_SCOPE(Profiler::intern("buildMesh:" + _source.name()));

    for (auto& builder : m_styleBuilder) {

        builder.second->addLayoutItems(m_labelLayout);
    }

    float tileSize = MapProjection::tileSize() * m_scene.pixelScale();

    m_labelLayout.process(tile.getID(), tile.getInverseScale(), tileSize);

    for (auto& builder : m_styleBuilder) {
        tile.setMesh(builder.second->style(), builder.second->build());
    }

    tile.setSelectionFeatures(m_selectionFeatures);
}

}
