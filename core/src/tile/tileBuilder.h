#pragma once

#include "data/tileSource.h"
#include "labels/labelCollider.h"
#include "scene/styleContext.h"
#include "scene/drawRule.h"
#include "style/style.h"

namespace Tangram {

class DataLayer;
class Tile;
class TileSource;
struct Feature;
struct Properties;
struct TileData;

class TileBuilder {

public:

    explicit TileBuilder(const Scene& _scene);

    StyleBuilder* getStyleBuilder(const std::string& _name);

    void build(Tile& tile, const TileData& _tileData, const TileSource& _source);

    const Scene& scene() const { return m_scene; }

    // For testing
    TileBuilder(const Scene& _scene, StyleContext* _styleContext);

    void init();

private:

    // Determine and apply DrawRules for a @_feature
    // Profiling-only per-build accumulators (TileBuilder is single-threaded per worker).
    bool m_profOn = false;
    uint32_t m_profFeatures = 0, m_profMatched = 0;
    double m_profMatchMs = 0, m_profAddMs = 0;

    void applyStyling(const Feature& _feature, const SceneLayer& _layer);

    const Scene& m_scene;

    std::unique_ptr<StyleContext> m_styleContext;
    DrawRuleMergeSet m_ruleSet;
    int64_t globalsGeneration = 0;

    LabelCollider m_labelLayout;

    fastmap<std::string, std::unique_ptr<StyleBuilder>> m_styleBuilder;

    fastmap<uint32_t, std::shared_ptr<Properties>> m_selectionFeatures;
};

}
