#pragma once

#include "data/tileData.h"
#include "pbf/pbf.hpp"
#include "util/variant.h"

#include <memory>
#include <string>
#include <vector>

namespace Tangram {

class Tile;
class TileTask;
class MapProjection;

namespace Mvt {

    struct Geometry {
        std::vector<Point> coordinates;
        std::vector<int> sizes;
    };

    struct ParserContext {
        ParserContext(int32_t _sourceId) : sourceId(_sourceId){}

        int32_t sourceId;
        std::vector<std::string> keys;
        std::vector<Value> values;
        std::vector<protobuf::message> featureMsgs;
        Geometry geometry;
        // Map Key ID -> Tag values
        std::vector<int> featureTags;
        // Key IDs sorted by Property key ordering
        std::vector<int> orderedKeys;

        int tileExtent = 0;
        int winding = 0;
    };

    enum GeomCmd {
        moveTo = 1,
        lineTo = 2,
        closePath = 7
    };

    void getGeometry(ParserContext& _ctx, protobuf::message _geomIn);

    Feature getFeature(ParserContext& _ctx, protobuf::message _featureIn);

    Layer getLayer(ParserContext& _ctx, protobuf::message _layerIn);

    // `_layerFilter`: if non-null and non-empty, only layers with these names are decoded
// (the rest are skipped without touching their features) - see TileSource::mvtLayers().
std::shared_ptr<TileData> parseTile(const TileTask& _task, int32_t _sourceId,
                                    const std::vector<std::string>* _layerFilter = nullptr);

} // namespace Mvt

} // namespace Tangram
