#include <cmath>
#include <algorithm>
#include "data/tileSource.h"

#include "data/formats/geoJson.h"
#include "data/formats/mvt.h"
#include "data/formats/topoJson.h"
#include "data/tileData.h"
#include "data/rasterSource.h"
#include "platform.h"
#include "tile/tileID.h"
#include "tile/tile.h"
#include "tile/tileTask.h"
#include "log.h"
#include "util/geom.h"

#include <atomic>
#include <functional>

namespace Tangram {

TileSource::TileSource(const std::string& _name, std::unique_ptr<DataSource> _sources,
                       ZoomOptions _zoomOptions) :
    m_name(_name),
    m_zoomOptions(_zoomOptions),
    m_sources(std::move(_sources)) {

    static std::atomic<int32_t> s_serial;

    m_id = s_serial++;
}

TileSource::~TileSource() {
    clearData();
}

int32_t TileSource::zoomBiasFromTileSize(int32_t tileSize) {
    const auto BaseTileSize = 256;

    // zero tileSize (log(0) is undefined)
    if (!tileSize) { return 0; }

    if (isPowerOfTwo(tileSize)) {
        return std::log(static_cast<float>(tileSize)/static_cast<float>(BaseTileSize)) / std::log(2);
    }

    LOGW("Illegal tile_size defined. Must be power of 2. Default tileSize of 256px set");
    return 0;
}

const char* TileSource::mimeType() const {
    switch (m_format) {
    case Format::GeoJson: return "application/geo+json";
    case Format::TopoJson: return "application/topo+json";
    case Format::Mvt: return "application/vnd.mapbox-vector-tile";
    }
    assert(false);
    return "";
}

std::shared_ptr<TileTask> TileSource::createTask(TileID _tileId) {
    auto task = std::make_shared<BinaryTileTask>(_tileId, this);

    addRasterTasks(*task);

    return task;
}

void TileSource::addRasterTasks(TileTask& _task) {

    for (auto& source : m_rasterSources) {
        source->addRasterTask(_task);
    }
}

void TileSource::clearData() {

    if (m_sources) { m_sources->clear(); }

    m_generation++;
}

void TileSource::loadTileData(std::shared_ptr<TileTask> _task, TileTaskCb _cb) {

    assert(_task->tileId().z <= m_zoomOptions.maxZoom);
    if (m_sources) {
        if (_task->needsLoading()) {
            if (m_sources->loadTileData(_task, _cb)) {
                _task->startedLoading();
            }
        } else if(_task->hasData() && _task->shareCount < 2) {
            _cb.func(_task);
        }
    }

    for (auto& subTask : _task->subTasks()) {
        subTask->source()->loadTileData(subTask, _cb);
    }
}

// The tile server writes a glacier's name only on its centroid point (poi layer); the glacier
// POLYGON (landuse layer) carries the same `area` but no name. Area labels need the polygon, so
// copy the name over to the polygon with the identical area (docs/label-typography.md).
static void joinGlacierNames(TileData& _data) {
    Layer* poi = nullptr;
    Layer* landuse = nullptr;
    for (auto& layer : _data.layers) {
        if (layer.name == "poi") { poi = &layer; }
        else if (layer.name == "landuse") { landuse = &layer; }
    }
    if (!poi || !landuse) { return; }

    struct Named { double area; const Properties* props; };
    std::vector<Named> named;
    for (auto& f : poi->features) {
        double area;
        std::string nat, name;
        if (f.props.getString("natural", nat) && nat == "glacier" &&
            f.props.getString("name", name) && !name.empty() && f.props.getNumber("area", area)) {
            named.push_back({area, &f.props});
        }
    }
    if (named.empty()) { return; }

    for (auto& f : landuse->features) {
        std::string nat, name;
        double area;
        if (f.geometryType != GeometryType::polygons) { continue; }
        if (!f.props.getString("natural", nat) || nat != "glacier") { continue; }
        if (f.props.getString("name", name) && !name.empty()) { continue; }
        if (!f.props.getNumber("area", area)) { continue; }
        for (const auto& n : named) {
            if (std::abs(n.area - area) <= 1e-6 * std::max(1.0, std::abs(area))) {
                for (const char* key : {"name", "name_en", "osm_id"}) {
                    std::string v;
                    if (n.props->getString(key, v)) { f.props.set(key, v); }
                }
                break;
            }
        }
    }
}

std::shared_ptr<TileData> TileSource::parse(const TileTask& _task) const {
    std::shared_ptr<TileData> data;
    switch (m_format) {
    case Format::TopoJson: data = TopoJson::parseTile(_task, m_id); break;
    case Format::GeoJson: data = GeoJson::parseTile(_task, m_id); break;
    case Format::Mvt:
        data = Mvt::parseTile(_task, m_id, m_mvtLayers.empty() ? nullptr : &m_mvtLayers);
        break;
    default:
        assert(false);
        return nullptr;
    }
    if (data) { joinGlacierNames(*data); }
    return data;
}

void TileSource::cancelLoadingTile(TileTask& _task) {
    // handling of shareCount and subtasks now done in TileManager::TileEntry::clearTask()
    if (m_sources) { m_sources->cancelLoadingTile(_task); }
}

void TileSource::addRasterSource(std::shared_ptr<TileSource> _rasterSource) {
    if (!_rasterSource) {
        LOGE("No raster source");
        return;
    }
    auto rasterSource = dynamic_cast<RasterSource*>(_rasterSource.get());
    if (!rasterSource) {
        LOGE("Not a raster source: %s", _rasterSource->name().c_str());
        return;
    }
    // We limit the parent source by any attached raster source's min/max.
    int32_t rasterMinDisplayZoom = _rasterSource->minDisplayZoom();
    int32_t rasterMaxDisplayZoom = _rasterSource->maxDisplayZoom();
    if (rasterMinDisplayZoom > m_zoomOptions.minDisplayZoom) {
        m_zoomOptions.minDisplayZoom = rasterMinDisplayZoom;
    }
    if (rasterMaxDisplayZoom < m_zoomOptions.maxDisplayZoom) {
        m_zoomOptions.maxDisplayZoom = rasterMaxDisplayZoom;
    }

    m_rasterSources.push_back(rasterSource);
}

}
