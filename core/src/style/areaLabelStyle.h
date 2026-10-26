#pragma once

#include "style/textStyle.h"

#include <memory>

namespace Tangram {

class AreaRegistry;

// Text style for world-space area labels (lakes, glaciers, ... docs/label-typography.md,
// "Round 6"). It does two jobs with one draw rule vocabulary:
//   * built from a TILE (polygon features of the osm-areas source): emits no label at all, it
//     only deposits the polygon fragments, in world meters, into the Scene's AreaRegistry;
//   * built from a MARKER (AreaLabelManager's spine features, one per label part): lays the
//     name out along the spine window with the right tracking.
class AreaLabelStyle : public TextStyle {

public:
    AreaLabelStyle(std::string _name, bool _sdf = true) : TextStyle(_name, _sdf) {}
    void build(const Scene& _scene) override;
    std::unique_ptr<StyleBuilder> createBuilder() const override;

    std::shared_ptr<AreaRegistry> registry() const { return m_registry; }

private:
    std::shared_ptr<AreaRegistry> m_registry;
};

}
