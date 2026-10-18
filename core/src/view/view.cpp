#include "view/view.h"
#include <cstdlib>

#include "log.h"
#include "scene/stops.h"
#include "util/elevationManager.h"

#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtx/rotate_vector.hpp"
#include <algorithm>
#include <cmath>

#define MAX_LOD 6

namespace Tangram {

double invLodFunc(double d) {
    return exp2(d) - 1.0;
}

View::View(int _width, int _height) :
    m_obliqueAxis(0, 1),
    m_width(0),
    m_height(0),
    m_type(CameraType::perspective),
    m_dirtyMatrices(true),
    m_dirtyTiles(true),
    m_changed(false) {

    auto bounds = MapProjection::mapProjectedMetersBounds();
    m_constraint.setLimitsY(bounds.min.y, bounds.max.y);

    setViewport(0, 0, _width, _height);
    setZoom(m_zoom);
    setPosition(0.0, 0.0);
}

void View::setPixelScale(float _pixelsPerPoint) {

    m_pixelScale = _pixelsPerPoint;
    m_dirtyMatrices = true;
    m_dirtyTiles = true;
    m_dirtyWorldBoundsMinZoom = true;

}

void View::setCamera(const Camera& _camera) {
    setCameraType(_camera.type);

    switch (_camera.type) {
    case CameraType::perspective:
        setVanishingPoint(_camera.vanishingPoint.x, _camera.vanishingPoint.y);
        if (_camera.fovStops) {
            setFieldOfViewStops(_camera.fovStops);
        } else {
            setFieldOfView(_camera.fieldOfView);
        }
        break;
    case CameraType::isometric:
        setObliqueAxis(_camera.obliqueAxis.x, _camera.obliqueAxis.y);
        break;
    case CameraType::flat:
        break;
    }

    if (_camera.maxTiltStops) {
        setMaxPitchStops(_camera.maxTiltStops);
    } else {
        setMaxPitch(_camera.maxTilt);
    }

    // reset zoom
    m_zoom = m_baseZoom;
}

void View::setCameraType(CameraType _type) {

    m_type = _type;
    m_dirtyMatrices = true;
    m_dirtyTiles = true;

}

ViewState View::state() const {

    return {
        //m_changed,
        glm::dvec2(m_pos),
        m_zoom,
        m_type == CameraType::perspective ? m_pos.z/exp2(-m_baseZoom) : powf(2.f, m_zoom),
        m_zoom - std::floor(m_zoom),
        glm::vec2(m_vpWidth, m_vpHeight),
        (float)MapProjection::tileSize() * m_pixelScale
    };
}

void View::setViewport(int _x, int _y, int _width, int _height) {

    m_vpX = _x;
    m_vpY = _y;
    m_vpWidth = std::max(_width, 1);
    m_vpHeight = std::max(_height, 1);
    m_aspect = (float)m_vpWidth/ (float)m_vpHeight;
    m_dirtyMatrices = true;
    m_dirtyTiles = true;
    m_dirtyWorldBoundsMinZoom = true;

    // Screen space orthographic projection matrix, top left origin, y pointing down
    m_orthoViewport = glm::ortho(0.f, (float)m_vpWidth, (float)m_vpHeight, 0.f, -1.f, 1.f);

}

void View::setFieldOfView(float radians) {

    m_fov = radians;
    m_fovStops = nullptr;
    m_dirtyMatrices = true;
    m_dirtyTiles = true;

}

void View::setFieldOfViewStops(std::shared_ptr<Stops> stops) {

    m_fovStops = stops;
    m_dirtyMatrices = true;
    m_dirtyTiles = true;

}

float View::getFieldOfView() const {

    if (m_fovStops) {
        return m_fovStops->evalFloat(m_zoom);
    }
    return m_fov;

}

void View::setFocalLength(float length) {

    setFieldOfView(focalLengthToFieldOfView(length));

}

void View::setFocalLengthStops(std::shared_ptr<Stops> stops) {

    for (auto& frame : stops->frames) {
        float length = frame.value.get<float>();
        frame.value = focalLengthToFieldOfView(length);
    }
    setFieldOfViewStops(stops);

}

float View::getFocalLength() const {

    return fieldOfViewToFocalLength(getFieldOfView());

}

void View::setMinZoom(float minZoom) {

    m_minZoom = std::max(minZoom, 0.f);
    m_maxZoom = std::max(minZoom, m_maxZoom);
    // Set the current zoom again to validate it.
    setZoom(m_zoom);
}

void View::setMaxZoom(float maxZoom) {

    m_maxZoom = std::min(maxZoom, 20.5f);
    m_minZoom = std::min(maxZoom, m_minZoom);
    // Set the current zoom again to validate it.
    setZoom(m_zoom);
}

void View::setMaxPitch(float degrees) {

    m_maxPitch = degrees;
    m_maxPitchStops = nullptr;
    setPitch(m_pitch);

}

void View::setMaxPitchStops(std::shared_ptr<Stops> stops) {

    m_maxPitchStops = stops;
    setPitch(m_pitch);

}

float View::getMaxPitch() const {

    if (m_maxPitchStops) {
        return m_maxPitchStops->evalFloat(m_zoom);
    }
    return m_maxPitch;

}

void View::setConstrainToWorldBounds(bool constrainToWorldBounds) {

    m_worldBoundsMinZoom = 0.f;
    m_constrainToWorldBounds = constrainToWorldBounds;
    if (m_constrainToWorldBounds) {
        applyWorldBounds();
    }

}

void View::setPosition(double _x, double _y) {
    // Wrap horizontal position around the 180th meridian, which corresponds to +/- HALF_CIRCUMFERENCE meters.
    m_pos.x = _x - std::round(_x / MapProjection::EARTH_CIRCUMFERENCE_METERS) * MapProjection::EARTH_CIRCUMFERENCE_METERS;
    // Clamp vertical position to the span of the map, which is +/- HALF_CIRCUMFERENCE meters.
    m_pos.y = glm::clamp(_y, -MapProjection::EARTH_HALF_CIRCUMFERENCE_METERS, MapProjection::EARTH_HALF_CIRCUMFERENCE_METERS);
    m_dirtyTiles = true;
    if (m_elevationManager) { m_dirtyMatrices = true; }  // elevation under pos and eye changed
    if (m_constrainToWorldBounds) {
        applyWorldBounds();
    }
}

void View::setZoom(float _z, bool setBaseZoom) {
    // ensure zoom value is allowed
    _z = glm::clamp(_z, m_minZoom, m_maxZoom);

    if (!m_elevationManager) {  //||m_baseZoom == prevZoom)
        m_baseZoom = m_zoom = _z;
    } else if (setBaseZoom) {
        m_zoom = -std::log2(std::exp2(-_z) - std::exp2(-m_baseZoom) + std::exp2(-m_zoom));
        m_baseZoom = _z;
    } else {
        m_baseZoom = -std::log2(std::exp2(-_z) - std::exp2(-m_zoom) + std::exp2(-m_baseZoom));
        m_zoom = _z;
    }

    m_dirtyMatrices = true;
    m_dirtyTiles = true;
    if (m_constrainToWorldBounds) {
        applyWorldBounds();
    }
}

void View::setYaw(float _rad) {

    m_yaw = glm::mod(_rad, (float)TWO_PI);
    m_dirtyMatrices = true;
    m_dirtyTiles = true;

}

void View::setPitch(float _rad) {

    m_pitch = _rad;
    m_dirtyMatrices = true;
    m_dirtyTiles = true;

}

void View::translate(double _dx, double _dy) {

    setPosition(m_pos.x + _dx, m_pos.y + _dy);

}

void View::applyWorldBounds() {
    // Approximate the view diameter in pixels by taking the maximum dimension.
    double viewDiameterPixels = std::fmax(getWidth(), getHeight()) / pixelScale();
    if (m_dirtyWorldBoundsMinZoom) {
        // Approximate the minimum zoom that keeps with view span within the drawable projection area. [1]
        m_worldBoundsMinZoom = static_cast<float>(std::log(viewDiameterPixels / MapProjection::tileSize() + 2) / std::log(2));
        m_dirtyWorldBoundsMinZoom = false;
    }
    if (m_zoom < m_worldBoundsMinZoom) {
        m_baseZoom = m_zoom = m_worldBoundsMinZoom;
    }
    // Constrain by moving map center to keep view in bounds.
    m_constraint.setRadius(0.5 * viewDiameterPixels / pixelsPerMeter());
    m_pos.x = m_constraint.getConstrainedX(m_pos.x);
    m_pos.y = m_constraint.getConstrainedY(m_pos.y);
}

bool View::update() {

    // ensure valid zoom
    if (!m_elevationManager && m_zoom != m_baseZoom) {
        setZoom(m_zoom);
    }

    // updateMatrices() sets m_changed = true
    if (m_dirtyMatrices) { updateMatrices(); }

    if (m_dirtyTiles) {
        m_changed = true;
        m_dirtyTiles = false;
    }

    return std::exchange(m_changed, false);
}

glm::dmat2 View::getBoundsRect() const {
    double hw = m_width * 0.5;
    double hh = m_height * 0.5;
    return glm::dmat2(m_pos.x - hw, m_pos.y - hh, m_pos.x + hw, m_pos.y + hh);
}

void View::setPadding(EdgePadding padding) {
    if (padding != m_padding) {
        m_padding = padding;
        m_dirtyMatrices = true;
    }
}

glm::vec2 View::normalizedWindowCoordinates(float _x, float _y) const {
    return { _x / m_vpWidth, 1.0 - _y / m_vpHeight };
}

glm::dvec2 View::screenToGroundPlane(float _screenX, float _screenY, float _elev, double* distOut) {

    if (m_dirtyMatrices) { updateMatrices(); }

    // Cast a ray and find its intersection with the z = _elev plane,
    // following the technique described here: http://antongerdelan.net/opengl/raycasting.html

    glm::dvec4 target_clip = { 2. * _screenX / m_vpWidth - 1., 1. - 2. * _screenY / m_vpHeight, -1., 1. };
    glm::dvec4 target_world = m_invViewProj * target_clip;
    target_world /= target_world.w;

    glm::dvec4 origin_world;
    switch (m_type) {
        case CameraType::perspective:
            origin_world = glm::dvec4(m_eye, 1);
            break;
        case CameraType::isometric:
        case CameraType::flat:
            origin_world = m_invViewProj * (target_clip * glm::dvec4(1, 1, 0, 1));
            break;
    }

    glm::dvec4 ray_world = target_world - origin_world;

    double t = 0; // Distance along ray to ground plane
    if (ray_world.z != 0.f) {
        t = -(origin_world.z - _elev) / ray_world.z;
    }

    ray_world *= std::abs(t);

    // Determine the maximum distance from the view position at which tiles can be drawn; If the projected point
    // is farther than this maximum or if the point is above the horizon (t < 0) then we set the distance of the
    // point to always be this maximum distance.
    double maxTileDistance = invLodFunc(MAX_LOD + 1) * 2.0 * MapProjection::EARTH_HALF_CIRCUMFERENCE_METERS * pow(2, -m_zoom);
    double rayDistanceXY = sqrt(ray_world.x * ray_world.x + ray_world.y * ray_world.y);
    if (rayDistanceXY > maxTileDistance || t < 0) {
        ray_world *= maxTileDistance / rayDistanceXY;
    }

    if (distOut) { *distOut = t; }

    return glm::dvec2(ray_world) + glm::dvec2(origin_world);
}

float View::pixelsPerMeter() const {

    double metersPerTile = MapProjection::EARTH_CIRCUMFERENCE_METERS * exp2(-m_zoom);
    return MapProjection::tileSize() / metersPerTile;
}

float View::focalLengthToFieldOfView(float length) {
    return 2.f * atanf(1.f / length);
}

float View::fieldOfViewToFocalLength(float radians) {
    return 1.f / tanf(radians / 2.f);
}

glm::dvec2 View::positionToLookAt(glm::dvec2 target, bool& elevOk) {
    elevOk = true;
    float elev = m_elevationManager ? m_elevationManager->getElevation(target, elevOk) : 0;
    glm::vec2 center = glm::vec2(m_vpWidth, m_vpHeight)/2.0f;
    if (!m_padding.isVisible) {
        center += glm::vec2(m_padding.right - m_padding.left, m_padding.top - m_padding.bottom)/2.0f;
    }
    return target - screenToGroundPlane(center.x, center.y, elev);
}

void View::updateMatrices() {

    // viewport height in world space is such that each tile is [m_pixelsPerTile] px square in screen space
    double screenTileSize = MapProjection::tileSize() * m_pixelScale;
    double worldHeight = m_vpHeight * MapProjection::EARTH_CIRCUMFERENCE_METERS / screenTileSize;

    // set vertical field-of-view, applying intended FOV to wider dimension
    float fovy = m_aspect > 1 ? getFieldOfView()/m_aspect : getFieldOfView();
    if (fovy <= 0 || fovy >= PI) { assert(false); fovy = PI/4; }  // invalid FOV causes NaNs

    double worldToCameraHeight = worldHeight * 0.5 / tan(fovy * 0.5);

    // set camera z to produce desired viewable area
    m_pos.z = exp2(-m_baseZoom) * worldToCameraHeight;

    //m_zoom = m_baseZoom; -- this creates large jumps in zoom if an elevation tile isn't loaded (yet)
    // get camera space depth (i.e. distance to terrain) at screen center - note that this unavoidably
    //  lags by one frame, since we need to render frame to get depth
    float prevViewZ = m_elevationManager ? m_elevationManager->getDepth({m_vpWidth/2, m_vpHeight/2}) : 0;
    if (m_type != CameraType::perspective) {
    } else if (prevViewZ > 0 && prevViewZ < 1E9f) {
        // Minimum camera clearance above terrain: a heightfield sampled at ~1 m per pixel is no
        //  longer a rock face, and looking straight at a wall from a few hundred meters shows
        //  nothing but DEM texels and procedural noise. The clearance corresponds to zoom
        //  ASCEND_TERRAIN_MAX_ZOOM (default 17.5, ~0.5 m/pixel at mid latitudes; was 16.5), independent of the
        //  (higher) overall max zoom used over flat/2D content.
        static const float terrainMaxZoom = [] {
            const char* e = std::getenv("ASCEND_TERRAIN_MAX_ZOOM");
            return e ? float(std::atof(e)) : 17.5f;
        }();
        double minCameraDist = exp2(-std::min(m_maxZoom, terrainMaxZoom)) * worldToCameraHeight;
        double prevCamDist = exp2(-m_elevationManager->getDepthData().zoom) * worldToCameraHeight;
        double viewZ = prevViewZ + m_pos.z - prevCamDist;
        // decrease base zoom if too close to terrain (but never increase)
        if (viewZ < minCameraDist) {
            m_pos.z += minCameraDist - viewZ;
            m_baseZoom = -log2( m_pos.z / worldToCameraHeight );
            viewZ = minCameraDist;
        }
        m_zoom = glm::clamp(-float(log2( viewZ / worldToCameraHeight )), m_baseZoom, m_maxZoom);
        //LOGW("viewZ: %f (prev: %f); base zoom: %.2f; zoom: %.2f", viewZ, prevViewZ, m_baseZoom, m_zoom);
    }

    // m_baseZoom now has final value
    m_height = exp2(-m_baseZoom) * worldHeight;
    m_width = m_height * m_aspect;

    // Ensure valid pitch angle.
    float maxPitchRadians = glm::radians(getMaxPitch());
    if (m_type != CameraType::perspective) {
        // Prevent projection plane from intersecting ground plane.
        float intersectingPitchRadians = atan2(m_pos.z, m_height * .5f);
        maxPitchRadians = glm::min(maxPitchRadians, intersectingPitchRadians);
    }
    m_pitch = glm::clamp(m_pitch, 0.f, maxPitchRadians);

    // using non-zero elevation for camera reference creates all kinds of problems
    glm::vec3 up = glm::rotateZ(glm::rotateX(glm::vec3(0.f, 1.f, 0.f), m_pitch), m_yaw);
    glm::vec3 at = { 0.f, 0.f, 0.f };
    m_eye = glm::rotateZ(glm::rotateX(glm::vec3(0.f, 0.f, m_pos.z), m_pitch), m_yaw);

    // keep eye above terrain
    bool elevOk;
    if (m_elevationManager) { // && elevOk) {
        double eyeElev = m_elevationManager->getElevation(glm::dvec2(m_eye) + glm::dvec2(m_pos), elevOk);
        if (elevOk && m_eye.z < eyeElev + 2) {
            m_eye.z = eyeElev + 2;
        }
    }

    // Generate view matrix
    m_view = glm::lookAt(m_eye, at, up);

    // find dimensions of tiles in world space at new zoom level
    float worldTileSize = MapProjection::EARTH_CIRCUMFERENCE_METERS * exp2(-m_baseZoom);
    float maxTileDistance = worldTileSize * invLodFunc(MAX_LOD + 1);
    float near = m_pos.z / 50.0;
    float far = 1;
    float hw = 0.5f * m_width;
    float hh = 0.5f * m_height;

    glm::vec2 viewportSize = { m_vpWidth, m_vpHeight };
    glm::vec2 paddingOffset = { m_padding.right - m_padding.left, m_padding.top - m_padding.bottom };
    glm::vec2 centerOffset = m_padding.isVisible ? (paddingOffset / viewportSize) : glm::vec2(0, 0);

    // Generate projection matrix based on camera type
    switch (m_type) {
        case CameraType::perspective:
            far = 2. * m_pos.z / std::max(0.f, std::cos(m_pitch + 0.5f * fovy));
            far = std::min(far, maxTileDistance);
            m_proj = glm::perspective(fovy, m_aspect, near, far);
            // Adjust projection center for edge padding.
            m_proj[2][0] = centerOffset.x;
            m_proj[2][1] = centerOffset.y;
            // Adjust for vanishing point.
            m_proj[2][0] -= m_vanishingPoint.x / getWidth();
            m_proj[2][1] -= m_vanishingPoint.y / getHeight();
            break;
        case CameraType::isometric:
        case CameraType::flat:
            far = 2. * (m_pos.z + hh * std::abs(tan(m_pitch)));
            far = std::min(far, maxTileDistance);
            m_proj = glm::ortho(-hw, hw, -hh, hh, near, far);
            // Adjust projection center for edge padding.
            m_proj[3][0] -= centerOffset.x;
            m_proj[3][1] -= centerOffset.y;
            break;
    }

    if (m_type == CameraType::isometric) {
        glm::mat4 shear = m_view;

        // Add the oblique projection scaling factors to the shear matrix
        shear[2][0] += m_obliqueAxis.x;
        shear[2][1] += m_obliqueAxis.y;

        // Remove the view from the shear matrix so we don't apply it two times
        shear *= glm::inverse(m_view);

        // Inject the shear in the projection matrix
        m_proj *= shear;
    }

    m_viewProj = m_proj * m_view;
    m_invViewProj = glm::inverse(m_viewProj);

    // The matrix that transforms normals from world space to camera space is the transpose of the inverse of the view matrix,
    // but since our view matrix is orthonormal transposing is equivalent to inverting, so the normal matrix is just the
    // original view matrix (cropped to the top-left 3 rows and columns, since we're applying it to 3d vectors)
    m_normalMatrix = glm::mat3(m_view);
    m_invNormalMatrix = glm::inverse(m_normalMatrix);

    m_dirtyMatrices = false;
    m_changed = true;
}

glm::vec2 View::lngLatToScreenPosition(double lng, double lat, bool& outsideViewport, bool clipToViewport) {

    if (m_dirtyMatrices) { updateMatrices(); } // Need the view matrices to be up-to-date

    glm::dvec2 absoluteMeters = MapProjection::lngLatToProjectedMeters({lng, lat});
    glm::dvec2 relativeMeters = getRelativeMeters(absoluteMeters);
    bool ok;
    double elev = m_elevationManager ? m_elevationManager->getElevation(absoluteMeters, ok) : 0;
    glm::vec4 worldPosition(relativeMeters, elev, 1.0);
    glm::vec4 clip = worldToClipSpace(m_viewProj, worldPosition);
    glm::vec3 ndc = clipSpaceToNdc(clip);
    outsideViewport = clipSpaceIsBehindCamera(clip) || abs(ndc.x) > 1 || abs(ndc.y) > 1;

    if (outsideViewport && clipToViewport) {
        // Get direction to the point and determine the point on the screen edge in that direction.
        glm::vec4 worldDirection(relativeMeters, 0, 0);
        glm::vec4 clipDirection = worldToClipSpace(m_viewProj, worldDirection);
        ndc = glm::vec3(clipDirection) / glm::max(abs(clipDirection.x), abs(clipDirection.y));
    }

    glm::vec2 screenSize(m_vpWidth, m_vpHeight);
    glm::vec2 screenPosition = ndcToScreenSpace(ndc, screenSize);

    if (!m_padding.isVisible && !outsideViewport) {
        outsideViewport = screenPosition.x < m_padding.left || screenPosition.x > m_vpWidth - m_padding.right
            || screenPosition.y < m_padding.top || screenPosition.y > m_vpHeight - m_padding.bottom;
    }

    return screenPosition;
}

LngLat View::screenPositionToLngLat(float x, float y, float* elevOut, bool* intersection) {

    if (m_dirtyMatrices) { updateMatrices(); } // Need the view matrices to be up-to-date

    glm::dvec2 dpos;
    float z = m_elevationManager ? m_elevationManager->getDepth({x, y}) : 0;
    if (z > 0 && z < 1E9f) {
        // ref: https://www.khronos.org/opengl/wiki/GluProject_and_gluUnProject_code (gluUnProject)
        //double ndcZ = m_elevationManager->getDepth({x, y});
        //glm::dvec4 target_clip = { 2*x/m_vpWidth - 1, 1 - 2*y/m_vpHeight, ndcZ, 1. };
        //glm::dvec4 target_world = m_invViewProj * target_clip;
        //target_world /= target_world.w;
        //dpos = glm::dvec2(target_world);
        //_intersection = ndcZ > -1;

        // z is -1 * view space z (to make it positive)
        glm::dvec4 target_clip = m_type == CameraType::perspective ?
            glm::dvec4(z*(2*x/m_vpWidth - 1), z*(1 - 2*y/m_vpHeight), -m_proj[2][2]*z + m_proj[3][2], z) :
            glm::dvec4(2*x/m_vpWidth - 1, 1 - 2*y/m_vpHeight, -m_proj[2][2]*z + m_proj[3][2], 1.0);

        glm::dvec4 target_world = m_invViewProj * target_clip;
        dpos = glm::dvec2(target_world);
        if (intersection) { *intersection = z > 0; }
        if (elevOut) { *elevOut = std::max(0.0, target_world.z); }
    } else {
        double distance = 0;
        dpos = screenToGroundPlane(x, y, 0, &distance);
        if (intersection) { *intersection = distance >= 0; }
        if (elevOut) { *elevOut = 0; }
    }
    LngLat lngLat = MapProjection::projectedMetersToLngLat(dpos + glm::dvec2(m_pos));
    return lngLat.wrapped();
}

glm::dvec2 View::getRelativeMeters(glm::dvec2 projectedMeters) const {
    return MapProjection::wrapProjectedMeters(projectedMeters - glm::dvec2(m_pos));
}

float View::horizonScreenPosition() {
    if (m_pitch == 0) { return std::numeric_limits<float>::infinity(); }
    if (m_dirtyMatrices) { updateMatrices(); } // Need the view matrices to be up-to-date

    float worldTileSize = MapProjection::EARTH_CIRCUMFERENCE_METERS * exp2(-m_baseZoom);
    float maxTileDistance = worldTileSize * invLodFunc(MAX_LOD + 1);
    glm::vec2 maxPos = -maxTileDistance*glm::normalize(glm::vec2(m_eye));
    glm::vec4 clip = worldToClipSpace(m_viewProj, glm::vec4(maxPos, 0., 1.));
    glm::vec3 ndc = clipSpaceToNdc(clip);
    //bool outsideViewport = clipSpaceIsBehindCamera(clip) || abs(ndc.x) > 1 || abs(ndc.y) > 1;
    glm::vec2 screenPosition = ndcToScreenSpace(ndc, glm::vec2(m_vpWidth, m_vpHeight));
    return screenPosition.y;
}

glm::vec4 View::tileCoordsToClipSpace(TileCoordinates tc, float elevation) const {
    glm::dvec2 absoluteMeters = MapProjection::tileCoordinatesToProjectedMeters(tc);
    glm::dvec2 relativeMeters = absoluteMeters - glm::dvec2(m_pos);  //getRelativeMeters(absoluteMeters);
    return worldToClipSpace(m_viewProj, glm::vec4(relativeMeters, elevation, 1.0));
}

// Proper way to get visible tiles in 3D is to create 3D AABB for tile using min and max elevation and
//  intersect it with frustum in world space ("frustum culling").  Since eye always looks down, we can use
//  eye elevation instead of actual (max) elevation at the cost of a few false positives underneath eye.
// ... but for now, we project to clip space and cull there instead, then subdivide if screen area of
//  tile exceeds threshold (an alternative approach is to just use distance to tile).
// Tangram originally "rasterized" trapezoid corresponding to screen in tile space plus triangle from eye to
//  screen bottom with fixed LOD ranges; attempted to combine with screen area based LOD calc, but was much
//  slower than recursive approach - see 3 Oct 2024 rev
// getVisibleTiles() (calling getTileScreenArea()) now moved to TileManager::updateTileSets()

static bool allLess(glm::vec4 a, glm::vec4 b) { return a.x < b.x && a.y < b.y && a.z < b.z && a.w < b.w; }
static bool allGreater(glm::vec4 a, glm::vec4 b) { return a.x > b.x && a.y > b.y && a.z > b.z && a.w > b.w; }

float View::getTileScreenArea(TileID tile) const {

    TileCoordinates tc = {double(tile.x), double(tile.y), tile.z};

    // handle wrapping at anti-meridian
    auto tcenter = MapProjection::tileCoordinatesToProjectedMeters({tc.x + 0.5, tc.y + 0.5, tc.z});
    double dx = tcenter.x - m_pos.x;
    constexpr double hc = MapProjection::EARTH_HALF_CIRCUMFERENCE_METERS;
    if (dx > hc) { tc.x -= 1 << tc.z; }
    else if (dx < -hc) { tc.x += 1 << tc.z; }

    // use elevation at center of screen (used to calc m_zoom) for tile bottom
    // 1 - 2^(base_z - z) gives normalized distance along pos -> eye vector of terrain intersection, so
    //  multiplying by eye elev gives terrain elev (similar triangles)
    float elev0 = m_elevationManager ? m_eye.z * (1 - std::exp2(m_baseZoom - m_zoom)) : 0;

    // Tile's real elevation range (cheap/cached - getMinMaxElev, elevationManager.cpp), used both
    // for elevation-aware culling below and for the steepness correction further down.
    float elevLo = elev0, elevHi = elev0;
    if (m_elevationManager) {
        glm::vec2 elevRange = m_elevationManager->getMinMaxElev(tile, 3);
        elevLo = std::min(elev0, elevRange.x);
        elevHi = std::max(elev0, elevRange.y);
    }

    auto a00 = tileCoordsToClipSpace(tc, elev0);
    auto a01 = tileCoordsToClipSpace({tc.x, tc.y + 1, tc.z}, elev0);
    auto a10 = tileCoordsToClipSpace({tc.x + 1, tc.y, tc.z}, elev0);
    auto a11 = tileCoordsToClipSpace({tc.x + 1, tc.y + 1, tc.z}, elev0);
    auto a = glm::transpose(glm::mat4(a00, a01, a10, a11));
    auto wa = glm::abs(a[3]);

    if (m_elevationManager) {  //&& m_pitch != 0) {
        // Elevation-aware culling using the tile's own real min/max range (elevLo/elevHi above)
        // instead of a fixed worst-case constant.
        auto b00 = tileCoordsToClipSpace(tc, elevHi);
        auto b01 = tileCoordsToClipSpace({tc.x, tc.y + 1, tc.z}, elevLo);
        auto b10 = tileCoordsToClipSpace({tc.x + 1, tc.y, tc.z}, elevLo);
        auto b11 = tileCoordsToClipSpace({tc.x + 1, tc.y + 1, tc.z}, elevHi);
        auto b = glm::transpose(glm::mat4(b00, b01, b10, b11));
        auto wb = glm::abs(b[3]);

        if (allLess(a[0], -wa) && allLess(b[0], -wb))     { return 0; }
        if (allGreater(a[0], wa) && allGreater(b[0], wb)) { return 0; }
        if (allLess(a[1], -wa) && allLess(b[1], -wb))     { return 0; }
        if (allGreater(a[1], wa) && allGreater(b[1], wb)) { return 0; }
        if (allLess(a[2], -wa) && allLess(b[2], -wb))     { return 0; }
        if (allGreater(a[2], wa) && allGreater(b[2], wb)) { return 0; }
    } else {
        if (allLess(a[0], -wa) || allGreater(a[0], wa)) { return 0; }
        if (allLess(a[1], -wa) || allGreater(a[1], wa)) { return 0; }
        if (allLess(a[2], -wa) || allGreater(a[2], wa)) { return 0; }
    }

    // Zenith-consistency fix: this used to also return FLT_MAX unconditionally at m_pitch == 0
    // ("always subdivide maximally, straight down") - a completely separate code path from every
    // other pitch, which used the real computed area below. That's exactly why resolution could
    // disagree between zenithal and tilted views (e.g. after raising lodAreaBias to compensate
    // for grazing tilt, zenith would revert to whatever the flat integer-zoom ceiling gives,
    // while tilted views - now using real area - could exceed it): two different mechanisms
    // governing the same decision. Only the near-camera-plane safety check remains; the real
    // area computed below now applies uniformly across every pitch, including 0.
    if (!allGreater(a[3], glm::vec4(0))) return FLT_MAX;

    glm::vec2 screenSize(m_vpWidth, m_vpHeight);
    auto r00 = ndcToScreenSpace(clipSpaceToNdc(a00), screenSize);
    auto r01 = ndcToScreenSpace(clipSpaceToNdc(a01), screenSize);
    auto r10 = ndcToScreenSpace(clipSpaceToNdc(a10), screenSize);
    auto r11 = ndcToScreenSpace(clipSpaceToNdc(a11), screenSize);
    auto pts = {r00, r01, r11, r10};
    float area = std::abs(signedArea(pts.begin(), pts.end()));

    // Grazing-tilt resolution fix, rigorous version: a flat quad's projected area already
    // correctly foreshortens by cos(theta) where theta is the angle between the GROUND normal
    // (vertical) and the view direction - that's exactly what perspective-projecting a flat,
    // horizontal quad computes. The bug was applying that same flat-ground foreshortening to
    // terrain that isn't flat: resolution demand for a real surface scales with cos(theta) using
    // the SURFACE'S OWN normal, not the ground's. A slope tilted toward the camera has a much
    // smaller theta (more face-on) than flat ground does at the same camera pitch, so it needs
    // correspondingly MORE resolution - "it gets shorter with the cosine as it flattens" is
    // exactly right, just about the surface's own tilt, not the camera pitch alone.
    //
    // Sample the LOCAL surface normal directly from real elevation data a fixed, small,
    // physically-meaningful distance apart, rather than inferring a "worst-case" slope from this
    // tile's own min/max relief over its (zoom-dependent) width - that earlier approach
    // systematically undershot true steepness for anything but the smallest/highest-zoom tiles,
    // because a real steep slope is steep over tens of meters, not over a multi-km tile; averaging
    // relief across the whole tile washes it out. A fixed real-world sample offset doesn't have
    // that problem.
    if (m_elevationManager) {
        double correction = steepnessCorrection(tile);
        area *= float(correction);
    }

    return area;
}

// Ratio between a surface's real local foreshortening and flat ground's, for use as a
// multiplier on a flat-quad area (see getTileScreenArea).
//
// Samples the local gradient at several locations spread across the tile (quarter-points), not
// just its center - a single center sample can land on a locally-gentle patch (e.g. a shoulder
// or saddle) even when the tile clearly contains a much steeper face elsewhere. Takes whichever
// sampled location gives the steepest (most camera-facing) result, per this project's established
// fidelity-over-cost preference for terrain LOD (docs/terrain-drape-resolution-plan.md).
double View::steepnessCorrection(TileID tile) const {
    if (!m_elevationManager) { return 1.0; }

    glm::dvec3 eye(m_pos.x + m_eye.x, m_pos.y + m_eye.y, m_eye.z);
    const double kSampleOffset = 100.0;  // meters - real-world scale a mountain slope is actually steep over

    double bestCorrection = 1.0;
    // Quarter-points, not corners/center: corners are shared with neighboring tiles (biases
    // toward whatever's already being sampled there) and a single center point can miss a real
    // steep feature elsewhere in the tile.
    for (double fx : {0.25, 0.75}) {
        for (double fy : {0.25, 0.75}) {
            TileCoordinates tc = {double(tile.x) + fx, double(tile.y) + fy, tile.z};
            glm::dvec2 sampleMeters = MapProjection::tileCoordinatesToProjectedMeters(tc);

            bool okC = false, okX = false, okY = false;
            double eC = m_elevationManager->getElevation(sampleMeters, okC);
            double eX = m_elevationManager->getElevation(sampleMeters + glm::dvec2(kSampleOffset, 0.), okX);
            double eY = m_elevationManager->getElevation(sampleMeters + glm::dvec2(0., kSampleOffset), okY);
            if (!okC || !okX || !okY) { continue; }

            // Standard height-field normal from the sampled gradient.
            glm::dvec3 normal = glm::normalize(glm::dvec3(-(eX - eC) / kSampleOffset, -(eY - eC) / kSampleOffset, 1.0));

            glm::dvec3 surfacePos(sampleMeters, eC);
            double dist = glm::distance(eye, surfacePos);
            if (dist <= 1e-3) { continue; }

            glm::dvec3 viewDir = (eye - surfacePos) / dist;  // surface -> eye
            double cosFlat = glm::clamp(viewDir.z, 0.0, 1.0);                       // flat ground's foreshortening (already baked into the caller's flat-quad area)
            double cosSlope = glm::clamp(glm::dot(normal, viewDir), 0.0, 1.0);      // this surface's actual foreshortening
            if (cosFlat <= 1e-4) { continue; }

            // Ratio converts the quad area's flat-ground foreshortening into this surface's
            // real one. Never reduce below the flat estimate (only ever raises resolution).
            double correction = std::max(1.0, cosSlope / cosFlat);
            bestCorrection = std::max(bestCorrection, correction);
        }
    }
    return bestCorrection;
}

float View::getTileDistance(TileID tile) const {
    glm::dvec3 eye(m_pos.x + m_eye.x, m_pos.y + m_eye.y, m_eye.z);
    return float(glm::distance(eye, glm::dvec3(MapProjection::tileCenter(tile), 0.)));
}

float View::getTileNearDistance(TileID tile, float topElev) const {
    glm::dvec3 eye(m_pos.x + m_eye.x, m_pos.y + m_eye.y, m_eye.z);
    BoundingBox b = MapProjection::tileBounds(tile);
    glm::dvec2 lo = glm::min(b.min, b.max), hi = glm::max(b.min, b.max);
    double dx = std::max({lo.x - eye.x, 0.0, eye.x - hi.x});
    double dy = std::max({lo.y - eye.y, 0.0, eye.y - hi.y});
    double dz = std::max(0.0, eye.z - double(topElev));
    return float(std::sqrt(dx*dx + dy*dy + dz*dz));
}

float View::pixelsPerMeterAtDistance(float distMeters) const {
    if (!(distMeters > 0)) { return 0.f; }
    return float(m_vpHeight) / (2.f * distMeters * std::tan(0.5f * m_fov));
}

bool View::demChildrenResident(TileID tile) const {
    if (!m_elevationManager) { return true; }
    for (int i = 0; i < 4; i++) {
        if (m_elevationManager->getTerrainDetail(tile.getChild(i, 100)).valid) { return true; }
    }
    return false;
}

bool View::demSaliencyWantsChildren(TileID tile, float tau, float nyquistBoost, int maxZoom,
                                    float massFrac) const {
    if (!m_elevationManager || tile.z >= maxZoom) { return false; }
    ++m_demStats.evaluated;

    // true-ground texel size of this node's children (256 texels per tile)
    LngLat ll = MapProjection::projectedMetersToLngLat(MapProjection::tileCenter(tile));
    double cosLat = std::cos(ll.latitude * M_PI / 180.0);
    double hChild = 0.5 * MapProjection::metersPerTileAtZoom(tile.z) * cosLat / 256.0;

    // Salient-texel table of the node itself, else of the nearest loaded ancestor (detail per
    // texel shrinks ~0.4x per level in the offline calibration, so an ancestor's q is scaled
    // accordingly; the salient FRACTION is assumed uniform over the ancestor's footprint).
    ElevationManager::TerrainDetail d;
    int up = 0;
    {
        TileID t = tile;
        for (; up <= 4 && t.z > 0; ++up, t = t.getParent()) {
            d = m_elevationManager->getTerrainDetail(t);
            if (d.valid) { break; }
        }
    }
    if (!d.valid || !d.facing || d.facing->texels == 0) { ++m_demStats.noData; return false; }
    const auto& fs = *d.facing;

    // Camera geometry at the tile: direction camera -> tile (center at the tile's real top
    // elevation), and pixel density at its nearest point.
    glm::vec2 mm = m_elevationManager->getMinMaxElev(tile, 3);
    float topElev = (mm == glm::vec2(0.f, 9000.f)) ? 0.f : mm.y;
    glm::dvec3 eye(m_pos.x + m_eye.x, m_pos.y + m_eye.y, m_eye.z);
    glm::dvec2 c = MapProjection::tileCenter(tile);
    glm::dvec3 dir = glm::dvec3(c, 0.5 * double(topElev)) - eye;
    double len = glm::length(dir);
    if (len < 1e-3) { return true; }
    dir /= len;
    float vx = float(dir.x), vy = float(dir.y), vz = float(dir.z);   // vz < 0 when looking down

    float dNear = getTileNearDistance(tile, topElev);
    float ppm = pixelsPerMeterAtDistance(dNear);
    if (!(ppm > 0.f)) { return true; }  // at/inside the near plane: always wants detail

    // Hidden behind terrain drawn last frame (a valley behind a ridge): extra detail there is
    // wasted. Needs a real elevation range for the box; the "unknown" sentinel skips the test.
    if (mm != glm::vec2(0.f, 9000.f)) {
        BoundingBox b = MapProjection::tileBounds(tile);
        glm::dvec2 lo = glm::min(b.min, b.max), hi = glm::max(b.min, b.max);
        if (m_elevationManager->isBoxOccluded(lo, hi, mm.x, mm.y)) { ++m_demStats.occluded; return false; }
    }

    const float rhoMin = std::exp2(-2.f * nyquistBoost);
    const float K = float(hChild * hChild) * ppm * ppm;   // pixels^2 per child texel at P = 1
    const float qScale = std::pow(0.4f, float(up));

    float mass = 0.f;
    for (int t = 0; t < ElevationManager::kDemTiers; ++t) {
        float lo = ElevationManager::kDemTierLo[t];
        float hi = ElevationManager::kDemTierLo[t + 1];
        float center = (hi > 1e8f) ? lo * 1.5f : std::sqrt(lo * hi);
        if (center * qScale < tau) { continue; }
        for (int o = 0; o < ElevationManager::kDemOctants; ++o) {
            uint32_t n = fs.cnt[t][o];
            if (!n) { continue; }
            float inv = 1.f / float(n);
            float P = (fs.sgx[t][o] * vx + fs.sgy[t][o] * vy) * inv - vz;
            if (P > 0.f && K * P >= rhoMin) { mass += float(n); }
        }
    }
    if (mass >= massFrac * float(fs.texels)) { ++m_demStats.accepted; return true; }
    ++m_demStats.belowMass;
    return false;
}

float View::getTileDistanceZoomBound(TileID tile) const {
    // Same distance->zoom relationship as updateMatrices()'s m_zoom = clamp(-log2(viewZ /
    // worldToCameraHeight), baseZoom, maxZoom), just algebraically reduced using
    // worldToCameraHeight = m_pos.z * exp2(m_baseZoom) (from m_pos.z's own definition a few lines
    // above that) and evaluated at this tile's distance instead of the screen-center depth.
    glm::dvec3 eye(m_pos.x + m_eye.x, m_pos.y + m_eye.y, m_eye.z);
    double dist = glm::distance(eye, glm::dvec3(MapProjection::tileCenter(tile), 0.));
    if (!(dist > 0) || !(m_pos.z > 0)) { return m_maxZoom; }
    return glm::clamp(m_baseZoom - float(log2(dist / m_pos.z)), m_baseZoom, m_maxZoom);
}

}

// [1]
// The maximum visible span horizontally is the span covered by 2^z - 2 tiles. We consider one tile to be
// effectively not visible because at the 180th meridian it will be drawn on one side or the other, not half
// on both sides. Tile coverage is calculated from the floor() of our zoom value but here we operate on the
// continuous zoom value, so we remove one more tile to under-approximate the coverage. Algebraically we get:
// (span of view in meters) = (view diameter in pixels) * (earth circumference) / ((tile size in pixels) * 2^z)
// If we substitute the desired view span at the minimum zoom:
// (span of view in meters) = (earth circumference) * (2^z - 2) / 2^z
// We can solve for the minimum zoom:
// z = log2((view diameter in pixels) / (tile size in pixels) + 2)
