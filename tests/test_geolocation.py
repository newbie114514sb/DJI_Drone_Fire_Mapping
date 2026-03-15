"""Tests for 3D detection geolocation utilities."""

import math

from src.visualization.map_generator import FireGeolocation


def _azimuth_elevation(camera_position, object_position):
    east = object_position[0] - camera_position[0]
    north = object_position[1] - camera_position[1]
    up = object_position[2] - camera_position[2]
    horizontal = (east ** 2 + north ** 2) ** 0.5
    azimuth = (math.degrees(math.atan2(east, north)) + 360.0) % 360.0
    elevation = math.degrees(math.atan2(up, horizontal))
    return azimuth, elevation


def _center_box(center_x, center_y, width=20, height=20):
    return (
        center_x - (width // 2),
        center_y - (height // 2),
        center_x + (width // 2),
        center_y + (height // 2),
    )


def test_triangulate_observations_returns_expected_3d_fix():
    geolocator = FireGeolocation(config={})
    reference = (34.145, -118.029, 250.0)
    object_enu = (12.0, 38.0, -8.0)
    camera_a = (0.0, 0.0, 0.0)
    camera_b = (-22.0, 9.0, 3.0)

    cam_a_lat, cam_a_lon, cam_a_alt = geolocator._enu_to_geodetic(*camera_a, *reference)
    cam_b_lat, cam_b_lon, cam_b_alt = geolocator._enu_to_geodetic(*camera_b, *reference)

    azimuth_a, elevation_a = _azimuth_elevation(camera_a, object_enu)
    azimuth_b, elevation_b = _azimuth_elevation(camera_b, object_enu)

    observations = [
        {
            'drone_lat': cam_a_lat,
            'drone_lon': cam_a_lon,
            'drone_altitude_m': cam_a_alt,
            'camera_heading': azimuth_a,
            'gimbal_pitch': elevation_a,
            'drone_pitch': 0.0,
            'bbox': _center_box(500, 400),
            'frame_width': 1000,
            'frame_height': 800,
            'confidence': 0.95,
        },
        {
            'drone_lat': cam_b_lat,
            'drone_lon': cam_b_lon,
            'drone_altitude_m': cam_b_alt,
            'camera_heading': azimuth_b,
            'gimbal_pitch': elevation_b,
            'drone_pitch': 0.0,
            'bbox': _center_box(500, 400),
            'frame_width': 1000,
            'frame_height': 800,
            'confidence': 0.90,
        },
    ]

    result = geolocator.triangulate_observations(observations)

    assert result is not None
    solved_enu = geolocator._geodetic_to_enu(result['latitude'], result['longitude'], result['altitude'], *reference)
    assert abs(solved_enu[0] - object_enu[0]) < 0.5
    assert abs(solved_enu[1] - object_enu[1]) < 0.5
    assert abs(solved_enu[2] - object_enu[2]) < 0.5
    assert result['method'] == 'triangulation'
    assert result['observation_count'] == 2


def test_single_view_projection_hits_requested_msl_altitude():
    geolocator = FireGeolocation(config={})
    reference = (34.145, -118.029, 250.0)
    object_enu = (8.0, 22.0, -6.0)
    camera = (0.0, 0.0, 0.0)

    camera_lat, camera_lon, camera_alt = geolocator._enu_to_geodetic(*camera, *reference)
    azimuth, elevation = _azimuth_elevation(camera, object_enu)
    target_altitude_msl = reference[2] + object_enu[2]

    result = geolocator.geolocate_detection(
        drone_lat=camera_lat,
        drone_lon=camera_lon,
        drone_altitude_m=camera_alt,
        detection_bbox=_center_box(500, 400),
        frame_width=1000,
        frame_height=800,
        camera_fov=70.0,
        gimbal_pitch=elevation,
        camera_heading=azimuth,
        drone_pitch=0.0,
        target_altitude_msl=target_altitude_msl,
        detection_confidence=0.88,
        vertical_fov=56.0,
    )

    assert result is not None
    solved_enu = geolocator._geodetic_to_enu(result['latitude'], result['longitude'], result['altitude'], *reference)
    assert abs(solved_enu[0] - object_enu[0]) < 0.5
    assert abs(solved_enu[1] - object_enu[1]) < 0.5
    assert abs(result['altitude'] - target_altitude_msl) < 1e-6
    assert result['method'] == 'plane-intersection'