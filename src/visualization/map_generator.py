"""
Visualization and geolocation.
- Map generation with detected fires or other detected objects
- Multi-view 3D object triangulation from drone telemetry + image detections
- Results export
"""

import json
import logging
from typing import Dict, List, Optional, Tuple

import folium
import numpy as np
from folium import plugins

logger = logging.getLogger(__name__)


class FireGeolocation:
    """Geolocate detections using single-view projection or multi-view triangulation."""

    EARTH_RADIUS_M = 6378137.0
    
    def __init__(self, config: Optional[dict]):
        """Initialize with config"""
        self.config = config or {}
        camera_cfg = self.config.get('camera', {})
        self.default_horizontal_fov = float(camera_cfg.get('horizontal_fov_deg', 70.0))
        self.default_vertical_fov = float(camera_cfg.get('vertical_fov_deg', 56.0))

    def _camera_heading(self, observation: Dict) -> Optional[float]:
        heading = observation.get('camera_heading')
        if heading is not None:
            return float(heading)

        drone_yaw = observation.get('drone_yaw')
        gimbal_yaw = observation.get('gimbal_yaw', 0.0)
        if drone_yaw is None:
            return None

        return (float(drone_yaw) + float(gimbal_yaw)) % 360.0

    def _camera_elevation(self, observation: Dict) -> float:
        return float(observation.get('drone_pitch', 0.0)) + float(observation.get('gimbal_pitch', 0.0))

    def _reference_origin(self, observations: List[Dict]) -> Tuple[float, float, float]:
        latitudes = [float(item['drone_lat']) for item in observations]
        longitudes = [float(item['drone_lon']) for item in observations]
        altitudes = [float(item.get('drone_altitude_m', item.get('drone_altitude'))) for item in observations]
        return (float(np.mean(latitudes)), float(np.mean(longitudes)), float(np.mean(altitudes)))

    def _geodetic_to_enu(
        self,
        latitude: float,
        longitude: float,
        altitude_msl: float,
        ref_latitude: float,
        ref_longitude: float,
        ref_altitude_msl: float,
    ) -> np.ndarray:
        lat_rad = np.radians(latitude)
        lon_rad = np.radians(longitude)
        ref_lat_rad = np.radians(ref_latitude)
        ref_lon_rad = np.radians(ref_longitude)

        east = self.EARTH_RADIUS_M * (lon_rad - ref_lon_rad) * np.cos(ref_lat_rad)
        north = self.EARTH_RADIUS_M * (lat_rad - ref_lat_rad)
        up = altitude_msl - ref_altitude_msl
        return np.array([east, north, up], dtype=float)

    def _enu_to_geodetic(
        self,
        east: float,
        north: float,
        up: float,
        ref_latitude: float,
        ref_longitude: float,
        ref_altitude_msl: float,
    ) -> Tuple[float, float, float]:
        ref_lat_rad = np.radians(ref_latitude)
        latitude = ref_latitude + np.degrees(north / self.EARTH_RADIUS_M)
        longitude = ref_longitude + np.degrees(east / (self.EARTH_RADIUS_M * np.cos(ref_lat_rad)))
        altitude_msl = ref_altitude_msl + up
        return latitude, longitude, altitude_msl

    @staticmethod
    def _unit_vector_from_angles(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
        azimuth_rad = np.radians(azimuth_deg)
        elevation_rad = np.radians(elevation_deg)
        horizontal = np.cos(elevation_rad)
        east = horizontal * np.sin(azimuth_rad)
        north = horizontal * np.cos(azimuth_rad)
        up = np.sin(elevation_rad)
        direction = np.array([east, north, up], dtype=float)
        return direction / np.linalg.norm(direction)

    def _observation_to_ray(
        self,
        observation: Dict,
        ref_latitude: float,
        ref_longitude: float,
        ref_altitude_msl: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        bbox = observation['bbox']
        frame_width = float(observation['frame_width'])
        frame_height = float(observation['frame_height'])
        horizontal_fov = float(observation.get('horizontal_fov_deg', observation.get('camera_fov', self.default_horizontal_fov)))
        vertical_fov = float(observation.get('vertical_fov_deg', self.default_vertical_fov))
        camera_heading = self._camera_heading(observation)
        if camera_heading is None:
            raise ValueError('Observation is missing camera_heading or drone_yaw')

        center_x = (float(bbox[0]) + float(bbox[2])) / 2.0
        center_y = (float(bbox[1]) + float(bbox[3])) / 2.0

        focal_x = (frame_width / 2.0) / np.tan(np.radians(horizontal_fov) / 2.0)
        focal_y = (frame_height / 2.0) / np.tan(np.radians(vertical_fov) / 2.0)

        image_x = (center_x - (frame_width / 2.0)) / focal_x
        image_y = ((frame_height / 2.0) - center_y) / focal_y

        azimuth_offset = np.degrees(np.arctan2(image_x, 1.0))
        elevation_offset = np.degrees(np.arctan2(image_y, np.sqrt(1.0 + image_x ** 2)))

        ray_azimuth = camera_heading + azimuth_offset
        ray_elevation = self._camera_elevation(observation) + elevation_offset
        direction = self._unit_vector_from_angles(ray_azimuth, ray_elevation)

        altitude_msl = float(observation.get('drone_altitude_m', observation.get('drone_altitude')))
        origin = self._geodetic_to_enu(
            latitude=float(observation['drone_lat']),
            longitude=float(observation['drone_lon']),
            altitude_msl=altitude_msl,
            ref_latitude=ref_latitude,
            ref_longitude=ref_longitude,
            ref_altitude_msl=ref_altitude_msl,
        )
        return origin, direction

    @staticmethod
    def _distance_to_ray(point: np.ndarray, origin: np.ndarray, direction: np.ndarray) -> float:
        return float(np.linalg.norm(np.cross(point - origin, direction)))
    
    def geolocate_detection(
        self,
        drone_lat: float,
        drone_lon: float,
        drone_altitude_m: float,
        detection_bbox: Tuple[int, int, int, int],
        frame_width: int,
        frame_height: int,
        camera_fov: float = 70,
        gimbal_pitch: float = -45,
        camera_heading: Optional[float] = None,
        drone_yaw: Optional[float] = None,
        gimbal_yaw: float = 0.0,
        drone_pitch: float = 0.0,
        target_altitude_msl: Optional[float] = None,
        detection_confidence: Optional[float] = None,
        vertical_fov: Optional[float] = None,
    ) -> Optional[Dict[str, float]]:
        """
        Geolocate a single detection by intersecting its view ray with a target altitude plane.
        
        Args:
            drone_lat, drone_lon: drone GPS position
            drone_altitude_m: drone altitude in meters above sea level
            detection_bbox: (x1, y1, x2, y2) in pixels
            frame_width, frame_height: image dimensions
            camera_fov: horizontal camera field of view (degrees)
            gimbal_pitch: gimbal pitch angle (degrees)
            camera_heading: world-relative camera heading in degrees clockwise from north
            drone_yaw: fallback drone yaw if camera heading is not provided
            gimbal_yaw: yaw offset from the drone heading
            drone_pitch: aircraft pitch contribution in degrees
            target_altitude_msl: altitude plane to intersect in meters above sea level
            detection_confidence: detector confidence score
            vertical_fov: optional vertical field of view in degrees
        
        Returns:
            {'latitude': float, 'longitude': float, 'altitude': float, ...}
        """
        try:
            if target_altitude_msl is None:
                target_altitude_msl = self.config.get('geolocation', {}).get('default_target_altitude_msl')

            if target_altitude_msl is None:
                logger.warning('Single-view geolocation requires target_altitude_msl. Use triangulate_observations for true 3D output.')
                return None

            observation = {
                'drone_lat': drone_lat,
                'drone_lon': drone_lon,
                'drone_altitude_m': drone_altitude_m,
                'bbox': detection_bbox,
                'frame_width': frame_width,
                'frame_height': frame_height,
                'horizontal_fov_deg': camera_fov,
                'vertical_fov_deg': vertical_fov or self.default_vertical_fov,
                'gimbal_pitch': gimbal_pitch,
                'camera_heading': camera_heading,
                'drone_yaw': drone_yaw,
                'gimbal_yaw': gimbal_yaw,
                'drone_pitch': drone_pitch,
                'confidence': detection_confidence,
            }

            result = self.intersect_observation_with_altitude(observation, float(target_altitude_msl))
            if result:
                logger.info(f"Geolocated detection at {result['latitude']:.6f}, {result['longitude']:.6f}")
            return result
        
        except Exception as e:
            logger.error(f"Geolocation failed: {e}")
            return None

    def intersect_observation_with_altitude(self, observation: Dict, target_altitude_msl: float) -> Optional[Dict[str, float]]:
        """Project a single observation ray onto a known altitude plane."""
        ref_latitude, ref_longitude, ref_altitude_msl = self._reference_origin([observation])
        origin, direction = self._observation_to_ray(observation, ref_latitude, ref_longitude, ref_altitude_msl)

        target_up = float(target_altitude_msl) - ref_altitude_msl
        if abs(direction[2]) < 1e-6:
            logger.warning('Observation ray is nearly parallel to the target altitude plane')
            return None

        ray_scale = (target_up - origin[2]) / direction[2]
        if ray_scale < 0:
            logger.warning('Target altitude plane lies behind the camera ray')
            return None

        point = origin + (ray_scale * direction)
        latitude, longitude, altitude = self._enu_to_geodetic(
            east=point[0],
            north=point[1],
            up=point[2],
            ref_latitude=ref_latitude,
            ref_longitude=ref_longitude,
            ref_altitude_msl=ref_altitude_msl,
        )

        return {
            'latitude': latitude,
            'longitude': longitude,
            'altitude': altitude,
            'confidence': float(observation.get('confidence', 0.0)),
            'method': 'plane-intersection',
            'observation_count': 1,
        }

    def triangulate_observations(self, observations: List[Dict]) -> Optional[Dict[str, float]]:
        """Triangulate a 3D object location from two or more detections of the same object."""
        if len(observations) < 2:
            logger.warning('Triangulation needs at least two observations of the same object')
            return None

        ref_latitude, ref_longitude, ref_altitude_msl = self._reference_origin(observations)
        rays = [
            self._observation_to_ray(item, ref_latitude, ref_longitude, ref_altitude_msl)
            for item in observations
        ]

        baselines = [
            np.linalg.norm(rays[i][0] - rays[j][0])
            for i in range(len(rays))
            for j in range(i + 1, len(rays))
        ]
        max_baseline_m = max(baselines) if baselines else 0.0
        if max_baseline_m < 0.5:
            logger.warning('Camera baseline is too small for a stable triangulation solution')
            return None

        system_matrix = np.zeros((3, 3), dtype=float)
        system_vector = np.zeros(3, dtype=float)
        identity = np.eye(3, dtype=float)
        for origin, direction in rays:
            projector = identity - np.outer(direction, direction)
            system_matrix += projector
            system_vector += projector @ origin

        point, *_ = np.linalg.lstsq(system_matrix, system_vector, rcond=None)

        ray_errors = [self._distance_to_ray(point, origin, direction) for origin, direction in rays]
        forward_scales = [float(np.dot(point - origin, direction)) for origin, direction in rays]
        if any(scale < 0 for scale in forward_scales):
            logger.warning('Triangulated point falls behind at least one camera; solution may be unstable')

        latitude, longitude, altitude = self._enu_to_geodetic(
            east=point[0],
            north=point[1],
            up=point[2],
            ref_latitude=ref_latitude,
            ref_longitude=ref_longitude,
            ref_altitude_msl=ref_altitude_msl,
        )

        confidences = [float(item.get('confidence', 0.0)) for item in observations if item.get('confidence') is not None]
        return {
            'latitude': latitude,
            'longitude': longitude,
            'altitude': altitude,
            'confidence': float(np.mean(confidences)) if confidences else 0.0,
            'method': 'triangulation',
            'observation_count': len(observations),
            'max_baseline_m': float(max_baseline_m),
            'mean_ray_error_m': float(np.mean(ray_errors)) if ray_errors else 0.0,
            'max_ray_error_m': float(np.max(ray_errors)) if ray_errors else 0.0,
        }

    def triangulate_tracks(self, tracks: List[List[Dict]]) -> List[Dict]:
        """Triangulate a collection of tracked detections."""
        results = []
        for observations in tracks:
            result = self.triangulate_observations(observations)
            if result:
                results.append(result)
        return results
    
    def geolocate_batch(self, detections_with_poses: List[dict]) -> List[dict]:
        """
        Geolocate multiple detections or triangulated detection tracks.
        
        Args:
            detections_with_poses: list of dicts with detection pose info, or
                a dict with an observations list for multi-view triangulation
        
        Returns:
            list of geolocated points
        """
        geolocations = []
        for item in detections_with_poses:
            if 'observations' in item:
                loc = self.triangulate_observations(item['observations'])
            else:
                loc = self.geolocate_detection(
                    drone_lat=item['drone_lat'],
                    drone_lon=item['drone_lon'],
                    drone_altitude_m=item.get('drone_altitude_m', item.get('drone_altitude')),
                    detection_bbox=item['bbox'],
                    frame_width=item['frame_width'],
                    frame_height=item['frame_height'],
                    camera_fov=item.get('horizontal_fov_deg', item.get('camera_fov', self.default_horizontal_fov)),
                    gimbal_pitch=item.get('gimbal_pitch', -45),
                    camera_heading=item.get('camera_heading'),
                    drone_yaw=item.get('drone_yaw'),
                    gimbal_yaw=item.get('gimbal_yaw', 0.0),
                    drone_pitch=item.get('drone_pitch', 0.0),
                    target_altitude_msl=item.get('target_altitude_msl'),
                    detection_confidence=item.get('confidence'),
                    vertical_fov=item.get('vertical_fov_deg'),
                )
            if loc:
                geolocations.append(loc)
        
        return geolocations


class MapGenerator:
    """Generate interactive maps with fire detections"""
    
    def __init__(self, config: dict):
        """Initialize map generator"""
        self.config = config
    
    def create_fire_map(self, geolocations: List[dict], output_path: str = 'fire_map.html'):
        """
        Create interactive map with detected fires
        
        Args:
            geolocations: list of {'latitude', 'longitude', 'confidence'} dicts
            output_path: where to save the map HTML
        """
        if not geolocations:
            logger.warning("No geolocations to map")
            return
        
        # Calculate center
        center_lat = np.mean([g['latitude'] for g in geolocations])
        center_lon = np.mean([g['longitude'] for g in geolocations])
        
        # Create base map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=16,
            tiles='OpenStreetMap'
        )
        
        # Add fire detections
        for i, geo in enumerate(geolocations):
            color = 'red' if geo.get('confidence', 0.5) > 0.8 else 'orange'
            popup_text = (
                f"Fire Detection #{i+1}<br>"
                f"Confidence: {geo.get('confidence', 0):.2%}<br>"
                f"Altitude (MSL): {geo.get('altitude', 0):.1f}m<br>"
                f"Method: {geo.get('method', 'unknown')}<br>"
                f"Observations: {geo.get('observation_count', 1)}"
            )
            
            folium.CircleMarker(
                location=[geo['latitude'], geo['longitude']],
                radius=20,
                popup=popup_text,
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.7,
            ).add_to(m)
        
        # Add heatmap layer
        if len(geolocations) > 1:
            heat_data = [
                [g['latitude'], g['longitude'], g.get('confidence', 0.5)]
                for g in geolocations
            ]
            plugins.HeatMap(heat_data, radius=30).add_to(m)
        
        # Save
        m.save(output_path)
        logger.info(f"Map saved to {output_path}")

    def export_geojson(self, geolocations: List[dict], output_path: str = 'detections.geojson'):
        """Export geolocated detections as a GeoJSON FeatureCollection."""
        features = []
        for index, geo in enumerate(geolocations, start=1):
            properties = {key: value for key, value in geo.items() if key not in {'latitude', 'longitude'}}
            properties['id'] = index
            features.append({
                'type': 'Feature',
                'geometry': {
                    'type': 'Point',
                    'coordinates': [geo['longitude'], geo['latitude'], geo.get('altitude', 0.0)],
                },
                'properties': properties,
            })

        with open(output_path, 'w', encoding='utf-8') as handle:
            json.dump({'type': 'FeatureCollection', 'features': features}, handle, indent=2)

        logger.info(f"GeoJSON saved to {output_path}")
    
    def create_summary_report(self, detections: List[dict], geolocations: List[dict], output_path: str):
        """Generate text summary report"""
        with open(output_path, 'w') as f:
            f.write("FIRE DETECTION SUMMARY REPORT\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Total Detections: {len(detections)}\n")
            f.write(f"Geolocated Points: {len(geolocations)}\n\n")
            
            if geolocations:
                f.write("Detected Fire Locations:\n")
                f.write("-" * 40 + "\n")
                for i, loc in enumerate(geolocations):
                    f.write(
                        f"{i+1}. Lat: {loc['latitude']:.6f}, "
                        f"Lon: {loc['longitude']:.6f}, "
                        f"Altitude(MSL): {loc.get('altitude', 0):.1f}m, "
                        f"Confidence: {loc.get('confidence', 0):.2%}, "
                        f"Method: {loc.get('method', 'unknown')}\n"
                    )
        
        logger.info(f"Report saved to {output_path}")
