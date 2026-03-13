"""
Visualization and geolocation
- Map generation with detected fires
- Fire point geolocation from drone GPS + image position
- Results export
"""

import folium
from folium import plugins
import numpy as np
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class FireGeolocation:
    """Geolocate fire detections using drone position and camera angles"""
    
    def __init__(self, config: dict):
        """Initialize with config"""
        self.config = config
    
    def geolocate_detection(
        self,
        drone_lat: float,
        drone_lon: float,
        drone_altitude_m: float,
        detection_bbox: Tuple[int, int, int, int],
        frame_width: int,
        frame_height: int,
        camera_fov: float = 70,
        gimbal_pitch: float = -45
    ) -> Dict[str, float]:
        """
        Geolocate a fire detection
        
        Args:
            drone_lat, drone_lon: drone GPS position
            drone_altitude_m: altitude in meters
            detection_bbox: (x1, y1, x2, y2) in pixels
            frame_width, frame_height: image dimensions
            camera_fov: camera field of view (degrees)
            gimbal_pitch: gimbal pitch angle (degrees)
        
        Returns:
            {'latitude': float, 'longitude': float, 'confidence': float}
        """
        try:
            # Get center of detection in image
            cx = (detection_bbox[0] + detection_bbox[2]) / 2
            cy = (detection_bbox[1] + detection_bbox[3]) / 2
            
            # Normalize to -1 to 1
            norm_x = (cx - frame_width / 2) / (frame_width / 2)
            norm_y = (cy - frame_height / 2) / (frame_height / 2)
            
            # Calculate angle offset from camera center
            half_fov = camera_fov / 2
            angle_x = norm_x * half_fov
            angle_y = norm_y * half_fov
            
            # Simple distance calculation based on altitude
            # (real implementation would use proper triangulation)
            distance = drone_altitude_m / np.cos(np.radians(gimbal_pitch))
            
            # Calculate lat/lon offset (simplified, ~111 km per degree)
            lat_offset = (distance / 111000) * np.cos(np.radians(angle_x))
            lon_offset = (distance / 111000) * np.sin(np.radians(angle_x))
            
            fire_lat = drone_lat + lat_offset / 111000
            fire_lon = drone_lon + lon_offset / (111000 * np.cos(np.radians(drone_lat)))
            
            logger.info(f"Geolocated fire at {fire_lat:.6f}, {fire_lon:.6f}")
            
            return {
                'latitude': fire_lat,
                'longitude': fire_lon,
                'altitude': drone_altitude_m,
                'confidence': 0.85,  # TODO: tie to detection confidence
            }
        
        except Exception as e:
            logger.error(f"Geolocation failed: {e}")
            return None
    
    def geolocate_batch(self, detections_with_poses: List[dict]) -> List[dict]:
        """
        Geolocate multiple detections
        
        Args:
            detections_with_poses: list of dicts with detection and drone pose info
        
        Returns:
            list of geolocated points
        """
        geolocations = []
        for item in detections_with_poses:
            loc = self.geolocate_detection(
                drone_lat=item['drone_lat'],
                drone_lon=item['drone_lon'],
                drone_altitude_m=item['drone_altitude'],
                detection_bbox=item['bbox'],
                frame_width=item['frame_width'],
                frame_height=item['frame_height'],
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
                f"Altitude: {geo.get('altitude', 0):.1f}m"
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
                        f"Confidence: {loc.get('confidence', 0):.2%}\n"
                    )
        
        logger.info(f"Report saved to {output_path}")
