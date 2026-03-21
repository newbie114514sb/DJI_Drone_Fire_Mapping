"""
Interactive video viewer for hyperlapse flight data
- Display images with telemetry overlay
- Show GPS location on minimap
- Display gimbal angles and altitude
"""

from pathlib import Path
from typing import Dict, Optional, List, Tuple
import logging
import json
import cv2
import numpy as np
import folium
from folium import plugins
from src.visualization.telemetry import TelemetrySequence
import shutil

# GUI dependencies may not always be available (headless environments)
try:
    import tkinter as tk
    from PIL import Image, ImageTk
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    import matplotlib.pyplot as plt
except ImportError:
    tk = None
    Image = None
    ImageTk = None
    FigureCanvasTkAgg = None
    plt = None

logger = logging.getLogger(__name__)


class HyperlapseViewer:
    """View hyperlapse images with telemetry overlay"""
    
    def __init__(self, image_folder: str, telemetry_sequence=None):
        """
        Initialize viewer
        Args:
            image_folder: path to hyperlapse images
            telemetry_sequence: TelemetrySequence object with extracted data
        """
        self.folder = Path(image_folder)
        patterns = (
            'IMG_*.jpg',
            'IMG_*.JPG',
            'HYPERLAPSE_*.jpg',
            'HYPERLAPSE_*.JPG',
            '*.jpg',
            '*.JPG',
            '*.jpeg',
            '*.JPEG',
        )
        discovered: List[Path] = []
        seen: set[Path] = set()
        for pattern in patterns:
            for image_path in self.folder.glob(pattern):
                if image_path in seen:
                    continue
                seen.add(image_path)
                discovered.append(image_path)

        self.images = sorted(discovered)
        self.telemetry_sequence = telemetry_sequence
        self.current_index = 0
        
        logger.info(f"Initialized viewer with {len(self.images)} images")
    
    def get_image_count(self) -> int:
        """Get total number of images"""
        return len(self.images)
    
    def get_image_at_index(self, index: int) -> Optional[np.ndarray]:
        """Get image at index as numpy array"""
        if index < 0 or index >= len(self.images):
            return None
        
        img_path = self.images[index]
        try:
            img = cv2.imread(str(img_path))
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            logger.error(f"Failed to load image {img_path}: {e}")
            return None
    
    def get_telemetry_at_index(self, index: int) -> Optional[Dict]:
        """Get telemetry for image at index"""
        if not self.telemetry_sequence:
            return None
        return self.telemetry_sequence.get_telemetry_at_index(index)
    
    def draw_telemetry_overlay(self, image: np.ndarray, telemetry: Dict) -> np.ndarray:
        """
        Draw telemetry on image
        Args:
            image: input image (RGB format)
            telemetry: telemetry dict with GPS, gimbal, etc.
        Returns:
            annotated image
        """
        overlay = image.copy()
        height, width = overlay.shape[:2]
        
        # Background panel for text (semi-transparent)
        panel_height = 120
        overlay[0:panel_height, 0:width] = cv2.addWeighted(
            overlay[0:panel_height, 0:width], 0.7,
            np.full_like(overlay[0:panel_height, 0:width], (0, 0, 0)), 0.3, 0
        )
        
        # Text color (white)
        text_color = (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        line_height = 25
        
        y = 25
        
        # GPS coordinates
        gps = telemetry.get('gps', {})
        if gps and gps.get('latitude'):
            gps_text = f"GPS: {gps['latitude']:.6f}, {gps['longitude']:.6f}"
            cv2.putText(overlay, gps_text, (10, y), font, font_scale, text_color, thickness)
            y += line_height
        
        # Altitude
        if gps and gps.get('altitude'):
            alt_text = f"Altitude: {gps['altitude']:.1f}m"
            cv2.putText(overlay, alt_text, (10, y), font, font_scale, text_color, thickness)
            y += line_height
        
        # Gimbal angles
        gimbal = telemetry.get('gimbal', {})
        if gimbal:
            gimbal_text = f"Gimbal: P={gimbal.get('pitch', 'N/A')} "
            gimbal_text += f"R={gimbal.get('roll', 'N/A')} Y={gimbal.get('yaw', 'N/A')}"
            cv2.putText(overlay, gimbal_text, (10, y), font, font_scale, text_color, thickness)
            y += line_height
        
        # Drone model
        drone_model = telemetry.get('drone_model', 'Unknown')
        model_text = f"Drone: {drone_model}"
        cv2.putText(overlay, model_text, (10, y), font, font_scale, text_color, thickness)
        
        return overlay
    
    def draw_compass_indicator(self, image: np.ndarray, yaw: float = 0) -> np.ndarray:
        """
        Draw compass/heading indicator on image
        Args:
            image: input image
            yaw: yaw angle in degrees
        Returns:
            image with compass
        """
        overlay = image.copy()
        height, width = overlay.shape[:2]
        
        # Compass position (bottom right)
        center_x = width - 60
        center_y = height - 60
        radius = 40
        
        # Draw circle
        cv2.circle(overlay, (center_x, center_y), radius, (255, 255, 255), 2)
        
        # Draw cardinal directions
        cv2.putText(overlay, 'N', (center_x - 8, center_y - radius - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(overlay, 'E', (center_x + radius + 5, center_y + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(overlay, 'S', (center_x - 8, center_y + radius + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(overlay, 'W', (center_x - radius - 15, center_y + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
        # Draw heading arrow (yaw)
        angle_rad = np.radians(yaw)
        end_x = int(center_x + radius * 0.7 * np.sin(angle_rad))
        end_y = int(center_y - radius * 0.7 * np.cos(angle_rad))
        cv2.arrowedLine(overlay, (center_x, center_y), (end_x, end_y),
                       (0, 255, 0), 2)
        
        return overlay


class TrajectoryMapGenerator:
    """Generate maps showing drone trajectory"""
    
    def __init__(self, telemetry_sequence, config: Optional[Dict] = None):
        """
        Initialize map generator
        Args:
            telemetry_sequence: TelemetrySequence object
        """
        self.telemetry_sequence = telemetry_sequence
        self.config = config or {}
        visualization_config = self.config.get('visualization', {})
        self.trajectory_offset_east_m = float(visualization_config.get('trajectory_offset_east_m', 0.0))
        self.trajectory_offset_north_m = float(visualization_config.get('trajectory_offset_north_m', 0.0))

    def _offset_latlon(self, latitude: float, longitude: float) -> Tuple[float, float]:
        if abs(self.trajectory_offset_east_m) < 1e-9 and abs(self.trajectory_offset_north_m) < 1e-9:
            return float(latitude), float(longitude)

        meters_per_deg_lat = 111320.0
        meters_per_deg_lon = max(111320.0 * np.cos(np.radians(float(latitude))), 1e-6)
        adjusted_lat = float(latitude) + (self.trajectory_offset_north_m / meters_per_deg_lat)
        adjusted_lon = float(longitude) + (self.trajectory_offset_east_m / meters_per_deg_lon)
        return adjusted_lat, adjusted_lon

    def _offset_trajectory(self, trajectory: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
        return [(*self._offset_latlon(lat, lon), alt) for lat, lon, alt in trajectory]

    @staticmethod
    def _trajectory_bounds(trajectory: List[Tuple[float, float, float]]) -> Optional[Dict[str, float]]:
        if not trajectory:
            return None

        latitudes = [point[0] for point in trajectory]
        longitudes = [point[1] for point in trajectory]
        altitudes = [point[2] for point in trajectory]
        return {
            'north': max(latitudes),
            'south': min(latitudes),
            'east': max(longitudes),
            'west': min(longitudes),
            'max_altitude': max(altitudes),
            'min_altitude': min(altitudes),
        }

    @staticmethod
    def _detection_altitude_range(detection_points: Optional[List[Dict]]) -> Tuple[float, float]:
        altitudes = [float(point.get('altitude', 0.0)) for point in (detection_points or [])]
        if not altitudes:
            return 0.0, 1.0
        minimum = min(altitudes)
        maximum = max(altitudes)
        if abs(maximum - minimum) < 1e-6:
            return minimum, minimum + 1.0
        return minimum, maximum

    @staticmethod
    def _altitude_color(altitude_msl: float, altitude_min: float, altitude_max: float) -> str:
        ratio = (float(altitude_msl) - altitude_min) / max(altitude_max - altitude_min, 1e-6)
        ratio = max(0.0, min(1.0, ratio))
        red = int(255 * ratio)
        green = int(190 - (110 * ratio))
        blue = int(255 * (1.0 - ratio))
        return f'#{red:02x}{green:02x}{blue:02x}'

    @staticmethod
    def _detection_altitude_display(point: Dict) -> Tuple[float, str, str]:
        if point.get('altitude_reference') == 'takeoff_relative':
            value = float(point.get('relative_altitude_m', point.get('altitude', 0.0)))
            return value, f'Height (relative to takeoff plane): {value:.1f}m', f'{value:.1f}m rel'

        value = float(point.get('altitude', 0.0))
        return value, f'Elevation (MSL): {value:.1f}m', f'{value:.1f}m MSL'
    
    def create_trajectory_map(self, output_path: str = 'trajectory_map.html', detection_points: Optional[List[Dict]] = None):
        """
        Create interactive map showing flight path
        Args:
            output_path: where to save HTML map
        """
        trajectory = self._offset_trajectory(self.telemetry_sequence.get_trajectory())
        bounds = self._trajectory_bounds(trajectory)
        
        if not trajectory or not bounds:
            logger.error("No trajectory data available")
            return False
        
        # Center map on flight area
        center_lat = (bounds['north'] + bounds['south']) / 2
        center_lon = (bounds['east'] + bounds['west']) / 2
        
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=16,
            tiles=None,
        )
        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Tiles &copy; Esri',
            name='Satellite',
            overlay=False,
            control=True,
            show=True,
        ).add_to(m)
        folium.TileLayer(
            tiles='OpenStreetMap',
            name='Street',
            overlay=False,
            control=True,
            show=False,
        ).add_to(m)
        
        # Draw trajectory line
        folium.PolyLine(
            [(lat, lon) for lat, lon, alt in trajectory],
            weight=3,
            color='blue',
            opacity=0.7,
            popup='Flight Path'
        ).add_to(m)
        
        # Mark waypoints
        for i, (lat, lon, alt) in enumerate(trajectory):
            # Color based on altitude
            if i == 0:
                color = 'green'
                prefix = 'Start'
            elif i == len(trajectory) - 1:
                color = 'red'
                prefix = 'End'
            else:
                color = 'blue'
                prefix = f'#{i}'
            
            popup_text = f"{prefix}<br>Lat: {lat:.6f}<br>Lon: {lon:.6f}<br>Alt: {alt:.1f}m"
            
            folium.CircleMarker(
                location=[lat, lon],
                radius=8 if color == 'green' or color == 'red' else 5,
                popup=popup_text,
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.7,
            ).add_to(m)
        
        if detection_points:
            altitude_min, altitude_max = self._detection_altitude_range(detection_points)
            for point in detection_points:
                altitude_value, altitude_line, tooltip_text = self._detection_altitude_display(point)
                marker_color = self._altitude_color(altitude_value, altitude_min, altitude_max)
                popup_text = (
                    f"{point.get('class', 'object').title()} #{point.get('track_id', '?')}<br>"
                    f"Lat: {point['latitude']:.6f}<br>"
                    f"Lon: {point['longitude']:.6f}<br>"
                    f"{altitude_line}<br>"
                    f"Method: {point.get('method', 'unknown')}<br>"
                    f"Confidence: {point.get('max_confidence', point.get('confidence', 0.0)):.2%}"
                )
                folium.CircleMarker(
                    location=[point['latitude'], point['longitude']],
                    radius=4,
                    popup=popup_text,
                    tooltip=tooltip_text,
                    color=marker_color,
                    fill=True,
                    fillColor=marker_color,
                    fillOpacity=0.95,
                    weight=1,
                ).add_to(m)
        
        # Add bounds box
        bounds_box = [
            [bounds['south'], bounds['west']],
            [bounds['north'], bounds['east']],
        ]
        folium.Rectangle(
            bounds_box,
            color='red',
            weight=2,
            fill=False,
            popup='Survey Area'
        ).add_to(m)
        
        # Add minimap plugin
        minimap = plugins.MiniMap(toggle_display=True)
        m.add_child(minimap)
        folium.LayerControl(collapsed=True).add_to(m)
        
        m.save(output_path)
        logger.info(f"Trajectory map saved to {output_path}")
        return True

    def create_trajectory_overview_image(
        self,
        output_path: str = 'trajectory_overview.png',
        detection_points: Optional[List[Dict]] = None,
    ):
        """Create a static trajectory overview image in local meter coordinates."""
        if plt is None:
            logger.error("Matplotlib is not available for static map export")
            return False

        raw_trajectory = self.telemetry_sequence.get_trajectory()
        trajectory = self._offset_trajectory(raw_trajectory)
        bounds = self._trajectory_bounds(trajectory)
        raw_bounds = self.telemetry_sequence.get_bounds()
        if not trajectory or not bounds or not raw_bounds:
            logger.error("No trajectory data available")
            return False

        center_lat = (raw_bounds['north'] + raw_bounds['south']) / 2.0
        center_lon = (raw_bounds['east'] + raw_bounds['west']) / 2.0
        meters_per_deg_lat = 111320.0
        meters_per_deg_lon = 111320.0 * np.cos(np.radians(center_lat))

        def to_local_xy(lat: float, lon: float) -> Tuple[float, float]:
            x_m = (float(lon) - center_lon) * meters_per_deg_lon
            y_m = (float(lat) - center_lat) * meters_per_deg_lat
            return x_m, y_m

        trajectory_xy = np.array([to_local_xy(lat, lon) for lat, lon, _ in trajectory], dtype=float)

        figure, axis = plt.subplots(figsize=(10, 8), dpi=180)
        axis.set_facecolor('#f6f4ef')
        figure.patch.set_facecolor('white')

        axis.plot(
            trajectory_xy[:, 0],
            trajectory_xy[:, 1],
            color='#2457c5',
            linewidth=2.2,
            alpha=0.9,
            label='Flight path',
            zorder=2,
        )
        axis.scatter(
            trajectory_xy[1:-1, 0] if len(trajectory_xy) > 2 else trajectory_xy[:, 0],
            trajectory_xy[1:-1, 1] if len(trajectory_xy) > 2 else trajectory_xy[:, 1],
            s=12,
            color='#6f90d9',
            alpha=0.7,
            zorder=3,
        )
        axis.scatter(
            trajectory_xy[0, 0],
            trajectory_xy[0, 1],
            s=90,
            color='#179b45',
            edgecolors='white',
            linewidths=0.9,
            label='Start',
            zorder=4,
        )
        axis.scatter(
            trajectory_xy[-1, 0],
            trajectory_xy[-1, 1],
            s=90,
            color='#d84a3a',
            edgecolors='white',
            linewidths=0.9,
            label='End',
            zorder=4,
        )

        plot_xy = trajectory_xy
        if detection_points:
            altitude_min, altitude_max = self._detection_altitude_range(detection_points)
            detection_xy = np.array(
                [to_local_xy(point['latitude'], point['longitude']) for point in detection_points],
                dtype=float,
            )
            plot_xy = np.vstack([trajectory_xy, detection_xy])
            colors = [
                self._altitude_color(float(point.get('altitude', 0.0)), altitude_min, altitude_max)
                for point in detection_points
            ]
            axis.scatter(
                detection_xy[:, 0],
                detection_xy[:, 1],
                s=22,
                c=colors,
                edgecolors='black',
                linewidths=0.25,
                alpha=0.95,
                label='Mapped detections',
                zorder=5,
            )

        span_x = max(float(plot_xy[:, 0].max() - plot_xy[:, 0].min()), 1e-9)
        span_y = max(float(plot_xy[:, 1].max() - plot_xy[:, 1].min()), 1e-9)
        padding_x = max(abs(span_x) * 0.08, 15.0)
        padding_y = max(abs(span_y) * 0.08, 15.0)
        axis.set_xlim(plot_xy[:, 0].min() - padding_x, plot_xy[:, 0].max() + padding_x)
        axis.set_ylim(plot_xy[:, 1].min() - padding_y, plot_xy[:, 1].max() + padding_y)
        axis.set_aspect('equal', adjustable='box')
        axis.grid(True, linestyle='--', linewidth=0.6, color='#b8b5ac', alpha=0.6)
        axis.set_xlabel('Easting offset (m)')
        axis.set_ylabel('Northing offset (m)')
        axis.set_title('Flight Trajectory Overview')
        axis.legend(loc='best')

        info_lines = [
            f"Frames: {len(trajectory)}",
            f"Altitude range: {bounds['min_altitude']:.1f}m to {bounds['max_altitude']:.1f}m MSL",
        ]
        if detection_points:
            info_lines.append(f"Mapped detections: {len(detection_points)}")
        axis.text(
            0.02,
            0.98,
            '\n'.join(info_lines),
            transform=axis.transAxes,
            va='top',
            ha='left',
            fontsize=9,
            bbox={'boxstyle': 'round,pad=0.35', 'facecolor': 'white', 'edgecolor': '#c8c3b8', 'alpha': 0.92},
            zorder=6,
        )

        figure.tight_layout()
        figure.savefig(output_path, bbox_inches='tight')
        plt.close(figure)
        logger.info(f"Trajectory overview image saved to {output_path}")
        return True

    def create_altitude_profile(self, output_path: str = 'altitude_profile.png'):
        """Create altitude profile chart"""
        try:
            import matplotlib.pyplot as plt
            trajectory = self.telemetry_sequence.get_trajectory()
            if not trajectory:
                logger.error("No trajectory data")
                return False
            distances = []
            altitudes = []
            cumdist = 0
            for i, (lat, lon, alt) in enumerate(trajectory):
                altitudes.append(alt)
                if i > 0:
                    prev_lat, prev_lon = trajectory[i-1][0], trajectory[i-1][1]
                    dlat = (lat - prev_lat) * 111000
                    dlon = (lon - prev_lon) * 111000 * np.cos(np.radians(lat))
                    dist = (dlat**2 + dlon**2)**0.5
                    cumdist += dist
                distances.append(cumdist)
            plt.figure(figsize=(8,3))
            plt.plot(distances, altitudes, '-o')
            plt.xlabel('Distance (m)')
            plt.ylabel('Altitude (m)')
            plt.title('Altitude Profile')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
            logger.info(f"Altitude profile saved to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Altitude profile error: {e}")
            return False

    @staticmethod
    def _burn_detections_into_frame(
        frame: np.ndarray,
        detections: Optional[List[Dict]],
        source_width: int,
        source_height: int,
    ) -> np.ndarray:
        if frame is None or not detections:
            return frame

        output = frame.copy()
        target_height, target_width = output.shape[:2]
        scale_x = target_width / max(float(source_width), 1.0)
        scale_y = target_height / max(float(source_height), 1.0)

        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            left = int(round(x1 * scale_x))
            top = int(round(y1 * scale_y))
            right = int(round(x2 * scale_x))
            bottom = int(round(y2 * scale_y))
            cv2.rectangle(output, (left, top), (right, bottom), (48, 48, 255), 2)

            label = f"{str(detection.get('class', 'object')).upper()} {float(detection.get('confidence', 0.0)) * 100:.0f}%"
            (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_top = max(0, top - label_height - baseline - 4)
            label_bottom = label_top + label_height + baseline + 4
            label_right = min(target_width, left + label_width + 8)
            cv2.rectangle(output, (left, label_top), (label_right, label_bottom), (24, 24, 160), thickness=-1)
            cv2.putText(
                output,
                label,
                (left + 4, label_bottom - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return output
    
    def _encode_video_variants(
        self,
        out_dir: Path,
        fps: int = 5,
        detections_by_frame: Optional[List[List[Dict]]] = None,
    ) -> List[Tuple[str, str]]:
        """Encode a browser-playable WebM file and return available (filename, mime_type)."""
        images = self.telemetry_sequence.images
        if not images:
            return []

        first = cv2.imread(str(images[0]))
        if first is None:
            return []

        h, w = first.shape[:2]
        max_width = 1280
        if w > max_width:
            scale = max_width / float(w)
            w = int(w * scale)
            h = int(h * scale)

        video_path = out_dir / 'hyperlapse.webm'
        fourcc = cv2.VideoWriter_fourcc(*'VP80')
        writer = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))
        if not writer.isOpened():
            logger.warning('Video writer failed to open for hyperlapse.webm using codec VP80')
            writer.release()
            return []

        for frame_index, img_path in enumerate(images):
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            source_height, source_width = frame.shape[:2]
            if frame.shape[1] != w or frame.shape[0] != h:
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            frame = self._burn_detections_into_frame(
                frame,
                detections_by_frame[frame_index] if detections_by_frame and frame_index < len(detections_by_frame) else None,
                source_width,
                source_height,
            )
            writer.write(frame)
        writer.release()

        if video_path.exists() and video_path.stat().st_size > 0:
            logger.info(f"Video encoded to {video_path}")
            return [('hyperlapse.webm', 'video/webm')]

        return []

    def _prepare_preview_frames(
        self,
        out_dir: Path,
        max_width: int = 1280,
        detections_by_frame: Optional[List[List[Dict]]] = None,
    ) -> List[str]:
        """Create downscaled JPG preview frames for browser-side fallback playback."""
        frames_dir = out_dir / 'frames'
        frames_dir.mkdir(parents=True, exist_ok=True)

        frame_paths = []
        for index, img_path in enumerate(self.telemetry_sequence.images):
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue

            source_height, source_width = frame.shape[:2]
            height, width = source_height, source_width
            if width > max_width:
                scale = max_width / float(width)
                frame = cv2.resize(frame, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)

            frame = self._burn_detections_into_frame(
                frame,
                detections_by_frame[index] if detections_by_frame and index < len(detections_by_frame) else None,
                source_width,
                source_height,
            )

            output_name = f"frame_{index:04d}.jpg"
            output_path = frames_dir / output_name
            cv2.imwrite(str(output_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            frame_paths.append(f"frames/{output_name}")

        return frame_paths

    def create_interactive_video_viewer(
        self,
        output_path: str,
        detections_by_frame: Optional[List[List[Dict]]] = None,
        map_points: Optional[List[Dict]] = None,
    ) -> bool:
        """Create an interactive HTML video viewer with telemetry overlay and toggleable minimap"""
        if not self.telemetry_sequence:
            logger.error("No telemetry data available")
            return False
        
        trajectory = self._offset_trajectory(self.telemetry_sequence.get_trajectory())
        bounds = self._trajectory_bounds(trajectory)
        
        if not trajectory:
            logger.error("No trajectory data available")
            return False
        
        import json
        out_dir = Path(output_path).parent

        # Encode browser-playable video variants for smooth native playback
        fps = 5
        logger.info("Encoding image sequence to video...")
        video_sources = self._encode_video_variants(out_dir, fps=fps, detections_by_frame=detections_by_frame)
        frame_files = self._prepare_preview_frames(out_dir, detections_by_frame=detections_by_frame)

        telemetry_json = json.dumps(self.telemetry_sequence.telemetry)
        trajectory_json = json.dumps([[lat, lon] for lat, lon, alt in trajectory])
        frame_files_json = json.dumps(frame_files)
        detections_json = json.dumps(detections_by_frame or [])
        map_points_json = json.dumps(map_points or [])
        report_name_json = json.dumps(out_dir.name)
        build_note_json = json.dumps("Latest build: HUD overlay refresh with restored attitude telemetry")
        video_source_count_json = json.dumps(len(video_sources))
        video_sources_html = '\n'.join(
            [f'            <source src="{filename}" type="{mime_type}">' for filename, mime_type in video_sources]
        )
        
        html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DJI Hyperlapse Fire Mapping Viewer</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/leaflet@1.9.3/dist/leaflet.css"/>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <script src="https://cdn.jsdelivr.net/npm/leaflet@1.9.3/dist/leaflet.js"></script>
    <style>
        *, *::before, *::after { box-sizing: border-box; }
        html, body {
            width: 100%; height: 100%; margin: 0; padding: 0;
            font-family: 'Segoe UI', sans-serif;
            background: #111; color: white; overflow: hidden;
        }
        .viewer-container { display: flex; flex-direction: column; width: 100%; height: 100%; }
        .video-section { flex: 1; position: relative; background: radial-gradient(circle at top, #3b3b3b 0%, #242424 38%, #111 100%); overflow: hidden; display: flex; align-items: center; justify-content: center; }
        .report-banner {
            position: absolute; top: 14px; left: 50%; transform: translateX(-50%);
            z-index: 960;
            padding: 8px 14px;
            border-radius: 999px;
            border: 1px solid #4a4a4a;
            background: rgba(0, 0, 0, 0.78);
            backdrop-filter: blur(10px);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
            font-size: 12px;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #d9d9d9;
            white-space: nowrap;
        }
        .build-note {
            position: absolute; left: 20px; bottom: 96px;
            z-index: 930;
            padding: 4px 10px;
            border-radius: 999px;
            border: 1px solid rgba(99, 255, 182, 0.24);
            background: rgba(0, 20, 12, 0.46);
            color: rgba(120, 255, 202, 0.88);
            font-size: 10px;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            white-space: nowrap;
        }
        #main-video {
            max-width: 100%; max-height: 100%;
            width: 100%; height: 100%;
            object-fit: contain;
            outline: none;
        }
        #frame-fallback {
            max-width: 100%; max-height: 100%;
            width: 100%; height: 100%; object-fit: contain;
        }
        .detection-layer {
            position: absolute;
            inset: 0;
            pointer-events: none;
            z-index: 944;
        }
        .detection-box {
            position: absolute;
            border: 2px solid rgba(255, 64, 64, 0.95);
            box-shadow: 0 0 16px rgba(255, 64, 64, 0.35);
            background: rgba(255, 40, 40, 0.06);
        }
        .detection-label {
            position: absolute;
            top: -20px;
            left: -2px;
            padding: 2px 6px;
            background: rgba(120, 0, 0, 0.92);
            color: #fff;
            font-size: 11px;
            font-weight: 700;
            border-radius: 4px;
            letter-spacing: 0.03em;
            white-space: nowrap;
        }
        .azimuth-bar,
        .elevation-bar,
        .hud-ladder,
        .hud-horizon {
            display: none;
        }
        .hud-center {
            position: absolute;
            left: 50%;
            top: 50%;
            transform: translate(-50%, -50%);
            width: 12px;
            height: 12px;
            z-index: 938;
            pointer-events: none;
        }
        .hud-reticle {
            position: absolute;
            left: 50%;
            top: 50%;
            width: 8px;
            height: 8px;
            transform: translate(-50%, -50%);
            border: none;
            border-radius: 50%;
            background: #63ffb6;
            box-shadow: 0 0 10px rgba(99, 255, 182, 0.8);
        }
        .hud-reticle::before,
        .hud-reticle::after,
        .hud-reticle-inner {
            display: none;
        }
        .azimuth-bar {
            position: absolute;
            top: 62px;
            left: 50%;
            transform: translateX(-50%);
            width: min(460px, calc(100vw - 420px));
            z-index: 950;
        }
        .elevation-bar {
            position: absolute;
            top: 50%;
            right: 20px;
            transform: translateY(-50%);
            height: min(360px, calc(100vh - 220px));
            z-index: 950;
        }
        .azimuth-ruler {
            position: relative;
            padding-top: 6px;
            color: #63ffb6;
            text-align: center;
            pointer-events: none;
        }
        .azimuth-label {
            margin-bottom: 10px;
            font-size: 11px;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            color: rgba(153, 255, 216, 0.72);
        }
        .azimuth-scale {
            position: relative;
            height: 38px;
            margin: 0 14px 10px;
            overflow: hidden;
        }
        .azimuth-line {
            position: absolute;
            left: 0;
            right: 0;
            top: 18px;
            height: 1px;
            background: linear-gradient(90deg, rgba(99,255,182,0) 0%, rgba(99,255,182,0.85) 18%, rgba(99,255,182,0.85) 82%, rgba(99,255,182,0) 100%);
        }
        .azimuth-line::before {
            content: '';
            position: absolute;
            inset: -10px 0 -10px 0;
            background:
                linear-gradient(90deg, transparent 0, transparent calc(50% - 1px), rgba(99,255,182,0.95) calc(50% - 1px), rgba(99,255,182,0.95) calc(50% + 1px), transparent calc(50% + 1px), transparent 100%),
                repeating-linear-gradient(90deg, transparent 0 34px, rgba(99,255,182,0.42) 34px 35px, transparent 35px 68px);
            opacity: 0.8;
        }
        .azimuth-ticks {
            position: absolute;
            top: -2px;
            left: 50%;
            display: flex;
            align-items: center;
            font-size: 18px;
            letter-spacing: 0.08em;
            color: rgba(153, 255, 216, 0.8);
            white-space: nowrap;
            transform: translateX(-50%);
            transition: transform 0.12s linear;
            will-change: transform;
        }
        .azimuth-ticks span {
            min-width: 34px;
            text-align: center;
            display: inline-block;
        }
        .azimuth-value {
            font-size: 46px;
            line-height: 1;
            font-weight: 700;
            text-shadow: 0 0 10px rgba(99, 255, 182, 0.18);
        }
        .elevation-ruler {
            position: relative;
            width: 112px;
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            pointer-events: none;
        }
        .elevation-track {
            position: absolute;
            top: 10px;
            bottom: 10px;
            right: 36px;
            width: 1px;
            background: linear-gradient(180deg, rgba(99,255,182,0) 0%, rgba(99,255,182,0.84) 15%, rgba(99,255,182,0.84) 85%, rgba(99,255,182,0) 100%);
        }
        .elevation-pointer {
            position: absolute;
            right: 25px;
            top: 50%;
            width: 20px;
            height: 1px;
            background: #63ffb6;
            box-shadow: 0 0 7px rgba(99, 255, 182, 0.45);
            transition: transform 0.12s linear;
            will-change: transform;
        }
        .elevation-track::before {
            content: '';
            position: absolute;
            inset: 0 -18px;
            background:
                linear-gradient(180deg, transparent 0, transparent calc(50% - 1px), rgba(99,255,182,0.95) calc(50% - 1px), rgba(99,255,182,0.95) calc(50% + 1px), transparent calc(50% + 1px), transparent 100%),
                repeating-linear-gradient(180deg, transparent 0 30px, rgba(99,255,182,0.62) 30px 32px, transparent 32px 62px);
        }
        .elevation-label {
            position: absolute;
            right: 0;
            top: 50%;
            transform: translateY(-50%);
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.14em;
            color: rgba(153, 255, 216, 0.72);
            writing-mode: vertical-rl;
            text-orientation: mixed;
        }
        .elevation-value {
            position: absolute;
            right: 52px;
            top: 50%;
            transform: translateY(-50%);
            color: #63ffb6;
            font-size: 30px;
            font-weight: 700;
            line-height: 1;
            text-shadow: 0 0 10px rgba(99, 255, 182, 0.16);
        }
        .telemetry-overlay {
            position: absolute; top: 14px; left: 14px;
            background: rgba(0, 14, 8, 0.38);
            padding: 14px 16px; border-radius: 14px;
            font-family: 'Courier New', monospace; font-size: 13px;
            width: 320px; max-width: calc(100vw - 32px); border: 1px solid rgba(99, 255, 182, 0.28); z-index: 940;
            backdrop-filter: blur(8px);
            color: #63ffb6;
            box-shadow: inset 0 0 24px rgba(99, 255, 182, 0.08);
        }
        .telemetry-title { font-weight: bold; color: #63ffb6; margin-bottom: 10px; font-size: 14px; letter-spacing: 0.12em; }
        .telemetry-item { margin: 3px 0; display: flex; justify-content: space-between; gap: 8px; }
        .telemetry-label { color: rgba(153, 255, 216, 0.78); text-transform: uppercase; letter-spacing: 0.06em; font-size: 11px; }
        .telemetry-value { color: #63ffb6; font-weight: bold; text-align: right; }
        .hud-center {
            position: absolute;
            left: 50%; top: 50%; transform: translate(-50%, -50%);
            width: 360px; height: 280px; z-index: 938; pointer-events: none;
        }
        .hud-ladder {
            position: absolute;
            left: 50%; top: 50%; width: 280px; height: 160px;
            transform: translate(-50%, -50%);
            opacity: 0.62;
        }
        .hud-ladder::before {
            content: '';
            position: absolute;
            inset: 0;
            background:
                linear-gradient(180deg, transparent 0, transparent calc(50% - 1px), rgba(99,255,182,0.5) calc(50% - 1px), rgba(99,255,182,0.5) calc(50% + 1px), transparent calc(50% + 1px), transparent 100%),
                repeating-linear-gradient(180deg, transparent 0 23px, rgba(99,255,182,0.2) 23px 24px, transparent 24px 48px);
            mask: linear-gradient(90deg, transparent 0, black 18%, black 82%, transparent 100%);
        }
        .hud-horizon {
            position: absolute;
            left: 50%; top: 50%; width: 250px; height: 2px;
            transform: translate(-50%, -50%);
            background: linear-gradient(90deg, rgba(0,0,0,0) 0%, #63ffb6 18%, #63ffb6 82%, rgba(0,0,0,0) 100%);
            box-shadow: 0 0 6px rgba(99, 255, 182, 0.28);
        }
        .hud-horizon::before,
        .hud-horizon::after {
            content: '';
            position: absolute;
            top: -12px;
            width: 34px;
            height: 26px;
            border-top: 2px solid rgba(99, 255, 182, 0.95);
        }
        .hud-horizon::before {
            left: 28px;
            border-left: 2px solid rgba(99, 255, 182, 0.95);
        }
        .hud-horizon::after {
            right: 28px;
            border-right: 2px solid rgba(99, 255, 182, 0.95);
        }
        .hud-reticle {
            position: absolute;
            left: 50%; top: 50%; width: 58px; height: 58px;
            transform: translate(-50%, -50%);
            border: 1px solid rgba(99, 255, 182, 0.85);
            border-radius: 50%;
            box-shadow: 0 0 8px rgba(99, 255, 182, 0.18);
            background: radial-gradient(circle, rgba(99,255,182,0.08) 0%, rgba(99,255,182,0.02) 45%, rgba(0,0,0,0) 70%);
        }
        .hud-reticle::before,
        .hud-reticle::after {
            content: '';
            position: absolute;
            background: #63ffb6;
            box-shadow: 0 0 5px rgba(99, 255, 182, 0.25);
        }
        .hud-reticle::before {
            left: 50%; top: -16px; width: 1px; height: 90px; transform: translateX(-50%);
        }
        .hud-reticle::after {
            top: 50%; left: -16px; width: 90px; height: 1px; transform: translateY(-50%);
        }
        .hud-reticle-inner {
            position: absolute;
            left: 50%; top: 50%; width: 12px; height: 12px;
            transform: translate(-50%, -50%);
            border: 1px solid rgba(99, 255, 182, 0.9);
            border-radius: 50%;
            box-shadow: 0 0 6px rgba(99, 255, 182, 0.22);
        }
        .controls {
            background: #1a1a1a; border-top: 1px solid #333;
            padding: 10px 16px; display: flex; align-items: center; gap: 10px;
        }
        .control-btn {
            background: #2a2a2a; border: 1px solid #444; color: white;
            padding: 7px 12px; border-radius: 7px; cursor: pointer;
            font-size: 14px; transition: background 0.15s; white-space: nowrap;
        }
        .control-btn:hover { background: #383838; }
        .slider-wrap { flex: 1; display: flex; align-items: center; gap: 8px; }
        .time-display { font-size: 12px; color: #aaa; min-width: 42px; text-align: center; }
        input[type=range] {
            flex: 1; -webkit-appearance: none; height: 5px;
            border-radius: 3px; background: #444; outline: none; cursor: pointer;
        }
        input[type=range]::-webkit-slider-thumb {
            -webkit-appearance: none; width: 14px; height: 14px;
            border-radius: 50%; background: #00ff88; cursor: pointer; border: 2px solid #111;
        }
        .minimap-container {
            position: absolute; top: 14px; right: 14px;
            width: 260px; background: rgba(0,0,0,0.85);
            border-radius: 10px; border: 1px solid #444; overflow: hidden; z-index: 900;
        }
        .minimap-container.collapsed { width: 90px; }
        .panel-header {
            padding: 7px 11px; background: rgba(0,0,0,0.9);
            cursor: pointer; display: flex; align-items: center; justify-content: space-between;
            font-size: 13px; font-weight: bold; user-select: none;
        }
        .panel-header span { font-size: 11px; color: #aaa; }
        .minimap-body { height: 170px; }
        .minimap-body.collapsed { display: none; }
        #minimap { width: 100%; height: 100%; }
        .profile-panel {
            position: absolute;
            right: 14px;
            bottom: 84px;
            width: 344px;
            display: flex;
            align-items: stretch;
            z-index: 920;
            transform: translateX(300px);
            transition: transform 0.22s ease;
        }
        .profile-panel.open { transform: translateX(0); }
        .profile-tab-button {
            width: 44px;
            border: 1px solid #444;
            border-right: none;
            border-radius: 12px 0 0 12px;
            background: rgba(0, 0, 0, 0.86);
            color: #fff;
            cursor: pointer;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            gap: 8px;
            padding: 12px 6px;
            font: inherit;
            writing-mode: vertical-rl;
            text-orientation: mixed;
        }
        .profile-content {
            width: 300px;
            background: rgba(0, 0, 0, 0.86);
            border: 1px solid #444;
            border-radius: 0 12px 12px 12px;
            overflow: hidden;
            backdrop-filter: blur(10px);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
        }
        .profile-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 10px 12px;
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: #d2d2d2;
            background: rgba(255, 255, 255, 0.04);
        }
        .profile-body {
            padding: 10px;
        }
        .profile-image {
            width: 100%;
            display: block;
            border-radius: 8px;
            border: 1px solid #333;
            background: #121212;
        }
        @media (max-width: 980px) {
            .report-banner {
                top: 10px;
                max-width: calc(100vw - 28px);
                overflow: hidden;
                text-overflow: ellipsis;
            }
            .build-note {
                left: 14px;
                bottom: 128px;
                max-width: calc(100vw - 28px);
                overflow: hidden;
                text-overflow: ellipsis;
                transform: none;
            }
            .azimuth-bar {
                top: 56px;
                width: calc(100vw - 60px);
            }
            .elevation-bar {
                top: auto;
                bottom: 104px;
                right: 14px;
                transform: none;
                height: 180px;
            }
            .profile-panel {
                right: 14px;
                left: 14px;
                width: auto;
                bottom: 132px;
                transform: translateY(calc(100% - 44px));
                flex-direction: column-reverse;
            }
            .profile-panel.open {
                transform: translateY(0);
            }
            .profile-tab-button {
                width: 100%;
                writing-mode: horizontal-tb;
                text-orientation: initial;
                border-right: 1px solid #444;
                border-top: none;
                border-radius: 0 0 12px 12px;
                flex-direction: row;
                padding: 10px 12px;
            }
            .profile-content {
                width: 100%;
                border-radius: 12px 12px 0 0;
            }
            .telemetry-overlay {
                width: 250px;
                font-size: 12px;
            }
            .azimuth-value { font-size: 34px; }
            .azimuth-ticks { font-size: 14px; }
            .elevation-value { font-size: 26px; }
            .hud-center { width: 250px; height: 220px; }
            .hud-ladder { width: 220px; height: 120px; }
            .hud-horizon { width: 180px; }
        }
    </style>
</head>
<body>
<div class="viewer-container">
    <div class="video-section">
        <div class="report-banner" id="report-name"></div>
        <div class="build-note" id="build-note"></div>
        <video id="main-video" preload="auto" playsinline>
    {video_sources_html}
            Your browser could not load the generated video.
        </video>
        <img id="frame-fallback" alt="Frame preview" style="display:none;" />
        <div class="detection-layer" id="detection-layer"></div>

        <div class="hud-center">
            <div class="hud-ladder"></div>
            <div class="hud-horizon" id="hud-horizon"></div>
            <div class="hud-reticle"><div class="hud-reticle-inner"></div></div>
        </div>

        <div class="azimuth-bar">
            <div class="azimuth-ruler">
                <div class="azimuth-label">Azimuth Relative To Earth</div>
                <div class="azimuth-scale">
                    <div class="azimuth-line"></div>
                    <div class="azimuth-ticks" id="azimuth-ticks"></div>
                </div>
                <div class="azimuth-value" id="earth-azimuth">--</div>
            </div>
        </div>

        <div class="elevation-bar">
            <div class="elevation-ruler">
                <div class="elevation-track"></div>
                <div class="elevation-pointer" id="elevation-pointer"></div>
                <div class="elevation-value" id="earth-elevation">--</div>
                <div class="elevation-label">Elevation Relative To Earth</div>
            </div>
        </div>

        <div class="telemetry-overlay">
            <div class="telemetry-title">HUD TELEMETRY</div>
            <div class="telemetry-item"><span class="telemetry-label">Frame:</span><span class="telemetry-value" id="t-frame">--</span></div>
            <div class="telemetry-item"><span class="telemetry-label">GPS:</span><span class="telemetry-value" id="t-gps">--</span></div>
            <div class="telemetry-item"><span class="telemetry-label">Altitude MSL:</span><span class="telemetry-value" id="t-alt">--</span></div>
            <div class="telemetry-item"><span class="telemetry-label">Camera Heading:</span><span class="telemetry-value" id="t-heading">--</span></div>
            <div class="telemetry-item"><span class="telemetry-label">Camera Attitude:</span><span class="telemetry-value" id="t-camera-attitude">--</span></div>
            <div class="telemetry-item"><span class="telemetry-label">Gimbal:</span><span class="telemetry-value" id="t-gimbal">--</span></div>
            <div class="telemetry-item"><span class="telemetry-label">Speed:</span><span class="telemetry-value" id="t-speed">--</span></div>
        </div>

        <div class="minimap-container collapsed" id="minimap-container">
            <div class="panel-header" onclick="toggleMinimap()">
                <div>&#128506; Map</div><span id="map-toggle">&#9654;</span>
            </div>
            <div class="minimap-body collapsed" id="minimap-body">
                <div id="minimap"></div>
            </div>
        </div>

        <div class="profile-panel" id="profile-panel">
            <button class="profile-tab-button" type="button" onclick="toggleProfilePanel()">
                <span>Altitude Profile</span>
                <span id="profile-toggle">&#9664;</span>
            </button>
            <div class="profile-content">
                <div class="profile-header">
                    <span>Altitude Profile</span>
                    <span>MSL</span>
                </div>
                <div class="profile-body">
                    <img class="profile-image" src="altitude_profile.png" alt="Altitude profile graph" />
                </div>
            </div>
        </div>
    </div>

    <div class="controls">
        <button class="control-btn" id="btn-play" onclick="togglePlay()"><i class="fas fa-play"></i></button>
        <button class="control-btn" onclick="stepFrame(-1)"><i class="fas fa-step-backward"></i></button>
        <button class="control-btn" onclick="stepFrame(1)"><i class="fas fa-step-forward"></i></button>
        <div class="slider-wrap">
            <span class="time-display" id="t-current">0:00</span>
            <input type="range" id="seek-bar" min="0" max="1000" value="0"
                   oninput="onSeek(this.value)" onmousedown="seekStart()" onmouseup="seekEnd()">
            <span class="time-display" id="t-total">0:00</span>
        </div>
        <button class="control-btn" id="btn-speed" onclick="cycleSpeed()">1x</button>
        <button class="control-btn" onclick="toggleMinimap()">Map</button>
    </div>
</div>

<script>
    const telemetryData = {telemetry_json};
    const trajectory   = {trajectory_json};
    const frameFiles   = {frame_files_json};
    const detectionsByFrame = {detections_json};
    const mapDetections = {map_points_json};
    const reportName   = {report_name_json};
    const buildNote    = {build_note_json};
    const videoSourceCount = {video_source_count_json};
    const FPS = {fps};

    const vid   = document.getElementById('main-video');
    const fallbackImg = document.getElementById('frame-fallback');
    const bar   = document.getElementById('seek-bar');
    let isSeeking = false;
    let map = null, mapMarker = null, mapInitialized = false;
    let staticDetectionMarkers = [];
    let useFrameFallback = false;
    let fallbackFrame = 0;
    let fallbackPlaying = false;
    let fallbackRaf = null;
    let fallbackLastTs = 0;
    let fallbackPlaybackRate = 1;
    let azimuthContinuousDeg = null;
    const preloadedFallbackFrames = new Map();

    function frameFromTime(seconds) {
        return Math.max(0, Math.min(Math.floor((seconds * FPS) + 1e-6), telemetryData.length - 1));
    }

    function preloadFallbackFrame(frameIdx) {
        if (!frameFiles.length) return;
        const safeFrame = Math.max(0, Math.min(frameIdx, frameFiles.length - 1));
        if (preloadedFallbackFrames.has(safeFrame)) return;
        const img = new Image();
        img.src = frameFiles[safeFrame];
        preloadedFallbackFrames.set(safeFrame, img);
    }

    function preloadFallbackWindow(centerFrame) {
        for (let offset = 0; offset <= 8; offset += 1) {
            preloadFallbackFrame(centerFrame + offset);
        }
    }

    function cardinalDirection(deg) {
        const headings = ['N', 'NW', 'W', 'SW', 'S', 'SE', 'E', 'NE'];
        const idx = Math.round((((deg % 360) + 360) % 360) / 45) % headings.length;
        return headings[idx];
    }

    function initAzimuthTicks() {
        const ticks = document.getElementById('azimuth-ticks');
        if (!ticks) return;
        ticks.innerHTML = '';
        for (let deg = -720; deg <= 720; deg += 15) {
            const marker = document.createElement('span');
            marker.textContent = (deg % 45 === 0) ? cardinalDirection(deg) : '|';
            ticks.appendChild(marker);
        }
    }

    function unwrapAzimuth(azimuthDeg) {
        const normalized = ((azimuthDeg % 360) + 360) % 360;
        if (azimuthContinuousDeg == null) {
            azimuthContinuousDeg = normalized;
            return azimuthContinuousDeg;
        }
        let candidate = normalized;
        while (candidate - azimuthContinuousDeg > 180) candidate -= 360;
        while (candidate - azimuthContinuousDeg < -180) candidate += 360;
        azimuthContinuousDeg = candidate;
        return azimuthContinuousDeg;
    }

    function updateRulers(azimuthDeg, elevationDeg, hasElevation) {
        const ticks = document.getElementById('azimuth-ticks');
        const scale = document.querySelector('.azimuth-scale');
        if (ticks && scale && azimuthDeg != null) {
            const continuous = unwrapAzimuth(azimuthDeg);
            const pxPerDeg = Math.max(2.2, scale.clientWidth / 140);
            ticks.style.transform = `translateX(calc(-50% - ${continuous * pxPerDeg}px))`;
        }

        const pointer = document.getElementById('elevation-pointer');
        const track = document.querySelector('.elevation-track');
        if (pointer && track) {
            if (!hasElevation) {
                pointer.style.opacity = '0';
                return;
            }
            pointer.style.opacity = '1';
            const minDeg = -90;
            const maxDeg = 30;
            const clamped = Math.max(minDeg, Math.min(maxDeg, elevationDeg || 0));
            const ratio = (clamped - minDeg) / (maxDeg - minDeg);
            const offsetPx = (0.5 - ratio) * track.clientHeight;
            pointer.style.transform = `translateY(${offsetPx}px)`;
        }
    }

    function currentDuration() {
        return useFrameFallback ? (frameFiles.length / FPS) : (vid.duration || 0);
    }

    function currentTimeValue() {
        return useFrameFallback ? (fallbackFrame / FPS) : vid.currentTime;
    }

    function syncFromFrame(frame) {
        const safeFrame = Math.max(0, Math.min(frame, telemetryData.length - 1));
        updateTelemetry(safeFrame);
        renderDetections(safeFrame);
        updateMapMarker(safeFrame);
        document.getElementById('t-current').textContent = fmt(safeFrame / FPS);
        const duration = currentDuration();
        if (duration > 0) {
            bar.value = ((safeFrame / FPS) / duration) * 1000;
        }
    }

    function showFallbackFrame(frame) {
        if (!frameFiles.length) return;
        fallbackFrame = Math.max(0, Math.min(frame, frameFiles.length - 1));
        preloadFallbackWindow(fallbackFrame);
        const cachedImage = preloadedFallbackFrames.get(fallbackFrame);
        fallbackImg.src = cachedImage ? cachedImage.src : frameFiles[fallbackFrame];
        syncFromFrame(fallbackFrame);
    }

    function activateFrameFallback() {
        if (useFrameFallback) return;
        useFrameFallback = true;
        vid.pause();
        vid.style.display = 'none';
        fallbackImg.style.display = 'block';
        document.getElementById('t-total').textContent = fmt(frameFiles.length / FPS);
        showFallbackFrame(fallbackFrame);
    }

    function startFallbackPlayback() {
        if (fallbackPlaying || !frameFiles.length) return;
        fallbackPlaying = true;
        fallbackLastTs = 0;
        document.querySelector('#btn-play i').className = 'fas fa-pause';

        const step = (timestamp) => {
            if (!fallbackPlaying) return;
            if (!fallbackLastTs) fallbackLastTs = timestamp;
            const frameIntervalMs = 1000 / (FPS * fallbackPlaybackRate);
            const elapsed = timestamp - fallbackLastTs;
            if (elapsed >= frameIntervalMs) {
                const framesToAdvance = Math.max(1, Math.floor(elapsed / frameIntervalMs));
                fallbackLastTs += framesToAdvance * frameIntervalMs;
                if (fallbackFrame >= frameFiles.length - 1) {
                    stopFallbackPlayback();
                    return;
                }
                showFallbackFrame(Math.min(fallbackFrame + framesToAdvance, frameFiles.length - 1));
            }
            fallbackRaf = requestAnimationFrame(step);
        };

        fallbackRaf = requestAnimationFrame(step);
    }

    function stopFallbackPlayback() {
        fallbackPlaying = false;
        if (fallbackRaf) {
            cancelAnimationFrame(fallbackRaf);
            fallbackRaf = null;
        }
        document.querySelector('#btn-play i').className = 'fas fa-play';
    }

    // ── Telemetry sync ────────────────────────────────────────────
    vid.addEventListener('timeupdate', () => {
        if (useFrameFallback) return;
        if (!isSeeking) updateSeekBar();
        const frame = frameFromTime(vid.currentTime);
        updateTelemetry(frame);
        renderDetections(frame);
        updateMapMarker(frame);
        document.getElementById('t-current').textContent = fmt(vid.currentTime);
    });

    vid.addEventListener('loadedmetadata', () => {
        document.getElementById('t-total').textContent = fmt(vid.duration);
        document.getElementById('t-current').textContent = fmt(0);
        updateTelemetry(0);
        renderDetections(0);
        updateMapMarker(0);
        updateSeekBar();
    });

    vid.addEventListener('loadeddata', () => {
        updateTelemetry(0);
        renderDetections(0);
        updateMapMarker(0);
    });

    vid.addEventListener('error', () => {
        activateFrameFallback();
    });

    window.addEventListener('load', () => {
        document.getElementById('report-name').textContent = reportName;
        document.getElementById('build-note').textContent = buildNote;
        initAzimuthTicks();
        updateTelemetry(0);
        renderDetections(0);
        preloadFallbackWindow(0);
        if (!videoSourceCount && frameFiles.length) {
            document.getElementById('t-total').textContent = fmt(frameFiles.length / FPS);
            activateFrameFallback();
        } else {
            document.getElementById('t-total').textContent = fmt(frameFiles.length / FPS);
            window.setTimeout(() => {
                if (!useFrameFallback && vid.readyState < 2) {
                    activateFrameFallback();
                }
            }, 1200);
        }
    });

    vid.addEventListener('play',  () => { document.querySelector('#btn-play i').className = 'fas fa-pause'; });
    vid.addEventListener('pause', () => { document.querySelector('#btn-play i').className = 'fas fa-play'; });
    vid.addEventListener('ended', () => { document.querySelector('#btn-play i').className = 'fas fa-play'; });

    function updateTelemetry(idx) {
        const t = telemetryData[idx] || {};
        const g = t.gps || {};
        const d = t.drone || {};
        const azimuth = t.camera_heading ?? null;
        const elevation = t.camera_pitch ?? (t.gimbal && t.gimbal.pitch != null ? t.gimbal.pitch : null);
        const hasElevation = elevation != null;
        const cameraRoll = t.camera_roll ?? (t.gimbal && t.gimbal.roll != null ? t.gimbal.roll : 0);
        const cameraYaw = azimuth ?? (t.gimbal && t.gimbal.yaw != null ? t.gimbal.yaw : null);
        document.getElementById('t-frame').textContent    = idx + 1;
        document.getElementById('t-gps').textContent      = g.latitude  ? `${g.latitude.toFixed(5)}, ${g.longitude.toFixed(5)}` : '--';
        document.getElementById('t-alt').textContent      = g.altitude  ? `${g.altitude.toFixed(1)}m` : '--';
        document.getElementById('t-heading').textContent  = t.camera_heading != null ? `${t.camera_heading.toFixed(1)}\\u00b0` : '--';
        document.getElementById('t-camera-attitude').textContent = hasElevation ? `Y:${cameraYaw != null ? cameraYaw.toFixed(1) : '--'}\\u00b0 P:${elevation.toFixed(1)}\\u00b0 R:${cameraRoll.toFixed(1)}\\u00b0` : '--';
        document.getElementById('t-gimbal').textContent   = t.gimbal ? `P:${t.gimbal.pitch||0}\\u00b0 R:${t.gimbal.roll||0}\\u00b0 Y:${t.gimbal.yaw||0}\\u00b0` : '--';
        const spd = d.speed ?? ((d.speed_x != null) ? Math.sqrt(d.speed_x**2 + d.speed_y**2 + d.speed_z**2) : null);
        document.getElementById('t-speed').textContent    = spd != null ? `${spd.toFixed(1)} m/s` : '--';
        document.getElementById('earth-azimuth').textContent = azimuth != null ? `${azimuth.toFixed(1)}\u00b0` : '--';
        document.getElementById('earth-elevation').textContent = hasElevation ? `${elevation.toFixed(1)}\u00b0` : '--';
        updateRulers(azimuth, elevation, hasElevation);
        updateHudAttitude(cameraRoll, elevation ?? 0);
    }

    function updateHudAttitude(roll, pitch) {
        const horizon = document.getElementById('hud-horizon');
        const clampedPitch = Math.max(-20, Math.min(20, pitch || 0));
        horizon.style.transform = `translate(-50%, calc(-50% + ${clampedPitch * 3}px)) rotate(${roll || 0}deg)`;
    }

    function getVisibleMediaRect(sourceWidth, sourceHeight) {
        const section = document.querySelector('.video-section');
        const element = useFrameFallback ? fallbackImg : vid;
        if (!section || !element || !sourceWidth || !sourceHeight) {
            return null;
        }

        const sectionRect = section.getBoundingClientRect();
        const mediaRect = element.getBoundingClientRect();
        if (!mediaRect.width || !mediaRect.height) {
            return null;
        }

        const sourceAspect = sourceWidth / sourceHeight;
        let drawWidth = mediaRect.width;
        let drawHeight = drawWidth / sourceAspect;
        if (drawHeight > mediaRect.height) {
            drawHeight = mediaRect.height;
            drawWidth = drawHeight * sourceAspect;
        }

        return {
            left: mediaRect.left - sectionRect.left + ((mediaRect.width - drawWidth) / 2),
            top: mediaRect.top - sectionRect.top + ((mediaRect.height - drawHeight) / 2),
            width: drawWidth,
            height: drawHeight,
        };
    }

    function renderDetections(frameIdx) {
        const layer = document.getElementById('detection-layer');
        if (layer) layer.innerHTML = '';
    }

    // ── Playback controls ─────────────────────────────────────────
    function togglePlay() {
        if (useFrameFallback) {
            fallbackPlaying ? stopFallbackPlayback() : startFallbackPlayback();
            return;
        }
        const playPromise = vid.paused ? vid.play() : Promise.resolve(vid.pause());
        if (playPromise && typeof playPromise.catch === 'function') {
            playPromise.catch(() => activateFrameFallback());
        }
    }

    function stepFrame(dir) {
        if (useFrameFallback) {
            stopFallbackPlayback();
            showFallbackFrame(fallbackFrame + dir);
            return;
        }
        vid.pause();
        vid.currentTime = Math.max(0, Math.min(vid.duration, vid.currentTime + dir / FPS));
        const frame = frameFromTime(vid.currentTime);
        updateTelemetry(frame);
        renderDetections(frame);
        updateMapMarker(frame);
        updateSeekBar();
        document.getElementById('t-current').textContent = fmt(vid.currentTime);
    }

    function cycleSpeed() {
        if (useFrameFallback) {
            const speeds = [0.5, 1, 2, 4];
            const idx = speeds.indexOf(fallbackPlaybackRate);
            fallbackPlaybackRate = speeds[(idx + 1) % speeds.length];
            document.getElementById('btn-speed').textContent = fallbackPlaybackRate + 'x';
            return;
        }
        const speeds = [0.25, 0.5, 1, 2, 4];
        const idx = speeds.indexOf(vid.playbackRate);
        vid.playbackRate = speeds[(idx + 1) % speeds.length];
        document.getElementById('btn-speed').textContent = vid.playbackRate + 'x';
    }

    // ── Seek bar ──────────────────────────────────────────────────
    function seekStart() { isSeeking = true; }
    function seekEnd()   {
        isSeeking = false;
        if (useFrameFallback) {
            showFallbackFrame(fallbackFrame);
            return;
        }
        const frame = Math.min(Math.round(vid.currentTime * FPS), telemetryData.length - 1);
        updateTelemetry(frame);
        updateMapMarker(frame);
        updateSeekBar();
        document.getElementById('t-current').textContent = fmt(vid.currentTime);
    }
    function onSeek(val) {
        if (useFrameFallback) {
            const target = Math.round((val / 1000) * (frameFiles.length - 1));
            showFallbackFrame(target);
            return;
        }
        if (vid.duration) {
            vid.currentTime = (val / 1000) * vid.duration;
            const frame = frameFromTime(vid.currentTime);
            updateTelemetry(frame);
            renderDetections(frame);
            updateMapMarker(frame);
            document.getElementById('t-current').textContent = fmt(vid.currentTime);
        }
    }
    function updateSeekBar() {
        if (useFrameFallback) {
            if (frameFiles.length > 1) {
                bar.value = (fallbackFrame / (frameFiles.length - 1)) * 1000;
            }
            return;
        }
        if (vid.duration) bar.value = (vid.currentTime / vid.duration) * 1000;
    }

    // ── Keyboard shortcuts ────────────────────────────────────────
    document.addEventListener('keydown', e => {
        if (e.code === 'Space')      { e.preventDefault(); togglePlay(); }
        if (e.code === 'ArrowRight') { e.preventDefault(); stepFrame(1); }
        if (e.code === 'ArrowLeft')  { e.preventDefault(); stepFrame(-1); }
    });

    // ── Minimap ───────────────────────────────────────────────────
    function toggleMinimap() {
        const c = document.getElementById('minimap-container');
        const b = document.getElementById('minimap-body');
        const t = document.getElementById('map-toggle');
        const open = c.classList.toggle('collapsed');
        b.classList.toggle('collapsed', open);
        t.textContent = open ? '\\u25b6' : '\\u25bc';
        if (!open) {
            if (!mapInitialized) {
                initMap();
            } else {
                map.invalidateSize();
                const frame = useFrameFallback ? fallbackFrame : frameFromTime(vid.currentTime);
                updateMapMarker(frame);
            }
        }
    }

    function toggleProfilePanel() {
        const panel = document.getElementById('profile-panel');
        const toggle = document.getElementById('profile-toggle');
        const open = panel.classList.toggle('open');
        toggle.textContent = open ? '\u25b6' : '\u25c0';
    }

    function initMap() {
        const lats = trajectory.map(p => p[0]);
        const lons = trajectory.map(p => p[1]);
        const cLat = (Math.max(...lats) + Math.min(...lats)) / 2;
        const cLon = (Math.max(...lons) + Math.min(...lons)) / 2;
        map = L.map('minimap').setView([cLat, cLon], 16);
        const satelliteLayer = L.tileLayer(
            'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            { attribution: 'Tiles \u00a9 Esri' }
        );
        const streetLayer = L.tileLayer(
            'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
            { attribution: '\u00a9 OSM' }
        );
        satelliteLayer.addTo(map);
        L.control.layers(
            {
                'Satellite': satelliteLayer,
                'Street': streetLayer,
            },
            {},
            { collapsed: true }
        ).addTo(map);
        L.polyline(trajectory, { color: '#00ff88', weight: 3 }).addTo(map);
        if (trajectory.length) {
            L.circleMarker(trajectory[0], { color:'green', fillColor:'green', fillOpacity:1, radius:7 }).addTo(map).bindPopup('START');
            L.circleMarker(trajectory[trajectory.length-1], { color:'red', fillColor:'red', fillOpacity:1, radius:7 }).addTo(map).bindPopup('END');
        }
        mapMarker = L.circleMarker(trajectory[0] || [cLat, cLon], { color:'red', fillColor:'red', fillOpacity:1, radius:9 }).addTo(map);
        const detectionAltitudes = mapDetections
            .map(item => Number(item.altitude ?? 0))
            .filter(value => Number.isFinite(value));
        const detectionAltitudeMin = detectionAltitudes.length ? Math.min(...detectionAltitudes) : 0;
        const detectionAltitudeMaxRaw = detectionAltitudes.length ? Math.max(...detectionAltitudes) : 1;
        const detectionAltitudeMax = Math.abs(detectionAltitudeMaxRaw - detectionAltitudeMin) < 1e-6
            ? detectionAltitudeMin + 1
            : detectionAltitudeMaxRaw;

        function detectionColorForAltitude(altitude) {
            const safeAltitude = Number.isFinite(Number(altitude)) ? Number(altitude) : detectionAltitudeMin;
            const ratio = Math.max(0, Math.min(1, (safeAltitude - detectionAltitudeMin) / (detectionAltitudeMax - detectionAltitudeMin)));
            const red = Math.round(255 * ratio);
            const green = Math.round(190 - (110 * ratio));
            const blue = Math.round(255 * (1 - ratio));
            return `rgb(${red}, ${green}, ${blue})`;
        }

        staticDetectionMarkers = mapDetections.map(item => {
            const altitudeValue = Number(item.altitude_reference === 'takeoff_relative' ? (item.relative_altitude_m ?? item.altitude ?? 0) : (item.altitude ?? 0));
            const altitudeText = altitudeValue.toFixed(1);
            const markerColor = detectionColorForAltitude(altitudeValue);
            const altitudeLine = item.altitude_reference === 'takeoff_relative'
                ? `Height (relative to takeoff plane): ${altitudeText}m`
                : `Elevation (MSL): ${altitudeText}m`;
            const tooltipText = item.altitude_reference === 'takeoff_relative'
                ? `${altitudeText}m rel`
                : `${altitudeText}m MSL`;
            const popupText = `${(item.class || 'object').toUpperCase()} #${item.track_id || '?'}<br>Lat: ${item.latitude.toFixed(6)}<br>Lon: ${item.longitude.toFixed(6)}<br>${altitudeLine}<br>Method: ${item.method || 'unknown'}`;
            return L.circleMarker([item.latitude, item.longitude], {
                color: markerColor,
                fillColor: markerColor,
                fillOpacity: 0.9,
                radius: 4,
                weight: 1,
            }).addTo(map).bindPopup(popupText).bindTooltip(tooltipText, {direction: 'top', opacity: 0.95});
        });
        mapInitialized = true;
        updateMapMarker(useFrameFallback ? fallbackFrame : frameFromTime(vid.currentTime));
    }

    function updateMapMarker(frameIdx) {
        if (!mapInitialized || !mapMarker) return;
        const point = trajectory[frameIdx] || null;
        if (point && point.length >= 2) mapMarker.setLatLng(point);
    }

    function fmt(s) {
        if (!isFinite(s)) return '0:00';
        const mins = Math.floor(s / 60);
        const secs = Math.floor(s % 60);
        return `${mins}:${String(secs).padStart(2, '0')}`;
    }

    window.addEventListener('resize', () => {
        const frame = useFrameFallback ? fallbackFrame : frameFromTime(vid.currentTime);
        renderDetections(frame);
        const t = telemetryData[frame] || {};
        const d = t.drone || {};
        const azimuth = t.camera_heading ?? null;
        const elevation = t.camera_pitch ?? (t.gimbal && t.gimbal.pitch != null ? t.gimbal.pitch : null);
        const hasElevation = elevation != null;
        updateRulers(azimuth, elevation, hasElevation);
    });

</script>
</body>
</html>"""

        # Replace templated placeholders with actual values
        html_content = html_content.replace("{telemetry_json}", telemetry_json)
        html_content = html_content.replace("{trajectory_json}", trajectory_json)
        html_content = html_content.replace("{frame_files_json}", frame_files_json)
        html_content = html_content.replace("{detections_json}", detections_json)
        html_content = html_content.replace("{map_points_json}", map_points_json)
        html_content = html_content.replace("{fps}", str(fps))
        html_content = html_content.replace("{video_sources_html}", video_sources_html)
        html_content = html_content.replace("{video_source_count_json}", video_source_count_json)
        html_content = html_content.replace("{report_name_json}", report_name_json)
        html_content = html_content.replace("{build_note_json}", build_note_json)
        # Write HTML file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Interactive video viewer saved to {output_path}")
        return True

