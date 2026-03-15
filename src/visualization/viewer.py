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
        self.images = sorted(
            list(self.folder.glob('IMG_*.jpg')) + 
            list(self.folder.glob('*.jpg'))
        )
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
    
    def __init__(self, telemetry_sequence):
        """
        Initialize map generator
        Args:
            telemetry_sequence: TelemetrySequence object
        """
        self.telemetry_sequence = telemetry_sequence
    
    def create_trajectory_map(self, output_path: str = 'trajectory_map.html'):
        """
        Create interactive map showing flight path
        Args:
            output_path: where to save HTML map
        """
        trajectory = self.telemetry_sequence.get_trajectory()
        bounds = self.telemetry_sequence.get_bounds()
        
        if not trajectory or not bounds:
            logger.error("No trajectory data available")
            return False
        
        # Center map on flight area
        center_lat = (bounds['north'] + bounds['south']) / 2
        center_lon = (bounds['east'] + bounds['west']) / 2
        
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=16,
            tiles='OpenStreetMap'
        )
        
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
        
        # Add heatmap layer for density
        heat_data = [[lat, lon] for lat, lon, _ in trajectory]
        if len(heat_data) > 1:
            plugins.HeatMap(heat_data, radius=20, blur=15).add_to(m)
        
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
        
        m.save(output_path)
        logger.info(f"Trajectory map saved to {output_path}")
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
    
    def _encode_video_variants(self, out_dir: Path, fps: int = 5) -> List[Tuple[str, str]]:
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

        for img_path in images:
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            if frame.shape[1] != w or frame.shape[0] != h:
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            writer.write(frame)
        writer.release()

        if video_path.exists() and video_path.stat().st_size > 0:
            logger.info(f"Video encoded to {video_path}")
            return [('hyperlapse.webm', 'video/webm')]

        return []

    def _prepare_preview_frames(self, out_dir: Path, max_width: int = 1280) -> List[str]:
        """Create downscaled JPG preview frames for browser-side fallback playback."""
        frames_dir = out_dir / 'frames'
        frames_dir.mkdir(parents=True, exist_ok=True)

        frame_paths = []
        for index, img_path in enumerate(self.telemetry_sequence.images):
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue

            height, width = frame.shape[:2]
            if width > max_width:
                scale = max_width / float(width)
                frame = cv2.resize(frame, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)

            output_name = f"frame_{index:04d}.jpg"
            output_path = frames_dir / output_name
            cv2.imwrite(str(output_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            frame_paths.append(f"frames/{output_name}")

        return frame_paths

    def create_interactive_video_viewer(self, output_path: str) -> bool:
        """Create an interactive HTML video viewer with telemetry overlay and toggleable minimap"""
        if not self.telemetry_sequence:
            logger.error("No telemetry data available")
            return False
        
        trajectory = self.telemetry_sequence.get_trajectory()
        bounds = self.telemetry_sequence.get_bounds()
        
        if not trajectory:
            logger.error("No trajectory data available")
            return False
        
        import json
        out_dir = Path(output_path).parent

        # Encode browser-playable video variants for smooth native playback
        fps = 5
        logger.info("Encoding image sequence to video...")
        video_sources = self._encode_video_variants(out_dir, fps=fps)
        frame_files = self._prepare_preview_frames(out_dir)

        telemetry_json = json.dumps(self.telemetry_sequence.telemetry)
        trajectory_json = json.dumps([[lat, lon] for lat, lon, alt in trajectory])
        frame_files_json = json.dumps(frame_files)
        report_name_json = json.dumps(out_dir.name)
        build_note_json = json.dumps("Latest build: HUD overlay refresh with restored attitude telemetry")
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
        }
        .azimuth-line {
            position: absolute;
            left: 0;
            right: 0;
            top: 18px;
            height: 2px;
            background: linear-gradient(90deg, rgba(99,255,182,0) 0%, rgba(99,255,182,0.9) 12%, rgba(99,255,182,0.9) 88%, rgba(99,255,182,0) 100%);
            box-shadow: 0 0 12px rgba(99, 255, 182, 0.32);
        }
        .azimuth-line::before {
            content: '';
            position: absolute;
            inset: -14px 0 -14px 0;
            background:
                linear-gradient(90deg, transparent 0, transparent calc(50% - 1px), rgba(99,255,182,0.95) calc(50% - 1px), rgba(99,255,182,0.95) calc(50% + 1px), transparent calc(50% + 1px), transparent 100%),
                repeating-linear-gradient(90deg, transparent 0 38px, rgba(99,255,182,0.6) 38px 40px, transparent 40px 78px);
            opacity: 0.95;
        }
        .azimuth-ticks {
            position: absolute;
            top: -2px;
            left: 8px;
            right: 8px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 18px;
            letter-spacing: 0.08em;
            color: rgba(153, 255, 216, 0.8);
        }
        .azimuth-value {
            font-size: 54px;
            line-height: 1;
            font-weight: 800;
            text-shadow: 0 0 18px rgba(99, 255, 182, 0.22);
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
            width: 2px;
            background: linear-gradient(180deg, rgba(99,255,182,0) 0%, rgba(99,255,182,0.88) 12%, rgba(99,255,182,0.88) 88%, rgba(99,255,182,0) 100%);
            box-shadow: 0 0 12px rgba(99, 255, 182, 0.3);
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
            font-size: 34px;
            font-weight: 800;
            line-height: 1;
            text-shadow: 0 0 18px rgba(99, 255, 182, 0.22);
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
            opacity: 0.9;
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
            box-shadow: 0 0 10px rgba(99, 255, 182, 0.4);
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
            left: 50%; top: 50%; width: 74px; height: 74px;
            transform: translate(-50%, -50%);
            border: 1px solid rgba(99, 255, 182, 0.85);
            border-radius: 50%;
            box-shadow: 0 0 12px rgba(99, 255, 182, 0.24);
            background: radial-gradient(circle, rgba(99,255,182,0.14) 0%, rgba(99,255,182,0.03) 42%, rgba(0,0,0,0) 70%);
        }
        .hud-reticle::before,
        .hud-reticle::after {
            content: '';
            position: absolute;
            background: #63ffb6;
            box-shadow: 0 0 8px rgba(99, 255, 182, 0.4);
        }
        .hud-reticle::before {
            left: 50%; top: -22px; width: 2px; height: 118px; transform: translateX(-50%);
        }
        .hud-reticle::after {
            top: 50%; left: -22px; width: 118px; height: 2px; transform: translateY(-50%);
        }
        .hud-reticle-inner {
            position: absolute;
            left: 50%; top: 50%; width: 16px; height: 16px;
            transform: translate(-50%, -50%);
            border: 2px solid rgba(99, 255, 182, 0.9);
            border-radius: 50%;
            box-shadow: 0 0 10px rgba(99, 255, 182, 0.35);
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
                    <div class="azimuth-ticks"><span>W</span><span>NW</span><span>N</span><span>NE</span><span>E</span></div>
                </div>
                <div class="azimuth-value" id="earth-azimuth">--</div>
            </div>
        </div>

        <div class="elevation-bar">
            <div class="elevation-ruler">
                <div class="elevation-track"></div>
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
            <div class="telemetry-item"><span class="telemetry-label">Drone Attitude:</span><span class="telemetry-value" id="t-attitude">--</span></div>
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
    const reportName   = {report_name_json};
    const buildNote    = {build_note_json};
    const FPS = {fps};

    const vid   = document.getElementById('main-video');
    const fallbackImg = document.getElementById('frame-fallback');
    const bar   = document.getElementById('seek-bar');
    let isSeeking = false;
    let map = null, mapMarker = null, mapInitialized = false;
    let useFrameFallback = false;
    let fallbackFrame = 0;
    let fallbackPlaying = false;
    let fallbackRaf = null;
    let fallbackLastTs = 0;
    let fallbackPlaybackRate = 1;

    function currentDuration() {
        return useFrameFallback ? (frameFiles.length / FPS) : (vid.duration || 0);
    }

    function currentTimeValue() {
        return useFrameFallback ? (fallbackFrame / FPS) : vid.currentTime;
    }

    function syncFromFrame(frame) {
        const safeFrame = Math.max(0, Math.min(frame, telemetryData.length - 1));
        updateTelemetry(safeFrame);
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
        fallbackImg.src = frameFiles[fallbackFrame];
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
            if (timestamp - fallbackLastTs >= (1000 / (FPS * fallbackPlaybackRate))) {
                fallbackLastTs = timestamp;
                if (fallbackFrame >= frameFiles.length - 1) {
                    stopFallbackPlayback();
                    return;
                }
                showFallbackFrame(fallbackFrame + 1);
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
        const frame = Math.min(Math.round(vid.currentTime * FPS), telemetryData.length - 1);
        updateTelemetry(frame);
        updateMapMarker(frame);
        document.getElementById('t-current').textContent = fmt(vid.currentTime);
    });

    vid.addEventListener('loadedmetadata', () => {
        document.getElementById('t-total').textContent = fmt(vid.duration);
        document.getElementById('t-current').textContent = fmt(0);
        updateTelemetry(0);
        updateSeekBar();
    });

    vid.addEventListener('loadeddata', () => {
        updateTelemetry(0);
        updateMapMarker(0);
    });

    vid.addEventListener('error', () => {
        activateFrameFallback();
    });

    window.addEventListener('load', () => {
        document.getElementById('report-name').textContent = reportName;
        document.getElementById('build-note').textContent = buildNote;
        updateTelemetry(0);
        document.getElementById('t-total').textContent = fmt(frameFiles.length / FPS);
        if (frameFiles.length) {
            activateFrameFallback();
        } else {
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
        const elevation = ((d.pitch ?? 0) + (t.gimbal?.pitch ?? 0));
        const hasElevation = d.pitch != null || (t.gimbal && t.gimbal.pitch != null);
        document.getElementById('t-frame').textContent    = idx + 1;
        document.getElementById('t-gps').textContent      = g.latitude  ? `${g.latitude.toFixed(5)}, ${g.longitude.toFixed(5)}` : '--';
        document.getElementById('t-alt').textContent      = g.altitude  ? `${g.altitude.toFixed(1)}m` : '--';
        document.getElementById('t-heading').textContent  = t.camera_heading != null ? `${t.camera_heading.toFixed(1)}\\u00b0` : '--';
        document.getElementById('t-attitude').textContent = (d.yaw != null) ? `Y:${d.yaw.toFixed(1)}\\u00b0 P:${(d.pitch ?? 0).toFixed(1)}\\u00b0 R:${(d.roll ?? 0).toFixed(1)}\\u00b0` : '--';
        document.getElementById('t-gimbal').textContent   = t.gimbal ? `P:${t.gimbal.pitch||0}\\u00b0 R:${t.gimbal.roll||0}\\u00b0 Y:${t.gimbal.yaw||0}\\u00b0` : '--';
        const spd = d.speed ?? ((d.speed_x != null) ? Math.sqrt(d.speed_x**2 + d.speed_y**2 + d.speed_z**2) : null);
        document.getElementById('t-speed').textContent    = spd != null ? `${spd.toFixed(1)} m/s` : '--';
        document.getElementById('earth-azimuth').textContent = azimuth != null ? `${azimuth.toFixed(1)}\u00b0` : '--';
        document.getElementById('earth-elevation').textContent = hasElevation ? `${elevation.toFixed(1)}\u00b0` : '--';
        updateHudAttitude(d.roll ?? 0, d.pitch ?? 0);
    }

    function updateHudAttitude(roll, pitch) {
        const horizon = document.getElementById('hud-horizon');
        const clampedPitch = Math.max(-20, Math.min(20, pitch || 0));
        horizon.style.transform = `translate(-50%, calc(-50% + ${clampedPitch * 3}px)) rotate(${roll || 0}deg)`;
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
        const frame = Math.min(Math.round(vid.currentTime * FPS), telemetryData.length - 1);
        updateTelemetry(frame);
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
            const frame = Math.min(Math.round(vid.currentTime * FPS), telemetryData.length - 1);
            updateTelemetry(frame);
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
                const frame = Math.min(Math.round(vid.currentTime * FPS), telemetryData.length - 1);
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
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', { attribution: '\\u00a9 OSM' }).addTo(map);
        L.polyline(trajectory, { color: '#00ff88', weight: 3 }).addTo(map);
        if (trajectory.length) {
            L.circleMarker(trajectory[0], { color:'green', fillColor:'green', fillOpacity:1, radius:7 }).addTo(map).bindPopup('START');
            L.circleMarker(trajectory[trajectory.length-1], { color:'red', fillColor:'red', fillOpacity:1, radius:7 }).addTo(map).bindPopup('END');
        }
        mapMarker = L.circleMarker(trajectory[0] || [cLat, cLon], { color:'red', fillColor:'red', fillOpacity:1, radius:9 }).addTo(map);
        mapInitialized = true;
        updateMapMarker(Math.min(Math.round(vid.currentTime * FPS), telemetryData.length - 1));
    }

    function updateMapMarker(frameIdx) {
        if (!mapInitialized || !mapMarker) return;
        const t = telemetryData[frameIdx] || {};
        const g = t.gps || {};
        if (g.latitude && g.longitude) mapMarker.setLatLng([g.latitude, g.longitude]);
    }

    function fmt(s) {
        if (!isFinite(s)) return '0:00';
        const mins = Math.floor(s / 60);
        const secs = Math.floor(s % 60);
        return `${mins}:${String(secs).padStart(2, '0')}`;
    }

</script>
</body>
</html>"""

        # Replace templated placeholders with actual values
        html_content = html_content.replace("{telemetry_json}", telemetry_json)
        html_content = html_content.replace("{trajectory_json}", trajectory_json)
        html_content = html_content.replace("{frame_files_json}", frame_files_json)
        html_content = html_content.replace("{fps}", str(fps))
        html_content = html_content.replace("{video_sources_html}", video_sources_html)
        html_content = html_content.replace("{report_name_json}", report_name_json)
        html_content = html_content.replace("{build_note_json}", build_note_json)
        # Write HTML file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Interactive video viewer saved to {output_path}")
        return True

