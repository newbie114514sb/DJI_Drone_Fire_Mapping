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
        
        # Create HTML content with relative image paths
        import json, os
        out_dir = Path(output_path).parent
        images_dir = out_dir / 'images'
        images_dir.mkdir(exist_ok=True)
        copied_images = []
        for img_path in self.telemetry_sequence.images:
            img_name = Path(img_path).name
            dest_path = images_dir / img_name
            shutil.copy(img_path, dest_path)
            copied_images.append(f"images/{img_name}")
        telemetry_json = json.dumps(self.telemetry_sequence.telemetry)
        images_json = json.dumps(copied_images)
        trajectory_json = json.dumps([[lat, lon] for lat, lon, alt in trajectory])
        
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DJI Hyperlapse Fire Mapping Viewer</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/leaflet@1.9.3/dist/leaflet.css"/>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.2/dist/css/bootstrap.min.css"/>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <script src="https://cdn.jsdelivr.net/npm/leaflet@1.9.3/dist/leaflet.js"></script>
    <script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.2.2/dist/js/bootstrap.bundle.min.js"></script>
    <style>
        body {{
            margin: 0;
            padding: 0;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #1a1a1a;
            color: white;
            overflow: hidden;
        }}
        
        .viewer-container {{
            position: relative;
            width: 100vw;
            height: 100vh;
            display: flex;
            flex-direction: column;
        }}
        
        .video-section {{
            flex: 1;
            position: relative;
            background: #000;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        
        .video-container {{
            position: relative;
            max-width: 90%;
            max-height: 90%;
        }}
        
        .video-frame {{
            max-width: 100%;
            max-height: 100%;
            border: 2px solid #333;
            border-radius: 8px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.5);            width: 100%;
            height: 100%;
            object-fit: contain;
            transition: opacity 0.2s ease-in-out;
            opacity: 1;        }}
        
        .telemetry-overlay {{
            position: absolute;
            top: 20px;
            left: 20px;
            background: rgba(0, 0, 0, 0.8);
            padding: 15px;
            border-radius: 8px;
            font-family: 'Courier New', monospace;
            font-size: 14px;
            min-width: 280px;
            border: 1px solid #444;
        }}
        
        .telemetry-title {{
            font-weight: bold;
            color: #00ff88;
            margin-bottom: 10px;
            font-size: 16px;
        }}
        
        .telemetry-item {{
            margin: 5px 0;
            display: flex;
            justify-content: space-between;
        }}
        
        .telemetry-label {{
            color: #ccc;
        }}
        
        .telemetry-value {{
            color: #fff;
            font-weight: bold;
        }}
        
        .controls {{
            position: absolute;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(0, 0, 0, 0.8);
            padding: 15px 20px;
            border-radius: 25px;
            display: flex;
            align-items: center;
            gap: 15px;
            border: 1px solid #444;
        }}
        
        .control-btn {{
            background: #333;
            border: none;
            color: white;
            padding: 8px 12px;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.2s;
            font-size: 14px;
        }}
        
        .control-btn:hover {{
            background: #555;
        }}
        
        .control-btn.active {{
            background: #00ff88;
            color: black;
        }}
        
        .progress-container {{
            flex: 1;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .progress-bar {{
            flex: 1;
            height: 6px;
            background: #333;
            border-radius: 3px;
            cursor: pointer;
            position: relative;
        }}
        
        .progress-fill {{
            height: 100%;
            background: #00ff88;
            border-radius: 3px;
            width: 0%;
            transition: width 0.1s;
        }}
        
        .time-display {{
            font-size: 12px;
            color: #ccc;
            min-width: 80px;
            text-align: center;
        }}
        
        .minimap-container {{
            position: absolute;
            top: 20px;
            right: 20px;
            width: 300px;
            height: 200px;
            background: rgba(0, 0, 0, 0.8);
            border-radius: 8px;
            border: 1px solid #444;
            overflow: hidden;
            transition: all 0.3s;
            z-index: 1000;
        }}
        
        .minimap-container.collapsed {{
            width: 50px;
            height: 50px;
        }}
        
        .minimap-header {{
            padding: 8px 12px;
            background: #333;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}
        
        .minimap-title {{
            font-size: 14px;
            font-weight: bold;
        }}
        
        .minimap-toggle {{
            font-size: 12px;
            color: #ccc;
        }}
        
        .minimap-content {{
            height: calc(100% - 40px);
            position: relative;
        }}
        
        .minimap-content.collapsed {{
            display: none;
        }}
        
        #minimap {{
            width: 100%;
            height: 100%;
        }}
        
        .current-position {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 12px;
            height: 12px;
            background: red;
            border: 2px solid white;
            border-radius: 50%;
            z-index: 1000;
            box-shadow: 0 0 10px red;
        }}
        
        .loading {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            color: #00ff88;
            font-size: 18px;
        }}
        
        @media (max-width: 768px) {{
            .telemetry-overlay {{
                top: 10px;
                left: 10px;
                min-width: 250px;
                font-size: 12px;
            }}
            
            .minimap-container {{
                top: 10px;
                right: 10px;
                width: 250px;
                height: 150px;
            }}
            
            .controls {{
                bottom: 10px;
                padding: 10px 15px;
                gap: 10px;
            }}
        }}
    </style>
</head>
<body>
    <div class="viewer-container">
        <div class="video-section">
            <div class="video-container">
                <img id="video-frame" class="video-frame" src="" alt="Loading...">
                <div class="telemetry-overlay">
                    <div class="telemetry-title">📡 TELEMETRY DATA</div>
                    <div class="telemetry-item">
                        <span class="telemetry-label">Frame:</span>
                        <span class="telemetry-value" id="frame-number">0</span>
                    </div>
                    <div class="telemetry-item">
                        <span class="telemetry-label">GPS:</span>
                        <span class="telemetry-value" id="gps-coords">--</span>
                    </div>
                    <div class="telemetry-item">
                        <span class="telemetry-label">Altitude:</span>
                        <span class="telemetry-value" id="altitude">--</span>
                    </div>
                    <div class="telemetry-item">
                        <span class="telemetry-label">Gimbal:</span>
                        <span class="telemetry-value" id="gimbal-angle">--</span>
                    </div>
                    <div class="telemetry-item">
                        <span class="telemetry-label">Speed:</span>
                        <span class="telemetry-value" id="speed">--</span>
                    </div>
                    <div class="telemetry-item">
                        <span class="telemetry-label">Detections:</span>
                        <span class="telemetry-value" id="detections">None</span>
                    </div>
                </div>
            </div>
            
        </div>
        
        <div class="controls">
            <button class="control-btn" id="play-pause" onclick="togglePlayPause()">
                <i class="fas fa-play"></i>
            </button>
            <button class="control-btn" onclick="previousFrame()">
                <i class="fas fa-step-backward"></i>
            </button>
            <button class="control-btn" onclick="nextFrame()">
                <i class="fas fa-step-forward"></i>
            </button>
            
            <div class="progress-container">
                <span class="time-display" id="current-time">0:00</span>
                <div class="progress-bar" id="progress-bar" onclick="seekTo(event)">
                    <div class="progress-fill" id="progress-fill"></div>
                </div>
                <span class="time-display" id="total-time">0:00</span>
            </div>
            
            <button class="control-btn" id="speed-btn" onclick="changeSpeed()">
                1x
            </button>
            <button class="control-btn" onclick="setOneMinute()" id="one-minute">1 min</button>
            <button class="control-btn" onclick="openMap()">Map</button>
        </div>
    </div>

    <script>
        // Telemetry data
        const telemetryData = {telemetry_json};
        const imageFiles = {images_json};
        
        // Viewer state
        let currentFrame = 0;
        let isPlaying = false;
        let playbackSpeed = 1;
        let intervalId = null;
        const frameRate = 30; // FPS
        
        // Map variables (only used if popup opened)
        let map = null;
        let trajectoryLayer = null;
        let currentMarker = null;
        
        // Initialize viewer
        document.addEventListener('DOMContentLoaded', function() {{
            initializeViewer();
            // no embedded map by default
            updateFrame();
        }});
        
        function initializeViewer() {{
            document.getElementById('total-time').textContent = formatTime(imageFiles.length / frameRate);
            document.addEventListener('keydown', handleKeyPress);
        }}
        
        function initializeMap() {{
            const centerLat = ({bounds['north']} + {bounds['south']}) / 2;
            const centerLon = ({bounds['east']} + {bounds['west']}) / 2;
            
            map = L.map('minimap').setView([centerLat, centerLon], 16);
            
            L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
                attribution: '© OpenStreetMap contributors'
            }}).addTo(map);
            
            // Add trajectory
            trajectoryLayer = L.polyline({trajectory_json}, {{
                color: '#00ff88',
                weight: 3,
                opacity: 0.8
            }}).addTo(map);
            
            // Fit bounds
            map.fitBounds(trajectoryLayer.getBounds());
            
            // Add start/end markers
            const trajectoryCoords = {trajectory_json};
            if (trajectoryCoords.length > 0) {{
                L.circleMarker(trajectoryCoords[0], {{
                    color: 'green',
                    fillColor: 'green',
                    fillOpacity: 0.8,
                    radius: 8
                }}).addTo(map).bindPopup('START');
                
                L.circleMarker(trajectoryCoords[trajectoryCoords.length - 1], {{
                    color: 'red',
                    fillColor: 'red',
                    fillOpacity: 0.8,
                    radius: 8
                }}).addTo(map).bindPopup('END');
            }}
        }}
        
        function updateFrame() {{
            if (currentFrame >= imageFiles.length) {{
                currentFrame = imageFiles.length - 1;
                stopPlayback();
            }}
            
            // Update image
            document.getElementById('video-frame').src = imageFiles[currentFrame];
            
            // Update telemetry
            const telemetry = telemetryData[currentFrame] || {{}};
            const gps = telemetry.gps || {{}};
            
            document.getElementById('frame-number').textContent = currentFrame + 1;
            document.getElementById('gps-coords').textContent = 
                gps.latitude ? `${{gps.latitude.toFixed(6)}}, ${{gps.longitude.toFixed(6)}}` : '--';
            document.getElementById('altitude').textContent = 
                gps.altitude ? `${{gps.altitude.toFixed(1)}}m` : '--';
            document.getElementById('gimbal-angle').textContent = 
                telemetry.gimbal ? `${{telemetry.gimbal.pitch || 0}}°` : '--';
            document.getElementById('speed').textContent = 
                telemetry.speed ? `${{telemetry.speed.toFixed(1)}} m/s` : '--';
            document.getElementById('detections').textContent = 'None (Future)';
            
            // Update progress
            const progress = (currentFrame / (imageFiles.length - 1)) * 100;
            document.getElementById('progress-fill').style.width = `${{progress}}%`;
            document.getElementById('current-time').textContent = formatTime(currentFrame / frameRate);
            
            // Update map marker
            updateMapMarker(gps);
        }}
        
        function updateMapMarker(gps) {{
            // no-op if map not created
            if (!map) return;
            if (currentMarker) {{
                map.removeLayer(currentMarker);
            }}
            
            if (gps.latitude && gps.longitude) {{
                currentMarker = L.circleMarker([gps.latitude, gps.longitude], {{
                    color: 'red',
                    fillColor: 'red',
                    fillOpacity: 1,
                    radius: 8,
                    weight: 2
                }}).addTo(map);
            }}
        }}
        
        function togglePlayPause() {{
            const btn = document.getElementById('play-pause');
            const icon = btn.querySelector('i');
            
            if (isPlaying) {{
                stopPlayback();
                icon.className = 'fas fa-play';
            }} else {{
                startPlayback();
                icon.className = 'fas fa-pause';
            }}
        }}
        
        function setOneMinute() {{
            // adjust frameRate so full sequence plays in 60s
            const total = imageFiles.length;
            playbackSpeed = (total / 60) / frameRate;
            document.getElementById('speed-btn').textContent = `${{(frameRate * playbackSpeed).toFixed(1)}} fps`;
            if (isPlaying) {{
                stopPlayback();
                startPlayback();
            }}
        }}
        
        function openMap() {{
            window.open('trajectory_map.html', '_blank');
        }}
        
        function startPlayback() {{
            if (isPlaying) return;
            isPlaying = true;
            intervalId = setInterval(() => {{
                currentFrame++;
                if (currentFrame >= imageFiles.length) {{
                    stopPlayback();
                }} else {{
                    updateFrame();
                }}
            }}, (1000 / frameRate) / playbackSpeed);
        }}
        
        function stopPlayback() {{
            isPlaying = false;
            if (intervalId) {{
                clearInterval(intervalId);
                intervalId = null;
            }}
        }}
        
        function nextFrame() {{
            if (currentFrame < imageFiles.length - 1) {{
                currentFrame++;
                updateFrame();
            }}
        }}
        
        function previousFrame() {{
            if (currentFrame > 0) {{
                currentFrame--;
                updateFrame();
            }}
        }}
        
        function seekTo(event) {{
            const rect = event.target.getBoundingClientRect();
            const x = event.clientX - rect.left;
            const percentage = x / rect.width;
            currentFrame = Math.floor(percentage * (imageFiles.length - 1));
            updateFrame();
        }}
        
        function changeSpeed() {{
            const speeds = [0.5, 1, 2, 4];
            const currentIndex = speeds.indexOf(playbackSpeed);
            playbackSpeed = speeds[(currentIndex + 1) % speeds.length];
            document.getElementById('speed-btn').textContent = `${{playbackSpeed}}x`;
            
            if (isPlaying) {{
                stopPlayback();
                startPlayback();
            }}
        }}
        
        function toggleMinimap() {{
            const container = document.getElementById('minimap-container');
            const content = document.getElementById('minimap-content');
            const toggle = document.getElementById('minimap-toggle');
            
            if (container.classList.contains('collapsed')) {{
                container.classList.remove('collapsed');
                content.classList.remove('collapsed');
                toggle.textContent = '▼';
            }} else {{
                container.classList.add('collapsed');
                content.classList.add('collapsed');
                toggle.textContent = '▶';
            }}
        }}
        
        function handleKeyPress(event) {{
            switch(event.code) {{
                case 'Space':
                    event.preventDefault();
                    togglePlayPause();
                    break;
                case 'ArrowLeft':
                    event.preventDefault();
                    previousFrame();
                    break;
                case 'ArrowRight':
                    event.preventDefault();
                    nextFrame();
                    break;
            }}
        }}
        
        function formatTime(seconds) {{
            const mins = Math.floor(seconds / 60);
            const secs = Math.floor(seconds % 60);
            return `${{mins}}:${{secs.toString().padStart(2, '0')}}`;
        }}
        
        // Handle map resize when minimap is toggled
        document.getElementById('minimap-container').addEventListener('transitionend', function() {{
            if (map) {{
                map.invalidateSize();
            }}
        }});
    </script>
</body>
</html>"""

        # Write HTML file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Interactive video viewer saved to {output_path}")
        return True

