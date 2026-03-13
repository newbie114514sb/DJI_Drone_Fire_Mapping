"""
Flight planning for DJI Mini 4 Pro
NOTE: Mini 4 Pro does NOT support onboard SDK control

Workflow:
1. Create flight plan with waypoint mapping tool (e.g., waypointmap.com)
2. Load plan onto drone via DJI Fly app
3. Start mission manually and use HYPERLAPSE function
4. Hyperlapse records flight telemetry in EXIF data
5. Download hyperlapse image folder for post-processing
"""

from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


class FlightPlanInfo:
    """Information about a pre-planned flight"""
    
    def __init__(self, name: str, waypoints: List[dict] = None):
        """
        Initialize flight plan info
        Args:
            name: flight plan name
            waypoints: list of waypoint dicts with lat/lon/altitude
        """
        self.name = name
        self.waypoints = waypoints or []
        logger.info(f"Flight plan '{name}' with {len(self.waypoints)} waypoints")
    
    def add_waypoint(self, lat: float, lon: float, altitude: float):
        """Add a waypoint to the plan"""
        self.waypoints.append({
            'latitude': lat,
            'longitude': lon,
            'altitude': altitude,
        })
    
    def export_dji_format(self, output_path: str):
        """
        Export flight plan in DJI format for import into DJI Fly app
        (Typically a .txt file with waypoint format)
        """
        logger.info(f"Export functionality requires DJI FlightKML library")
        logger.info(f"Use waypointmap.com or similar tool to create flight plans")


class HyperlapseFlightGuide:
    """
    Guide for performing hyperlapse flights with DJI Mini 4 Pro
    
    Hyperlapse automatically:
    - Records images at regular intervals
    - Embeds GPS, altitude, gimbal angle in EXIF data
    - Creates a folder with timestamped images
    """
    
    @staticmethod
    def get_setup_instructions():
        """Get step-by-step hyperlapse setup"""
        return """
DJI Mini 4 Pro Hyperlapse Flight Plan:

1. PREPARE FLIGHT PLAN:
   - Use waypointmap.com or similar waypoint planning tool
   - Create a survey pattern (zigzag recommended for fire mapping)
   - Set waypoint altitude (recommend 50m for good resolution)
   - Export plan and import into DJI Fly app

2. HYPERLAPSE SETTINGS (in DJI Fly):
   - Select "Hyperlapse" mode (not regular video)
   - Set interval: 2-5 seconds between frames
   - Gimbal: Set to -90° (straight down) for fire detection
   - Resolution: 4K recommended (4096×2160)

3. PRE-FLIGHT:
   - Check battery level (need ≥80% for long survey)
   - Format SD card
   - Verify GPS signal strength
   - Test gimbal movement to ensure sensor works

4. EXECUTE FLIGHT:
   - Press START in DJI Fly app
   - Let drone fly complete mission automatically
   - Watch for return-to-home trigger

5. POST-FLIGHT:
   - Retrieve SD card
   - Copy "Hyperlapse_XXXX" folder to computer
   - Run through fire detection pipeline

6. EXPECTED OUTPUTS:
   - Folder: IMG_XXXX.jpg files (numbered sequentially)
   - Each image contains EXIF data with:
     * GPS coordinates (latitude, longitude, altitude)
     * Gimbal/camera angles (pitch, roll, yaw)
     * Flight time / image index
     * Drone model and firmware version
        """
    
    @staticmethod
    def verify_hyperlapse_folder(folder_path: str) -> dict:
        """
        Verify that a folder contains valid hyperlapse output
        Args:
            folder_path: path to hyperlapse image folder
        Returns:
            validation report dict
        """
        from pathlib import Path
        folder = Path(folder_path)
        
        if not folder.exists():
            return {'valid': False, 'error': 'Folder does not exist'}
        
        # Count images
        images = list(folder.glob('IMG_*.jpg')) + list(folder.glob('*.jpg'))
        
        if not images:
            return {'valid': False, 'error': 'No JPG images found'}
        
        logger.info(f"Found {len(images)} hyperlapse images")
        return {
            'valid': True,
            'image_count': len(images),
            'folder': str(folder),
            'sample_image': str(images[0])
        }
