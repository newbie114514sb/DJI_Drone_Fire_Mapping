"""
Telemetry extraction from DJI hyperlapse image EXIF data
- Extract GPS coordinates, altitude, gimbal angles from image metadata
- Build flight trajectory from image sequence
"""

from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)


class ExifTelemetryExtractor:
    """Extract telemetry from image EXIF data"""
    
    def __init__(self):
        """Initialize extractor"""
        try:
            from PIL import Image
            from PIL.ExifTags import TAGS, GPSTAGS
            self.Image = Image
            self.TAGS = TAGS
            self.GPSTAGS = GPSTAGS
            self.available = True
            logger.info("PIL/Pillow available for EXIF extraction")
        except ImportError:
            logger.warning("PIL not available; install with: pip install Pillow")
            self.available = False
    
    def extract_gps_from_exif(self, exif_data: dict) -> Optional[Dict]:
        """
        Extract GPS coordinates from EXIF data
        Returns: {'latitude': float, 'longitude': float, 'altitude': float}
        """
        if not self.available:
            return None
        
        try:
            gps_ifd = exif_data.get(0x8825)  # GPS IFD tag
            if not gps_ifd:
                return None
            
            gps_data = {}
            for tag, value in gps_ifd.items():
                tag_name = self.GPSTAGS.get(tag, tag)
                gps_data[tag_name] = value
            
            # Parse latitude/longitude
            lat = self._convert_gps_coord(gps_data.get('GPSLatitude'))
            lon = self._convert_gps_coord(gps_data.get('GPSLongitude'))
            alt = gps_data.get('GPSAltitude')
            
            if lat is None or lon is None:
                return None
            
            # Adjust sign based on direction
            if gps_data.get('GPSLatitudeRef') == 'S':
                lat = -lat
            if gps_data.get('GPSLongitudeRef') == 'W':
                lon = -lon
            if alt and gps_data.get('GPSAltitudeRef') == 1:  # Below sea level
                alt = -alt
            
            return {
                'latitude': lat,
                'longitude': lon,
                'altitude': float(alt) if alt else None,
            }
        
        except Exception as e:
            logger.debug(f"GPS extraction error: {e}")
            return None
    
    @staticmethod
    def _convert_gps_coord(coord_data) -> Optional[float]:
        """Convert GPS coordinate tuple to decimal degrees"""
        if not coord_data or len(coord_data) < 3:
            return None
        try:
            deg = float(coord_data[0])
            min = float(coord_data[1])
            sec = float(coord_data[2])
            return deg + (min / 60.0) + (sec / 3600.0)
        except (TypeError, ValueError):
            return None
    
    def extract_drone_info(self, image_path: str) -> Dict:
        """
        Extract drone telemetry from image
        Returns: {
            'gps': {'latitude', 'longitude', 'altitude'},
            'gimbal': {'pitch', 'roll', 'yaw'},
            'drone_model': str,
            'timestamp': datetime,
            'other_exif': dict
        }
        """
        if not self.available:
            logger.error("PIL not available")
            return {}
        
        try:
            image = self.Image.open(image_path)
            exif_data = image._getexif() if hasattr(image, '_getexif') else {}
            
            # Extract XMP data
            xmp_data = self._extract_xmp(image)
            
            if not exif_data and not xmp_data:
                logger.debug(f"No EXIF or XMP data in {image_path}")
                return {}
            
            # Extract GPS
            gps = self.extract_gps_from_exif(exif_data)
            
            # Extract gimbal and drone attitude
            gimbal = self._extract_gimbal(exif_data, xmp_data)
            drone_attitude = self._extract_drone_attitude(exif_data, xmp_data, gimbal)
            
            # Extract other metadata
            telemetry = {
                'image_path': str(image_path),
                'gps': gps,
                'gimbal': gimbal,
                'drone': drone_attitude,
                'camera_heading': drone_attitude.get('camera_heading') if drone_attitude else None,
                'drone_model': self._extract_drone_model(exif_data),
                'timestamp': self._extract_timestamp(exif_data),
            }
            
            return telemetry
        
        except Exception as e:
            logger.error(f"Failed to extract telemetry from {image_path}: {e}")
            return {}
    
    def _extract_xmp(self, image) -> Optional[str]:
        """Extract XMP metadata from image"""
        try:
            xmp_data = image.getxmp()
            return xmp_data if xmp_data else None
        except Exception as e:
            logger.debug(f"XMP extraction error: {e}")
            return None
    
    def _extract_gimbal(self, exif_data: dict, xmp_data: Optional[str] = None) -> Optional[Dict]:
        """Extract gimbal angles from DJI XMP data or MakerNotes"""
        gimbal_data = {}
        
        # First try XMP data (preferred for DJI)
        if xmp_data:
            try:
                # XMP data is returned as a nested dict by PIL
                # Navigate to the Description section
                if 'xmpmeta' in xmp_data and 'RDF' in xmp_data['xmpmeta']:
                    rdf = xmp_data['xmpmeta']['RDF']
                    if 'Description' in rdf:
                        desc = rdf['Description']
                        
                        # Extract gimbal angles
                        if 'GimbalPitchDegree' in desc:
                            gimbal_data['pitch'] = float(desc['GimbalPitchDegree'])
                        if 'GimbalRollDegree' in desc:
                            gimbal_data['roll'] = float(desc['GimbalRollDegree'])
                        if 'GimbalYawDegree' in desc:
                            gimbal_data['yaw'] = float(desc['GimbalYawDegree'])
                        
                        if gimbal_data:
                            logger.debug(f"Found gimbal data in XMP: {gimbal_data}")
                            return gimbal_data
                    
            except Exception as e:
                logger.debug(f"XMP gimbal parsing error: {e}")
        
        # Fallback to MakerNotes parsing (legacy method)
        try:
            maker_notes = exif_data.get(0x927c)  # MakerNote tag
            if not maker_notes or not isinstance(maker_notes, bytes):
                return None
            
            # DJI MakerNotes parsing - look for gimbal data
            # This is a simplified parser; DJI format can vary by model/firmware
            
            # Try to find gimbal angles in the binary data
            # Common DJI gimbal data starts around offset 0x1A-0x26
            if len(maker_notes) > 50:
                try:
                    # Pitch (usually 2 bytes signed int, degrees * 10)
                    if len(maker_notes) > 26:
                        pitch_raw = int.from_bytes(maker_notes[26:28], byteorder='little', signed=True)
                        gimbal_data['pitch'] = pitch_raw / 10.0
                    
                    # Roll
                    if len(maker_notes) > 28:
                        roll_raw = int.from_bytes(maker_notes[28:30], byteorder='little', signed=True)
                        gimbal_data['roll'] = roll_raw / 10.0
                    
                    # Yaw
                    if len(maker_notes) > 30:
                        yaw_raw = int.from_bytes(maker_notes[30:32], byteorder='little', signed=True)
                        gimbal_data['yaw'] = yaw_raw / 10.0
                    
                except (ValueError, IndexError):
                    pass
            
            # If we found any gimbal data, return it
            return gimbal_data if gimbal_data else None
            
        except Exception as e:
            logger.debug(f"DJI gimbal extraction error: {e}")
            return None
    
    def _extract_drone_attitude(self, exif_data: dict, xmp_data: Optional[str], gimbal_data: Optional[Dict]) -> Optional[Dict]:
        """Extract drone attitude (gyro) and compute camera heading"""
        attitude = {}
        
        # Parse from XMP if available
        if xmp_data:
            try:
                if 'xmpmeta' in xmp_data and 'RDF' in xmp_data['xmpmeta']:
                    rdf = xmp_data['xmpmeta']['RDF']
                    if 'Description' in rdf:
                        desc = rdf['Description']
                        # Drone attitude (yaw/pitch/roll)
                        if 'FlightPitchDegree' in desc:
                            attitude['pitch'] = float(desc['FlightPitchDegree'])
                        if 'FlightRollDegree' in desc:
                            attitude['roll'] = float(desc['FlightRollDegree'])
                        if 'FlightYawDegree' in desc:
                            attitude['yaw'] = float(desc['FlightYawDegree'])
                        # Drone velocity
                        if 'FlightXSpeed' in desc:
                            attitude['speed_x'] = float(desc['FlightXSpeed'])
                        if 'FlightYSpeed' in desc:
                            attitude['speed_y'] = float(desc['FlightYSpeed'])
                        if 'FlightZSpeed' in desc:
                            attitude['speed_z'] = float(desc['FlightZSpeed'])
                        # magnitude (m/s) if all components exist
                        if all(k in attitude for k in ('speed_x', 'speed_y', 'speed_z')):
                            attitude['speed'] = (attitude['speed_x']**2 + attitude['speed_y']**2 + attitude['speed_z']**2) ** 0.5
            except Exception as e:
                logger.debug(f"XMP attitude parsing error: {e}")
        
        # DJI GimbalYawDegree is already world-referenced heading on Mini series.
        # Use it directly for camera azimuth and only fall back to drone yaw.
        gimbal_yaw = gimbal_data.get('yaw') if isinstance(gimbal_data, dict) else None
        if gimbal_yaw is not None:
            attitude['camera_heading'] = (float(gimbal_yaw) + 360) % 360
        elif attitude.get('yaw') is not None:
            attitude['camera_heading'] = (float(attitude.get('yaw', 0.0)) + 360) % 360
        
        return attitude if attitude else None
    
    @staticmethod
    def _extract_drone_model(exif_data: dict) -> Optional[str]:
        """Extract drone model from EXIF"""
        try:
            # Tag 0x010f is "Make" in most formats
            if 0x010f in exif_data:
                return str(exif_data[0x010f])
        except Exception:
            pass
        return None
    
    @staticmethod
    def _extract_timestamp(exif_data: dict) -> Optional[str]:
        """Extract image timestamp from EXIF"""
        try:
            # Tag 0x0132 is "DateTime"
            if 0x0132 in exif_data:
                return str(exif_data[0x0132])
        except Exception:
            pass
        return None


class TelemetrySequence:
    """Build and manage flight telemetry from image sequence"""
    
    def __init__(self, hyperlapse_folder: str):
        """
        Initialize from hyperlapse image folder
        Args:
            hyperlapse_folder: path to folder with hyperlapse images
        """
        self.folder = Path(hyperlapse_folder)
        self.images: List[Path] = []
        self.telemetry: List[Dict] = []
        self.extractor = ExifTelemetryExtractor()
        
        self._load_images()
    
    def _load_images(self):
        """Load all images from folder"""
        self.images = sorted(
            list(self.folder.glob('IMG_*.jpg')) + 
            list(self.folder.glob('*.jpg'))
        )
        logger.info(f"Loaded {len(self.images)} images from {self.folder}")
    
    def extract_telemetry(self) -> List[Dict]:
        """Extract telemetry from all images"""
        logger.info(f"Extracting telemetry from {len(self.images)} images...")
        
        self.telemetry = []
        for i, img_path in enumerate(self.images):
            telem = self.extractor.extract_drone_info(str(img_path))
            if telem.get('gps'):
                self.telemetry.append({
                    'index': i,
                    'image_path': str(img_path),
                    **telem
                })
            
            if (i + 1) % 50 == 0:
                logger.info(f"Processed {i + 1}/{len(self.images)} images")
        
        logger.info(f"Extracted telemetry from {len(self.telemetry)} images")
        return self.telemetry
    
    def get_trajectory(self) -> List[Tuple[float, float, float]]:
        """Get flight path as list of (lat, lon, altitude) tuples"""
        trajectory = []
        for telem in self.telemetry:
            gps = telem.get('gps')
            if gps and gps['latitude'] and gps['longitude']:
                trajectory.append((
                    gps['latitude'],
                    gps['longitude'],
                    gps.get('altitude', 0)
                ))
        return trajectory
    
    def get_telemetry_at_index(self, index: int) -> Optional[Dict]:
        """Get telemetry for specific image index"""
        if index < len(self.telemetry):
            return self.telemetry[index]
        return None
    
    def get_bounds(self) -> Optional[Dict]:
        """Get geographical bounds of flight"""
        trajectory = self.get_trajectory()
        if not trajectory:
            return None
        
        lats = [pt[0] for pt in trajectory]
        lons = [pt[1] for pt in trajectory]
        alts = [pt[2] for pt in trajectory]
        
        return {
            'north': max(lats),
            'south': min(lats),
            'east': max(lons),
            'west': min(lons),
            'max_altitude': max(alts),
            'min_altitude': min(alts),
        }
