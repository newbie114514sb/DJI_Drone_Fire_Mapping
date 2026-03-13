"""
Telemetry extraction from DJI hyperlapse image EXIF data
- Extract GPS coordinates, altitude, gimbal angles from image metadata
- Build flight trajectory from image sequence
"""

from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging

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
            
            if not exif_data:
                logger.debug(f"No EXIF data in {image_path}")
                return {}
            
            # Extract GPS
            gps = self.extract_gps_from_exif(exif_data)
            
            # Extract other metadata
            telemetry = {
                'image_path': str(image_path),
                'gps': gps,
                'gimbal': self._extract_gimbal(exif_data),
                'drone_model': self._extract_drone_model(exif_data),
                'timestamp': self._extract_timestamp(exif_data),
            }
            
            return telemetry
        
        except Exception as e:
            logger.error(f"Failed to extract telemetry from {image_path}: {e}")
            return {}
    
    def _extract_gimbal(self, exif_data: dict) -> Optional[Dict]:
        """Extract gimbal angles if available in EXIF"""
        # DJI stores gimbal info in maker notes or specific tags
        # This is a simplified version - actual tag varies by DJI model
        try:
            # Common DJI gimbal tags
            gimbal_info = {}
            for tag_id, value in exif_data.items():
                tag_name = self.TAGS.get(tag_id, tag_id)
                # Look for gimbal-related tags
                if 'gimbal' in str(tag_name).lower():
                    gimbal_info[tag_name] = value
            
            return gimbal_info if gimbal_info else None
        except Exception as e:
            logger.debug(f"Gimbal extraction error: {e}")
            return None
    
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
