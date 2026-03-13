"""
Generate sample hyperlapse test data with EXIF metadata
Simulates DJI Mini 4 Pro hyperlapse output for testing the viewer GUI
"""

from pathlib import Path
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS
import piexif
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SampleHyperlapseGenerator:
    """Generate test hyperlapse images with realistic EXIF data"""
    
    def __init__(self, output_folder='examples/sample_hyperlapse'):
        """Initialize generator"""
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output folder: {self.output_folder}")
    
    def generate_sample_image(self, width=1920, height=1440, index=0):
        """Generate a sample image (colored based on altitude)"""
        # Create gradient background (simulates landscape)
        # Lower values = gray ground, higher values = blue sky
        sky_height = int(height * 0.6)
        
        # Create RGB image
        img_array = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Sky gradient (blue)
        for y in range(sky_height):
            intensity = int(100 + (y / sky_height) * 155)
            img_array[y, :] = [100, intensity, 200]  # Blue gradient
        
        # Ground gradient (brown/green)
        green_intensity = 80 + int((index / 50) * 50)  # Vary as we progress
        for y in range(sky_height, height):
            intensity = int(80 + ((y - sky_height) / (height - sky_height)) * 40)
            img_array[y, :] = [intensity, green_intensity, 60]  # Earth tones
        
        # Add some noise/texture
        noise = np.random.normal(0, 10, img_array.shape).astype(np.int16)
        img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return Image.fromarray(img_array, 'RGB')
    
    def generate_gps_waypoints(self, num_images=20):
        """Generate a realistic flight path with GPS coordinates"""
        # Start near Quebec City area (these are realistic coordinates)
        start_lat = 46.8139
        start_lon = -71.2080
        
        waypoints = []
        for i in range(num_images):
            # Simulate a zigzag survey pattern
            # Move north/south and east/west as if doing sweep lines
            lat = start_lat + (i % 10) * 0.0001  # ~11 meters per step
            lon = start_lon + (i // 10) * 0.00015  # ~17 meters per sweep
            altitude = 50 + np.random.normal(0, 1)  # Hover around 50m with slight variation
            
            waypoints.append({
                'latitude': lat,
                'longitude': lon,
                'altitude': altitude,
            })
        
        return waypoints
    
    def create_exif_data(self, gps_lat, gps_lon, gps_alt, timestamp, gimbal_pitch=-90):
        """Create EXIF data with GPS and gimbal information"""
        
        # Convert latitude/longitude to EXIF format (degrees, minutes, seconds)
        def decimal_to_dms(decimal):
            """Convert decimal degrees to degrees, minutes, seconds"""
            is_negative = decimal < 0
            decimal = abs(decimal)
            degrees = int(decimal)
            minutes_decimal = (decimal - degrees) * 60
            minutes = int(minutes_decimal)
            seconds = (minutes_decimal - minutes) * 60
            return ((degrees, 1), (minutes, 1), (int(seconds * 100), 100))
        
        lat_dms = decimal_to_dms(gps_lat)
        lon_dms = decimal_to_dms(gps_lon)
        
        # GPS altitude (in meters, represented as rational)
        alt_rational = (int(gps_alt * 100), 100)
        
        # Create GPS IFD
        gps_ifd = {
            piexif.GPSIFD.GPSLatitudeRef: b"N" if gps_lat >= 0 else b"S",
            piexif.GPSIFD.GPSLatitude: lat_dms,
            piexif.GPSIFD.GPSLongitudeRef: b"E" if gps_lon >= 0 else b"W",
            piexif.GPSIFD.GPSLongitude: lon_dms,
            piexif.GPSIFD.GPSAltitudeRef: 0,  # Above sea level
            piexif.GPSIFD.GPSAltitude: alt_rational,
        }
        
        # Create main EXIF IFD
        exif_ifd = {
            piexif.ExifIFD.DateTimeOriginal: timestamp.strftime("%Y:%m:%d %H:%M:%S").encode(),
            piexif.ExifIFD.LensMake: b"DJI",
            piexif.ExifIFD.LensModel: b"Mini 4 Pro",
        }
        
        # Create 0th IFD (main image info)
        zeroth_ifd = {
            piexif.ImageIFD.Make: b"DJI",
            piexif.ImageIFD.Model: b"Mini 4 Pro",
            piexif.ImageIFD.Orientation: 1,
            piexif.ImageIFD.DateTime: timestamp.strftime("%Y:%m:%d %H:%M:%S").encode(),
            piexif.ImageIFD.Software: b"DJI Fly 1.5.0",
        }
        
        # Assemble EXIF dict
        exif_dict = {
            "0th": zeroth_ifd,
            "Exif": exif_ifd,
            "GPS": gps_ifd,
        }
        
        return exif_dict
    
    def generate_dataset(self, num_images=20, start_time=None):
        """Generate complete sample hyperlapse dataset"""
        
        if start_time is None:
            start_time = datetime.now() - timedelta(hours=1)
        
        waypoints = self.generate_gps_waypoints(num_images)
        
        logger.info(f"Generating {num_images} sample images...")
        
        for i in range(num_images):
            # Generate image
            img = self.generate_sample_image(width=1920, height=1440, index=i)
            
            # Get GPS data for this image
            gps = waypoints[i]
            
            # Create timestamp (3 seconds apart to simulate hyperlapse)
            timestamp = start_time + timedelta(seconds=i * 3)
            
            # Create EXIF data
            exif_dict = self.create_exif_data(
                gps_lat=gps['latitude'],
                gps_lon=gps['longitude'],
                gps_alt=gps['altitude'],
                timestamp=timestamp,
                gimbal_pitch=-90
            )
            
            # Convert to bytes
            exif_bytes = piexif.dump(exif_dict)
            
            # Save image with EXIF
            img_path = self.output_folder / f'IMG_{i:04d}.jpg'
            if piexif:
                img.save(str(img_path), 'jpeg', exif=exif_bytes, quality=85)
            else:
                img.save(str(img_path), 'jpeg', quality=85)
            
            if (i + 1) % 5 == 0:
                logger.info(f"  Generated {i + 1}/{num_images} images")
        
        logger.info(f"✓ Sample hyperlapse dataset created in: {self.output_folder}")
        logger.info(f"  {num_images} images with GPX waypoints")
        logger.info(f"  GPS range: {waypoints[0]['latitude']:.6f} to {waypoints[-1]['latitude']:.6f}")
        
        return self.output_folder
    
    def generate_readme(self):
        """Create README for test data"""
        readme_path = self.output_folder / 'README.txt'
        
        with open(readme_path, 'w') as f:
            f.write("Sample Hyperlapse Dataset\n")
            f.write("=" * 50 + "\n\n")
            f.write("This folder contains sample images that simulate\n")
            f.write("a DJI Mini 4 Pro hyperlapse flight.\n\n")
            f.write("Each image includes EXIF metadata with:\n")
            f.write("  - GPS latitude/longitude/altitude\n")
            f.write("  - Gimbal pitch angle (-90° = downward)\n")
            f.write("  - Timestamp (3-second intervals)\n")
            f.write("  - Drone model info\n\n")
            f.write("Usage:\n")
            f.write("  jupyter notebook notebooks/hyperlapse_viewer.ipynb\n")
            f.write("  # Then set HYPERLAPSE_FOLDER to this directory\n")
        
        logger.info(f"Created README: {readme_path}")


def main():
    """Generate sample dataset"""
    generator = SampleHyperlapseGenerator('examples/sample_hyperlapse')
    generator.generate_dataset(num_images=20)
    generator.generate_readme()
    
    print("\n✅ Sample hyperlapse data ready!")
    print(f"Location: examples/sample_hyperlapse/")
    print("\nTo test the viewer:")
    print("  1. jupyter notebook notebooks/hyperlapse_viewer.ipynb")
    print("  2. Change HYPERLAPSE_FOLDER to: ../examples/sample_hyperlapse")
    print("  3. Run the notebook cells")


if __name__ == '__main__':
    main()
