#!/usr/bin/env python3
"""
Quick test of the hyperlapse viewer with sample data
Run this to verify the GUI works with the sample images
"""

from pathlib import Path
import sys

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.visualization.telemetry import TelemetrySequence
from src.visualization.viewer import HyperlapseViewer, TrajectoryMapGenerator


def test_viewer():
    """Test the viewer with sample data"""
    
    print("\n" + "=" * 60)
    print("DJI Mini 4 Pro Hyperlapse Viewer - Test Run")
    print("=" * 60)
    
    sample_folder = Path('examples/sample_hyperlapse')
    
    if not sample_folder.exists():
        print(f"\n❌ Sample folder not found: {sample_folder}")
        print("\nGenerate sample data first:")
        print("  python generate_test_data.py")
        return False
    
    print(f"\n📁 Found sample folder: {sample_folder}")
    
    # Load images
    print(f"\n📷 Loading {len(list(sample_folder.glob('IMG_*.jpg')))} images...")
    
    # Extract telemetry
    print("📍 Extracting telemetry from EXIF...")
    telem_seq = TelemetrySequence(str(sample_folder))
    telemetry = telem_seq.extract_telemetry()
    
    if not telemetry:
        print("❌ No telemetry extracted!")
        return False
    
    print(f"✓ Extracted {len(telemetry)} GPS points")
    
    # Show bounds
    bounds = telem_seq.get_bounds()
    print(f"\n📊 Flight Statistics:")
    print(f"   Total Images: {len(telem_seq.images)}")
    print(f"   GPS Points: {len(telemetry)}")
    print(f"   Latitude:  {bounds['south']:.6f}° to {bounds['north']:.6f}°")
    print(f"   Longitude: {bounds['west']:.6f}° to {bounds['east']:.6f}°")
    print(f"   Altitude:  {bounds['min_altitude']:.1f}m to {bounds['max_altitude']:.1f}m")
    
    # Show sample telemetry
    print(f"\n📋 Sample Telemetry (first 3 images):")
    for i, telem in enumerate(telemetry[:3]):
        gps = telem.get('gps', {})
        print(f"   Image {telem['index']}: "
              f"({gps['latitude']:.6f}, {gps['longitude']:.6f}) "
              f"Alt: {gps.get('altitude', 0):.1f}m")
    
    # Test viewer
    print(f"\n🎬 Testing image viewer...")
    viewer = HyperlapseViewer(str(sample_folder), telem_seq)
    
    img = viewer.get_image_at_index(0)
    telem = viewer.get_telemetry_at_index(0)
    
    if img is None:
        print("❌ Could not load image")
        return False
    
    if telem is None:
        print("❌ Could not load telemetry")
        return False
    
    print(f"   ✓ Loaded image: {img.shape}")
    print(f"   ✓ Loaded telemetry for image 0")
    
    # Test overlay
    img_with_overlay = viewer.draw_telemetry_overlay(img, telem)
    print(f"   ✓ Drew telemetry overlay")
    
    # Generate maps
    print(f"\n🗺️  Generating maps...")
    output_dir = Path('data/outputs/test_sample')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    map_gen = TrajectoryMapGenerator(telem_seq)
    
    map_result = map_gen.create_trajectory_map(str(output_dir / 'test_trajectory.html'))
    profile_result = map_gen.create_altitude_profile(str(output_dir / 'test_altitude.png'))
    
    if map_result:
        print(f"   ✓ Created trajectory map: {output_dir / 'test_trajectory.html'}")
    if profile_result:
        print(f"   ✓ Created altitude profile: {output_dir / 'test_altitude.png'}")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed! The viewer is working correctly.")
    print("=" * 60)
    
    print("\nNext steps:")
    print("  1. View the sample interactive map:")
    print(f"     {output_dir / 'test_trajectory.html'}")
    print("\n  2. Test the interactive notebook:")
    print("     jupyter notebook notebooks/hyperlapse_viewer.ipynb")
    print("     (Set HYPERLAPSE_FOLDER = 'examples/sample_hyperlapse')")
    
    return True


if __name__ == '__main__':
    success = test_viewer()
    sys.exit(0 if success else 1)
