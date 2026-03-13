#!/usr/bin/env python3
"""
Example: Analyze a hyperlapse flight
Shows how to use the TelemetrySequence, HyperlapseViewer, and mapping tools
"""

from pathlib import Path
from src.visualization.telemetry import TelemetrySequence
from src.visualization.viewer import HyperlapseViewer, TrajectoryMapGenerator


def analyze_hyperlapse_example():
    """Example analysis of hyperlapse flight data"""
    
    # Path to your hyperlapse images
    hyperlapse_folder = Path('data/raw/hyperlapse_images')
    output_dir = Path('data/outputs')
    
    print("=" * 60)
    print("DJI Mini 4 Pro Hyperlapse Fire Mapping - Analysis Example")
    print("=" * 60)
    
    # Check if folder exists
    if not hyperlapse_folder.exists():
        print(f"\n❌ Hyperlapse folder not found: {hyperlapse_folder}")
        print(f"\nTo use this script:")
        print(f"  1. Download your hyperlapse images from drone SD card")
        print(f"  2. Copy folder to: {hyperlapse_folder.resolve()}")
        print(f"  3. Run this script again")
        return
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Loading hyperlapse images from: {hyperlapse_folder.resolve()}")
    
    # Initialize sequence
    telem_seq = TelemetrySequence(str(hyperlapse_folder))
    print(f"✓ Found {len(telem_seq.images)} images")
    
    # Extract telemetry
    print("\n📍 Extracting telemetry from EXIF metadata...")
    telemetry = telem_seq.extract_telemetry()
    
    if not telemetry:
        print("❌ No telemetry data found in images")
        print("   Ensure images have GPS data recorded (check EXIF)")
        return
    
    print(f"✓ Extracted {len(telemetry)} telemetry points")
    
    # Display statistics
    bounds = telem_seq.get_bounds()
    print("\n📊 Flight Statistics:")
    print(f"   Images: {len(telem_seq.images)}")
    print(f"   GPS Points: {len(telemetry)}")
    print(f"   Latitude:  {bounds['south']:.6f}° to {bounds['north']:.6f}°")
    print(f"   Longitude: {bounds['west']:.6f}° to {bounds['east']:.6f}°")
    print(f"   Altitude:  {bounds['min_altitude']:.1f}m to {bounds['max_altitude']:.1f}m")
    
    # Display sample telemetry
    print("\n📋 Sample Telemetry (first 3 images):")
    for i, telem in enumerate(telemetry[:3]):
        gps = telem.get('gps', {})
        print(f"   Image {telem['index']}: "
              f"({gps['latitude']:.6f}, {gps['longitude']:.6f}) "
              f"Alt: {gps.get('altitude', 0):.1f}m")
    
    # Generate maps
    print("\n🗺️  Generating maps...")
    map_gen = TrajectoryMapGenerator(telem_seq)
    
    map_path = output_dir / 'trajectory_map.html'
    map_gen.create_trajectory_map(str(map_path))
    print(f"✓ Trajectory map: {map_path}")
    
    profile_path = output_dir / 'altitude_profile.png'
    map_gen.create_altitude_profile(str(profile_path))
    print(f"✓ Altitude profile: {profile_path}")
    
    # Create report
    report_path = output_dir / 'flight_report.txt'
    with open(report_path, 'w') as f:
        f.write("HYPERLAPSE FLIGHT ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Date: {telem_seq.images[0].stat().st_mtime if telem_seq.images else 'Unknown'}\n")
        f.write(f"Images: {len(telem_seq.images)}\n")
        f.write(f"GPS Points: {len(telemetry)}\n\n")
        f.write("Flight Bounds:\n")
        f.write(f"  North: {bounds['north']:.6f}°\n")
        f.write(f"  South: {bounds['south']:.6f}°\n")
        f.write(f"  East:  {bounds['east']:.6f}°\n")
        f.write(f"  West:  {bounds['west']:.6f}°\n")
        f.write(f"\nAltitude Range:\n")
        f.write(f"  Min: {bounds['min_altitude']:.1f}m\n")
        f.write(f"  Max: {bounds['max_altitude']:.1f}m\n")
    
    print(f"✓ Report: {report_path}")
    
    # Display viewer
    print("\n🎬 Initializing image viewer...")
    viewer = HyperlapseViewer(str(hyperlapse_folder), telem_seq)
    
    # Show first and last images
    print(f"\n   First image with overlay:")
    img = viewer.get_image_at_index(0)
    telem = viewer.get_telemetry_at_index(0)
    if img is not None and telem:
        annotated = viewer.draw_telemetry_overlay(img, telem)
        # Save sample
        sample_path = output_dir / 'sample_frame_annotated.jpg'
        import cv2
        cv2.imwrite(str(sample_path), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
        print(f"✓ Sample: {sample_path}")
    
    print("\n" + "=" * 60)
    print("✅ Analysis complete!")
    print(f"\n📁 Output folder: {output_dir.resolve()}")
    print(f"   - trajectory_map.html (interactive map)")
    print(f"   - altitude_profile.png (elevation chart)")
    print(f"   - flight_report.txt (statistics)")
    print(f"   - sample_frame_annotated.jpg (image with telemetry)")
    
    print("\n💡 Next steps:")
    print("   1. Open trajectory_map.html in web browser")
    print("   2. Use hyperlapse_viewer.ipynb for interactive viewing")
    print("   3. Implement fire detection (coming soon)")
    print("=" * 60)


if __name__ == '__main__':
    analyze_hyperlapse_example()
