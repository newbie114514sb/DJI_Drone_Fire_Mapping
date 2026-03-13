# DJI Drone Fire Mapping

Science fair project: Automated wildfire detection and mapping using DJI Mini 4 Pro and Mini 5 drones with post-flight processing.

## Overview

This system detects wildfires from drone hyperlapse flights using post-processing:

**Workflow:**
1. Create flight plan with [waypointmap.com](https://waypointmap.com) or similar
2. Load plan onto drone via DJI Fly app
3. Start mission manually, use HYPERLAPSE mode
4. Hyperlapse records images with GPS/gimbal data in EXIF metadata
5. Download images and run through analysis pipeline

**Current Features:**
- 📷 **Hyperlapse Image Viewer** - Browse images with telemetry overlay
- 📍 **GPS Telemetry Extraction** - Parse EXIF data for position, altitude, gimbal angles
- 🗺️ **Interactive Maps** - View flight trajectory on Folium maps
- 📊 **Altitude Profiles** - Visualize flight elevation changes
- 📈 **Flight Reports** - Generate summary statistics

**Future:**
- 🔥 Fire detection with YOLOv8 (coming next)
- 🌡️ Thermal camera support (if using M300 instead)

Based on research from [forest_fire_detection_system](https://github.com/lee-shun/forest_fire_detection_system).

## Quick Start

### 1. Setup
```bash
pip install -r requirements.txt
```

### 2. Prepare Flight Plan
- Go to [waypointmap.com](https://waypointmap.com)
- Create a survey mission (zigzag pattern recommended)
- Export and import into DJI Fly app

### 3. Execute Flight
- Launch DJI Fly app
- Select your waypoint mission
- Set HYPERLAPSE mode (2-5 second interval)
- Point gimbal at -90° (straight down)
- Start flight

### 4. Download & Analyze
```bash
# Copy downloaded hyperlapse folder to data/raw/

# View results in interactive notebook
jupyter notebook notebooks/hyperlapse_viewer.ipynb

# Or analyze from command line
python main.py --analyze data/raw/hyperlapse_images --output data/outputs/
```

## Project Structure

```
src/
├── drone_control/
│   └── flight_manager.py       # Flight plan guide, hyperlapse info
├── image_processing/
│   └── image_handler.py        # Frame preprocessing (future use)
├── detection/
│   └── detector.py             # Fire detection model (future)
└── visualization/
    ├── telemetry.py            # EXIF parsing, flight trajectory
    ├── viewer.py               # Image viewer + telemetry overlay
    └── map_generator.py        # Folium maps, altitude profiles

notebooks/
├── hyperlapse_viewer.ipynb     # Interactive viewer & analysis
└── fire_detection_exploration.ipynb  # Fire detection notebook (future)

config/
└── config.yaml                 # Settings & drone parameters

data/
├── raw/                        # Hyperlapse folders from drone
├── processed/                  # Processed images
└── outputs/                    # Maps and reports
```

## Dependencies

**Core:**
- `opencv-python` - Image processing
- `pillow` - EXIF metadata extraction
- `numpy` - Array operations

**Visualization:**
- `matplotlib` - Charts and plots
- `folium` - Interactive maps

**Optional (for fire detection):**
- `ultralytics` - YOLOv8 model
- `tensorflow` - Deep learning inference

See `requirements.txt` for versions.

## Usage

### Using Jupyter Notebook (Recommended)
```bash
jupyter notebook notebooks/hyperlapse_viewer.ipynb
```

Then:
1. Edit `HYPERLAPSE_FOLDER` to point to your image folder
2. Run cells in order
3. Use slider to browse images with telemetry
4. View generated maps in `data/outputs/`

### Using Command Line
```bash
# Show flight instructions
python main.py --help-flight

# Analyze hyperlapse folder
python main.py --analyze data/raw/hyperlapse_images --output data/outputs/
```

## Telemetry Data

Each hyperlapse image contains metadata in EXIF:

- **GPS**: Latitude, longitude, altitude
- **Gimbal**: Pitch, roll, yaw angles
- **Timing**: Image capture timestamp
- **Drone**: Model, firmware version

The system extracts this automatically and displays:
- Live telemetry overlay on images
- GPS position minimap
- Altitude profile chart
- Flight trajectory visualization

## DJI Mini 4 Pro Notes

⚠️ **SDK Limitation:** Mini 4 Pro does NOT support onboard SDK control (DJI Fly Mobile SDK only).

**Why Hyperlapse?**
- ✓ Automatic GPS logging in EXIF
- ✓ Gimbal angle recording
- ✓ Works with autonomous waypoint plans
- ✓ Mobile app only (no complex SDK setup)
- ✗ Post-processing only (not real-time)

**Flight Specifications:**
- Max altitude: 2500m (limited by regulation)
- Recommended survey altitude: 40-60m (good image resolution)
- Flight time: ~31 min max (limit at 20% battery)
- Payload: ~249g (no external cameras)
- Video: 4K/30fps max (hyperlapse uses high-res stills)

## Hyperlapse Flight Setup

1. **DJI Fly Settings:**
   - Mode: HYPERLAPSE
   - Interval: 2-5 seconds between frames
   - Gimbal Pitch: -90° (nadir/downward)
   - Resolution: 4K (4096×2160)

2. **Pre-flight:**
   - Format SD card on drone
   - Verify GPS lock (wait ≥20 satellites)
   - Check battery ≥80%

3. **Expected Output:**
   - Folder: `Hyperlapse_XXXX/` with IMG_XXXX.jpg files
   - Each image has GPS + gimbal data in EXIF
   - ~1 image per waypoint in survey area

## Example Workflow

```python
from src.visualization.telemetry import TelemetrySequence
from src.visualization.viewer import HyperlapseViewer, TrajectoryMapGenerator

# Load hyperlapse images
telem_seq = TelemetrySequence('data/raw/my_hyperlapse/')

# Extract GPS, altitude, gimbal from EXIF
telemetry = telem_seq.extract_telemetry()

# Get flight bounds
bounds = telem_seq.get_bounds()
print(f"Flight area: {bounds['north']} - {bounds['south']} lat")

# Create viewer
viewer = HyperlapseViewer('data/raw/my_hyperlapse/', telem_seq)
img = viewer.get_image_at_index(0)  # Get first image
telem = viewer.get_telemetry_at_index(0)
img_with_overlay = viewer.draw_telemetry_overlay(img, telem)

# Generate maps
map_gen = TrajectoryMapGenerator(telem_seq)
map_gen.create_trajectory_map('trajectory.html')
map_gen.create_altitude_profile('altitude_profile.png')
```

## Fire Detection (Future)

Once implemented, will use:
- **Model**: YOLOv8 (real-time object detection)
- **Input**: Hyperlapse images or extracted frames
- **Output**: Fire bounding boxes with confidence scores → GPS locations
- **Training**: Custom dataset from fire/drone imagery

## Safety

⚠️ **Before any flight:**
- Obtain necessary permits (check local regulations)
- Check airspace restrictions (use B4UFLY app)
- Test all systems on ground first
- Have emergency procedures
- Fly only in safe, designated areas
- Maintain visual line of sight

## References

- [DJI Mini 4 Pro Specs](https://www.dji.com/ca/mini-4-pro/specs)
- [DJI Fly App](https://www.dji.com/downloads/djiflysafe)
- [Waypointmap.com](https://waypointmap.com/)
- [Lee Shun's Fire Detection System](https://github.com/lee-shun/forest_fire_detection_system)
- [YOLOv8 Documentation](https://docs.ultralytics.com/)

## License

TBD
