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
- 📍 **GPS Telemetry Extraction** - Parse EXIF/XMP data for position, altitude, gimbal angles, drone attitude, and camera heading
- 🗺️ **Interactive Maps** - View flight trajectory on Folium maps
- 📊 **Altitude Profiles** - Visualize flight elevation changes
- 📈 **Flight Reports** - Generate summary statistics
- 🎛️ **HUD Viewer Overlay** - Show azimuth, elevation, attitude, minimap, and altitude profile in the served viewer
- 🌐 **Timestamped Report Serving** - Generate timestamped output folders and serve the newest or a selected report
- 🚗 **YOLOv8-Ready Vehicle Overlay** - Optionally render red detection boxes in the HUD and geolocated car markers on the map when a model is provided

**Future:**
- 🔥 Fire detection with YOLOv8 (custom model)
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

# Serve the newest generated report
python main.py --serve-latest

# Serve a specific report folder by name
python main.py --serve 03-15-2026_13-06

# List reports and choose one interactively
python main.py --list-reports

# Optional: run a YOLOv8 model and keep only cars
python main.py --analyze data/raw/hyperlapse_images --output data/outputs/ --detect-model path/to/model.pt

# Optional: keep multiple classes and change the threshold
python main.py --analyze data/raw/hyperlapse_images --output data/outputs/ --detect-model path/to/model.pt --detect-class car --detect-class truck --detect-confidence 0.35
```

Generated reports are written to timestamped folders under `data/outputs/`. When the viewer changes, generate a new report rather than editing an older output folder in place.
If `--detect-model` is provided, the report also writes `object_detections.json`, `car_detection_report.txt`, and `car_detections.geojson` when tracks can be geolocated.

## Telemetry Data

Each hyperlapse image contains metadata in EXIF/XMP:

- **GPS**: Latitude, longitude, altitude
- **Gimbal**: Pitch, roll, yaw angles
- **Drone Attitude**: Pitch, roll, yaw, and speed components when present in DJI XMP
- **Camera Heading**: Earth-relative heading derived from drone yaw and gimbal yaw
- **Timing**: Image capture timestamp
- **Drone**: Model, firmware version

The system extracts this automatically and displays:
- Live telemetry overlay on images
- GPS position minimap
- Altitude profile chart
- Flight trajectory visualization
- Earth-relative azimuth and elevation readouts in the HUD viewer

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
from src.visualization.viewer import TrajectoryMapGenerator

# Load hyperlapse images
telem_seq = TelemetrySequence('data/raw/my_hyperlapse/')

# Extract GPS, altitude, gimbal, and attitude from EXIF/XMP
telemetry = telem_seq.extract_telemetry()

# Get flight bounds
bounds = telem_seq.get_bounds()
print(f"Flight area: {bounds['north']} - {bounds['south']} lat")

# Generate maps and interactive report assets
map_gen = TrajectoryMapGenerator(telem_seq)
map_gen.create_trajectory_map('trajectory.html')
map_gen.create_altitude_profile('altitude_profile.png')
map_gen.create_interactive_video_viewer('hyperlapse_viewer.html')
```

## Viewer And Reports

- The HTML viewer is generated into each report folder as `hyperlapse_viewer.html`.
- Reports are timestamped like `03-15-2026_13-06/` so older analysis snapshots remain inspectable.
- `--serve-latest` serves the newest report on `http://localhost:8001/hyperlapse_viewer.html` with cache disabled.
- The viewer includes a HUD-style center sight, top azimuth ruler, right elevation ruler, minimap, and slide-out altitude profile.
- The altitude shown in the telemetry/report is GPS altitude above mean sea level, not above local ground.

## Fire Detection (Future)

Once implemented, will use:
- **Model**: YOLOv8 (real-time object detection)
- **Input**: Hyperlapse images or extracted frames
- **Output**: Fire bounding boxes with confidence scores → GPS locations
- **Training**: Custom dataset from fire/drone imagery

## Vehicle Detection Overlay

The analysis pipeline can now accept a placeholder YOLOv8 model for cars or other object classes.

- `--detect-model`: path to a YOLOv8 `.pt` model
- `--detect-class`: class label to keep from the model output; repeat the flag for multiple classes
- `--detect-confidence`: minimum confidence threshold for retained detections

When detection mode is enabled:

- The HUD viewer draws red bounding boxes over the current frame.
- Multi-frame tracks are triangulated when possible to estimate object latitude, longitude, and MSL altitude.
- The minimap and saved trajectory map show red markers for geolocated detections.

## 3D Object Geolocation

The repo now includes a multi-view triangulation path for future YOLOv8 detections.

- Single-view detections can be projected only if you provide a known target altitude plane.
- True 3D object placement uses two or more detections of the same object from different frames.
- Output is latitude, longitude, and altitude above sea level.

Use [src/visualization/map_generator.py](src/visualization/map_generator.py) through `FireGeolocation.triangulate_observations(...)` with observations shaped like:

```python
observations = [
   {
      'drone_lat': 34.145123,
      'drone_lon': -118.029201,
      'drone_altitude_m': 242.4,
      'camera_heading': 91.3,
      'drone_pitch': 0.0,
      'gimbal_pitch': -34.6,
      'bbox': (1180, 420, 1300, 560),
      'frame_width': 3840,
      'frame_height': 2160,
      'horizontal_fov_deg': 70.0,
      'vertical_fov_deg': 56.0,
      'confidence': 0.93,
   },
   {
      'drone_lat': 34.145044,
      'drone_lon': -118.029015,
      'drone_altitude_m': 242.3,
      'camera_heading': 102.4,
      'drone_pitch': 0.0,
      'gimbal_pitch': -31.9,
      'bbox': (980, 440, 1100, 575),
      'frame_width': 3840,
      'frame_height': 2160,
      'horizontal_fov_deg': 70.0,
      'vertical_fov_deg': 56.0,
      'confidence': 0.91,
   },
]

geolocator = FireGeolocation(config={})
object_fix = geolocator.triangulate_observations(observations)
```

That result can be written to HTML maps or GeoJSON using `MapGenerator.export_geojson(...)`.

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
