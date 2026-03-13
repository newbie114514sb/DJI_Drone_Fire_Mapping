# DJI Mini 4 Pro Fire Mapping - Setup Guide

## Overview

This is a post-processing system for analyzing DJI Mini 4 Pro hyperlapse flights. The Mini 4 Pro does NOT support onboard SDK control, so all flight planning is done via the DJI Fly mobile app, and analysis happens after downloading images.

## Installation

### 1. Clone Repository
```bash
git clone <your-repo-url>
cd DJI_Drone_Fire_Mapping
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## Pre-Flight Setup

### 1. Create Flight Plan
Visit [waypointmap.com](https://waypointmap.com/) and:
- Draw survey area (rectangle or polygon)
- Set altitude: 40-60m recommended
- Set waypoint spacing: 10-20m
- Use "Zigzag" pattern for optimal coverage
- Export as KML/WML format

### 2. Import to DJI Fly App
1. Open DJI Fly on your phone/tablet
2. Go to: Menu → Flight Plan → Create → Import
3. Select exported waypoint file
4. Review and adjust if needed

### 3. Configure Hyperlapse Settings
In DJI Fly before flight:
- **Mode**: Hyperlapse (not regular Video)
- **Interval**: 3-5 seconds between frames
- **Resolution**: 4K (4096×2160)
- **Gimbal Pitch**: -90° (point straight down)

## Flight Execution

### 1. Pre-Flight Check
- [ ] Battery: ≥80% charged
- [ ] SD Card: Formatted on drone
- [ ] GPS: Lock acquired (≥20 satellites)
- [ ] Weather: No strong wind or rain
- [ ] Propellers: Secure and undamaged

### 2. Start Mission
1. Power on drone and controller
2. Open DJI Fly app
3. Select your flight plan
4. Review flight path on map
5. Tap START and confirm

### 3. Monitor Flight
- Watch battery level in app
- Ensure drone maintains altitude
- Hyperlapse will auto-return when done

## Post-Flight Analysis

### 1. Download Images
- Insert SD card into computer
- Copy `Hyperlapse_XXXX` folder to:
  ```
  data/raw/hyperlapse_images/
  ```

### 2. Run Analysis
**Option A: Interactive Notebook (Recommended)**
```bash
jupyter notebook notebooks/hyperlapse_viewer.ipynb
```
Then edit the `HYPERLAPSE_FOLDER` path and run cells.

**Option B: Command Line**
```bash
python example_analyze_hyperlapse.py
```

**Option C: Main Script**
```bash
python main.py --analyze data/raw/hyperlapse_images --output data/outputs/
```

### 3. View Results
- **Maps**: Open `data/outputs/trajectory_map.html` in web browser
- **Charts**: View `data/outputs/altitude_profile.png`
- **Report**: Read `data/outputs/flight_report.txt`

## Project Structure

```
DJI_Drone_Fire_Mapping/
├── src/
│   ├── drone_control/
│   │   └── flight_manager.py         # Flight planning info
│   ├── image_processing/
│   │   └── image_handler.py          # Image preprocessing
│   ├── detection/
│   │   └── detector.py               # Fire detection (future)
│   └── visualization/
│       ├── telemetry.py              # EXIF extraction
│       ├── viewer.py                 # Image viewer + telemetry
│       └── map_generator.py          # Map generation
│
├── main.py                           # CLI entry point
├── example_analyze_hyperlapse.py     # Example script
├── config/
│   └── config.yaml                   # Settings
├── data/
│   ├── raw/                          # Hyperlapse folders
│   └── outputs/                      # Generated maps/reports
├── notebooks/
│   ├── hyperlapse_viewer.ipynb       # Interactive viewer
│   └── fire_detection_exploration.ipynb  # Fire detection notebook
└── requirements.txt
```

## Telemetry Data

### What Gets Recorded
Each hyperlapse image automatically records in EXIF metadata:

| Data | Source | Description |
|------|--------|-------------|
| Latitude | GPS | Decimal degrees (-90 to 90) |
| Longitude | GPS | Decimal degrees (-180 to 180) |
| Altitude | Barometer | Height above takeoff point (meters) |
| Gimbal Pitch | Gimbal sensor | Downward angle (-90° = straight down) |
| Gimbal Roll | Gimbal sensor | Roll angle |
| Gimbal Yaw | Compass | Heading (0-360°) |
| Timestamp | RTC | Image capture time |
| Drone Model | Firmware | "DJI Mini 4 Pro" etc |

### View Raw EXIF
```bash
# Linux/macOS
exiftool data/raw/hyperlapse_images/IMG_0001.jpg

# Python
from PIL import Image
img = Image.open('IMG_0001.jpg')
print(img._getexif())
```

## Configuration

Edit `config/config.yaml` to adjust:

```yaml
drone:
  survey_altitude_m: 50      # Flight altitude
  gimbal_pitch: -90          # Camera angle

hyperlapse:
  interval_seconds: 3        # Frames per 3 seconds
  target_resolution: "4K"    # Image size

detection:
  confidence_threshold: 0.75 # For fire detection (future)
```

## Troubleshooting

### "No telemetry extracted"
- Ensure images have GPS EXIF data
- Check that images are in JPG format
- Verify drone had GPS lock during flight
- Try opening images in phone gallery to confirm metadata

### "Folder not found"
- Check path is correct: `data/raw/hyperlapse_images/`
- Ensure folder name is exactly right
- File browser shows hidden files (on some OS)

### "Map not generating"
- Ensure you have folium installed: `pip install folium`
- Check internet connection (for map tiles)
- Verify telemetry extracted successfully first

### Slow Processing
- Reduce number of images (extract every 2nd/3rd frame)
- Use smaller image resolution for preview
- Crop survey area in image viewer

## Next Steps

1. **Practice Flights**: Do 2-3 test flights to perfect settings
2. **Model Training**: Collect labeled fire/ground images for ML model
3. **Integration**: Add fire detection to image viewer
4. **Validation**: Test system in controlled burn scenarios
5. **Deployment**: Use in actual wildfire monitoring

## References

- [DJI Fly App Guide](https://support.dji.com/hc/en-us/articles/360019233051-DJI-Fly-Quick-Start-Guide)
- [Waypointmap.com](https://waypointmap.com/)
- [Mini 4 Pro Specs](https://www.dji.com/ca/mini-4-pro/specs)
- [EXIF Data Format](https://www.exif.org/)
- [YOLOv8 Fire Detection](https://docs.ultralytics.com/)

## Support

For issues or questions:
1. Check README.md for overview
2. Review example_analyze_hyperlapse.py
3. See notebook examples in notebooks/
4. Check GitHub issues (if applicable)

## License

TBD
