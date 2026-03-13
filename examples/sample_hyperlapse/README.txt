Sample Hyperlapse Dataset
==================================================

This folder contains sample images that simulate
a DJI Mini 4 Pro hyperlapse flight.

Each image includes EXIF metadata with:
  - GPS latitude/longitude/altitude
  - Gimbal pitch angle (-90° = downward)
  - Timestamp (3-second intervals)
  - Drone model info

Usage:
  jupyter notebook notebooks/hyperlapse_viewer.ipynb
  # Then set HYPERLAPSE_FOLDER to this directory
