"""
DJI Mini 4 Pro Fire Mapping System - Hyperlapse Analysis Pipeline

Workflow:
1. Create flight plan using waypointmap.com or similar tool
2. Load plan onto drone via DJI Fly app
3. Start mission manually with HYPERLAPSE enabled
4. Drone records images with embedded GPS/gimbal data in EXIF
5. Download hyperlapse folder and input to this pipeline
6. View telemetry overlay and create maps
7. Run fire detection on images (future feature)
"""

import logging
from pathlib import Path
import json
import yaml
from typing import Dict, List, Optional
import argparse
import http.server
import socketserver
import socket
import os
from datetime import datetime

import cv2

# Import modules from src
from src.detection.detector import DetectionTracker, YoloObjectDetector
from src.drone_control.flight_manager import HyperlapseFlightGuide
from src.visualization.telemetry import TelemetrySequence
from src.visualization.map_generator import FireGeolocation, MapGenerator
from src.visualization.viewer import TrajectoryMapGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fire_mapping.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class NoCacheHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Static file handler that disables browser caching for generated reports."""

    def end_headers(self):
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        super().end_headers()


class ReusableTCPServer(socketserver.TCPServer):
    """TCP server that can reuse a recently closed port on Windows."""

    allow_reuse_address = True


def _find_available_port(start_port: int = 8001, max_tries: int = 25) -> int:
    """Return the first available localhost port starting from start_port."""
    for port in range(start_port, start_port + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind(("", port))
                return port
            except OSError:
                continue
    raise OSError(f"No free port found in range {start_port}-{start_port + max_tries - 1}")


def _serve_report_directory(output_dir: str):
    """Serve a generated report directory on the first available local port."""
    os.chdir(output_dir)
    port = _find_available_port(8001)
    handler = NoCacheHTTPRequestHandler
    with ReusableTCPServer(("", port), handler) as httpd:
        url = f"http://localhost:{port}/hyperlapse_viewer.html"
        print(f"\nServing interactive viewer at: {url}")
        try:
            import webbrowser
            webbrowser.open(url)
        except Exception:
            pass
        print("Press Ctrl+C to stop the server")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")


class HyperlapseFireMappingSystem:
    """Main system for analyzing hyperlapse fire mapping flights"""
    
    def __init__(self, config_path: str = 'config/config.yaml'):
        """Initialize the system"""
        self.config = self._load_config(config_path)
        logger.info("Hyperlapse Fire Mapping System initialized")
    
    @staticmethod
    def _load_config(config_path: str) -> dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except FileNotFoundError:
            logger.error(f"Config file not found: {config_path}")
            return {}

    @staticmethod
    def _find_default_detection_model() -> Optional[str]:
        """Return the most recently modified best.pt model if one exists in common training folders."""
        search_roots = [Path('runs'), Path('notebooks') / 'runs']
        candidates: List[Path] = []
        for root in search_roots:
            if not root.exists():
                continue
            candidates.extend(path for path in root.glob('**/weights/best.pt') if path.is_file())

        if not candidates:
            return None

        latest = max(candidates, key=lambda item: item.stat().st_mtime)
        logger.info("Using latest trained YOLO model: %s", latest)
        return str(latest)
    
    def analyze_hyperlapse_folder(
        self,
        hyperlapse_folder: str,
        output_dir: str = 'data/outputs',
        detection_model_path: Optional[str] = None,
        detection_classes: Optional[List[str]] = None,
        detection_confidence: float = 0.35,
    ) -> dict:
        """
        Analyze a hyperlapse image folder
        
        Args:
            hyperlapse_folder: path to folder with hyperlapse images
            output_dir: where to save maps and reports
        
        Returns:
            analysis results dict
        """
        logger.info(f"Analyzing hyperlapse folder: {hyperlapse_folder}")

        resolved_detection_model = detection_model_path or self._find_default_detection_model()
        
        # Verify folder
        folder = Path(hyperlapse_folder)
        if not folder.exists():
            logger.error(f"Folder not found: {hyperlapse_folder}")
            return {}
        
        # Create timestamped output directory
        timestamp = datetime.now().strftime('%m-%d-%Y_%H-%M')
        output_dir = f"{output_dir}/{timestamp}"
        
        # Extract telemetry
        logger.info("Extracting telemetry from images...")
        telem_seq = TelemetrySequence(hyperlapse_folder)
        telemetry = telem_seq.extract_telemetry()
        
        if not telemetry:
            logger.error("No telemetry data extracted")
            return {}
        
        logger.info(f"Successfully extracted telemetry from {len(telemetry)} images")
        
        # Get trajectory info
        bounds = telem_seq.get_bounds()
        logger.info(f"Flight bounds: N={bounds['north']:.6f}, S={bounds['south']:.6f}, "
                   f"E={bounds['east']:.6f}, W={bounds['west']:.6f}")
        logger.info(f"Altitude range: {bounds['min_altitude']:.1f}m - {bounds['max_altitude']:.1f}m")
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Generate maps
        logger.info("Generating trajectory maps...")
        map_gen = TrajectoryMapGenerator(telem_seq, config=self.config)
        detection_bundle = None
        if resolved_detection_model:
            detection_bundle = self._run_object_detection(
                telem_seq=telem_seq,
                output_dir=output_dir,
                model_path=resolved_detection_model,
                detection_classes=detection_classes,
                detection_confidence=detection_confidence,
            )

        map_gen.create_trajectory_map(
            f"{output_dir}/trajectory_map.html",
            detection_points=detection_bundle['geolocations'] if detection_bundle else None,
        )
        map_gen.create_trajectory_overview_image(
            f"{output_dir}/trajectory_overview.png",
            detection_points=detection_bundle['geolocations'] if detection_bundle else None,
        )
        map_gen.create_altitude_profile(f"{output_dir}/altitude_profile.png")
        
        # Generate interactive video viewer
        logger.info("Generating interactive video viewer...")
        html_path = f"{output_dir}/hyperlapse_viewer.html"
        map_gen.create_interactive_video_viewer(
            html_path,
            detections_by_frame=detection_bundle['frame_detections'] if detection_bundle else None,
            map_points=detection_bundle['geolocations'] if detection_bundle else None,
        )
        
        # Generate summary report
        self._generate_report(telem_seq, output_dir, detection_bundle=detection_bundle)
        
        return {
            'hyperlapse_folder': hyperlapse_folder,
            'image_count': len(telem_seq.images),
            'telemetry_count': len(telemetry),
            'bounds': bounds,
            'output_dir': output_dir,
            'detection_model_path': resolved_detection_model,
            'detection_count': detection_bundle['stats']['count'] if detection_bundle else 0,
            'geolocated_detection_count': len(detection_bundle['geolocations']) if detection_bundle else 0,
        }
    
    def get_flight_instructions(self):
        """Get instructions for performing hyperlapse flights"""
        return HyperlapseFlightGuide.get_setup_instructions()

    def _estimate_target_altitude_msl(self, telem_seq: TelemetrySequence) -> Optional[float]:
        geolocation_cfg = self.config.get('geolocation', {})
        configured_altitude = geolocation_cfg.get('default_target_altitude_msl')
        if configured_altitude is not None:
            return float(configured_altitude)

        bounds = telem_seq.get_bounds()
        if not bounds or bounds.get('min_altitude') is None:
            return None

        survey_altitude_m = float(self.config.get('drone', {}).get('survey_altitude_m', 50.0))
        inferred_altitude = float(bounds['min_altitude']) - survey_altitude_m
        logger.info("Using inferred target altitude plane at %.1fm MSL", inferred_altitude)
        return inferred_altitude
    
    def _run_object_detection(
        self,
        telem_seq: TelemetrySequence,
        output_dir: str,
        model_path: str,
        detection_classes: Optional[List[str]],
        detection_confidence: float,
    ) -> Optional[Dict]:
        target_label = ', '.join(detection_classes) if detection_classes else 'all classes'
        logger.info(
            "Running object detection with model %s for classes: %s",
            model_path,
            target_label,
        )
        detector = YoloObjectDetector(
            model_path=model_path,
            confidence_threshold=detection_confidence,
            target_classes=detection_classes,
        )
        if detector.model is None:
            logger.warning("Detection model did not load; continuing without detections")
            return None

        frame_detections: List[List[dict]] = []
        all_detections: List[dict] = []
        for frame_index, image_path in enumerate(telem_seq.images):
            frame = cv2.imread(str(image_path))
            if frame is None:
                frame_detections.append([])
                continue

            frame_height, frame_width = frame.shape[:2]
            raw_detections = detector.detect(frame)
            filtered = detector.filter_detections(raw_detections, min_area_pixels=160)
            enriched: List[dict] = []
            for detection in filtered:
                item = dict(detection)
                item['frame_index'] = frame_index
                item['frame_width'] = frame_width
                item['frame_height'] = frame_height
                item['image_path'] = str(image_path)
                enriched.append(item)
                all_detections.append(item)
            frame_detections.append(enriched)

        tracker = DetectionTracker()
        tracks = tracker.build_tracks(frame_detections)

        geolocator = FireGeolocation(self.config)
        geolocations: List[dict] = []
        triangulated_tracks: List[dict] = []
        for track in tracks:
            observations = []
            for detection in track['detections']:
                observation = self._build_detection_observation(telem_seq, detection)
                if observation:
                    observations.append(observation)

            if not observations:
                continue

            best_observation = max(observations, key=lambda item: float(item.get('confidence', 0.0)))
            geolocation = geolocator.intersect_observation_with_ground_plane(best_observation, ground_up_m=0.0)
            if not geolocation:
                continue

            geolocation['class'] = track['class']
            geolocation['track_id'] = track['track_id']
            geolocation['max_confidence'] = max(item['confidence'] for item in observations)
            geolocation['frame_indices'] = [item['frame_index'] for item in observations]
            geolocation['source_frame_index'] = best_observation['frame_index']
            geolocation['target_plane_up_m'] = 0.0
            geolocations.append(geolocation)

        payload = {
            'model_path': model_path,
            'target_classes': detection_classes or [],
            'projection_mode': 'relative_ground_plane',
            'stats': detector.get_detection_stats(all_detections),
            'frame_detections': frame_detections,
            'tracks': tracks,
            'triangulated_tracks': triangulated_tracks,
            'geolocations': geolocations,
        }

        output_dir_path = Path(output_dir)
        detections_json_path = output_dir_path / 'object_detections.json'
        with open(detections_json_path, 'w', encoding='utf-8') as handle:
            json.dump(payload, handle, indent=2)

        if geolocations:
            map_generator = MapGenerator(self.config)
            map_generator.export_geojson(geolocations, str(output_dir_path / 'car_detections.geojson'))
        self._generate_detection_report(output_dir_path / 'car_detection_report.txt', payload)

        logger.info(
            "Detection complete: %s frame detections, %s mapped points",
            payload['stats']['count'],
            len(geolocations),
        )
        return payload

    def _build_detection_observation(self, telem_seq: TelemetrySequence, detection: Dict) -> Optional[Dict]:
        telemetry = telem_seq.get_telemetry_at_index(detection['frame_index'])
        if not telemetry:
            return None
        gps = telemetry.get('gps') or {}
        if gps.get('latitude') is None or gps.get('longitude') is None or gps.get('altitude') is None:
            return None

        drone = telemetry.get('drone') or {}
        gimbal = telemetry.get('gimbal') or {}

        relative_altitude_m = drone.get('relative_altitude_m')
        if relative_altitude_m is None:
            reference_altitude_m = None
            for item in telem_seq.telemetry:
                gps_item = (item or {}).get('gps') or {}
                if gps_item.get('altitude') is not None:
                    reference_altitude_m = float(gps_item['altitude'])
                    break
            if reference_altitude_m is not None:
                relative_altitude_m = float(gps['altitude']) - reference_altitude_m

        return {
            'frame_index': detection['frame_index'],
            'drone_lat': gps['latitude'],
            'drone_lon': gps['longitude'],
            'drone_altitude_m': gps['altitude'],
            'drone_relative_altitude_m': relative_altitude_m,
            'camera_heading': telemetry.get('camera_heading'),
            'camera_heading_compass': telemetry.get('drone', {}).get('camera_heading_compass') if telemetry.get('drone') else None,
            'drone_yaw': drone.get('yaw'),
            'drone_pitch': drone.get('pitch', 0.0),
            'drone_roll': drone.get('roll', 0.0),
            'gimbal_pitch': gimbal.get('pitch', 0.0),
            'gimbal_roll': gimbal.get('roll', 0.0),
            'gimbal_yaw': gimbal.get('yaw', 0.0),
            'bbox': detection['bbox'],
            'frame_width': detection['frame_width'],
            'frame_height': detection['frame_height'],
            'confidence': detection['confidence'],
            'class': detection['class'],
            'track_id': detection.get('track_id'),
        }

    @staticmethod
    def _generate_detection_report(report_path: Path, detection_bundle: Dict):
        with open(report_path, 'w', encoding='utf-8') as handle:
            handle.write("OBJECT DETECTION REPORT\n")
            handle.write("=" * 50 + "\n\n")
            handle.write(f"Model: {detection_bundle['model_path']}\n")
            target_classes = detection_bundle.get('target_classes') or []
            handle.write(
                f"Target classes: {', '.join(target_classes) if target_classes else 'all classes'}\n"
            )
            handle.write(f"Frame detections: {detection_bundle['stats']['count']}\n")
            handle.write(f"Mapped points: {len(detection_bundle['geolocations'])}\n")
            if detection_bundle.get('projection_mode') == 'relative_ground_plane':
                handle.write("Projection plane: z=0 relative to takeoff altitude\n")
            handle.write("\n")

            if detection_bundle['stats']['classes']:
                handle.write("Class counts:\n")
                for class_name, count in sorted(detection_bundle['stats']['classes'].items()):
                    handle.write(f"  {class_name}: {count}\n")
                handle.write("\n")

            if detection_bundle['geolocations']:
                handle.write("Mapped objects:\n")
                handle.write("-" * 50 + "\n")
                for item in detection_bundle['geolocations']:
                    if item.get('altitude_reference') == 'takeoff_relative':
                        altitude_text = f"Height(rel takeoff) {item.get('relative_altitude_m', item.get('altitude', 0.0)):.1f}m"
                    else:
                        altitude_text = f"Alt {item['altitude']:.1f}m MSL"
                    handle.write(
                        f"Track {item['track_id']} ({item['class']}): "
                        f"Lat {item['latitude']:.6f}, Lon {item['longitude']:.6f}, "
                        f"{altitude_text}, "
                        f"Conf {item.get('max_confidence', item.get('confidence', 0.0)):.2%}, "
                        f"Obs {item['observation_count']}\n"
                    )

        logger.info(f"Detection report saved to {report_path}")

    @staticmethod
    def _generate_report(telem_seq: TelemetrySequence, output_dir: str, detection_bundle: Optional[Dict] = None):
        """Generate text report with flight statistics"""
        report_path = Path(output_dir) / 'flight_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("HYPERLAPSE FLIGHT ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Total Images: {len(telem_seq.images)}\n")
            f.write(f"Images with GPS: {len(telem_seq.telemetry)}\n\n")
            
            bounds = telem_seq.get_bounds()
            if bounds:
                f.write("Flight Bounds:\n")
                f.write(f"  North: {bounds['north']:.6f}\n")
                f.write(f"  South: {bounds['south']:.6f}\n")
                f.write(f"  East: {bounds['east']:.6f}\n")
                f.write(f"  West: {bounds['west']:.6f}\n\n")
                
                f.write("Altitude:\n")
                f.write(f"  Max: {bounds['max_altitude']:.1f}m\n")
                f.write(f"  Min: {bounds['min_altitude']:.1f}m\n\n")

            if detection_bundle:
                f.write("Object Detection Summary:\n")
                f.write(f"  Frame detections: {detection_bundle['stats']['count']}\n")
                f.write(f"  Mapped points: {len(detection_bundle['geolocations'])}\n")
                if detection_bundle.get('projection_mode') == 'relative_ground_plane':
                    f.write("  Projection plane: z=0 relative to takeoff altitude\n")
                if detection_bundle['stats']['classes']:
                    for class_name, count in sorted(detection_bundle['stats']['classes'].items()):
                        f.write(f"  {class_name}: {count}\n")
                f.write("\n")
            
            f.write("Sample Telemetry Points:\n")
            f.write("-" * 50 + "\n")
            for i, telem in enumerate(telem_seq.telemetry[:10]):
                gps = telem.get('gps', {})
                gimbal = telem.get('gimbal', {})
                if gps:
                    line = f"  Image {i}: Lat {gps.get('latitude', 0):.6f}, "
                    line += f"Lon {gps.get('longitude', 0):.6f}, "
                    line += f"Alt {gps.get('altitude', 0):.1f}m"
                    if gimbal:
                        line += f", Gimbal P:{gimbal.get('pitch', 0):.1f}° R:{gimbal.get('roll', 0):.1f}° Y:{gimbal.get('yaw', 0):.1f}°"
                    f.write(line + "\n")
        
        logger.info(f"Report saved to {report_path}")


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description='DJI Mini 4 Pro Hyperlapse Fire Mapping Analysis'
    )
    
    parser.add_argument(
        '--help-flight',
        action='store_true',
        help='Show hyperlapse flight instructions'
    )
    
    parser.add_argument(
        '--analyze',
        type=str,
        metavar='FOLDER',
        help='Analyze hyperlapse image folder'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='data/outputs',
        help='Output directory for maps and reports (default: data/outputs)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Configuration file (default: config/config.yaml)'
    )
    parser.add_argument(
        '--sample',
        action='store_true',
        help='Analyze the built-in sample hyperlapse dataset'
    )
    parser.add_argument(
        '--serve',
        type=str,
        nargs='?',
        const='newest',
        help='Start a local web server to serve the generated HTML viewer. Optionally specify a report folder name to serve a specific report, or use "newest" to serve the most recent.'
    )
    
    parser.add_argument(
        '--serve-latest',
        action='store_true',
        help='Serve the most recently generated report folder (auto-select newest)'
    )
    
    parser.add_argument(
        '--list-reports',
        action='store_true',
        help='List available reports and prompt to choose which one to serve'
    )
    parser.add_argument(
        '--detect-model',
        type=str,
        help='Optional YOLOv8 model path for object detection during analysis. If omitted, the newest weights/best.pt under runs/ or notebooks/runs/ is used when available.'
    )
    parser.add_argument(
        '--detect-class',
        dest='detect_classes',
        action='append',
        help='Object class to keep from the YOLO model. Repeat the flag for multiple classes. If omitted, all classes produced by the model are kept.'
    )
    parser.add_argument(
        '--detect-confidence',
        type=float,
        default=0.35,
        help='Confidence threshold for YOLO detections (default: 0.35)'
    )
    
    args = parser.parse_args()

    if args.serve_latest:
        args.serve = 'newest'
    
    system = HyperlapseFireMappingSystem(args.config)
    
    if args.help_flight:
        print(system.get_flight_instructions())
    
    elif args.analyze or args.sample:
        if args.sample:
            input_folder = 'examples/sample_hyperlapse'
        else:
            input_folder = args.analyze
        results = system.analyze_hyperlapse_folder(
            input_folder,
            args.output,
            detection_model_path=args.detect_model,
            detection_classes=args.detect_classes,
            detection_confidence=args.detect_confidence,
        )
        if results:
            print(f"\n✓ Analysis complete!")
            print(f"  Output folder: {results['output_dir']}")
            print(f"  Maps generated: {results['output_dir']}/trajectory_map.html")
            print(f"  Static map image: {results['output_dir']}/trajectory_overview.png")
            print(f"  Report: {results['output_dir']}/flight_report.txt")
            if results.get('detection_model_path'):
                print(f"  Detection model: {results['detection_model_path']}")
                print(f"  Detected objects: {results['detection_count']}")
                print(f"  Geolocated objects: {results['geolocated_detection_count']}")
            if args.serve:
                _serve_report_directory(results['output_dir'])
    
    elif args.serve is not None or args.list_reports:
        # Find available output directories or files
        output_base = Path(args.output)
        viewer_file = output_base / 'hyperlapse_viewer.html'
        
        if viewer_file.exists():
            # Single output in base dir
            output_dirs = [output_base]
        else:
            # Look for subdirs with viewer
            output_dirs = [d for d in output_base.glob('*/') if (d / 'hyperlapse_viewer.html').exists()]
        
        output_dirs.sort(key=lambda x: x.stat().st_mtime if x.is_dir() else (x / 'hyperlapse_viewer.html').stat().st_mtime, reverse=True)
        
        selected_dir = None
        if args.serve and args.serve != 'newest':
            # Serve specific folder
            specific_dir = output_base / args.serve
            if (specific_dir / 'hyperlapse_viewer.html').exists():
                selected_dir = specific_dir
            else:
                print(f"Report not found: {args.serve}")
                return
        elif args.list_reports:
            print("Available reports:")
            for i, d in enumerate(output_dirs):
                mtime = datetime.fromtimestamp(d.stat().st_mtime if d.is_dir() else (d / 'hyperlapse_viewer.html').stat().st_mtime).strftime('%Y-%m-%d %H:%M')
                print(f"{i+1}. {d.name} (modified: {mtime})")
            try:
                choice = int(input("Enter the number of the report to serve: ")) - 1
                if 0 <= choice < len(output_dirs):
                    selected_dir = output_dirs[choice]
                else:
                    print("Invalid choice.")
                    return
            except ValueError:
                print("Invalid input.")
                return
        else:
            # Serve the newest
            if output_dirs:
                selected_dir = output_dirs[0]
                print(f"Serving newest report: {selected_dir.name}")
            else:
                print(f"No reports found in {output_base}")
                return
        if selected_dir:
            print(f"Serving report: {selected_dir.name}")
            _serve_report_directory(str(selected_dir))
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
