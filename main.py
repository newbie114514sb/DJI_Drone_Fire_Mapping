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
import yaml
from typing import Optional
import argparse
import http.server
import socketserver
import os
from datetime import datetime

# Import modules from src
from src.drone_control.flight_manager import HyperlapseFlightGuide
from src.visualization.telemetry import TelemetrySequence, ExifTelemetryExtractor
from src.visualization.viewer import HyperlapseViewer, TrajectoryMapGenerator

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


class HyperlapseFiringMappingSystem:
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
    
    def analyze_hyperlapse_folder(self, hyperlapse_folder: str, output_dir: str = 'data/outputs') -> dict:
        """
        Analyze a hyperlapse image folder
        
        Args:
            hyperlapse_folder: path to folder with hyperlapse images
            output_dir: where to save maps and reports
        
        Returns:
            analysis results dict
        """
        logger.info(f"Analyzing hyperlapse folder: {hyperlapse_folder}")
        
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
        map_gen = TrajectoryMapGenerator(telem_seq)
        map_gen.create_trajectory_map(f"{output_dir}/trajectory_map.html")
        map_gen.create_altitude_profile(f"{output_dir}/altitude_profile.png")
        
        # Generate interactive video viewer
        logger.info("Generating interactive video viewer...")
        html_path = f"{output_dir}/hyperlapse_viewer.html"
        map_gen.create_interactive_video_viewer(html_path)
        
        # Generate summary report
        self._generate_report(telem_seq, output_dir)
        
        return {
            'hyperlapse_folder': hyperlapse_folder,
            'image_count': len(telem_seq.images),
            'telemetry_count': len(telemetry),
            'bounds': bounds,
            'output_dir': output_dir,
        }
    
    def get_flight_instructions(self):
        """Get instructions for performing hyperlapse flights"""
        return HyperlapseFlightGuide.get_setup_instructions()
        viewer.run()
        return True
    
    @staticmethod
    def _generate_report(telem_seq: TelemetrySequence, output_dir: str):
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
            
            f.write("Sample Telemetry Points:\n")
            f.write("-" * 50 + "\n")
            for i, telem in enumerate(telem_seq.telemetry[:10]):
                gps = telem.get('gps', {})
                if gps:
                    f.write(f"  Image {i}: Lat {gps.get('latitude', 0):.6f}, "
                           f"Lon {gps.get('longitude', 0):.6f}, "
                           f"Alt {gps.get('altitude', 0):.1f}m\n")
        
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
        action='store_true',
        help='Start a local web server to serve the generated HTML viewer'
    )
    
    args = parser.parse_args()
    
    system = HyperlapseFiringMappingSystem(args.config)
    
    if args.help_flight:
        print(system.get_flight_instructions())
    
    elif args.analyze or args.sample:
        if args.sample:
            input_folder = 'examples/sample_hyperlapse'
        else:
            input_folder = args.analyze
        results = system.analyze_hyperlapse_folder(input_folder, args.output)
        if results:
            print(f"\n✓ Analysis complete!")
            print(f"  Output folder: {results['output_dir']}")
            print(f"  Maps generated: {results['output_dir']}/trajectory_map.html")
            print(f"  Report: {results['output_dir']}/flight_report.txt")
            if args.serve:
                # Start local web server
                output_dir = results['output_dir']
                os.chdir(output_dir)
                port = 8001
                handler = http.server.SimpleHTTPRequestHandler
                with socketserver.TCPServer(("", port), handler) as httpd:
                    url = f"http://localhost:{port}/hyperlapse_viewer.html"
                    print(f"\n🌐 Serving interactive viewer at: {url}")
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
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()

