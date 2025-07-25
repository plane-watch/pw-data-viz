#!/usr/bin/env python3
"""
Aircraft Path Visualizer CLI
Command-line interface for creating aircraft flight path visualizations.
"""

import argparse
import sys
import os
from typing import List

from aircraft_viz import AircraftPathVisualizer


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Create aircraft path visualizations from ClickHouse CSV data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s data.csv.gz
  %(prog)s data.csv.gz --format png svg pdf --dpi 600
  %(prog)s data.csv.gz --high-res --size large --output flight_paths_hd
  %(prog)s data.csv.gz --format svg --size large
        """
    )
    
    # Required arguments
    parser.add_argument(
        'input_file',
        help='Input .csv.gz file containing CSV data'
    )
    
    # Output options
    parser.add_argument(
        '--output', '-o',
        default='aircraft_paths',
        help='Base filename for output files (default: aircraft_paths)'
    )
    
    parser.add_argument(
        '--format', '-f',
        nargs='+',
        choices=['png', 'svg', 'pdf'],
        default=['png'],
        help='Output format(s) (default: png)'
    )
    
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='DPI for raster formats (default: 300)'
    )
    
    parser.add_argument(
        '--high-res',
        action='store_true',
        help='Enable high resolution output (minimum 600 DPI)'
    )
    
    parser.add_argument(
        '--size',
        choices=['normal', 'large'],
        default='normal',
        help='Figure size (default: normal)'
    )
    
    # Visualization options
    parser.add_argument(
        '--color-scheme',
        choices=['altitude', 'density', 'speed'],
        default='altitude',
        help='Color scheme for paths (default: altitude)'
    )
    
    parser.add_argument(
        '--line-width',
        type=float,
        default=0.8,
        help='Line width for paths (default: 0.8)'
    )
    
    parser.add_argument(
        '--no-cartopy',
        action='store_true',
        help='Disable cartopy map features (use basic matplotlib)'
    )
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_arguments()
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found")
        sys.exit(1)
    
    # Create configuration
    config = {
        'color_scheme': args.color_scheme,
        'line_width': args.line_width,
        'use_cartopy': not args.no_cartopy
    }
    
    try:
        # Create visualizer instance
        visualizer = AircraftPathVisualizer(config=config)
        
        # Generate visualization
        saved_files = visualizer.generate_visualization(
            csv_gz_path=args.input_file,
            output_base=args.output,
            formats=args.format,
            dpi=args.dpi,
            size=args.size,
            high_res=args.high_res
        )
        
        print(f"\n🎉 Success! Created {len(saved_files)} output file(s)")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
