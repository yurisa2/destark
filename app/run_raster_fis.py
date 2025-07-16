#!/usr/bin/env python3
"""
Command-line interface for the Raster Fuzzy Inference System.
Processes 3 input TIFF files (social, environmental, strategic) and outputs a single TIFF.
"""

import argparse
import sys
import os
from pathlib import Path
from raster_fuzzy_system import RasterFuzzyInferenceSystem, create_raster_config_template
import json


def create_config_file(config_path: str):
    """Create a template configuration file."""
    config = create_raster_config_template()
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Created configuration template: {config_path}")
    print("Edit this file to customize your fuzzy logic rules and membership functions.")


def main():
    parser = argparse.ArgumentParser(
        description="Raster Fuzzy Inference System - Process 3 TIFF files into 1 output TIFF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create a configuration template
  python run_raster_fis.py --create-config

  # Process TIFF files with default configuration
  python run_raster_fis.py social.tif environmental.tif strategic.tif output.tif

  # Process TIFF files with custom configuration
  python run_raster_fis.py social.tif environmental.tif strategic.tif output.tif --config my_config.json

  # Process TIFF files with custom NoData value
  python run_raster_fis.py social.tif environmental.tif strategic.tif output.tif --nodata -32768
        """
    )
    
    parser.add_argument('social_tiff', nargs='?', help='Path to social factor TIFF file')
    parser.add_argument('environmental_tiff', nargs='?', help='Path to environmental factor TIFF file')
    parser.add_argument('strategic_tiff', nargs='?', help='Path to strategic factor TIFF file')
    parser.add_argument('output_tiff', nargs='?', help='Path for output TIFF file')
    
    parser.add_argument('--config', '-c', default='raster_fis_config.json',
                       help='Path to configuration JSON file (default: raster_fis_config.json)')
    parser.add_argument('--create-config', action='store_true',
                       help='Create a template configuration file')
    parser.add_argument('--nodata', type=float, default=-9999.0,
                       help='NoData value for output raster (default: -9999.0)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Handle create-config option
    if args.create_config:
        create_config_file(args.config)
        return
    
    # Check if all required arguments are provided
    if not all([args.social_tiff, args.environmental_tiff, args.strategic_tiff, args.output_tiff]):
        parser.print_help()
        print("\nError: All four TIFF file paths are required.")
        print("Use --create-config to create a configuration template first.")
        sys.exit(1)
    
    # Check if input files exist
    for tiff_file in [args.social_tiff, args.environmental_tiff, args.strategic_tiff]:
        if not os.path.exists(tiff_file):
            print(f"Error: Input file not found: {tiff_file}")
            sys.exit(1)
    
    # Check if config file exists, create if not
    if not os.path.exists(args.config):
        print(f"Configuration file {args.config} not found. Creating template...")
        create_config_file(args.config)
        print("Please edit the configuration file and run again.")
        sys.exit(1)
    
    # Check if output directory exists
    output_dir = os.path.dirname(args.output_tiff)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        # Initialize the fuzzy inference system
        print(f"Loading configuration from: {args.config}")
        fis = RasterFuzzyInferenceSystem(args.config)
        
        # Process the rasters
        print(f"Processing rasters...")
        print(f"  Social: {args.social_tiff}")
        print(f"  Environmental: {args.environmental_tiff}")
        print(f"  Strategic: {args.strategic_tiff}")
        print(f"  Output: {args.output_tiff}")
        print(f"  NoData value: {args.nodata}")
        
        fis.process_rasters(
            social_tiff=args.social_tiff,
            environmental_tiff=args.environmental_tiff,
            strategic_tiff=args.strategic_tiff,
            output_tiff=args.output_tiff,
            nodata_value=args.nodata
        )
        
        print("Processing completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 