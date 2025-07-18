#!/usr/bin/env python3
"""
Command-line interface for the Unified Raster Fuzzy Inference System.
Processes 3 input TIFF files (social, environmental, strategic) and outputs a single TIFF.
Supports both sequential and parallel processing with automatic fallback.
"""

import argparse
import sys
import os
import multiprocessing as mp
from pathlib import Path
from raster_fuzzy_lib import UnifiedRasterFuzzyInferenceSystem, create_raster_config_template
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
        description="Unified Raster Fuzzy Inference System - Process 3 TIFF files into 1 output TIFF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create a configuration template
  python raster_fuzzy_cli.py --create-config

  # Sequential processing (default)
  python raster_fuzzy_cli.py social.tif environmental.tif strategic.tif output.tif

  # Parallel processing with all available cores
  python raster_fuzzy_cli.py social.tif environmental.tif strategic.tif output.tif --parallel

  # Parallel processing with custom number of CPU cores
  python raster_fuzzy_cli.py social.tif environmental.tif strategic.tif output.tif --parallel --cores 8

  # Parallel processing with custom chunk size and cores
  python raster_fuzzy_cli.py social.tif environmental.tif strategic.tif output.tif --parallel --cores 4 --chunk-size 200

  # Sequential processing with custom configuration and NoData value
  python raster_fuzzy_cli.py social.tif environmental.tif strategic.tif output.tif --config my_config.json --nodata -32768
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
    parser.add_argument('--nodata', type=float, default=5.0,
                       help='NoData value for output raster (default: 5.0)')
    parser.add_argument('--parallel', '-p', action='store_true',
                       help='Enable parallel processing (default: sequential)')
    parser.add_argument('--cores', type=int, default=None,
                       help=f'Number of CPU cores to use for parallel processing (default: all available, max: {mp.cpu_count()})')
    parser.add_argument('--chunk-size', type=int, default=100,
                       help='Number of rows per chunk for parallel processing (default: 100)')
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
    
    # Validate parallel processing parameters
    if args.parallel:
        if args.cores is not None:
            if args.cores <= 0:
                print(f"Error: Number of cores must be positive, got: {args.cores}")
                sys.exit(1)
            if args.cores > mp.cpu_count():
                print(f"Warning: Requested {args.cores} cores but only {mp.cpu_count()} available. Using {mp.cpu_count()} cores.")
                args.cores = mp.cpu_count()
        
        if args.chunk_size <= 0:
            print(f"Error: Chunk size must be positive, got: {args.chunk_size}")
            sys.exit(1)
    
    # Check if output directory exists
    output_dir = os.path.dirname(args.output_tiff)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        # Initialize the unified fuzzy inference system
        print(f"Loading configuration from: {args.config}")
        fis = UnifiedRasterFuzzyInferenceSystem(args.config)
        
        # Process the rasters
        print(f"Processing rasters...")
        print(f"  Social: {args.social_tiff}")
        print(f"  Environmental: {args.environmental_tiff}")
        print(f"  Strategic: {args.strategic_tiff}")
        print(f"  Output: {args.output_tiff}")
        print(f"  NoData value: {args.nodata}")
        if args.parallel:
            print(f"  Processing mode: Parallel")
            print(f"  CPU cores: {args.cores or 'all available'}")
            print(f"  Chunk size: {args.chunk_size} rows")
        else:
            print(f"  Processing mode: Sequential")
        
        fis.process_rasters(
            social_tiff=args.social_tiff,
            environmental_tiff=args.environmental_tiff,
            strategic_tiff=args.strategic_tiff,
            output_tiff=args.output_tiff,
            nodata_value=args.nodata,
            parallel=args.parallel,
            num_cores=args.cores,
            chunk_size=args.chunk_size
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