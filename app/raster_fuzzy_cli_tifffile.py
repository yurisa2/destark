#!/usr/bin/env python3
"""
Command-line interface for the tifffile-based raster fuzzy inference system.
Processes 3 input TIFF files (social, environmental, strategic) and outputs a single TIFF.
"""

import argparse
import sys
import os
from pathlib import Path

# Add the app directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from raster_fuzzy_lib_tifffile import UnifiedRasterFuzzyInferenceSystem, create_raster_config_template


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description='Raster Fuzzy Inference System using tifffile (GDAL-free)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process rasters with custom config
  python raster_fuzzy_cli_tifffile.py \\
    --social input/social.tif \\
    --environmental input/environmental.tif \\
    --strategic input/strategic.tif \\
    --output output/result.tif \\
    --config config.json

  # Process rasters with parallel processing
  python raster_fuzzy_cli_tifffile.py \\
    --social input/social.tif \\
    --environmental input/environmental.tif \\
    --strategic input/strategic.tif \\
    --output output/result.tif \\
    --config config.json \\
    --parallel \\
    --cores 4

  # Create a template configuration file
  python raster_fuzzy_cli_tifffile.py --create-config template_config.json
        """
    )
    
    # Input/output arguments
    parser.add_argument('--social', 
                       help='Path to social factor TIFF file')
    parser.add_argument('--environmental', 
                       help='Path to environmental factor TIFF file')
    parser.add_argument('--strategic', 
                       help='Path to strategic factor TIFF file')
    parser.add_argument('--output', 
                       help='Path for output TIFF file')
    parser.add_argument('--config', 
                       help='Path to configuration JSON file')
    
    # Processing options
    parser.add_argument('--parallel', 
                       action='store_true',
                       help='Use parallel processing')
    parser.add_argument('--cores', 
                       type=int,
                       help='Number of CPU cores to use (default: auto-detect)')
    parser.add_argument('--chunk-size', 
                       type=int, 
                       default=100,
                       help='Chunk size for parallel processing (default: 100)')
    parser.add_argument('--nodata', 
                       type=float, 
                       default=5.0,
                       help='NoData value (default: 5.0)')
    
    # Utility options
    parser.add_argument('--create-config', 
                       help='Create a template configuration file at the specified path')
    parser.add_argument('--verbose', '-v',
                       action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Handle configuration template creation
    if args.create_config:
        print(f"Creating template configuration file: {args.create_config}")
        config = create_raster_config_template()
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(args.create_config), exist_ok=True)
        
        # Write configuration file
        import json
        with open(args.create_config, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✓ Template configuration created: {args.create_config}")
        print("You can now edit this file to customize your fuzzy logic rules.")
        return
    
    # Validate required arguments for processing
    if not all([args.social, args.environmental, args.strategic, args.output, args.config]):
        parser.error("For processing, all of --social, --environmental, --strategic, --output, and --config are required")
    
    # Validate input files exist
    for input_file in [args.social, args.environmental, args.strategic]:
        if not os.path.exists(input_file):
            parser.error(f"Input file does not exist: {input_file}")
    
    # Validate config file exists
    if not os.path.exists(args.config):
        parser.error(f"Configuration file does not exist: {args.config}")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Print processing information
    print("=" * 60)
    print("Raster Fuzzy Inference System (tifffile version)")
    print("=" * 60)
    print(f"Input files:")
    print(f"  Social: {args.social}")
    print(f"  Environmental: {args.environmental}")
    print(f"  Strategic: {args.strategic}")
    print(f"Output file: {args.output}")
    print(f"Configuration: {args.config}")
    print(f"Processing mode: {'Parallel' if args.parallel else 'Sequential'}")
    if args.parallel and args.cores:
        print(f"CPU cores: {args.cores}")
    print(f"Chunk size: {args.chunk_size}")
    print(f"NoData value: {args.nodata}")
    print("=" * 60)
    
    try:
        # Create FIS instance
        print("Loading configuration and creating fuzzy inference system...")
        fis = UnifiedRasterFuzzyInferenceSystem(args.config)
        
        # Process rasters
        print("Starting raster processing...")
        fis.process_rasters(
            social_tiff=args.social,
            environmental_tiff=args.environmental,
            strategic_tiff=args.strategic,
            output_tiff=args.output,
            nodata_value=args.nodata,
            parallel=args.parallel,
            num_cores=args.cores,
            chunk_size=args.chunk_size
        )
        
        print("=" * 60)
        print("✓ Processing completed successfully!")
        print(f"Output saved to: {args.output}")
        print("=" * 60)
        
    except Exception as e:
        print(f"✗ Processing failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 