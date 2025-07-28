#!/usr/bin/env python3
"""
AWS Glue-compatible Raster Fuzzy Inference System (Simplified Local Version)
Optimized for AWS Glue with S3 input/output support.
This version can run locally without Spark for testing.
"""

import os
import sys
import json
import time
import logging
import numpy as np
import rasterio
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from datetime import datetime
import argparse
import multiprocessing as mp
from functools import partial

# Global fuzzy system cache to avoid recreation
_FUZZY_SYSTEM_CACHE = {}

def create_fuzzy_system_from_config(config: Dict):
    """Create fuzzy system with caching for performance."""
    # Use config hash as cache key
    config_hash = hash(json.dumps(config, sort_keys=True))
    
    if config_hash in _FUZZY_SYSTEM_CACHE:
        return _FUZZY_SYSTEM_CACHE[config_hash]
    
    import skfuzzy as fuzz
    from skfuzzy import control as ctrl
    
    input_variables = {}
    for var_name, var_config in config['input_variables'].items():
        resolution = min(50, var_config.get('resolution', 50))
        universe = np.linspace(var_config['min'], var_config['max'], resolution)
        input_variables[var_name] = ctrl.Antecedent(universe, var_name)
        
        for mf_name, mf_config in var_config['membership_functions'].items():
            if mf_config['type'] == 'trapmf':
                input_variables[var_name][mf_name] = fuzz.trapmf(
                    input_variables[var_name].universe, mf_config['params'])
            elif mf_config['type'] == 'trimf':
                input_variables[var_name][mf_name] = fuzz.trimf(
                    input_variables[var_name].universe, mf_config['params'])
    
    output_config = config['output_variable']
    resolution = min(50, output_config.get('resolution', 50))
    universe = np.linspace(output_config['min'], output_config['max'], resolution)
    output_variable = ctrl.Consequent(universe, output_config['name'])
    
    for mf_name, mf_config in output_config['membership_functions'].items():
        if mf_config['type'] == 'trapmf':
            output_variable[mf_name] = fuzz.trapmf(
                output_variable.universe, mf_config['params'])
        elif mf_config['type'] == 'trimf':
            output_variable[mf_name] = fuzz.trimf(
                output_variable.universe, mf_config['params'])
    
    rules = []
    for rule_config in config['rules']:
        antecedent = rule_config['antecedent']
        consequent = rule_config['consequent']
        
        antecedent_conditions = []
        for condition in antecedent:
            var_name = condition['variable']
            membership = condition['membership']
            antecedent_conditions.append(input_variables[var_name][membership])
        
        if len(antecedent_conditions) == 1:
            rule = ctrl.Rule(antecedent_conditions[0], output_variable[consequent])
        else:
            combined_antecedent = antecedent_conditions[0]
            for condition in antecedent_conditions[1:]:
                combined_antecedent = combined_antecedent & condition
            rule = ctrl.Rule(combined_antecedent, output_variable[consequent])
        
        rules.append(rule)
    
    control_system = ctrl.ControlSystem(rules)
    simulation = ctrl.ControlSystemSimulation(control_system)
    
    # Cache the result
    result = (simulation, list(config['input_variables'].keys()), output_config['name'])
    _FUZZY_SYSTEM_CACHE[config_hash] = result
    
    return result

def process_block_simple(chunk_data):
    """
    Simple block processing without Spark dependencies.
    """
    import skfuzzy as fuzz
    from skfuzzy import control as ctrl
    import numpy as np
    
    config, social_block, env_block, strat_block, block_start_row, nodata_value = chunk_data
    
    try:
        # Create fuzzy system
        simulation, input_vars, output_var = create_fuzzy_system_from_config(config)
        
        # Process the block
        rows, cols = social_block.shape
        result_block = np.full((rows, cols), nodata_value, dtype=np.float32)
        
        for i in range(rows):
            for j in range(cols):
                # Skip nodata values
                if (social_block[i, j] == nodata_value or 
                    env_block[i, j] == nodata_value or 
                    strat_block[i, j] == nodata_value):
                    continue
                
                # Set input values - match the order in config: environmental, social, strategic
                simulation.input[input_vars[0]] = float(env_block[i, j])      # environmental
                simulation.input[input_vars[1]] = float(social_block[i, j])   # social
                simulation.input[input_vars[2]] = float(strat_block[i, j])    # strategic
                
                # Compute
                simulation.compute()
                
                # Get result
                result_block[i, j] = float(simulation.output[output_var])
        
        return (block_start_row, result_block)
        
    except Exception as e:
        print(f"Error processing block starting at row {block_start_row}: {e}")
        return (block_start_row, np.full((rows, cols), nodata_value, dtype=np.float32))

def process_block_wrapper_global(args):
    """
    Global wrapper function for multiprocessing.
    """
    block_info, social_tiff, environmental_tiff, strategic_tiff, config, nodata_value = args
    start_row, end_row = block_info
    
    # Read raster blocks
    with rasterio.open(social_tiff) as social_src:
        social_block = social_src.read(1, window=((start_row, end_row), (0, social_src.width)))
    
    with rasterio.open(environmental_tiff) as env_src:
        env_block = env_src.read(1, window=((start_row, end_row), (0, env_src.width)))
    
    with rasterio.open(strategic_tiff) as strat_src:
        strat_block = strat_src.read(1, window=((start_row, end_row), (0, strat_src.width)))
    
    # Process block
    chunk_data = (config, social_block, env_block, strat_block, start_row, nodata_value)
    return process_block_simple(chunk_data)

def process_rasters_simple(social_tiff, environmental_tiff, strategic_tiff, output_tiff,
                          config_file, nodata_value=5.0, block_size=1000, num_workers=None):
    """
    Process rasters using multiprocessing (local version).
    """
    print(f"=== Simple Raster Processing (Local) ===")
    print(f"Input files:")
    print(f"  Social: {social_tiff}")
    print(f"  Environmental: {environmental_tiff}")
    print(f"  Strategic: {strategic_tiff}")
    print(f"Output: {output_tiff}")
    print(f"Config: {config_file}")
    print(f"Block size: {block_size} rows")
    print(f"Workers: {num_workers or 'auto'}")
    
    start_time = time.time()
    
    try:
        # Load configuration
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Read raster metadata
        with rasterio.open(social_tiff) as src:
            height = src.height
            width = src.width
            transform = src.transform
            crs = src.crs
            nodata = src.nodata
        
        print(f"Raster dimensions: {width}x{height}")
        print(f"Processing in blocks of {block_size} rows...")
        
        # Create blocks
        blocks = []
        for start_row in range(0, height, block_size):
            end_row = min(start_row + block_size, height)
            blocks.append((start_row, end_row))
        
        print(f"Created {len(blocks)} blocks")
        
        # Prepare arguments for multiprocessing
        mp_args = [(block, social_tiff, environmental_tiff, strategic_tiff, config, nodata_value) 
                   for block in blocks]
        
        # Process all blocks
        if num_workers and num_workers > 1:
            with mp.Pool(processes=num_workers) as pool:
                results = pool.map(process_block_wrapper_global, mp_args)
        else:
            # Single-threaded processing
            results = [process_block_wrapper_global(args) for args in mp_args]
        
        # Combine results
        print("Combining results...")
        output_raster = np.full((height, width), nodata_value, dtype=np.float32)
        
        for start_row, result_block in results:
            end_row = start_row + result_block.shape[0]
            output_raster[start_row:end_row, :] = result_block
        
        # Write output
        print(f"Writing output to {output_tiff}...")
        with rasterio.open(
            output_tiff, 'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=np.float32,
            crs=crs,
            transform=transform,
            nodata=nodata_value
        ) as dst:
            dst.write(output_raster, 1)
        
        elapsed_time = time.time() - start_time
        print(f"✓ Processing completed in {elapsed_time:.2f} seconds")
        print(f"✓ Output saved to: {output_tiff}")
        
        # Get file size
        file_size = os.path.getsize(output_tiff) / (1024 * 1024)  # MB
        print(f"✓ Output file size: {file_size:.1f} MB")
        
    except Exception as e:
        print(f"✗ Processing failed: {e}")
        import traceback
        traceback.print_exc()
        raise

def download_from_s3(s3_path, local_path):
    """
    Download file from S3 to local path.
    """
    if s3_path.startswith('s3://'):
        try:
            import boto3
            s3 = boto3.client('s3')
            
            # Parse S3 path
            bucket = s3_path.split('/')[2]
            key = '/'.join(s3_path.split('/')[3:])
            
            # Create directory if needed
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            # Download file
            s3.download_file(bucket, key, local_path)
            print(f"✓ Downloaded {s3_path} to {local_path}")
            return True
            
        except Exception as e:
            print(f"✗ Failed to download from S3: {e}")
            return False
    else:
        # Local file, just copy
        import shutil
        shutil.copy2(s3_path, local_path)
        return True

def upload_to_s3(local_path, s3_path):
    """
    Upload file from local path to S3.
    """
    if s3_path.startswith('s3://'):
        try:
            import boto3
            s3 = boto3.client('s3')
            
            # Parse S3 path
            bucket = s3_path.split('/')[2]
            key = '/'.join(s3_path.split('/')[3:])
            
            # Upload file
            s3.upload_file(local_path, bucket, key)
            print(f"✓ Uploaded {local_path} to {s3_path}")
            return True
            
        except Exception as e:
            print(f"✗ Failed to upload to S3: {e}")
            return False
    else:
        # Local file, just copy
        import shutil
        shutil.copy2(local_path, s3_path)
        return True

def main():
    """Main function for simple local processing."""
    parser = argparse.ArgumentParser(
        description="Simple Local Raster Fuzzy Inference System (Glue-compatible)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Local testing with local paths
  python raster_fuzzy_glue_simple.py social.tif env.tif strat.tif output.tif --config config.json

  # Local testing with S3 paths
  python raster_fuzzy_glue_simple.py s3://bucket/input/social.tif s3://bucket/input/env.tif s3://bucket/input/strat.tif s3://bucket/output/result.tif --config config.json

  # AWS Glue deployment (no arguments needed - uses environment variables)
  python raster_fuzzy_glue_simple.py
        """
    )
    
    parser.add_argument('social_tiff', nargs='?', help='Path to social factor TIFF file (S3 or local)')
    parser.add_argument('environmental_tiff', nargs='?', help='Path to environmental factor TIFF file (S3 or local)')
    parser.add_argument('strategic_tiff', nargs='?', help='Path to strategic factor TIFF file (S3 or local)')
    parser.add_argument('output_tiff', nargs='?', help='Path for output TIFF file (S3 or local)')
    
    parser.add_argument('--config', '-c', default='raster_fis_config.json',
                       help='Path to configuration JSON file (default: raster_fis_config.json)')
    parser.add_argument('--nodata', type=float, default=5.0,
                       help='NoData value for output raster (default: 5.0)')
    parser.add_argument('--block-size', type=int, default=1000,
                       help='Number of rows per block (default: 1000)')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of worker processes (default: auto)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Check if running in AWS Glue (no arguments provided)
    if not args.social_tiff and os.environ.get('AWS_EXECUTION_ENV'):
        # AWS Glue mode - use environment variables
        social_tiff = os.environ.get('SOCIAL_TIFF')
        environmental_tiff = os.environ.get('ENVIRONMENTAL_TIFF')
        strategic_tiff = os.environ.get('STRATEGIC_TIFF')
        output_tiff = os.environ.get('OUTPUT_TIFF')
        config_file = os.environ.get('CONFIG_FILE', 'raster_fis_config.json')
        
        if not all([social_tiff, environmental_tiff, strategic_tiff, output_tiff]):
            print("Error: Missing required environment variables in AWS Glue mode")
            print("Required: SOCIAL_TIFF, ENVIRONMENTAL_TIFF, STRATEGIC_TIFF, OUTPUT_TIFF")
            sys.exit(1)
    else:
        # Local mode or explicit arguments
        if not args.social_tiff:
            print("Error: Input files required for local mode")
            parser.print_help()
            sys.exit(1)
        
        social_tiff = args.social_tiff
        environmental_tiff = args.environmental_tiff
        strategic_tiff = args.strategic_tiff
        output_tiff = args.output_tiff
        config_file = args.config
    
    # Handle S3 paths for local testing
    temp_dir = "/tmp/raster_fuzzy_glue_simple"
    os.makedirs(temp_dir, exist_ok=True)
    
    local_social = os.path.join(temp_dir, "social.tif")
    local_env = os.path.join(temp_dir, "environmental.tif")
    local_strat = os.path.join(temp_dir, "strategic.tif")
    local_output = os.path.join(temp_dir, "output.tif")
    local_config = os.path.join(temp_dir, "config.json")
    
    # Download input files if they're S3 paths
    if social_tiff.startswith('s3://'):
        if not download_from_s3(social_tiff, local_social):
            sys.exit(1)
        social_tiff = local_social
    
    if environmental_tiff.startswith('s3://'):
        if not download_from_s3(environmental_tiff, local_env):
            sys.exit(1)
        environmental_tiff = local_env
    
    if strategic_tiff.startswith('s3://'):
        if not download_from_s3(strategic_tiff, local_strat):
            sys.exit(1)
        strategic_tiff = local_strat
    
    if config_file.startswith('s3://'):
        if not download_from_s3(config_file, local_config):
            sys.exit(1)
        config_file = local_config
    
    # Validate inputs
    for tiff_file in [social_tiff, environmental_tiff, strategic_tiff]:
        if not os.path.exists(tiff_file):
            print(f"Error: Input file not found: {tiff_file}")
            sys.exit(1)
    
    if not os.path.exists(config_file):
        print(f"Error: Configuration file not found: {config_file}")
        sys.exit(1)
    
    try:
        # Process rasters
        print(f"Processing rasters with simple local processing...")
        process_rasters_simple(
            social_tiff, environmental_tiff, strategic_tiff, local_output,
            config_file, args.nodata, args.block_size, args.workers
        )
        
        # Upload output if it's an S3 path
        if args.output_tiff and args.output_tiff.startswith('s3://'):
            if upload_to_s3(local_output, args.output_tiff):
                print(f"✓ Final output uploaded to: {args.output_tiff}")
            else:
                print(f"⚠ Output saved locally at: {local_output}")
        elif args.output_tiff and not args.output_tiff.startswith('s3://'):
            # Copy to local output path
            import shutil
            output_dir = os.path.dirname(args.output_tiff)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            shutil.copy2(local_output, args.output_tiff)
            print(f"✓ Output saved to: {args.output_tiff}")
        
        print("✓ Processing completed successfully!")
        
    except Exception as e:
        print(f"✗ Processing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 