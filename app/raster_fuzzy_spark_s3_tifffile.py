#!/usr/bin/env python3
"""
S3-enabled Spark-based Raster Fuzzy Inference System using tifffile
Compatible with EMR 7.9.0 and S3 file storage
"""

import os
import sys
import json
import time
import logging
import numpy as np
import tifffile
import boto3
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from datetime import datetime
import psutil

# Configure boto3 for S3 access
boto3.setup_default_session(region_name='us-east-1')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Global fuzzy system cache
_FUZZY_SYSTEM_CACHE = {}

def read_tiff_from_s3(s3_path: str) -> Tuple[np.ndarray, Dict]:
    """Read TIFF file from S3 using tifffile."""
    try:
        # Parse S3 path
        if s3_path.startswith('s3://'):
            bucket = s3_path.split('/')[2]
            key = '/'.join(s3_path.split('/')[3:])
        else:
            raise ValueError(f"Invalid S3 path: {s3_path}")
        
        # Download to temporary file
        s3_client = boto3.client('s3')
        temp_file = f"/tmp/{os.path.basename(s3_path)}"
        
        print(f"Downloading {s3_path} to {temp_file}")
        s3_client.download_file(bucket, key, temp_file)
        
        # Read with tifffile
        data = tifffile.imread(temp_file)
        
        # Get metadata
        metadata = {
            'shape': data.shape,
            'dtype': str(data.dtype),
            'min': float(data.min()),
            'max': float(data.max())
        }
        
        # Clean up temp file
        os.remove(temp_file)
        
        return data, metadata
        
    except Exception as e:
        print(f"Error reading {s3_path}: {e}")
        raise

def write_tiff_to_s3(data: np.ndarray, s3_path: str, metadata: Dict = None) -> None:
    """Write TIFF file to S3 using tifffile."""
    try:
        # Parse S3 path
        if s3_path.startswith('s3://'):
            bucket = s3_path.split('/')[2]
            key = '/'.join(s3_path.split('/')[3:])
        else:
            raise ValueError(f"Invalid S3 path: {s3_path}")
        
        # Write to temporary file
        temp_file = f"/tmp/{os.path.basename(s3_path)}"
        
        print(f"Writing data to {temp_file}")
        tifffile.imwrite(temp_file, data)
        
        # Upload to S3
        s3_client = boto3.client('s3')
        print(f"Uploading {temp_file} to {s3_path}")
        s3_client.upload_file(temp_file, bucket, key)
        
        # Clean up temp file
        os.remove(temp_file)
        
    except Exception as e:
        print(f"Error writing {s3_path}: {e}")
        raise

def create_fuzzy_system_from_config(config: Dict):
    """Create fuzzy system with caching for performance."""
    config_hash = hash(json.dumps(config, sort_keys=True))
    
    if config_hash in _FUZZY_SYSTEM_CACHE:
        return _FUZZY_SYSTEM_CACHE[config_hash]
    
    import skfuzzy as fuzz
    from skfuzzy import control as ctrl
    
    input_variables = {}
    for var_name, var_config in config['input_variables'].items():
        universe = np.arange(var_config['min'], var_config['max'] + 1, var_config['step'])
        input_variables[var_name] = ctrl.Antecedent(universe, var_name)
        
        for mf_name, mf_config in var_config['membership_functions'].items():
            if mf_config['type'] == 'trapmf':
                input_variables[var_name][mf_name] = fuzz.trapmf(
                    input_variables[var_name].universe, mf_config['params'])
            elif mf_config['type'] == 'trimf':
                input_variables[var_name][mf_name] = fuzz.trimf(
                    input_variables[var_name].universe, mf_config['params'])
    
    output_config = config['output_variable']
    universe = np.arange(output_config['min'], output_config['max'] + 1, output_config['step'])
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
    
    result = (simulation, list(config['input_variables'].keys()), output_config['name'])
    _FUZZY_SYSTEM_CACHE[config_hash] = result
    
    return result

def process_block_s3_tifffile(chunk_data):
    """Process a block of raster data using tifffile and S3."""
    import skfuzzy as fuzz
    
    try:
        # Unpack chunk data
        block_id, social_block, env_block, strat_block, config = chunk_data
        
        # Create fuzzy system
        simulation, input_vars, output_var = create_fuzzy_system_from_config(config)
        
        # Process each pixel
        result_block = np.zeros_like(social_block, dtype=np.float32)
        
        for i in range(social_block.shape[0]):
            for j in range(social_block.shape[1]):
                try:
                    # Set input values
                    simulation.social = float(social_block[i, j])
                    simulation.environmental = float(env_block[i, j])
                    simulation.strategic = float(strat_block[i, j])
                    
                    # Compute
                    simulation.compute()
                    
                    # Get result
                    result_block[i, j] = float(getattr(simulation, output_var))
                    
                except Exception as e:
                    print(f"Warning: Fuzzy computation failed at pixel ({i}, {j}): {e}")
                    result_block[i, j] = 5.0  # Default value
        
        return block_id, result_block
        
    except Exception as e:
        print(f"Error processing block {block_id}: {e}")
        return block_id, np.zeros_like(social_block, dtype=np.float32)

def process_rasters_spark_s3_tifffile(
    spark, social_tiff_s3, environmental_tiff_s3, strategic_tiff_s3, output_tiff_s3,
    config_file, nodata_value=5.0, block_size=100, num_partitions=None):
    """
    Process rasters using Spark with S3 input/output and tifffile.
    """
    print("Starting S3-enabled raster fuzzy inference processing...")
    print(f"Input files:")
    print(f"  Social: {social_tiff_s3}")
    print(f"  Environmental: {environmental_tiff_s3}")
    print(f"  Strategic: {strategic_tiff_s3}")
    print(f"Output file: {output_tiff_s3}")
    
    start_time = time.time()
    
    # Load configuration
    def load_config_safe(config_path):
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
            raise
    
    config = load_config_safe(config_file)
    
    # Read input rasters from S3
    print("Reading input rasters from S3...")
    social_data, social_meta = read_tiff_from_s3(social_tiff_s3)
    env_data, env_meta = read_tiff_from_s3(environmental_tiff_s3)
    strat_data, strat_meta = read_tiff_from_s3(strategic_tiff_s3)
    
    print(f"Raster shapes: {social_data.shape}, {env_data.shape}, {strat_data.shape}")
    
    # Create blocks
    height, width = social_data.shape
    blocks = []
    
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            end_i = min(i + block_size, height)
            end_j = min(j + block_size, width)
            
            social_block = social_data[i:end_i, j:end_j]
            env_block = env_data[i:end_i, j:end_j]
            strat_block = strat_data[i:end_i, j:end_j]
            
            block_id = f"{i}_{j}"
            blocks.append((block_id, social_block, env_block, strat_block, config))
    
    print(f"Created {len(blocks)} blocks for processing")
    
    # Process with Spark
    if num_partitions is None:
        num_partitions = max(1, len(blocks) // 10)
    
    rdd = spark.sparkContext.parallelize(blocks, num_partitions)
    results = rdd.map(process_block_s3_tifffile).collect()
    
    # Reconstruct output raster
    print("Reconstructing output raster...")
    output_data = np.zeros((height, width), dtype=np.float32)
    
    for block_id, result_block in results:
        i, j = map(int, block_id.split('_'))
        end_i = min(i + result_block.shape[0], height)
        end_j = min(j + result_block.shape[1], width)
        output_data[i:end_i, j:end_j] = result_block
    
    # Write output to S3
    print("Writing output raster to S3...")
    output_metadata = {
        'shape': output_data.shape,
        'dtype': str(output_data.dtype),
        'min': float(output_data.min()),
        'max': float(output_data.max()),
        'mean': float(output_data.mean())
    }
    
    write_tiff_to_s3(output_data, output_tiff_s3, output_metadata)
    
    processing_time = time.time() - start_time
    print(f"Processing completed in {processing_time:.2f} seconds")
    print(f"Output saved to: {output_tiff_s3}")
    
    return output_data, processing_time

def create_spark_session_s3(app_name: str = "RasterFuzzyInferenceS3", 
                           master_url: str = "spark://spark-master:7077", 
                           local_mode: bool = False):
    """Create Spark session optimized for S3 operations."""
    from pyspark.sql import SparkSession
    
    builder = SparkSession.builder.appName(app_name)
    
    if local_mode:
        builder = builder.master("local[*]")
    else:
        builder = builder.master(master_url)
    
    # Configure for S3 and performance
    spark = builder.config("spark.sql.adaptive.enabled", "true") \
                   .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
                   .config("spark.sql.adaptive.skewJoin.enabled", "true") \
                   .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
                   .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
                   .config("spark.sql.execution.arrow.maxRecordsPerBatch", "10000") \
                   .config("spark.executor.memory", "4g") \
                   .config("spark.driver.memory", "4g") \
                   .config("spark.executor.cores", "2") \
                   .config("spark.driver.cores", "2") \
                   .config("spark.sql.shuffle.partitions", "200") \
                   .config("spark.default.parallelism", "200") \
                   .getOrCreate()
    
    return spark

def main():
    """Main function for S3-enabled Spark processing."""
    import argparse
    
    parser = argparse.ArgumentParser(description='S3-enabled Spark Raster FIS')
    parser.add_argument('--social', required=True, help='S3 path to social TIFF')
    parser.add_argument('--environmental', required=True, help='S3 path to environmental TIFF')
    parser.add_argument('--strategic', required=True, help='S3 path to strategic TIFF')
    parser.add_argument('--output', required=True, help='S3 path for output TIFF')
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--master', default='spark://spark-master:7077', help='Spark master URL')
    parser.add_argument('--local', action='store_true', help='Run in local mode')
    parser.add_argument('--block-size', type=int, default=100, help='Block size for processing')
    parser.add_argument('--partitions', type=int, help='Number of partitions')
    
    args = parser.parse_args()
    
    # Create Spark session
    spark = create_spark_session_s3(
        app_name="RasterFuzzyInferenceS3",
        master_url=args.master,
        local_mode=args.local
    )
    
    try:
        # Process rasters
        output_data, processing_time = process_rasters_spark_s3_tifffile(
            spark=spark,
            social_tiff_s3=args.social,
            environmental_tiff_s3=args.environmental,
            strategic_tiff_s3=args.strategic,
            output_tiff_s3=args.output,
            config_file=args.config,
            block_size=args.block_size,
            num_partitions=args.partitions
        )
        
        print(f"✅ Processing completed successfully in {processing_time:.2f} seconds")
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        spark.stop()

if __name__ == "__main__":
    main() 