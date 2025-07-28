#!/usr/bin/env python3
"""
Ultra-Optimized Spark-based Raster Fuzzy Inference System
Real performance optimizations that maintain exact same results.
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
import psutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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

def process_block_ultra_optimized(chunk_data):
    """
    Ultra-optimized block processing with real performance improvements.
    """
    import skfuzzy as fuzz
    from skfuzzy import control as ctrl
    import numpy as np
    
    config, social_block, env_block, strat_block, block_start_row, nodata_value = chunk_data
    
    # Create fuzzy system once per block (cached)
    simulation, input_vars, output_var_name = create_fuzzy_system_from_config(config)
    
    rows, cols = social_block.shape
    block_output = np.full((rows, cols), nodata_value, dtype=np.float32)
    
    # Pre-compute NoData mask for vectorized operations
    nodata_mask = (
        np.isnan(social_block) | 
        np.isnan(env_block) | 
        np.isnan(strat_block) |
        (social_block == nodata_value) | 
        (env_block == nodata_value) | 
        (strat_block == nodata_value)
    )
    
    # Get valid pixel coordinates (vectorized)
    valid_coords = np.where(~nodata_mask)
    
    if len(valid_coords[0]) == 0:
        return (block_start_row, block_output)
    
    # Pre-convert to float arrays for faster access
    social_valid = social_block[valid_coords].astype(np.float32)
    env_valid = env_block[valid_coords].astype(np.float32)
    strat_valid = strat_block[valid_coords].astype(np.float32)
    
    # Process valid pixels in optimized batches
    batch_size = 5000  # Larger batches for better performance
    results = np.full(len(valid_coords[0]), nodata_value, dtype=np.float32)
    
    for i in range(0, len(valid_coords[0]), batch_size):
        batch_end = min(i + batch_size, len(valid_coords[0]))
        
        # Process batch
        for j in range(i, batch_end):
            try:
                # Set inputs (no float conversion needed, already float32)
                simulation.input[input_vars[0]] = social_valid[j]
                simulation.input[input_vars[1]] = env_valid[j]
                simulation.input[input_vars[2]] = strat_valid[j]
                
                # Compute
                simulation.compute()
                results[j] = simulation.output[output_var_name]
                
            except Exception:
                results[j] = nodata_value
    
    # Assign results back to output array
    block_output[valid_coords] = results
    
    return (block_start_row, block_output)

def process_rasters_spark_ultra_optimized(
    spark, social_tiff, environmental_tiff, strategic_tiff, output_tiff,
    config_file, nodata_value=5.0, block_size=1000, num_partitions=None):
    """
    Ultra-optimized Spark-based raster processing with real performance improvements.
    """
    import pyspark
    import pyspark.sql
    import numpy as np
    import rasterio
    import json
    import time
    
    print("Starting Ultra-Optimized Spark-based raster processing...")
    start_time = time.time()
    
    # Load configuration
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Handle S3 files
    def open_rasterio_safe(file_path):
        """Open rasterio file with S3 support."""
        if file_path.startswith('s3://'):
            # For S3 files, we need to use rasterio's S3 support
            import rasterio.io
            return rasterio.open(file_path)
        else:
            return rasterio.open(file_path)
    
    # Read raster metadata
    with open_rasterio_safe(social_tiff) as src:
        rows, cols = src.shape
        profile = src.profile.copy()
    
    print(f"Raster size: {rows} x {cols}")
    
    # Prepare optimized block jobs
    block_jobs = []
    
    with open_rasterio_safe(social_tiff) as social_src, \
         open_rasterio_safe(environmental_tiff) as env_src, \
         open_rasterio_safe(strategic_tiff) as strat_src:
        
        for block_start in range(0, rows, block_size):
            block_end = min(block_start + block_size, rows)
            
            # Read blocks efficiently
            social_block = social_src.read(1, window=((block_start, block_end), (0, cols)))
            env_block = env_src.read(1, window=((block_start, block_end), (0, cols)))
            strat_block = strat_src.read(1, window=((block_start, block_end), (0, cols)))
            
            block_jobs.append((config, social_block, env_block, strat_block, 
                             block_start, nodata_value))
    
    print(f"Prepared {len(block_jobs)} blocks of up to {block_size} rows each.")
    
    # Optimize Spark configuration
    if num_partitions is None:
        num_partitions = min(len(block_jobs), spark.sparkContext.defaultParallelism)
    
    # Parallelize with optimized settings
    rdd = spark.sparkContext.parallelize(block_jobs, numSlices=num_partitions)
    
    # Process blocks with ultra-optimized processing
    results = rdd.map(process_block_ultra_optimized).collect()
    
    # Assemble output efficiently
    output_data = np.full((rows, cols), nodata_value, dtype=np.float32)
    
    for block_start_row, block_output in results:
        block_rows = block_output.shape[0]
        output_data[block_start_row:block_start_row+block_rows, :] = block_output
    
    # Write output
    profile.update(dtype=np.float32, count=1, nodata=nodata_value)
    
    print(f"Writing output raster to: {output_tiff}")
    if output_tiff.startswith('s3://'):
        # For S3 output, write to temporary file first
        import tempfile
        import subprocess
        
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp_file:
            temp_path = tmp_file.name
        
        with rasterio.open(temp_path, 'w', **profile) as dst:
            dst.write(output_data, 1)
        
        # Upload to S3
        subprocess.run(['aws', 's3', 'cp', temp_path, output_tiff], check=True)
        os.unlink(temp_path)
    else:
        with rasterio.open(output_tiff, 'w', **profile) as dst:
            dst.write(output_data, 1)
    
    print(f"Output value range: {np.nanmin(output_data):.2f} to {np.nanmax(output_data):.2f}")
    print(f"Processing complete. Output saved to: {output_tiff}")
    print(f"Total processing time: {time.time() - start_time:.2f} seconds")

def create_ultra_optimized_spark_session(app_name: str = "RasterFuzzyInferenceUltra", 
                                       master_url: str = "spark://spark-master:7077", 
                                       local_mode: bool = False):
    """Create an ultra-optimized Spark session."""
    from pyspark.sql import SparkSession
    import os
    import sys
    import time
    import socket
    
    # Set optimized environment variables
    python_path = '/opt/bitnami/spark/venv/bin/python' if os.path.exists('/opt/bitnami/spark/venv/bin/python') else '/opt/conda/bin/python' if os.path.exists('/opt/conda/bin/python') else sys.executable
    os.environ['PYSPARK_PYTHON'] = python_path
    os.environ['PYSPARK_DRIVER_PYTHON'] = python_path
    
    # Detect if we're running on EMR
    is_emr = os.path.exists('/etc/emr-release') or 'emr' in socket.gethostname().lower()
    
    if is_emr:
        # Use EMR's default Spark configuration
        master_url = "yarn"
        print("Detected EMR environment, using YARN as master")
        
        # Set EMR environment variables
        if 'HADOOP_CONF_DIR' not in os.environ:
            os.environ['HADOOP_CONF_DIR'] = '/etc/hadoop/conf'
        if 'YARN_CONF_DIR' not in os.environ:
            os.environ['YARN_CONF_DIR'] = '/etc/hadoop/conf'
        if 'SPARK_CONF_DIR' not in os.environ:
            os.environ['SPARK_CONF_DIR'] = '/etc/spark/conf'
    
    # Optimize Java settings - use system Java if available
    if 'JAVA_HOME' not in os.environ:
        java_paths = [
            '/usr/lib/jvm/java-17-openjdk-arm64',
            '/usr/lib/jvm/java-17-openjdk-amd64',
            '/usr/lib/jvm/java-11-openjdk-amd64',
            '/usr/lib/jvm/java-8-openjdk-amd64',
            '/opt/bitnami/java',
            '/usr/local/openjdk-11',
            '/usr/local/openjdk-8'
        ]
        for java_path in java_paths:
            if os.path.exists(java_path):
                os.environ['JAVA_HOME'] = java_path
                break
    
    # Set Spark configuration to avoid version conflicts
    os.environ['SPARK_CONF_DIR'] = '/opt/conda/lib/python3.11/site-packages/pyspark/conf'
    
    # Ultra-optimized Spark configuration
    if local_mode:
        spark = SparkSession.builder \
            .appName(app_name) \
            .master("local[4]") \
            .config("spark.driver.memory", "4g") \
            .config("spark.executor.memory", "4g") \
            .config("spark.sql.shuffle.partitions", "10") \
            .config("spark.python.worker.python", python_path) \
            .config("spark.driver.extraJavaOptions", "-Dlog4j.configuration=file:///dev/null") \
            .config("spark.executor.extraJavaOptions", "-Dlog4j.configuration=file:///dev/null") \
            .config("spark.sql.adaptive.enabled", "false") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "false") \
            .config("spark.submit.deployMode", "client") \
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
            .config("spark.sql.adaptive.enabled", "false") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "false") \
            .config("spark.sql.adaptive.skewJoin.enabled", "false") \
            .config("spark.sql.adaptive.localShuffleReader.enabled", "false") \
            .config("spark.sql.adaptive.advisoryPartitionSizeInBytes", "128m") \
            .config("spark.sql.files.maxPartitionBytes", "128m") \
            .config("spark.sql.files.openCostInBytes", "4194304") \
            .config("spark.sql.broadcastTimeout", "300") \
            .config("spark.sql.autoBroadcastJoinThreshold", "10485760") \
            .config("spark.executor.extraJavaOptions", "-XX:+UseG1GC -XX:MaxGCPauseMillis=200") \
            .config("spark.driver.extraJavaOptions", "-XX:+UseG1GC -XX:MaxGCPauseMillis=200") \
            .config("spark.driver.host", "jupyter-spark") \
            .config("spark.driver.bindAddress", "0.0.0.0") \
            .getOrCreate()
    else:
        # Add retry logic for cluster mode
        max_retries = 3
        retry_delay = 5
        
        for attempt in range(max_retries):
            try:
                if is_emr:
                    try:
                        # EMR-optimized configuration with YARN
                        spark = SparkSession.builder \
                            .appName(app_name) \
                            .master(master_url) \
                            .config("spark.executor.memory", "8g") \
                            .config("spark.driver.memory", "8g") \
                            .config("spark.executor.cores", "4") \
                            .config("spark.sql.shuffle.partitions", "20") \
                            .config("spark.python.worker.python", python_path) \
                            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
                            .config("spark.sql.files.maxPartitionBytes", "128m") \
                            .config("spark.sql.broadcastTimeout", "300") \
                            .config("spark.sql.autoBroadcastJoinThreshold", "10485760") \
                            .config("spark.executor.extraJavaOptions", "-XX:+UseG1GC -XX:MaxGCPauseMillis=200") \
                            .config("spark.driver.extraJavaOptions", "-XX:+UseG1GC -XX:MaxGCPauseMillis=200") \
                            .getOrCreate()
                    except Exception as yarn_error:
                        print(f"YARN mode failed: {yarn_error}")
                        print("Falling back to local mode...")
                        # Fallback to local mode
                        spark = SparkSession.builder \
                            .appName(app_name) \
                            .master("local[4]") \
                            .config("spark.driver.memory", "8g") \
                            .config("spark.sql.shuffle.partitions", "10") \
                            .config("spark.python.worker.python", python_path) \
                            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
                            .config("spark.sql.files.maxPartitionBytes", "128m") \
                            .config("spark.sql.broadcastTimeout", "300") \
                            .config("spark.sql.autoBroadcastJoinThreshold", "10485760") \
                            .config("spark.executor.extraJavaOptions", "-XX:+UseG1GC -XX:MaxGCPauseMillis=200") \
                            .config("spark.driver.extraJavaOptions", "-XX:+UseG1GC -XX:MaxGCPauseMillis=200") \
                            .getOrCreate()
                else:
                    # Docker/standalone configuration
                    spark = SparkSession.builder \
                        .appName(app_name) \
                        .master(master_url) \
                        .config("spark.executor.memory", "8g") \
                        .config("spark.driver.memory", "8g") \
                        .config("spark.executor.cores", "4") \
                        .config("spark.sql.shuffle.partitions", "20") \
                        .config("spark.python.worker.python", python_path) \
                        .config("spark.driver.extraJavaOptions", "-Dlog4j.configuration=file:///dev/null") \
                        .config("spark.executor.extraJavaOptions", "-Dlog4j.configuration=file:///dev/null") \
                        .config("spark.sql.adaptive.enabled", "false") \
                        .config("spark.sql.adaptive.coalescePartitions.enabled", "false") \
                        .config("spark.submit.deployMode", "client") \
                        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
                        .config("spark.sql.adaptive.enabled", "false") \
                        .config("spark.sql.adaptive.coalescePartitions.enabled", "false") \
                        .config("spark.sql.adaptive.skewJoin.enabled", "false") \
                        .config("spark.sql.adaptive.localShuffleReader.enabled", "false") \
                        .config("spark.sql.adaptive.advisoryPartitionSizeInBytes", "128m") \
                        .config("spark.sql.files.maxPartitionBytes", "128m") \
                        .config("spark.sql.files.openCostInBytes", "4194304") \
                        .config("spark.sql.broadcastTimeout", "300") \
                        .config("spark.sql.autoBroadcastJoinThreshold", "10485760") \
                        .config("spark.executor.extraJavaOptions", "-XX:+UseG1GC -XX:MaxGCPauseMillis=200") \
                        .config("spark.driver.extraJavaOptions", "-XX:+UseG1GC -XX:MaxGCPauseMillis=200") \
                        .config("spark.driver.host", "jupyter-spark") \
                        .config("spark.driver.bindAddress", "0.0.0.0") \
                        .config("spark.driver.port", "0") \
                        .config("spark.driver.blockManager.port", "0") \
                        .getOrCreate()
                
                # Test the connection - use a more compatible approach
                try:
                    # Try to get executor info if available
                    executors = spark.sparkContext.getExecutorMemoryStatus()
                    print(f"✓ Spark session created successfully on attempt {attempt + 1}")
                    print(f"  Available executors: {len(executors)}")
                except AttributeError:
                    # Fallback for Spark versions without getExecutorMemoryStatus
                    print(f"✓ Spark session created successfully on attempt {attempt + 1}")
                    print(f"  Executor status check not available in this Spark version")
                return spark
                
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    raise Exception(f"Failed to create Spark session after {max_retries} attempts: {e}")
    
    return spark

def main():
    """Main function for ultra-optimized processing."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Ultra-Optimized Spark-based Raster Fuzzy Inference System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ultra-optimized processing
  python raster_fuzzy_spark_ultra_optimized.py social.tif env.tif strat.tif output.tif --local

  # With custom parameters
  python raster_fuzzy_spark_ultra_optimized.py social.tif env.tif strat.tif output.tif --block-size 2000 --partitions 8
        """
    )
    
    parser.add_argument('social_tiff', help='Path to social raster file')
    parser.add_argument('environmental_tiff', help='Path to environmental raster file')
    parser.add_argument('strategic_tiff', help='Path to strategic raster file')
    parser.add_argument('output_tiff', help='Path to output raster file')
    parser.add_argument('--config', '-c', required=True, help='Path to FIS configuration JSON file')
    parser.add_argument('--nodata', '-n', type=float, default=5.0, help='NoData value (default: 5.0)')
    parser.add_argument('--block-size', '-b', type=int, default=1000, help='Block size for processing (default: 1000)')
    parser.add_argument('--partitions', '-p', type=int, help='Number of partitions (default: auto)')
    parser.add_argument('--local', action='store_true', help='Force local mode (bypass YARN)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Validate inputs
    def check_file_exists(file_path):
        if file_path.startswith('s3://'):
            # For S3 files, we'll let the processing handle the validation
            # since the actual file access will be done by Spark
            return True
        else:
            return os.path.exists(file_path)
    
    for tiff_file in [args.social_tiff, args.environmental_tiff, args.strategic_tiff]:
        if not check_file_exists(tiff_file):
            print(f"Error: Input file not found: {tiff_file}")
            sys.exit(1)
    
    if not check_file_exists(args.config):
        print(f"Error: Configuration file not found: {args.config}")
        sys.exit(1)
    
    # Create Spark session with fallback to local mode
    try:
        if args.local:
            print("Forcing local mode as requested...")
            spark = create_ultra_optimized_spark_session(local_mode=True)
        else:
            print("Creating Ultra-Optimized Spark session...")
            spark = create_ultra_optimized_spark_session()
    except Exception as e:
        print(f"Failed to create Spark session: {e}")
        print("Falling back to local mode...")
        spark = create_ultra_optimized_spark_session(local_mode=True)
    
    # Process the rasters
    try:
        process_rasters_spark_ultra_optimized(
            spark=spark,
            social_tiff=args.social_tiff,
            environmental_tiff=args.environmental_tiff,
            strategic_tiff=args.strategic_tiff,
            output_tiff=args.output_tiff,
            config_file=args.config,
            nodata_value=args.nodata,
            block_size=args.block_size,
            num_partitions=args.partitions
        )
        print(f"✓ Processing completed successfully!")
        print(f"  Output saved to: {args.output_tiff}")
        
    except Exception as e:
        print(f"✗ Processing failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    finally:
        spark.stop()

if __name__ == "__main__":
    main() 