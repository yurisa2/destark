#!/usr/bin/env python3
"""
Spark-based Raster Fuzzy Inference System - Full Raster Processing
Distributes raster processing across a Spark cluster for improved performance.
"""

import os
import sys
import json
import time
import numpy as np
import rasterio
from typing import Dict, List, Tuple, Optional
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from raster_fuzzy_lib import create_raster_config_template


def create_fuzzy_system_from_config(config: Dict):
    import skfuzzy as fuzz
    from skfuzzy import control as ctrl
    import numpy as np
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
            rule = ctrl.Rule(antecedent_conditions[0] & antecedent_conditions[1] & antecedent_conditions[2],
                             output_variable[consequent])
        rules.append(rule)
    control_system = ctrl.ControlSystem(rules)
    simulation = ctrl.ControlSystemSimulation(control_system)
    return simulation, list(config['input_variables'].keys()), output_config['name']


def process_block(chunk_data):
    """
    Process a block of rows from the raster.
    chunk_data: (config, block_rows, block_start_row, block_shape)
    Returns: (block_start_row, block_output)
    """
    import skfuzzy as fuzz
    from skfuzzy import control as ctrl
    import numpy as np
    config, social_block, env_block, strat_block, block_start_row, nodata_value = chunk_data
    simulation, input_vars, output_var_name = create_fuzzy_system_from_config(config)
    rows, cols = social_block.shape
    block_output = np.full((rows, cols), nodata_value, dtype=np.float32)
    for i in range(rows):
        for j in range(cols):
            s = social_block[i, j]
            e = env_block[i, j]
            t = strat_block[i, j]
            if (np.isnan(s) or np.isnan(e) or np.isnan(t) or
                s == nodata_value or e == nodata_value or t == nodata_value):
                continue
            try:
                simulation.input[input_vars[0]] = float(s)
                simulation.input[input_vars[1]] = float(e)
                simulation.input[input_vars[2]] = float(t)
                simulation.compute()
                block_output[i, j] = float(simulation.output[output_var_name])
            except Exception:
                block_output[i, j] = nodata_value
    return (block_start_row, block_output)


def process_rasters_spark_full(
    spark, social_tiff, environmental_tiff, strategic_tiff, output_tiff,
    config_file, nodata_value=5.0, block_size=500, num_partitions=None):
    import pyspark
    import pyspark.sql
    import numpy as np
    import rasterio
    import json
    import time
    print("Starting Spark-based full raster processing...")
    start_time = time.time()
    with open(config_file, 'r') as f:
        config = json.load(f)
    with rasterio.open(social_tiff) as src:
        rows, cols = src.shape
        profile = src.profile.copy()
    print(f"Raster size: {rows} x {cols}")
    # Prepare block jobs
    block_jobs = []
    with rasterio.open(social_tiff) as social_src, \
         rasterio.open(environmental_tiff) as env_src, \
         rasterio.open(strategic_tiff) as strat_src:
        for block_start in range(0, rows, block_size):
            block_end = min(block_start + block_size, rows)
            social_block = social_src.read(1, window=((block_start, block_end), (0, cols)))
            env_block = env_src.read(1, window=((block_start, block_end), (0, cols)))
            strat_block = strat_src.read(1, window=((block_start, block_end), (0, cols)))
            block_jobs.append((config, social_block, env_block, strat_block, block_start, nodata_value))
    print(f"Prepared {len(block_jobs)} blocks of up to {block_size} rows each.")
    # Parallelize blocks
    rdd = spark.sparkContext.parallelize(block_jobs, numSlices=num_partitions or len(block_jobs))
    results = rdd.map(process_block).collect()
    # Assemble output
    output_data = np.full((rows, cols), nodata_value, dtype=np.float32)
    for block_start_row, block_output in results:
        block_rows = block_output.shape[0]
        output_data[block_start_row:block_start_row+block_rows, :] = block_output
    profile.update(dtype=np.float32, count=1, nodata=nodata_value)
    print(f"Writing output raster to: {output_tiff}")
    with rasterio.open(output_tiff, 'w', **profile) as dst:
        dst.write(output_data, 1)
    print(f"Output value range: {np.nanmin(output_data):.2f} to {np.nanmax(output_data):.2f}")
    print(f"Processing complete. Output saved to: {output_tiff}")
    print(f"Total processing time: {time.time() - start_time:.2f} seconds")


def create_spark_session(app_name: str = "RasterFuzzyInference", master_url: str = "spark://spark-master:7077", local_mode: bool = False):
    from pyspark.sql import SparkSession
    import os
    import sys
    
    # Set critical environment variables for Java/Python compatibility
    # Use system Python for local development, Docker Python for containerized environments
    python_path = '/opt/conda/bin/python' if os.path.exists('/opt/conda/bin/python') else sys.executable
    os.environ['PYSPARK_PYTHON'] = python_path
    os.environ['PYSPARK_DRIVER_PYTHON'] = python_path
    
    # Set Java environment variables if not already set
    if 'JAVA_HOME' not in os.environ:
        # Try to find Java in common locations
        java_paths = [
            '/usr/lib/jvm/java-17-openjdk-arm64',  # Docker container Java 17
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
    
    # For Docker containers, override JAVA_HOME if it's incorrect
    if os.environ.get('JAVA_HOME') == '/usr/lib/jvm/java-11-openjdk-amd64':
        # Check if we're in a Docker container with Java 17
        java_17_path = '/usr/lib/jvm/java-17-openjdk-arm64'
        if os.path.exists(java_17_path):
            os.environ['JAVA_HOME'] = java_17_path
    
    # For local mode, explicitly unset SPARK_HOME to prevent spark-submit usage
    if local_mode:
        if 'SPARK_HOME' in os.environ:
            del os.environ['SPARK_HOME']
    else:
        # Set Spark environment variables only if the path actually exists
        if 'SPARK_HOME' not in os.environ:
            spark_paths = [
                '/opt/bitnami/spark',
                '/usr/local/spark',
                '/opt/spark',
                '/opt/conda'  # Add this for Docker container
            ]
            for spark_path in spark_paths:
                if os.path.exists(spark_path):
                    os.environ['SPARK_HOME'] = spark_path
                    break
            # If no valid SPARK_HOME found, don't set it to avoid spark-submit issues
        
        # For Docker containers, ensure SPARK_HOME points to the PySpark installation
        if os.environ.get('SPARK_HOME') == '/opt/conda':
            # Check if we're in a Docker container with PySpark installed
            pyspark_path = '/opt/conda/lib/python3.11/site-packages/pyspark'
            if os.path.exists(pyspark_path):
                os.environ['SPARK_HOME'] = pyspark_path
    
    # Configure Spark session with minimal settings to avoid compatibility issues
    if local_mode:
        spark = SparkSession.builder \
            .appName(app_name) \
            .master("local[2]") \
            .config("spark.driver.memory", "2g") \
            .config("spark.executor.memory", "2g") \
            .config("spark.sql.shuffle.partitions", "5") \
            .config("spark.python.worker.python", python_path) \
            .config("spark.driver.extraJavaOptions", "-Dlog4j.configuration=file:///dev/null") \
            .config("spark.executor.extraJavaOptions", "-Dlog4j.configuration=file:///dev/null") \
            .config("spark.sql.adaptive.enabled", "false") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "false") \
            .config("spark.submit.deployMode", "client") \
            .getOrCreate()
    else:
        spark = SparkSession.builder \
            .appName(app_name) \
            .master(master_url) \
            .config("spark.executor.memory", "4g") \
            .config("spark.driver.memory", "4g") \
            .config("spark.executor.cores", "2") \
            .config("spark.sql.shuffle.partitions", "10") \
            .config("spark.python.worker.python", python_path) \
            .config("spark.driver.extraJavaOptions", "-Dlog4j.configuration=file:///dev/null") \
            .config("spark.executor.extraJavaOptions", "-Dlog4j.configuration=file:///dev/null") \
            .config("spark.sql.adaptive.enabled", "false") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "false") \
            .config("spark.submit.deployMode", "client") \
            .getOrCreate()
    
    return spark


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Spark-based Raster Fuzzy Inference System - Full Raster Processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python raster_fuzzy_spark_simple.py social.tif env.tif strat.tif output.tif --config config.json --local
        """
    )
    parser.add_argument('social_tiff', help='Path to social factor TIFF file')
    parser.add_argument('environmental_tiff', help='Path to environmental factor TIFF file')
    parser.add_argument('strategic_tiff', help='Path to strategic factor TIFF file')
    parser.add_argument('output_tiff', help='Path for output TIFF file')
    parser.add_argument('--config', '-c', default='raster_fis_config.json', help='Path to configuration JSON file (default: raster_fis_config.json)')
    parser.add_argument('--nodata', type=float, default=5.0, help='NoData value for output raster (default: 5.0)')
    parser.add_argument('--block-size', type=int, default=500, help='Number of rows per block (default: 500)')
    parser.add_argument('--partitions', type=int, default=None, help='Number of Spark partitions (default: auto)')
    parser.add_argument('--local', action='store_true', help='Run in local mode for testing')
    parser.add_argument('--master', default='spark://spark-master:7077', help='Spark master URL (default: spark://spark-master:7077)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    args = parser.parse_args()
    for tiff_file in [args.social_tiff, args.environmental_tiff, args.strategic_tiff]:
        if not os.path.exists(tiff_file):
            print(f"Error: Input file not found: {tiff_file}")
            sys.exit(1)
    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found: {args.config}")
        sys.exit(1)
    output_dir = os.path.dirname(args.output_tiff)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    try:
        print(f"Creating Spark session...")
        spark = create_spark_session(
            app_name="RasterFuzzyInference",
            master_url=args.master,
            local_mode=args.local
        )
        print(f"Processing rasters with Spark...")
        print(f"  Social: {args.social_tiff}")
        print(f"  Environmental: {args.environmental_tiff}")
        print(f"  Strategic: {args.strategic_tiff}")
        print(f"  Output: {args.output_tiff}")
        print(f"  Config: {args.config}")
        print(f"  NoData value: {args.nodata}")
        print(f"  Block size: {args.block_size} rows")
        print(f"  Partitions: {args.partitions or 'auto'}")
        print(f"  Mode: {'Local' if args.local else 'Cluster'}")
        process_rasters_spark_full(
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
        print("Spark processing completed successfully!")
        spark.stop()
    except Exception as e:
        print(f"Spark processing failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        print("Spark processing failed. Please check your Spark configuration or use the dedicated local processing script.")
        sys.exit(1)

if __name__ == "__main__":
    main() 