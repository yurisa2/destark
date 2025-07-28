#!/usr/bin/env python3
"""
Ultra-Optimized Spark-based Raster Fuzzy Inference System
Features:
- Comprehensive benchmarking and metrics
- Detailed logging for performance analysis
- Sampling for testing (10% of data)
- Extreme performance optimizations
- Memory-efficient processing
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
import gc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from raster_fuzzy_lib import create_raster_config_template

# Configure comprehensive logging
def setup_logging(log_file: str = None):
    """Setup comprehensive logging for performance analysis."""
    if log_file is None:
        log_file = f"logs/raster_processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Ensure logs directory exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def get_system_metrics():
    """Get current system metrics for benchmarking."""
    return {
        'memory_usage': psutil.virtual_memory().percent,
        'cpu_usage': psutil.cpu_percent(interval=1),
        'memory_available': psutil.virtual_memory().available / (1024**3),  # GB
        'disk_usage': psutil.disk_usage('/').percent
    }

class PerformanceBenchmark:
    """Comprehensive performance benchmarking class."""
    
    def __init__(self, logger):
        self.logger = logger
        self.start_time = None
        self.checkpoints = {}
        self.metrics = {}
    
    def start(self, operation_name: str):
        """Start timing an operation."""
        self.start_time = time.time()
        self.logger.info(f"🚀 Starting: {operation_name}")
        self.log_metrics(f"START_{operation_name}")
    
    def checkpoint(self, checkpoint_name: str):
        """Record a checkpoint with timing."""
        if self.start_time is None:
            self.logger.warning("No start time recorded")
            return
        
        elapsed = time.time() - self.start_time
        self.checkpoints[checkpoint_name] = elapsed
        self.logger.info(f"⏱️  Checkpoint '{checkpoint_name}': {elapsed:.3f}s")
        self.log_metrics(f"CHECKPOINT_{checkpoint_name}")
    
    def end(self, operation_name: str):
        """End timing and log final metrics."""
        if self.start_time is None:
            self.logger.warning("No start time recorded")
            return
        
        total_time = time.time() - self.start_time
        self.metrics[operation_name] = {
            'total_time': total_time,
            'checkpoints': self.checkpoints.copy()
        }
        
        self.logger.info(f"✅ Completed: {operation_name} in {total_time:.3f}s")
        self.log_metrics(f"END_{operation_name}")
        
        # Reset for next operation
        self.start_time = None
        self.checkpoints.clear()
    
    def log_metrics(self, operation: str):
        """Log system metrics."""
        metrics = get_system_metrics()
        self.logger.info(f"📊 Metrics for {operation}: CPU={metrics['cpu_usage']:.1f}%, "
                        f"Memory={metrics['memory_usage']:.1f}%, "
                        f"Available={metrics['memory_available']:.2f}GB")

def create_optimized_fuzzy_system_from_config(config: Dict):
    """Create an optimized fuzzy system with reduced resolution for speed."""
    import skfuzzy as fuzz
    from skfuzzy import control as ctrl
    import numpy as np
    
    # Use very low resolution for maximum speed
    resolution = 25  # Reduced from 50 for speed
    
    input_variables = {}
    for var_name, var_config in config['input_variables'].items():
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
    universe = np.linspace(output_config['min'], output_config['max'], resolution)
    output_variable = ctrl.Consequent(universe, output_config['name'])
    
    for mf_name, mf_config in output_config['membership_functions'].items():
        if mf_config['type'] == 'trapmf':
            output_variable[mf_name] = fuzz.trapmf(
                output_variable.universe, mf_config['params'])
        elif mf_config['type'] == 'trimf':
            output_variable[mf_name] = fuzz.trimf(
                output_variable.universe, mf_config['params'])
    
    # Optimize rules creation
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
            # Use & operator for multiple conditions
            combined_antecedent = antecedent_conditions[0]
            for condition in antecedent_conditions[1:]:
                combined_antecedent = combined_antecedent & condition
            rule = ctrl.Rule(combined_antecedent, output_variable[consequent])
        
        rules.append(rule)
    
    control_system = ctrl.ControlSystem(rules)
    simulation = ctrl.ControlSystemSimulation(control_system)
    return simulation, list(config['input_variables'].keys()), output_config['name']

def process_block_optimized(chunk_data):
    """
    Ultra-optimized block processing with vectorized operations.
    """
    import skfuzzy as fuzz
    from skfuzzy import control as ctrl
    import numpy as np
    
    config, social_block, env_block, strat_block, block_start_row, nodata_value, sample_rate = chunk_data
    
    # Create fuzzy system once per block
    simulation, input_vars, output_var_name = create_optimized_fuzzy_system_from_config(config)
    
    rows, cols = social_block.shape
    block_output = np.full((rows, cols), nodata_value, dtype=np.float32)
    
    # Apply sampling if specified
    if sample_rate < 1.0:
        # Create sampling mask
        sample_mask = np.random.random((rows, cols)) < sample_rate
    else:
        sample_mask = np.ones((rows, cols), dtype=bool)
    
    # Vectorized processing for valid pixels
    valid_mask = (
        ~np.isnan(social_block) & 
        ~np.isnan(env_block) & 
        ~np.isnan(strat_block) &
        (social_block != nodata_value) & 
        (env_block != nodata_value) & 
        (strat_block != nodata_value) &
        sample_mask
    )
    
    # Get valid pixel coordinates
    valid_coords = np.where(valid_mask)
    
    # Process valid pixels in batches for better performance
    batch_size = 1000
    for i in range(0, len(valid_coords[0]), batch_size):
        batch_end = min(i + batch_size, len(valid_coords[0]))
        batch_rows = valid_coords[0][i:batch_end]
        batch_cols = valid_coords[1][i:batch_end]
        
        for idx in range(len(batch_rows)):
            row, col = batch_rows[idx], batch_cols[idx]
            
            try:
                # Set inputs
                simulation.input[input_vars[0]] = float(social_block[row, col])
                simulation.input[input_vars[1]] = float(env_block[row, col])
                simulation.input[input_vars[2]] = float(strat_block[row, col])
                
                # Compute
                simulation.compute()
                block_output[row, col] = float(simulation.output[output_var_name])
                
            except Exception:
                block_output[row, col] = nodata_value
    
    return (block_start_row, block_output, len(valid_coords[0]))

def process_rasters_spark_optimized(
    spark, social_tiff, environmental_tiff, strategic_tiff, output_tiff,
    config_file, nodata_value=5.0, block_size=1000, num_partitions=None, 
    sample_rate=0.1, benchmark=None):
    """
    Ultra-optimized Spark-based raster processing with comprehensive benchmarking.
    """
    import pyspark
    import pyspark.sql
    import numpy as np
    import rasterio
    import json
    import time
    
    if benchmark is None:
        benchmark = PerformanceBenchmark(logging.getLogger(__name__))
    
    benchmark.start("SPARK_RASTER_PROCESSING")
    
    # Load configuration
    benchmark.checkpoint("config_loading")
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Read raster metadata
    benchmark.checkpoint("metadata_reading")
    with rasterio.open(social_tiff) as src:
        rows, cols = src.shape
        profile = src.profile.copy()
    
    benchmark.logger.info(f"📐 Raster size: {rows} x {cols} = {rows * cols:,} pixels")
    benchmark.logger.info(f"📊 Sample rate: {sample_rate:.1%} ({int(rows * cols * sample_rate):,} pixels)")
    
    # Prepare optimized block jobs
    benchmark.checkpoint("block_preparation")
    block_jobs = []
    
    with rasterio.open(social_tiff) as social_src, \
         rasterio.open(environmental_tiff) as env_src, \
         rasterio.open(strategic_tiff) as strat_src:
        
        for block_start in range(0, rows, block_size):
            block_end = min(block_start + block_size, rows)
            
            # Read blocks efficiently
            social_block = social_src.read(1, window=((block_start, block_end), (0, cols)))
            env_block = env_src.read(1, window=((block_start, block_end), (0, cols)))
            strat_block = strat_src.read(1, window=((block_start, block_end), (0, cols)))
            
            block_jobs.append((config, social_block, env_block, strat_block, 
                             block_start, nodata_value, sample_rate))
    
    benchmark.logger.info(f"📦 Prepared {len(block_jobs)} blocks of up to {block_size} rows each")
    
    # Optimize Spark configuration
    benchmark.checkpoint("spark_optimization")
    if num_partitions is None:
        num_partitions = min(len(block_jobs), spark.sparkContext.defaultParallelism)
    
    # Parallelize with optimized settings
    benchmark.checkpoint("rdd_creation")
    rdd = spark.sparkContext.parallelize(block_jobs, numSlices=num_partitions)
    
    # Process blocks with detailed metrics
    benchmark.checkpoint("block_processing_start")
    results = rdd.map(process_block_optimized).collect()
    benchmark.checkpoint("block_processing_complete")
    
    # Assemble output efficiently
    benchmark.checkpoint("output_assembly")
    output_data = np.full((rows, cols), nodata_value, dtype=np.float32)
    total_processed_pixels = 0
    
    for block_start_row, block_output, processed_pixels in results:
        block_rows = block_output.shape[0]
        output_data[block_start_row:block_start_row+block_rows, :] = block_output
        total_processed_pixels += processed_pixels
    
    benchmark.logger.info(f"✅ Processed {total_processed_pixels:,} pixels")
    
    # Write output with optimized settings
    benchmark.checkpoint("output_writing")
    profile.update(dtype=np.float32, count=1, nodata=nodata_value)
    
    with rasterio.open(output_tiff, 'w', **profile) as dst:
        dst.write(output_data, 1)
    
    benchmark.checkpoint("output_complete")
    
    # Final metrics
    output_stats = {
        'min': float(np.nanmin(output_data)),
        'max': float(np.nanmax(output_data)),
        'mean': float(np.nanmean(output_data)),
        'std': float(np.nanstd(output_data)),
        'processed_pixels': total_processed_pixels,
        'total_pixels': rows * cols,
        'sample_rate': sample_rate
    }
    
    benchmark.logger.info(f"📈 Output statistics: min={output_stats['min']:.3f}, "
                        f"max={output_stats['max']:.3f}, mean={output_stats['mean']:.3f}")
    benchmark.logger.info(f"📊 Processing efficiency: {total_processed_pixels/(rows*cols)*100:.1f}% of pixels processed")
    
    benchmark.end("SPARK_RASTER_PROCESSING")
    
    return output_stats

def create_optimized_spark_session(app_name: str = "RasterFuzzyInferenceOptimized", 
                                 master_url: str = "spark://spark-master:7077", 
                                 local_mode: bool = False):
    """Create an optimized Spark session with performance tuning."""
    from pyspark.sql import SparkSession
    import os
    import sys
    
    # Set optimized environment variables
    python_path = '/opt/conda/bin/python' if os.path.exists('/opt/conda/bin/python') else sys.executable
    os.environ['PYSPARK_PYTHON'] = python_path
    os.environ['PYSPARK_DRIVER_PYTHON'] = python_path
    
    # Optimize Java settings
    if 'JAVA_HOME' not in os.environ:
        java_paths = [
            '/usr/lib/jvm/java-17-openjdk-arm64',
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
    
    # Optimized Spark configuration
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
            .getOrCreate()
    else:
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
            .getOrCreate()
    
    return spark

def main():
    """Main function with comprehensive benchmarking."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Ultra-Optimized Spark-based Raster Fuzzy Inference System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Benchmark with 10% sampling
  python raster_fuzzy_spark_optimized.py social.tif env.tif strat.tif output.tif --sample-rate 0.1 --local

  # Full processing with optimized settings
  python raster_fuzzy_spark_optimized.py social.tif env.tif strat.tif output.tif --block-size 2000 --partitions 8
        """
    )
    
    parser.add_argument('social_tiff', help='Path to social factor TIFF file')
    parser.add_argument('environmental_tiff', help='Path to environmental factor TIFF file')
    parser.add_argument('strategic_tiff', help='Path to strategic factor TIFF file')
    parser.add_argument('output_tiff', help='Path for output TIFF file')
    
    parser.add_argument('--config', '-c', default='raster_fis_config.json',
                       help='Path to configuration JSON file (default: raster_fis_config.json)')
    parser.add_argument('--nodata', type=float, default=5.0,
                       help='NoData value for output raster (default: 5.0)')
    parser.add_argument('--block-size', type=int, default=1000,
                       help='Number of rows per block (default: 1000)')
    parser.add_argument('--partitions', type=int, default=None,
                       help='Number of Spark partitions (default: auto)')
    parser.add_argument('--sample-rate', type=float, default=0.1,
                       help='Sample rate for testing (0.1 = 10%%, default: 0.1)')
    parser.add_argument('--local', action='store_true',
                       help='Run in local mode for testing')
    parser.add_argument('--master', default='spark://spark-master:7077',
                       help='Spark master URL (default: spark://spark-master:7077)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--log-file', help='Custom log file path')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_file)
    benchmark = PerformanceBenchmark(logger)
    
    # Validate inputs
    for tiff_file in [args.social_tiff, args.environmental_tiff, args.strategic_tiff]:
        if not os.path.exists(tiff_file):
            logger.error(f"Input file not found: {tiff_file}")
            sys.exit(1)
    
    if not os.path.exists(args.config):
        logger.error(f"Configuration file not found: {args.config}")
        sys.exit(1)
    
    # Create output directory
    output_dir = os.path.dirname(args.output_tiff)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        # Log system information
        logger.info("🚀 Starting Ultra-Optimized Spark Raster Processing")
        logger.info(f"📁 Input files: {args.social_tiff}, {args.environmental_tiff}, {args.strategic_tiff}")
        logger.info(f"📁 Output file: {args.output_tiff}")
        logger.info(f"📊 Sample rate: {args.sample_rate:.1%}")
        logger.info(f"⚙️  Block size: {args.block_size}")
        logger.info(f"🔧 Partitions: {args.partitions or 'auto'}")
        logger.info(f"🏠 Mode: {'Local' if args.local else 'Cluster'}")
        
        # Create optimized Spark session
        benchmark.start("SPARK_SESSION_CREATION")
        spark = create_optimized_spark_session(
            app_name="RasterFuzzyInferenceOptimized",
            master_url=args.master,
            local_mode=args.local
        )
        benchmark.end("SPARK_SESSION_CREATION")
        
        # Process rasters with comprehensive benchmarking
        output_stats = process_rasters_spark_optimized(
            spark=spark,
            social_tiff=args.social_tiff,
            environmental_tiff=args.environmental_tiff,
            strategic_tiff=args.strategic_tiff,
            output_tiff=args.output_tiff,
            config_file=args.config,
            nodata_value=args.nodata,
            block_size=args.block_size,
            num_partitions=args.partitions,
            sample_rate=args.sample_rate,
            benchmark=benchmark
        )
        
        # Final summary
        logger.info("🎉 Processing completed successfully!")
        logger.info(f"📊 Final statistics: {output_stats}")
        
        # Cleanup
        spark.stop()
        
    except Exception as e:
        logger.error(f"❌ Processing failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 