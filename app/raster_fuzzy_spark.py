#!/usr/bin/env python3
"""
Spark-based Raster Fuzzy Inference System.
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

# Add the current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from raster_fuzzy_lib import UnifiedRasterFuzzyInferenceSystem, create_raster_config_template


def create_fuzzy_system_from_config(config: Dict):
    """
    Create a fuzzy inference system from configuration.
    This function will be serialized and sent to worker nodes.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Fuzzy inference system simulation object
    """
    import skfuzzy as fuzz
    from skfuzzy import control as ctrl
    import numpy as np
    
    # Create input variables with reduced resolution to save memory
    input_variables = {}
    for var_name, var_config in config['input_variables'].items():
        # Use very low resolution to save memory
        resolution = min(50, var_config.get('resolution', 50))
        universe = np.linspace(var_config['min'], var_config['max'], resolution)
        input_variables[var_name] = ctrl.Antecedent(universe, var_name)
        
        # Add membership functions
        for mf_name, mf_config in var_config['membership_functions'].items():
            if mf_config['type'] == 'trapmf':
                input_variables[var_name][mf_name] = fuzz.trapmf(
                    input_variables[var_name].universe, mf_config['params']
                )
            elif mf_config['type'] == 'trimf':
                input_variables[var_name][mf_name] = fuzz.trimf(
                    input_variables[var_name].universe, mf_config['params']
                )
    
    # Create output variable
    output_config = config['output_variable']
    resolution = min(50, output_config.get('resolution', 50))
    universe = np.linspace(output_config['min'], output_config['max'], resolution)
    output_variable = ctrl.Consequent(universe, output_config['name'])
    
    # Add output membership functions
    for mf_name, mf_config in output_config['membership_functions'].items():
        if mf_config['type'] == 'trapmf':
            output_variable[mf_name] = fuzz.trapmf(
                output_variable.universe, mf_config['params']
            )
        elif mf_config['type'] == 'trimf':
            output_variable[mf_name] = fuzz.trimf(
                output_variable.universe, mf_config['params']
            )
    
    # Create rules (simplified for memory efficiency)
    rules = []
    for rule_config in config['rules']:
        antecedent = rule_config['antecedent']
        consequent = rule_config['consequent']
        
        # Build antecedent conditions
        antecedent_conditions = []
        for condition in antecedent:
            var_name = condition['variable']
            membership = condition['membership']
            antecedent_conditions.append(input_variables[var_name][membership])
        
        # Create rule with AND logic for all conditions
        if len(antecedent_conditions) == 1:
            rule = ctrl.Rule(antecedent_conditions[0], output_variable[consequent])
        else:
            rule = ctrl.Rule(antecedent_conditions[0] & antecedent_conditions[1] & antecedent_conditions[2], 
                           output_variable[consequent])
        
        rules.append(rule)
    
    # Create control system
    control_system = ctrl.ControlSystem(rules)
    simulation = ctrl.ControlSystemSimulation(control_system)
    
    return simulation, list(config['input_variables'].keys()), output_config['name']


def process_pixel_chunk_standalone(chunk_data):
    """
    Standalone function to process a chunk of pixels using fuzzy logic.
    This function will be serialized and sent to worker nodes.
    
    Args:
        chunk_data: Tuple of (config, chunk_pixels) where chunk_pixels is a list of (row, col, social_val, env_val, strat_val) tuples
        
    Returns:
        List of (row, col, output_value) tuples
    """
    # Import here to avoid serialization issues
    import skfuzzy as fuzz
    from skfuzzy import control as ctrl
    import numpy as np
    
    config, chunk_pixels = chunk_data
    
    try:
        # Create fuzzy system from config
        simulation, input_vars, output_var_name = create_fuzzy_system_from_config(config)
        
        results = []
        
        for row, col, social_val, env_val, strat_val in chunk_pixels:
            try:
                # Set input values for fuzzy system
                simulation.input[input_vars[0]] = float(social_val)
                simulation.input[input_vars[1]] = float(env_val)
                simulation.input[input_vars[2]] = float(strat_val)
                
                # Compute output
                simulation.compute()
                output_value = float(simulation.output[output_var_name])
                
                results.append((row, col, output_value))
                
            except Exception as e:
                # If there's an error, skip this pixel
                print(f"Error processing pixel ({row}, {col}): {e}")
                continue
        
        return results
        
    except Exception as e:
        print(f"Error creating fuzzy system: {e}")
        return []


class SparkRasterFuzzyInferenceSystem:
    """
    Spark-based fuzzy inference system for distributed raster processing.
    Uses Spark to distribute raster chunks across cluster nodes.
    """
    
    def __init__(self, spark_session, config_file_path: str):
        """
        Initialize the Spark-based raster fuzzy inference system.
        
        Args:
            spark_session: Active Spark session
            config_file_path: Path to the JSON configuration file
        """
        self.spark = spark_session
        self.config_file_path = config_file_path
        self.config = self._load_config()
        
        # Create a fuzzy system instance for serialization
        self.fuzzy_system = UnifiedRasterFuzzyInferenceSystem(config_file_path)
    
    def _load_config(self) -> Dict:
        """Load configuration from JSON file."""
        try:
            with open(self.config_file_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file {self.config_file_path} not found.")
        except json.JSONDecodeError as e:
            raise ValueError(f"Error parsing JSON configuration: {e}")
    
    def _read_raster_to_dataframe(self, raster_path: str, chunk_size: int = 1000) -> 'DataFrame':
        """
        Read a raster file and convert it to a Spark DataFrame with pixel coordinates and values.
        
        Args:
            raster_path: Path to the raster file
            chunk_size: Size of chunks for processing
            
        Returns:
            Spark DataFrame with columns: row, col, value
        """
        from pyspark.sql.types import StructType, StructField, IntegerType, FloatType
        
        # Read raster metadata
        with rasterio.open(raster_path) as src:
            raster_data = src.read(1)
            profile = src.profile
            nodata = profile.get('nodata', 5.0)
        
        # Convert raster to list of (row, col, value) tuples
        rows, cols = raster_data.shape
        pixel_data = []
        
        for row in range(rows):
            for col in range(cols):
                value = float(raster_data[row, col])
                if value != nodata and not np.isnan(value):
                    pixel_data.append((row, col, value))
        
        # Create DataFrame schema
        schema = StructType([
            StructField("row", IntegerType(), False),
            StructField("col", IntegerType(), False),
            StructField("value", FloatType(), False)
        ])
        
        # Create DataFrame
        df = self.spark.createDataFrame(pixel_data, schema)
        return df
    
    def _process_pixel_chunk(self, chunk_data):
        """
        Process a chunk of pixels using fuzzy logic.
        This function will be serialized and sent to worker nodes.
        
        Args:
            chunk_data: Tuple of (config, chunk_pixels) where chunk_pixels is a list of (row, col, social_val, env_val, strat_val) tuples
            
        Returns:
            List of (row, col, output_value) tuples
        """
        # Use the standalone function for consistency
        return process_pixel_chunk_standalone(chunk_data)
    
    def process_rasters_spark(self, 
                             social_tiff: str, 
                             environmental_tiff: str, 
                             strategic_tiff: str, 
                             output_tiff: str,
                             nodata_value: float = 5.0,
                             chunk_size: int = 1000,
                             num_partitions: int = None) -> None:
        """
        Process three input TIFF files using Spark and output a single TIFF file.
        
        Args:
            social_tiff: Path to social factor TIFF file
            environmental_tiff: Path to environmental factor TIFF file
            strategic_tiff: Path to strategic factor TIFF file
            output_tiff: Path for output TIFF file
            nodata_value: Value to use for NoData pixels
            chunk_size: Number of pixels per chunk for processing
            num_partitions: Number of Spark partitions (default: auto)
        """
        print("Starting Spark-based raster processing...")
        start_time = time.time()
        
        try:
            # Read input rasters to DataFrames
            print("Reading input rasters...")
            social_df = self._read_raster_to_dataframe(social_tiff, chunk_size)
            env_df = self._read_raster_to_dataframe(environmental_tiff, chunk_size)
            strat_df = self._read_raster_to_dataframe(strategic_tiff, chunk_size)
            
            print(f"Social DataFrame: {social_df.count()} rows")
            print(f"Environmental DataFrame: {env_df.count()} rows")
            print(f"Strategic DataFrame: {strat_df.count()} rows")
            
            # Get raster dimensions for output
            with rasterio.open(social_tiff) as src:
                profile = src.profile.copy()
                rows, cols = src.shape
            
            print(f"Raster dimensions: {rows} rows x {cols} columns")
            
            # Join the three DataFrames on row and col
            print("Joining raster data...")
            
            # Rename columns before joining to avoid conflicts
            social_df_renamed = social_df.withColumnRenamed("value", "social_val")
            env_df_renamed = env_df.withColumnRenamed("value", "env_val")
            strat_df_renamed = strat_df.withColumnRenamed("value", "strat_val")
            
            joined_df = social_df_renamed.join(env_df_renamed, ["row", "col"], "inner") \
                                .join(strat_df_renamed, ["row", "col"], "inner")
            
            print(f"Joined DataFrame: {joined_df.count()} rows")
            print(f"Joined DataFrame columns: {joined_df.columns}")
            
            # Repartition if specified
            if num_partitions:
                joined_df = joined_df.repartition(num_partitions)
            
            # Convert to RDD for distributed processing
            print("Converting to RDD for distributed processing...")
            pixel_rdd = joined_df.rdd.map(lambda row: (
                row.row, row.col, row.social_val, row.env_val, row.strat_val
            ))
            
            # Group pixels into chunks
            print("Grouping pixels into chunks...")
            chunked_rdd = pixel_rdd.mapPartitions(lambda partition: self._chunk_partition(partition, chunk_size))
            
            # Process chunks using fuzzy logic with config
            print("Processing chunks with fuzzy logic...")
            processed_rdd = chunked_rdd.map(lambda chunk: (self.config, chunk))
            processed_rdd = processed_rdd.map(self._process_pixel_chunk)
            
            # Flatten results
            print("Flattening results...")
            result_rdd = processed_rdd.flatMap(lambda x: x)
            
            # Collect results
            print("Collecting results...")
            results = result_rdd.collect()
            
            print(f"Processed {len(results)} pixels")
            
            # Create output raster
            print("Creating output raster...")
            output_data = np.full((rows, cols), nodata_value, dtype=np.float32)
            
            for row, col, value in results:
                output_data[row, col] = value
            
            # Update profile for output
            profile.update(
                dtype=np.float32,
                count=1,
                nodata=nodata_value
            )
            
            # Write output raster
            print(f"Writing output raster to: {output_tiff}")
            with rasterio.open(output_tiff, 'w', **profile) as dst:
                dst.write(output_data, 1)
            
            processing_time = time.time() - start_time
            print(f"Spark processing completed in {processing_time:.2f} seconds")
            print(f"Processing complete. Output saved to: {output_tiff}")
            print(f"Output value range: {np.nanmin(output_data):.2f} to {np.nanmax(output_data):.2f}")
            
        except Exception as e:
            print(f"Error during Spark processing: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _chunk_partition(self, partition, chunk_size):
        """
        Group pixels in a partition into chunks.
        
        Args:
            partition: Iterator of (row, col, social_val, env_val, strat_val) tuples
            chunk_size: Size of each chunk
            
        Yields:
            Lists of pixel tuples
        """
        chunk = []
        for pixel in partition:
            chunk.append(pixel)
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        
        # Yield remaining pixels
        if chunk:
            yield chunk


def create_spark_session(app_name: str = "RasterFuzzyInference", 
                        master_url: str = "spark://spark-master:7077",
                        local_mode: bool = False) -> 'SparkSession':
    """
    Create a Spark session for raster processing.
    
    Args:
        app_name: Name of the Spark application
        master_url: URL of the Spark master
        local_mode: Whether to run in local mode (for testing)
        
    Returns:
        SparkSession object
    """
    from pyspark.sql import SparkSession
    
    if local_mode:
        # Local mode for testing with very low memory
        spark = SparkSession.builder \
            .appName(app_name) \
            .master("local[2]") \
            .config("spark.driver.memory", "512m") \
            .config("spark.executor.memory", "512m") \
            .config("spark.sql.shuffle.partitions", "5") \
            .getOrCreate()
    else:
        # Cluster mode with memory management
        spark = SparkSession.builder \
            .appName(app_name) \
            .master(master_url) \
            .config("spark.executor.memory", "1g") \
            .config("spark.driver.memory", "1g") \
            .config("spark.executor.cores", "1") \
            .config("spark.sql.shuffle.partitions", "10") \
            .getOrCreate()
    
    return spark


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Spark-based Raster Fuzzy Inference System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Local mode testing
  python raster_fuzzy_spark.py social.tif env.tif strat.tif output.tif --config config.json --local

  # Cluster mode
  python raster_fuzzy_spark.py social.tif env.tif strat.tif output.tif --config config.json --master spark://spark-master:7077
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
    parser.add_argument('--chunk-size', type=int, default=1000,
                       help='Number of pixels per chunk (default: 1000)')
    parser.add_argument('--partitions', type=int, default=None,
                       help='Number of Spark partitions (default: auto)')
    parser.add_argument('--local', action='store_true',
                       help='Run in local mode for testing')
    parser.add_argument('--master', default='spark://spark-master:7077',
                       help='Spark master URL (default: spark://spark-master:7077)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Check if input files exist
    for tiff_file in [args.social_tiff, args.environmental_tiff, args.strategic_tiff]:
        if not os.path.exists(tiff_file):
            print(f"Error: Input file not found: {tiff_file}")
            sys.exit(1)
    
    # Check if config file exists
    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found: {args.config}")
        sys.exit(1)
    
    # Check if output directory exists
    output_dir = os.path.dirname(args.output_tiff)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        # Create Spark session
        print(f"Creating Spark session...")
        spark = create_spark_session(
            app_name="RasterFuzzyInference",
            master_url=args.master,
            local_mode=args.local
        )
        
        # Initialize the Spark-based fuzzy inference system
        print(f"Loading configuration from: {args.config}")
        spark_fis = SparkRasterFuzzyInferenceSystem(spark, args.config)
        
        # Process the rasters
        print(f"Processing rasters with Spark...")
        print(f"  Social: {args.social_tiff}")
        print(f"  Environmental: {args.environmental_tiff}")
        print(f"  Strategic: {args.strategic_tiff}")
        print(f"  Output: {args.output_tiff}")
        print(f"  Config: {args.config}")
        print(f"  NoData value: {args.nodata}")
        print(f"  Chunk size: {args.chunk_size} pixels")
        print(f"  Partitions: {args.partitions or 'auto'}")
        print(f"  Mode: {'Local' if args.local else 'Cluster'}")
        
        spark_fis.process_rasters_spark(
            social_tiff=args.social_tiff,
            environmental_tiff=args.environmental_tiff,
            strategic_tiff=args.strategic_tiff,
            output_tiff=args.output_tiff,
            nodata_value=args.nodata,
            chunk_size=args.chunk_size,
            num_partitions=args.partitions
        )
        
        print("Spark processing completed successfully!")
        
        # Stop Spark session
        spark.stop()
        
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 