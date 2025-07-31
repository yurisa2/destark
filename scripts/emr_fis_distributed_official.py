#!/usr/bin/env python3
"""
EMR FIS Processing - Official EMR Image Version
Runs distributed Spark processing with S3 integration using official EMR image
"""

import os
import sys
import json
import numpy as np
import boto3
import rasterio
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from pyspark.sql import SparkSession
from pyspark import SparkContext

def create_fuzzy_system(config):
    """Create fuzzy inference system from configuration"""
    # Define input variables
    social = ctrl.Antecedent(np.arange(0, 11, 1), 'social')
    environmental = ctrl.Antecedent(np.arange(0, 11, 1), 'environmental')
    strategic = ctrl.Antecedent(np.arange(0, 11, 1), 'strategic')
    
    # Define output variable
    output = ctrl.Consequent(np.arange(0, 11, 1), 'output')
    
    # Define membership functions
    social['low'] = fuzz.trimf(social.universe, [0, 0, 5])
    social['medium'] = fuzz.trimf(social.universe, [0, 5, 10])
    social['high'] = fuzz.trimf(social.universe, [5, 10, 10])
    
    environmental['low'] = fuzz.trimf(environmental.universe, [0, 0, 5])
    environmental['medium'] = fuzz.trimf(environmental.universe, [0, 5, 10])
    environmental['high'] = fuzz.trimf(environmental.universe, [5, 10, 10])
    
    strategic['low'] = fuzz.trimf(strategic.universe, [0, 0, 5])
    strategic['medium'] = fuzz.trimf(strategic.universe, [0, 5, 10])
    strategic['high'] = fuzz.trimf(strategic.universe, [5, 10, 10])
    
    output['low'] = fuzz.trimf(output.universe, [0, 0, 5])
    output['medium'] = fuzz.trimf(output.universe, [0, 5, 10])
    output['high'] = fuzz.trimf(output.universe, [5, 10, 10])
    
    # Define rules based on config
    rules = []
    
    if config.get('aggregation_method') == 'median':
        # Median-based rules
        rules.append(ctrl.Rule(social['low'] & environmental['low'] & strategic['low'], output['low']))
        rules.append(ctrl.Rule(social['medium'] & environmental['medium'] & strategic['medium'], output['medium']))
        rules.append(ctrl.Rule(social['high'] & environmental['high'] & strategic['high'], output['high']))
    elif config.get('aggregation_method') == 'minimum':
        # Minimum-based rules
        rules.append(ctrl.Rule(social['low'] | environmental['low'] | strategic['low'], output['low']))
        rules.append(ctrl.Rule(social['medium'] & environmental['medium'] & strategic['medium'], output['medium']))
        rules.append(ctrl.Rule(social['high'] & environmental['high'] & strategic['high'], output['high']))
    else:
        # Default rules
        rules.append(ctrl.Rule(social['low'] & environmental['low'] & strategic['low'], output['low']))
        rules.append(ctrl.Rule(social['medium'] & environmental['medium'] & strategic['medium'], output['medium']))
        rules.append(ctrl.Rule(social['high'] & environmental['high'] & strategic['high'], output['high']))
    
    # Create control system
    ctrl_system = ctrl.ControlSystem(rules)
    simulation = ctrl.ControlSystemSimulation(ctrl_system)
    
    return simulation

def process_small_block_spark(chunk_data):
    """Process a small block of data using fuzzy logic"""
    config, start_row, end_row, start_col, end_col, social_block, env_block, strat_block, default_value = chunk_data
    
    try:
        # Create fuzzy system
        simulation = create_fuzzy_system(config)
        
        # Process each pixel in the block
        result_block = np.full(social_block.shape, default_value, dtype=np.float32)
        
        for i in range(social_block.shape[0]):
            for j in range(social_block.shape[1]):
                try:
                    # Get input values
                    social_val = float(social_block[i, j])
                    env_val = float(env_block[i, j])
                    strat_val = float(strat_block[i, j])
                    
                    # Skip if any value is NaN or invalid
                    if np.isnan(social_val) or np.isnan(env_val) or np.isnan(strat_val):
                        continue
                    
                    # Set input values
                    simulation.social = social_val
                    simulation.environmental = env_val
                    simulation.strategic = strat_val
                    
                    # Compute
                    simulation.compute()
                    
                    # Get result
                    result_block[i, j] = simulation.output
                    
                except Exception as e:
                    # If fuzzy computation fails, use default value
                    result_block[i, j] = default_value
                    continue
        
        return (start_row, end_row, start_col, end_col, result_block)
        
    except Exception as e:
        print(f"Error processing block: {e}")
        # Return default values if processing fails
        result_block = np.full(social_block.shape, default_value, dtype=np.float32)
        return (start_row, end_row, start_col, end_col, result_block)

def process_config_spark(config_name):
    """Process a single FIS configuration with distributed Spark"""
    print(f"Processing {config_name} with distributed Spark...")
    
    # Download config and input files from S3
    s3 = boto3.client("s3")
    bucket = "<AWS-BUCKET>-unifile"
    prefix = "unifile_test/"
    
    # Download config
    s3.download_file(bucket, prefix + config_name, f"/tmp/{config_name}")
    with open(f"/tmp/{config_name}", "r") as f:
        config = json.load(f)
    
    # Download input files from S3
    s3.download_file(bucket, prefix + "so300m.in", "/tmp/social.tif")
    s3.download_file(bucket, prefix + "e300m.in", "/tmp/environmental.tif")
    s3.download_file(bucket, prefix + "s300m.in", "/tmp/strategic.tif")
    
    # Read raster data
    with rasterio.open("/tmp/social.tif") as src:
        social_data = src.read(1)
        profile = src.profile
        height, width = social_data.shape
    
    with rasterio.open("/tmp/environmental.tif") as src:
        env_data = src.read(1)
    
    with rasterio.open("/tmp/strategic.tif") as src:
        strat_data = src.read(1)
    
    print(f"Processing {height}x{width} raster with distributed Spark...")
    
    # Create Spark session for distributed processing
    # Use 'spark://localhost:7077' for distributed mode in container
    spark = SparkSession.builder \
        .appName(f"FIS-Distributed-{config_name}") \
        .master("spark://localhost:7077") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .config("spark.kryo.registrationRequired", "false") \
        .config("spark.driver.memory", "4g") \
        .config("spark.driver.maxResultSize", "1g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.executor.cores", "2") \
        .config("spark.dynamicAllocation.enabled", "true") \
        .config("spark.dynamicAllocation.minExecutors", "1") \
        .config("spark.dynamicAllocation.maxExecutors", "4") \
        .getOrCreate()
    
    sc = spark.sparkContext
    
    # Create small blocks for distributed processing
    block_size = 100
    chunks = []
    
    for start_row in range(0, height, block_size):
        end_row = min(start_row + block_size, height)
        for start_col in range(0, width, block_size):
            end_col = min(start_col + block_size, width)
            
            social_block = social_data[start_row:end_row, start_col:end_col]
            env_block = env_data[start_row:end_row, start_col:end_col]
            strat_block = strat_data[start_row:end_row, start_col:end_col]
            
            chunks.append((config, start_row, end_row, start_col, end_col, 
                         social_block, env_block, strat_block, 5.0))
    
    print(f"Created {len(chunks)} blocks for distributed processing")
    
    # Distribute processing across cluster nodes
    num_partitions = min(50, len(chunks))  # Use fewer partitions for container testing
    rdd = sc.parallelize(chunks, numSlices=num_partitions)
    
    print(f"Distributing {num_partitions} partitions across cluster...")
    results = rdd.map(process_small_block_spark).collect()
    
    # Reconstruct result
    result_data = np.full((height, width), 5.0, dtype=np.float32)
    for start_row, end_row, start_col, end_col, result_block in results:
        result_data[start_row:end_row, start_col:end_col] = result_block
    
    # Save result
    config_base = config_name.replace(".json", "")
    output_name = f"result_official_{config_base}.tif"
    output_path = f"/tmp/{output_name}"
    
    profile.update(dtype=np.float32, count=1)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(result_data, 1)
    
    # Upload to S3
    s3.upload_file(output_path, bucket, prefix + output_name)
    print(f"Completed {config_name} -> {output_name}")
    
    spark.stop()
    return True

def main():
    """Main function to process all FIS configurations"""
    print("=== OFFICIAL EMR DISTRIBUTED SPARK FIS PROCESSING STARTED ===")
    print("This will process all configurations using distributed Spark on official EMR image")
    
    # Process all configurations
    configs = [
        "config_median.json",
        "config_minimum.json", 
        "config_mode.json",
        "config_round_down.json",
        "config_round_up.json"
    ]
    
    for i, config in enumerate(configs, 1):
        print(f"\n--- Processing Config {i}/{len(configs)}: {config} ---")
        try:
            process_config_spark(config)
            print(f"✅ SUCCESS: {config}")
        except Exception as e:
            print(f"❌ ERROR: {config} - {e}")
            continue
    
    print("\n=== OFFICIAL EMR DISTRIBUTED SPARK FIS PROCESSING COMPLETED ===")

if __name__ == "__main__":
    main() 