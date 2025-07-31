#!/usr/bin/env python3
import os
import sys
import json
import numpy as np
import rasterio
import boto3
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from pyspark.sql import SparkSession
from pyspark import SparkContext

def create_fuzzy_system(config):
    """Create fuzzy system from config."""
    resolution = 25  # Fast resolution
    input_vars = {}
    
    for var_name, var_config in config['input_variables'].items():
        universe = np.linspace(var_config['min'], var_config['max'], resolution)
        input_vars[var_name] = ctrl.Antecedent(universe, var_name)
        
        for mf_name, mf_config in var_config['membership_functions'].items():
            if mf_config['type'] == 'trapmf':
                input_vars[var_name][mf_name] = fuzz.trapmf(input_vars[var_name].universe, mf_config['params'])
            elif mf_config['type'] == 'trimf':
                input_vars[var_name][mf_name] = fuzz.trimf(input_vars[var_name].universe, mf_config['params'])
    
    output_config = config['output_variable']
    universe = np.linspace(output_config['min'], output_config['max'], resolution)
    output_var = ctrl.Consequent(universe, output_config['name'])
    
    for mf_name, mf_config in output_config['membership_functions'].items():
        if mf_config['type'] == 'trapmf':
            output_var[mf_name] = fuzz.trapmf(output_var.universe, mf_config['params'])
        elif mf_config['type'] == 'trimf':
            output_var[mf_name] = fuzz.trimf(output_var.universe, mf_config['params'])
    
    rules = []
    for rule_config in config['rules']:
        antecedent = rule_config['antecedent']
        consequent = rule_config['consequent']
        
        antecedent_conditions = []
        for condition in antecedent:
            var_name = condition['variable']
            membership = condition['membership']
            antecedent_conditions.append(input_vars[var_name][membership])
        
        if len(antecedent_conditions) == 1:
            rule = ctrl.Rule(antecedent_conditions[0], output_var[consequent])
        else:
            combined = antecedent_conditions[0]
            for condition in antecedent_conditions[1:]:
                combined = combined & condition
            rule = ctrl.Rule(combined, output_var[consequent])
        rules.append(rule)
    
    control_system = ctrl.ControlSystem(rules)
    simulation = ctrl.ControlSystemSimulation(control_system)
    return simulation, list(config['input_variables'].keys()), output_config['name']

def process_block_spark(chunk_data):
    """Process a block with Spark."""
    config, social_block, env_block, strat_block, block_start_row, nodata_value = chunk_data
    
    simulation, input_vars, output_var_name = create_fuzzy_system(config)
    rows, cols = social_block.shape
    result_block = np.full((rows, cols), nodata_value, dtype=np.float32)
    
    for i in range(rows):
        for j in range(cols):
            if (social_block[i, j] == nodata_value or 
                env_block[i, j] == nodata_value or 
                strat_block[i, j] == nodata_value):
                continue
            
            simulation.input[input_vars[0]] = float(social_block[i, j])
            simulation.input[input_vars[1]] = float(env_block[i, j])
            simulation.input[input_vars[2]] = float(strat_block[i, j])
            simulation.compute()
            result_block[i, j] = float(simulation.output[output_var_name])
    
    return (block_start_row, result_block)

def process_config_spark(config_name):
    """Process a single config with Spark."""
    print(f"\n=== Processing {config_name} with Spark FIS ===")
    
    s3 = boto3.client("s3")
    bucket = "<AWS-BUCKET>-unifile"
    prefix = "unifile_test/"
    
    # Download config
    s3.download_file(bucket, prefix + config_name, f"/tmp/{config_name}")
    with open(f"/tmp/{config_name}", "r") as f:
        config = json.load(f)
    
    # Read input files
    with rasterio.open("/tmp/social.tif") as src:
        social_data = src.read(1)
        profile = src.profile
        height, width = social_data.shape
    
    with rasterio.open("/tmp/environmental.tif") as src:
        env_data = src.read(1)
    
    with rasterio.open("/tmp/strategic.tif") as src:
        strat_data = src.read(1)
    
    print(f"Processing {height}x{width} raster with Spark FIS...")
    
    # Create Spark session
    spark = SparkSession.builder \
        .appName(f"FIS-{config_name}") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.sql.adaptive.skewJoin.enabled", "true") \
        .getOrCreate()
    
    sc = spark.sparkContext
    
    # Prepare data for Spark
    block_size = 1000
    chunks = []
    
    for start_row in range(0, height, block_size):
        end_row = min(start_row + block_size, height)
        social_block = social_data[start_row:end_row, :]
        env_block = env_data[start_row:end_row, :]
        strat_block = strat_data[start_row:end_row, :]
        
        chunks.append((config, social_block, env_block, strat_block, start_row, 5.0))
    
    # Process with Spark
    rdd = sc.parallelize(chunks, numSlices=len(chunks))
    results = rdd.map(process_block_spark).collect()
    
    # Combine results
    result_data = np.full((height, width), 5.0, dtype=np.float32)
    for start_row, result_block in results:
        end_row = min(start_row + result_block.shape[0], height)
        result_data[start_row:end_row, :] = result_block
    
    # Save result
    config_base = config_name.replace(".json", "")
    output_name = f"result_spark_{config_base}.tif"
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
    print("=== SPARK FIS PROCESSING STARTED ===")
    
    configs = ["config_median.json", "config_minimum.json", "config_mode.json", "config_round_down.json", "config_round_up.json"]
    
    for i, config in enumerate(configs, 1):
        print(f"\n--- Config {i}/{len(configs)} ---")
        try:
            process_config_spark(config)
            print(f"SUCCESS: {config}")
        except Exception as e:
            print(f"ERROR: {config} - {e}")
            continue
    
    print("\n=== SPARK FIS PROCESSING COMPLETED ===")

if __name__ == "__main__":
    main() 