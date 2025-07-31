#!/usr/bin/env python3
"""
Spark FIS Processing - Local Version
Runs on local Spark cluster (single node or multi-node)
"""

import os
import sys
import json
import numpy as np
import boto3

# Set environment to use local Spark
os.environ['SPARK_HOME'] = '/usr/lib/spark'
os.environ['PYSPARK_PYTHON'] = '/usr/bin/python3'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/bin/python3'

# Add Spark to Python path
sys.path.insert(0, '/usr/lib/spark/python')
sys.path.insert(0, '/usr/lib/spark/python/lib/py4j-0.10.9-src.zip')

# Import packages
import rasterio
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from pyspark.sql import SparkSession
from pyspark import SparkContext

def create_fuzzy_system(config):
    """Create fuzzy inference system from configuration"""
    resolution = 25
    input_vars = {}
    
    # Create input variables
    for var_name, var_config in config["input_variables"].items():
        universe = np.linspace(var_config["min"], var_config["max"], resolution)
        input_vars[var_name] = ctrl.Antecedent(universe, var_name)
        
        # Add membership functions
        for mf_name, mf_config in var_config["membership_functions"].items():
            if mf_config["type"] == "trapmf":
                input_vars[var_name][mf_name] = fuzz.trapmf(input_vars[var_name].universe, mf_config["params"])
            elif mf_config["type"] == "trimf":
                input_vars[var_name][mf_name] = fuzz.trimf(input_vars[var_name].universe, mf_config["params"])
    
    # Create output variable
    output_config = config["output_variable"]
    universe = np.linspace(output_config["min"], output_config["max"], resolution)
    output_var = ctrl.Consequent(universe, output_config["name"])
    
    for mf_name, mf_config in output_config["membership_functions"].items():
        if mf_config["type"] == "trapmf":
            output_var[mf_name] = fuzz.trapmf(output_var.universe, mf_config["params"])
        elif mf_config["type"] == "trimf":
            output_var[mf_name] = fuzz.trimf(output_var.universe, mf_config["params"])
    
    # Create rules
    rules = []
    for rule_config in config["rules"]:
        antecedent = rule_config["antecedent"]
        consequent = rule_config["consequent"]
        antecedent_conditions = []
        
        for condition in antecedent:
            var_name = condition["variable"]
            membership = condition["membership"]
            antecedent_conditions.append(input_vars[var_name][membership])
        
        if len(antecedent_conditions) == 1:
            rule = ctrl.Rule(antecedent_conditions[0], output_var[consequent])
        else:
            combined = antecedent_conditions[0]
            for condition in antecedent_conditions[1:]:
                combined = combined & condition
            rule = ctrl.Rule(combined, output_var[consequent])
        
        rules.append(rule)
    
    # Create control system
    control_system = ctrl.ControlSystem(rules)
    simulation = ctrl.ControlSystemSimulation(control_system)
    
    return simulation, list(config["input_variables"].keys()), output_config["name"]

def process_small_block_spark(chunk_data):
    """Process a small block of raster data with FIS"""
    config, start_row, end_row, start_col, end_col, social_data, env_data, strat_data, nodata_value = chunk_data
    
    simulation, input_vars, output_var_name = create_fuzzy_system(config)
    
    social_block = social_data[start_row:end_row, start_col:end_col]
    env_block = env_data[start_row:end_row, start_col:end_col]
    strat_block = strat_data[start_row:end_row, start_col:end_col]
    
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
    
    return (start_row, end_row, start_col, end_col, result_block)

def process_config_spark(config_name):
    """Process a single FIS configuration with Spark"""
    print(f"Processing {config_name} with local Spark...")
    
    # Download config and input files
    s3 = boto3.client("s3")
    bucket = "<AWS-BUCKET>-unifile"
    prefix = "unifile_test/"
    
    # Download config
    s3.download_file(bucket, prefix + config_name, f"/tmp/{config_name}")
    with open(f"/tmp/{config_name}", "r") as f:
        config = json.load(f)
    
    # Download input files
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
    
    print(f"Processing {height}x{width} raster with local Spark...")
    
    # Create Spark session for local processing
    spark = SparkSession.builder \
        .appName(f"FIS-Local-{config_name}") \
        .master("local[*]") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .config("spark.kryo.registrationRequired", "false") \
        .config("spark.driver.memory", "4g") \
        .config("spark.driver.maxResultSize", "1g") \
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
    
    print(f"Created {len(chunks)} blocks for local processing")
    
    # Distribute processing across local cores
    num_partitions = min(8, len(chunks))  # Use up to 8 partitions for local processing
    rdd = sc.parallelize(chunks, numSlices=num_partitions)
    results = rdd.map(process_small_block_spark).collect()
    
    # Reconstruct result
    result_data = np.full((height, width), 5.0, dtype=np.float32)
    for start_row, end_row, start_col, end_col, result_block in results:
        result_data[start_row:end_row, start_col:end_col] = result_block
    
    # Save result
    config_base = config_name.replace(".json", "")
    output_name = f"result_local_{config_base}.tif"
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
    print("=== LOCAL SPARK FIS PROCESSING STARTED ===")
    print("This will process all configurations using local Spark")
    
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
    
    print("\n=== LOCAL SPARK FIS PROCESSING COMPLETED ===")

if __name__ == "__main__":
    main() 