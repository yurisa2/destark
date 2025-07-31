#!/usr/bin/env python3
import numpy as np
import rasterio
import boto3
import json
import skfuzzy as fuzz
from skfuzzy import control as ctrl

def create_fuzzy_system_from_config(config):
    input_variables = {}
    for var_name, var_config in config["input_variables"].items():
        resolution = min(50, var_config.get("resolution", 50))
        universe = np.linspace(var_config["min"], var_config["max"], resolution)
        input_variables[var_name] = ctrl.Antecedent(universe, var_name)
        for mf_name, mf_config in var_config["membership_functions"].items():
            if mf_config["type"] == "trapmf":
                input_variables[var_name][mf_name] = fuzz.trapmf(input_variables[var_name].universe, mf_config["params"])
            elif mf_config["type"] == "trimf":
                input_variables[var_name][mf_name] = fuzz.trimf(input_variables[var_name].universe, mf_config["params"])
    
    output_config = config["output_variable"]
    resolution = min(50, output_config.get("resolution", 50))
    universe = np.linspace(output_config["min"], output_config["max"], resolution)
    output_variable = ctrl.Consequent(universe, output_config["name"])
    for mf_name, mf_config in output_config["membership_functions"].items():
        if mf_config["type"] == "trapmf":
            output_variable[mf_name] = fuzz.trapmf(output_variable.universe, mf_config["params"])
        elif mf_config["type"] == "trimf":
            output_variable[mf_name] = fuzz.trimf(output_variable.universe, mf_config["params"])
    
    rules = []
    for rule_config in config["rules"]:
        antecedent = rule_config["antecedent"]
        consequent = rule_config["consequent"]
        antecedent_conditions = []
        for condition in antecedent:
            var_name = condition["variable"]
            membership = condition["membership"]
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
    return (simulation, list(config["input_variables"].keys()), output_config["name"])

def process_block_fis(config, social_block, env_block, strat_block, nodata_value=5.0):
    try:
        simulation, input_vars, output_var = create_fuzzy_system_from_config(config)
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
                result_block[i, j] = float(simulation.output[output_var])
        
        return result_block
    except Exception as e:
        print(f"Error in FIS processing: {e}")
        return np.full((rows, cols), nodata_value, dtype=np.float32)

def process_single_config(config_name):
    print(f"\n=== Processing {config_name} with REAL FIS ===")
    s3 = boto3.client("s3")
    bucket = "<AWS-BUCKET>-unifile"
    prefix = "unifile_test/"
    
    print(f"Downloading {config_name}...")
    s3.download_file(bucket, prefix + config_name, f"/tmp/{config_name}")
    
    with open(f"/tmp/{config_name}", "r") as f:
        config = json.load(f)
    
    print(f"Loaded FIS config: {config_name}")
    print("Reading input files...")
    
    with rasterio.open("/tmp/social.tif") as src:
        social_data = src.read(1)
        profile = src.profile
        height, width = social_data.shape
    
    with rasterio.open("/tmp/environmental.tif") as src:
        env_data = src.read(1)
    
    with rasterio.open("/tmp/strategic.tif") as src:
        strat_data = src.read(1)
    
    print(f"Processing {height}x{width} raster with FIS...")
    block_size = 500
    result_data = np.full((height, width), 5.0, dtype=np.float32)
    
    for start_row in range(0, height, block_size):
        end_row = min(start_row + block_size, height)
        print(f"Processing rows {start_row}-{end_row-1}...")
        
        social_block = social_data[start_row:end_row, :]
        env_block = env_data[start_row:end_row, :]
        strat_block = strat_data[start_row:end_row, :]
        
        result_block = process_block_fis(config, social_block, env_block, strat_block)
        result_data[start_row:end_row, :] = result_block
    
    config_base = config_name.replace(".json", "")
    output_name = f"result_{config_base}.tif"
    output_path = f"/tmp/{output_name}"
    
    print(f"Creating {output_name}...")
    profile.update(dtype=np.float32, count=1)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(result_data, 1)
    
    print(f"Uploading {output_name}...")
    s3.upload_file(output_path, bucket, prefix + output_name)
    print(f"Completed {config_name} -> {output_name}")
    return True

def main():
    print("Input files downloaded")
    configs = ["config_median.json", "config_minimum.json", "config_mode.json", "config_round_down.json", "config_round_up.json"]
    print(f"Processing {len(configs)} configs with REAL FIS logic...")
    
    for i, config in enumerate(configs, 1):
        print(f"\n--- Config {i}/{len(configs)} ---")
        try:
            process_single_config(config)
            print(f"SUCCESS: {config}")
        except Exception as e:
            print(f"ERROR: {config} - {e}")
            continue
    
    print("\n=== FULL FIS PROCESSING COMPLETED ===")

if __name__ == "__main__":
    main() 