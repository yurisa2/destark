import rasterio
import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import json
from typing import Dict, List, Tuple, Optional
import os
from pathlib import Path
import time
from tqdm import tqdm


class RasterFuzzyInferenceSystem:
    """
    A fuzzy inference system designed to process raster TIFF files.
    Takes 3 input TIFF files (social, environmental, strategic) and outputs a single TIFF.
    """
    
    def __init__(self, config_file_path: str):
        """
        Initialize the raster fuzzy inference system.
        
        Args:
            config_file_path: Path to the JSON configuration file
        """
        self.config_file_path = config_file_path
        self.config = self._load_config()
        self.fuzzy_inputs = {}
        self.fuzzy_output = None
        self.control_system = None
        self.simulation = None
        
        self._create_fuzzy_variables()
        self._create_rules()
        self._create_control_system()
    
    def _load_config(self) -> Dict:
        """Load configuration from JSON file."""
        try:
            with open(self.config_file_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file {self.config_file_path} not found.")
        except json.JSONDecodeError as e:
            raise ValueError(f"Error parsing JSON configuration: {e}")
    
    def _create_fuzzy_variables(self):
        """Create fuzzy antecedent and consequent variables from configuration."""
        # Create input variables
        for var_name, var_config in self.config['input_variables'].items():
            universe = np.arange(var_config['min'], var_config['max'] + 1, var_config['step'])
            antecedent = ctrl.Antecedent(universe, var_name)
            
            # Create membership functions for each input variable
            for membership_name, membership_config in var_config['membership_functions'].items():
                if membership_config['type'] == 'trapmf':
                    antecedent[membership_name] = fuzz.trapmf(
                        antecedent.universe, 
                        membership_config['params']
                    )
                elif membership_config['type'] == 'trimf':
                    antecedent[membership_name] = fuzz.trimf(
                        antecedent.universe, 
                        membership_config['params']
                    )
            
            self.fuzzy_inputs[var_name] = antecedent
        
        # Create output variable
        output_config = self.config['output_variable']
        universe = np.arange(output_config['min'], output_config['max'] + 1, output_config['step'])
        self.fuzzy_output = ctrl.Consequent(universe, output_config['name'])
        
        # Create membership functions for output variable
        for membership_name, membership_config in output_config['membership_functions'].items():
            if membership_config['type'] == 'trapmf':
                self.fuzzy_output[membership_name] = fuzz.trapmf(
                    self.fuzzy_output.universe, 
                    membership_config['params']
                )
            elif membership_config['type'] == 'trimf':
                self.fuzzy_output[membership_name] = fuzz.trimf(
                    self.fuzzy_output.universe, 
                    membership_config['params']
                )
    
    def _create_rules(self):
        """Create fuzzy rules from the rule definitions."""
        self.fuzzy_rules = []
        
        for rule_config in self.config['rules']:
            # Build antecedent conditions
            antecedent_conditions = []
            for condition in rule_config['antecedent']:
                var_name = condition['variable']
                membership_name = condition['membership']
                antecedent_conditions.append(self.fuzzy_inputs[var_name][membership_name])
            
            # Combine conditions with AND operator
            if len(antecedent_conditions) == 1:
                antecedent = antecedent_conditions[0]
            else:
                antecedent = antecedent_conditions[0]
                for condition in antecedent_conditions[1:]:
                    antecedent = antecedent & condition
            
            # Create consequent
            consequent = self.fuzzy_output[rule_config['consequent']]
            
            # Create rule
            rule = ctrl.Rule(antecedent, consequent)
            self.fuzzy_rules.append(rule)
    
    def _create_control_system(self):
        """Create the control system and simulation."""
        self.control_system = ctrl.ControlSystem(self.fuzzy_rules)
        self.simulation = ctrl.ControlSystemSimulation(self.control_system)
    
    def process_rasters(self, 
                       social_tiff: str, 
                       environmental_tiff: str, 
                       strategic_tiff: str, 
                       output_tiff: str,
                       nodata_value: float = -9999.0) -> None:
        """
        Process three input TIFF files and output a single TIFF file.
        
        Args:
            social_tiff: Path to social factor TIFF file
            environmental_tiff: Path to environmental factor TIFF file
            strategic_tiff: Path to strategic factor TIFF file
            output_tiff: Path for output TIFF file
            nodata_value: Value to use for NoData pixels
        """
        # Read input rasters
        with rasterio.open(social_tiff) as social_src:
            social_data = social_src.read(1)
            profile = social_src.profile.copy()
        
        with rasterio.open(environmental_tiff) as env_src:
            environmental_data = env_src.read(1)
        
        with rasterio.open(strategic_tiff) as strat_src:
            strategic_data = strat_src.read(1)
        
        # Check if all rasters have the same dimensions
        if (social_data.shape != environmental_data.shape or 
            social_data.shape != strategic_data.shape):
            raise ValueError("All input rasters must have the same dimensions")
        
        # Create output array
        output_data = np.full(social_data.shape, nodata_value, dtype=np.float32)
        
        # Get input variable names from config
        input_vars = list(self.config['input_variables'].keys())
        
        # Process each pixel with progress tracking
        rows, cols = social_data.shape
        total_pixels = rows * cols
        
        print(f"Processing {total_pixels} pixels...")
        start_time = time.time()
        
        # Create a progress bar for pixel processing
        with tqdm(total=total_pixels, desc="Processing pixels", unit="pixel", ncols=80) as pbar:
            for row in range(rows):
                for col in range(cols):
                    # Get pixel values
                    social_val = social_data[row, col]
                    env_val = environmental_data[row, col]
                    strat_val = strategic_data[row, col]
                    
                    # Skip NoData pixels
                    if (social_val == profile['nodata'] or 
                        env_val == profile['nodata'] or 
                        strat_val == profile['nodata'] or
                        np.isnan(social_val) or np.isnan(env_val) or np.isnan(strat_val)):
                        pbar.update(1)
                        continue
                    
                    try:
                        # Set input values for fuzzy system
                        self.simulation.input[input_vars[0]] = float(social_val)
                        self.simulation.input[input_vars[1]] = float(env_val)
                        self.simulation.input[input_vars[2]] = float(strat_val)
                        
                        # Compute output
                        self.simulation.compute()
                        output_value = float(self.simulation.output[self.config['output_variable']['name']])
                        
                        # Store result
                        output_data[row, col] = output_value
                        
                    except Exception as e:
                        print(f"Error processing pixel ({row}, {col}): {e}")
                        print(f"Values: social={social_val}, env={env_val}, strat={strat_val}")
                        output_data[row, col] = nodata_value
                    
                    pbar.update(1)
        
        processing_time = time.time() - start_time
        print(f"Pixel processing completed in {processing_time:.2f} seconds")
        
        # Update profile for output
        profile.update(
            dtype=np.float32,
            count=1,
            nodata=nodata_value
        )
        
        # Write output raster
        with rasterio.open(output_tiff, 'w', **profile) as dst:
            dst.write(output_data, 1)
        
        print(f"Processing complete. Output saved to: {output_tiff}")
        print(f"Output value range: {np.nanmin(output_data):.2f} to {np.nanmax(output_data):.2f}")


def create_raster_config_template():
    """
    Create a template configuration file for raster processing.
    """
    config = {
        "description": "Raster fuzzy inference system for environmental assessment",
        "input_variables": {
            "social": {
                "min": 0,
                "max": 10,
                "step": 0.1,
                "membership_functions": {
                    "low": {
                        "type": "trapmf",
                        "params": [0, 0, 2, 4]
                    },
                    "medium": {
                        "type": "trapmf",
                        "params": [2, 4, 6, 7]
                    },
                    "high": {
                        "type": "trapmf",
                        "params": [6, 7, 10, 10]
                    }
                }
            },
            "environmental": {
                "min": 0,
                "max": 10,
                "step": 0.1,
                "membership_functions": {
                    "low": {
                        "type": "trapmf",
                        "params": [0, 0, 2, 5]
                    },
                    "medium": {
                        "type": "trapmf",
                        "params": [2, 5, 6, 8]
                    },
                    "high": {
                        "type": "trapmf",
                        "params": [6, 8, 10, 10]
                    }
                }
            },
            "strategic": {
                "min": 0,
                "max": 10,
                "step": 0.1,
                "membership_functions": {
                    "low": {
                        "type": "trapmf",
                        "params": [0, 0, 3, 5]
                    },
                    "medium": {
                        "type": "trapmf",
                        "params": [3, 5, 7, 8]
                    },
                    "high": {
                        "type": "trapmf",
                        "params": [7, 8, 10, 10]
                    }
                }
            }
        },
        "output_variable": {
            "name": "priority",
            "min": 0,
            "max": 10,
            "step": 0.1,
            "membership_functions": {
                "very_low": {
                    "type": "trimf",
                    "params": [0, 0, 2.5]
                },
                "low": {
                    "type": "trimf",
                    "params": [0, 2.5, 5]
                },
                "medium": {
                    "type": "trimf",
                    "params": [2.5, 5, 7.5]
                },
                "high": {
                    "type": "trimf",
                    "params": [5, 5.5, 10]
                },
                "very_high": {
                    "type": "trimf",
                    "params": [7.5, 10, 10]
                }
            }
        },
        "rules": [
            # Social low rules
            {"antecedent": [{"variable": "social", "membership": "low"}, 
                           {"variable": "environmental", "membership": "low"}, 
                           {"variable": "strategic", "membership": "high"}], "consequent": "very_low"},
            {"antecedent": [{"variable": "social", "membership": "low"}, 
                           {"variable": "environmental", "membership": "low"}, 
                           {"variable": "strategic", "membership": "medium"}], "consequent": "very_low"},
            {"antecedent": [{"variable": "social", "membership": "low"}, 
                           {"variable": "environmental", "membership": "low"}, 
                           {"variable": "strategic", "membership": "low"}], "consequent": "very_low"},
            {"antecedent": [{"variable": "social", "membership": "low"}, 
                           {"variable": "environmental", "membership": "medium"}, 
                           {"variable": "strategic", "membership": "high"}], "consequent": "medium"},
            {"antecedent": [{"variable": "social", "membership": "low"}, 
                           {"variable": "environmental", "membership": "medium"}, 
                           {"variable": "strategic", "membership": "medium"}], "consequent": "medium"},
            {"antecedent": [{"variable": "social", "membership": "low"}, 
                           {"variable": "environmental", "membership": "medium"}, 
                           {"variable": "strategic", "membership": "low"}], "consequent": "medium"},
            {"antecedent": [{"variable": "social", "membership": "low"}, 
                           {"variable": "environmental", "membership": "high"}, 
                           {"variable": "strategic", "membership": "high"}], "consequent": "low"},
            {"antecedent": [{"variable": "social", "membership": "low"}, 
                           {"variable": "environmental", "membership": "high"}, 
                           {"variable": "strategic", "membership": "medium"}], "consequent": "medium"},
            {"antecedent": [{"variable": "social", "membership": "low"}, 
                           {"variable": "environmental", "membership": "high"}, 
                           {"variable": "strategic", "membership": "low"}], "consequent": "high"},
            
            # Social medium rules
            {"antecedent": [{"variable": "social", "membership": "medium"}, 
                           {"variable": "environmental", "membership": "low"}, 
                           {"variable": "strategic", "membership": "high"}], "consequent": "low"},
            {"antecedent": [{"variable": "social", "membership": "medium"}, 
                           {"variable": "environmental", "membership": "low"}, 
                           {"variable": "strategic", "membership": "medium"}], "consequent": "medium"},
            {"antecedent": [{"variable": "social", "membership": "medium"}, 
                           {"variable": "environmental", "membership": "low"}, 
                           {"variable": "strategic", "membership": "low"}], "consequent": "medium"},
            {"antecedent": [{"variable": "social", "membership": "medium"}, 
                           {"variable": "environmental", "membership": "medium"}, 
                           {"variable": "strategic", "membership": "high"}], "consequent": "low"},
            {"antecedent": [{"variable": "social", "membership": "medium"}, 
                           {"variable": "environmental", "membership": "medium"}, 
                           {"variable": "strategic", "membership": "medium"}], "consequent": "medium"},
            {"antecedent": [{"variable": "social", "membership": "medium"}, 
                           {"variable": "environmental", "membership": "medium"}, 
                           {"variable": "strategic", "membership": "low"}], "consequent": "medium"},
            {"antecedent": [{"variable": "social", "membership": "medium"}, 
                           {"variable": "environmental", "membership": "high"}, 
                           {"variable": "strategic", "membership": "high"}], "consequent": "medium"},
            {"antecedent": [{"variable": "social", "membership": "medium"}, 
                           {"variable": "environmental", "membership": "high"}, 
                           {"variable": "strategic", "membership": "medium"}], "consequent": "medium"},
            {"antecedent": [{"variable": "social", "membership": "medium"}, 
                           {"variable": "environmental", "membership": "high"}, 
                           {"variable": "strategic", "membership": "low"}], "consequent": "high"},
            
            # Social high rules
            {"antecedent": [{"variable": "social", "membership": "high"}, 
                           {"variable": "environmental", "membership": "low"}, 
                           {"variable": "strategic", "membership": "high"}], "consequent": "low"},
            {"antecedent": [{"variable": "social", "membership": "high"}, 
                           {"variable": "environmental", "membership": "low"}, 
                           {"variable": "strategic", "membership": "medium"}], "consequent": "high"},
            {"antecedent": [{"variable": "social", "membership": "high"}, 
                           {"variable": "environmental", "membership": "low"}, 
                           {"variable": "strategic", "membership": "low"}], "consequent": "high"},
            {"antecedent": [{"variable": "social", "membership": "high"}, 
                           {"variable": "environmental", "membership": "medium"}, 
                           {"variable": "strategic", "membership": "high"}], "consequent": "medium"},
            {"antecedent": [{"variable": "social", "membership": "high"}, 
                           {"variable": "environmental", "membership": "medium"}, 
                           {"variable": "strategic", "membership": "medium"}], "consequent": "high"},
            {"antecedent": [{"variable": "social", "membership": "high"}, 
                           {"variable": "environmental", "membership": "medium"}, 
                           {"variable": "strategic", "membership": "low"}], "consequent": "very_high"},
            {"antecedent": [{"variable": "social", "membership": "high"}, 
                           {"variable": "environmental", "membership": "high"}, 
                           {"variable": "strategic", "membership": "high"}], "consequent": "very_high"},
            {"antecedent": [{"variable": "social", "membership": "high"}, 
                           {"variable": "environmental", "membership": "high"}, 
                           {"variable": "strategic", "membership": "medium"}], "consequent": "very_high"},
            {"antecedent": [{"variable": "social", "membership": "high"}, 
                           {"variable": "environmental", "membership": "high"}, 
                           {"variable": "strategic", "membership": "low"}], "consequent": "very_high"}
        ]
    }
    
    return config


def main():
    """
    Main function demonstrating how to use the raster fuzzy inference system.
    """
    # Create a sample configuration file if it doesn't exist
    config_file = "app/raster_fis_config.json"
    if not os.path.exists(config_file):
        config = create_raster_config_template()
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Created configuration template: {config_file}")
    
    # Example usage (you would replace these with your actual TIFF files)
    print("Raster Fuzzy Inference System")
    print("=" * 40)
    print("This system processes 3 input TIFF files (social, environmental, strategic)")
    print("and outputs a single TIFF file using fuzzy logic.")
    print()
    print("To use this system:")
    print("1. Create a configuration file (JSON format)")
    print("2. Prepare your 3 input TIFF files")
    print("3. Run the system with your files")
    print()
    print("Example usage:")
    print("python raster_fuzzy_system.py")
    print()
    print("Or programmatically:")
    print("fis = RasterFuzzyInferenceSystem('raster_fis_config.json')")
    print("fis.process_rasters('social.tif', 'environmental.tif', 'strategic.tif', 'output.tif')")


if __name__ == "__main__":
    main() 