#!/usr/bin/env python3
"""
Unified Raster Fuzzy Inference System.
Processes 3 input TIFF files (social, environmental, strategic) and outputs a single TIFF.
Supports both sequential and parallel processing with automatic fallback.
"""

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
import multiprocessing as mp
from functools import partial
import sys
from tqdm import tqdm


class UnifiedRasterFuzzyInferenceSystem:
    """
    A unified fuzzy inference system designed to process raster TIFF files.
    Supports both sequential and parallel processing with automatic fallback.
    Takes 3 input TIFF files (social, environmental, strategic) and outputs a single TIFF.
    """
    
    def __init__(self, config_file_path: str):
        """
        Initialize the unified raster fuzzy inference system.
        
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
    
    def _process_rasters_sequential(self, 
                                   social_tiff: str, 
                                   environmental_tiff: str, 
                                   strategic_tiff: str, 
                                   output_tiff: str,
                                   nodata_value: float = -9999.0) -> None:
        """
        Process three input TIFF files sequentially and output a single TIFF file.
        
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
        
        print(f"Processing {total_pixels} pixels sequentially...")
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
        print(f"Sequential processing completed in {processing_time:.2f} seconds")
        
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
    
    @staticmethod
    def _create_fis_from_config(config: Dict) -> 'UnifiedRasterFuzzyInferenceSystem':
        """Create a fuzzy inference system from configuration (for multiprocessing)."""
        fis = UnifiedRasterFuzzyInferenceSystem.__new__(UnifiedRasterFuzzyInferenceSystem)
        fis.config = config
        fis.fuzzy_inputs = {}
        fis.fuzzy_output = None
        fis.control_system = None
        fis.simulation = None
        
        fis._create_fuzzy_variables()
        fis._create_rules()
        fis._create_control_system()
        
        return fis
    
    def _process_chunk_parallel(self, chunk_data: Tuple[int, int, np.ndarray, np.ndarray, np.ndarray, 
                                                        Dict, List[str], float]) -> Tuple[int, int, np.ndarray]:
        """
        Process a chunk of raster data using fuzzy logic (for parallel processing).
        
        Args:
            chunk_data: Tuple containing (start_row, end_row, social_data, environmental_data, 
                        strategic_data, config, input_vars, nodata_value)
        
        Returns:
            Tuple of (start_row, end_row, output_chunk)
        """
        start_row, end_row, social_chunk, env_chunk, strat_chunk, config, input_vars, nodata_value = chunk_data
        
        # Create a new fuzzy system for this process
        fis = UnifiedRasterFuzzyInferenceSystem._create_fis_from_config(config)
        
        # Get chunk dimensions
        rows, cols = social_chunk.shape
        output_chunk = np.full((rows, cols), nodata_value, dtype=np.float32)
        
        # Process each pixel in the chunk
        for row in range(rows):
            for col in range(cols):
                # Get pixel values
                social_val = social_chunk[row, col]
                env_val = env_chunk[row, col]
                strat_val = strat_chunk[row, col]
                
                # Skip NoData pixels
                if (np.isnan(social_val) or np.isnan(env_val) or np.isnan(strat_val) or
                    social_val == nodata_value or env_val == nodata_value or strat_val == nodata_value):
                    continue
                
                try:
                    # Set input values for fuzzy system
                    fis.simulation.input[input_vars[0]] = float(social_val)
                    fis.simulation.input[input_vars[1]] = float(env_val)
                    fis.simulation.input[input_vars[2]] = float(strat_val)
                    
                    # Compute output
                    fis.simulation.compute()
                    output_value = float(fis.simulation.output[config['output_variable']['name']])
                    
                    # Store result
                    output_chunk[row, col] = output_value
                    
                except Exception as e:
                    # If there's an error, set to NoData
                    output_chunk[row, col] = nodata_value
        
        return start_row, end_row, output_chunk
    
    def _process_rasters_parallel(self, 
                                 social_tiff: str, 
                                 environmental_tiff: str, 
                                 strategic_tiff: str, 
                                 output_tiff: str,
                                 nodata_value: float = -9999.0,
                                 num_cores: int = None,
                                 chunk_size: int = 100) -> None:
        """
        Process three input TIFF files in parallel and output a single TIFF file.
        
        Args:
            social_tiff: Path to social factor TIFF file
            environmental_tiff: Path to environmental factor TIFF file
            strategic_tiff: Path to strategic factor TIFF file
            output_tiff: Path for output TIFF file
            nodata_value: Value to use for NoData pixels
            num_cores: Number of CPU cores to use (default: all available)
            chunk_size: Number of rows per chunk for parallel processing
        """
        # Determine number of cores to use
        if num_cores is None:
            num_cores = mp.cpu_count()
        num_cores = min(num_cores, mp.cpu_count())
        
        print(f"Using {num_cores} CPU cores for parallel processing")
        print(f"Chunk size: {chunk_size} rows per chunk")
        
        # Read input rasters
        print("Reading input rasters...")
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
        
        rows, cols = social_data.shape
        print(f"Raster dimensions: {rows} rows x {cols} columns")
        
        # Create chunks for parallel processing
        chunks = []
        for i, start_row in enumerate(range(0, rows, chunk_size)):
            end_row = min(start_row + chunk_size, rows)
            social_chunk = social_data[start_row:end_row, :]
            env_chunk = environmental_data[start_row:end_row, :]
            strat_chunk = strategic_data[start_row:end_row, :]
            
            chunks.append((start_row, end_row, social_chunk, env_chunk, strat_chunk, 
                         self.config, list(self.config['input_variables'].keys()), nodata_value))
        
        total_chunks = len(chunks)
        print(f"Created {total_chunks} chunks for parallel processing")
        
        # Process chunks in parallel with progress tracking
        print("Starting parallel processing...")
        start_time = time.time()
        
        # Initialize results list
        results = []
        
        # Create a progress bar
        with tqdm(total=total_chunks, desc="Processing chunks", unit="chunk", ncols=80) as pbar:
            try:
                with mp.Pool(processes=num_cores) as pool:
                    # Use imap_unordered for potentially faster completion
                    for result in pool.imap_unordered(self._process_chunk_parallel, chunks):
                        pbar.update(1)
                        results.append(result)
            except Exception as e:
                print(f"Parallel processing failed: {e}")
                raise
        
        # Sort results by start_row to maintain order
        results.sort(key=lambda x: x[0])
        
        processing_time = time.time() - start_time
        print(f"Parallel processing completed in {processing_time:.2f} seconds")
        
        # Combine results
        print("Combining results...")
        output_data = np.full((rows, cols), nodata_value, dtype=np.float32)
        
        for start_row, end_row, output_chunk in tqdm(results, desc="Combining chunks", unit="chunk"):
            output_data[start_row:end_row, :] = output_chunk
        
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
        
        print(f"Processing complete. Output saved to: {output_tiff}")
        print(f"Output value range: {np.nanmin(output_data):.2f} to {np.nanmax(output_data):.2f}")
        print(f"Total processing time: {time.time() - start_time:.2f} seconds")
    
    def process_rasters(self, 
                       social_tiff: str, 
                       environmental_tiff: str, 
                       strategic_tiff: str, 
                       output_tiff: str,
                       nodata_value: float = -9999.0,
                       parallel: bool = False,
                       num_cores: int = None,
                       chunk_size: int = 100) -> None:
        """
        Process three input TIFF files and output a single TIFF file.
        Supports both sequential and parallel processing with automatic fallback.
        
        Args:
            social_tiff: Path to social factor TIFF file
            environmental_tiff: Path to environmental factor TIFF file
            strategic_tiff: Path to strategic factor TIFF file
            output_tiff: Path for output TIFF file
            nodata_value: Value to use for NoData pixels
            parallel: Whether to use parallel processing (default: False)
            num_cores: Number of CPU cores to use for parallel processing (default: all available)
            chunk_size: Number of rows per chunk for parallel processing (default: 100)
        """
        if parallel:
            try:
                print("Attempting parallel processing...")
                self._process_rasters_parallel(
                    social_tiff=social_tiff,
                    environmental_tiff=environmental_tiff,
                    strategic_tiff=strategic_tiff,
                    output_tiff=output_tiff,
                    nodata_value=nodata_value,
                    num_cores=num_cores,
                    chunk_size=chunk_size
                )
                print("Parallel processing completed successfully!")
            except Exception as e:
                print(f"Parallel processing failed: {e}")
                print("Falling back to sequential processing...")
                self._process_rasters_sequential(
                    social_tiff=social_tiff,
                    environmental_tiff=environmental_tiff,
                    strategic_tiff=strategic_tiff,
                    output_tiff=output_tiff,
                    nodata_value=nodata_value
                )
                print("Sequential processing completed successfully!")
        else:
            self._process_rasters_sequential(
                social_tiff=social_tiff,
                environmental_tiff=environmental_tiff,
                strategic_tiff=strategic_tiff,
                output_tiff=output_tiff,
                nodata_value=nodata_value
            )


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
                           {"variable": "strategic", "membership": "high"}], "consequent": "low"},
            {"antecedent": [{"variable": "social", "membership": "low"}, 
                           {"variable": "environmental", "membership": "medium"}, 
                           {"variable": "strategic", "membership": "medium"}], "consequent": "low"},
            {"antecedent": [{"variable": "social", "membership": "low"}, 
                           {"variable": "environmental", "membership": "medium"}, 
                           {"variable": "strategic", "membership": "low"}], "consequent": "very_low"},
            {"antecedent": [{"variable": "social", "membership": "low"}, 
                           {"variable": "environmental", "membership": "high"}, 
                           {"variable": "strategic", "membership": "high"}], "consequent": "medium"},
            {"antecedent": [{"variable": "social", "membership": "low"}, 
                           {"variable": "environmental", "membership": "high"}, 
                           {"variable": "strategic", "membership": "medium"}], "consequent": "low"},
            {"antecedent": [{"variable": "social", "membership": "low"}, 
                           {"variable": "environmental", "membership": "high"}, 
                           {"variable": "strategic", "membership": "low"}], "consequent": "low"},
            
            # Social medium rules
            {"antecedent": [{"variable": "social", "membership": "medium"}, 
                           {"variable": "environmental", "membership": "low"}, 
                           {"variable": "strategic", "membership": "high"}], "consequent": "low"},
            {"antecedent": [{"variable": "social", "membership": "medium"}, 
                           {"variable": "environmental", "membership": "low"}, 
                           {"variable": "strategic", "membership": "medium"}], "consequent": "low"},
            {"antecedent": [{"variable": "social", "membership": "medium"}, 
                           {"variable": "environmental", "membership": "low"}, 
                           {"variable": "strategic", "membership": "low"}], "consequent": "very_low"},
            {"antecedent": [{"variable": "social", "membership": "medium"}, 
                           {"variable": "environmental", "membership": "medium"}, 
                           {"variable": "strategic", "membership": "high"}], "consequent": "medium"},
            {"antecedent": [{"variable": "social", "membership": "medium"}, 
                           {"variable": "environmental", "membership": "medium"}, 
                           {"variable": "strategic", "membership": "medium"}], "consequent": "medium"},
            {"antecedent": [{"variable": "social", "membership": "medium"}, 
                           {"variable": "environmental", "membership": "medium"}, 
                           {"variable": "strategic", "membership": "low"}], "consequent": "low"},
            {"antecedent": [{"variable": "social", "membership": "medium"}, 
                           {"variable": "environmental", "membership": "high"}, 
                           {"variable": "strategic", "membership": "high"}], "consequent": "high"},
            {"antecedent": [{"variable": "social", "membership": "medium"}, 
                           {"variable": "environmental", "membership": "high"}, 
                           {"variable": "strategic", "membership": "medium"}], "consequent": "medium"},
            {"antecedent": [{"variable": "social", "membership": "medium"}, 
                           {"variable": "environmental", "membership": "high"}, 
                           {"variable": "strategic", "membership": "low"}], "consequent": "medium"},
            
            # Social high rules
            {"antecedent": [{"variable": "social", "membership": "high"}, 
                           {"variable": "environmental", "membership": "low"}, 
                           {"variable": "strategic", "membership": "high"}], "consequent": "medium"},
            {"antecedent": [{"variable": "social", "membership": "high"}, 
                           {"variable": "environmental", "membership": "low"}, 
                           {"variable": "strategic", "membership": "medium"}], "consequent": "medium"},
            {"antecedent": [{"variable": "social", "membership": "high"}, 
                           {"variable": "environmental", "membership": "low"}, 
                           {"variable": "strategic", "membership": "low"}], "consequent": "low"},
            {"antecedent": [{"variable": "social", "membership": "high"}, 
                           {"variable": "environmental", "membership": "medium"}, 
                           {"variable": "strategic", "membership": "high"}], "consequent": "high"},
            {"antecedent": [{"variable": "social", "membership": "high"}, 
                           {"variable": "environmental", "membership": "medium"}, 
                           {"variable": "strategic", "membership": "medium"}], "consequent": "high"},
            {"antecedent": [{"variable": "social", "membership": "high"}, 
                           {"variable": "environmental", "membership": "medium"}, 
                           {"variable": "strategic", "membership": "low"}], "consequent": "medium"},
            {"antecedent": [{"variable": "social", "membership": "high"}, 
                           {"variable": "environmental", "membership": "high"}, 
                           {"variable": "strategic", "membership": "high"}], "consequent": "very_high"},
            {"antecedent": [{"variable": "social", "membership": "high"}, 
                           {"variable": "environmental", "membership": "high"}, 
                           {"variable": "strategic", "membership": "medium"}], "consequent": "very_high"},
            {"antecedent": [{"variable": "social", "membership": "high"}, 
                           {"variable": "environmental", "membership": "high"}, 
                           {"variable": "strategic", "membership": "low"}], "consequent": "high"}
        ]
    }
    
    return config


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Unified Raster Fuzzy Inference System - Process 3 TIFF files into 1 output TIFF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Sequential processing (default)
  python raster_fuzzy_system_unified.py social.tif environmental.tif strategic.tif output.tif

  # Parallel processing
  python raster_fuzzy_system_unified.py social.tif environmental.tif strategic.tif output.tif --parallel --cores 8

  # Parallel processing with custom chunk size
  python raster_fuzzy_system_unified.py social.tif environmental.tif strategic.tif output.tif --parallel --cores 4 --chunk-size 200
        """
    )
    
    parser.add_argument('social_tiff', help='Path to social factor TIFF file')
    parser.add_argument('environmental_tiff', help='Path to environmental factor TIFF file')
    parser.add_argument('strategic_tiff', help='Path to strategic factor TIFF file')
    parser.add_argument('output_tiff', help='Path for output TIFF file')
    
    parser.add_argument('--config', '-c', default='raster_fis_config.json',
                       help='Path to configuration JSON file (default: raster_fis_config.json)')
    parser.add_argument('--nodata', type=float, default=-9999.0,
                       help='NoData value for output raster (default: -9999.0)')
    parser.add_argument('--parallel', '-p', action='store_true',
                       help='Enable parallel processing (default: sequential)')
    parser.add_argument('--cores', type=int, default=None,
                       help=f'Number of CPU cores to use for parallel processing (default: all available, max: {mp.cpu_count()})')
    parser.add_argument('--chunk-size', type=int, default=100,
                       help='Number of rows per chunk for parallel processing (default: 100)')
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
        print(f"Configuration file {args.config} not found.")
        print("Creating template configuration file...")
        config = create_raster_config_template()
        with open(args.config, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Created configuration template: {args.config}")
        print("Please edit the configuration file and run again.")
        sys.exit(1)
    
    # Validate parallel processing parameters
    if args.parallel:
        if args.cores is not None and args.cores <= 0:
            print(f"Error: Number of cores must be positive, got: {args.cores}")
            sys.exit(1)
        if args.chunk_size <= 0:
            print(f"Error: Chunk size must be positive, got: {args.chunk_size}")
            sys.exit(1)
    
    # Check if output directory exists
    output_dir = os.path.dirname(args.output_tiff)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        # Initialize the unified fuzzy inference system
        print(f"Loading configuration from: {args.config}")
        fis = UnifiedRasterFuzzyInferenceSystem(args.config)
        
        # Process the rasters
        print(f"Processing rasters...")
        print(f"  Social: {args.social_tiff}")
        print(f"  Environmental: {args.environmental_tiff}")
        print(f"  Strategic: {args.strategic_tiff}")
        print(f"  Output: {args.output_tiff}")
        print(f"  NoData value: {args.nodata}")
        if args.parallel:
            print(f"  Processing mode: Parallel")
            print(f"  CPU cores: {args.cores or 'all available'}")
            print(f"  Chunk size: {args.chunk_size} rows")
        else:
            print(f"  Processing mode: Sequential")
        
        fis.process_rasters(
            social_tiff=args.social_tiff,
            environmental_tiff=args.environmental_tiff,
            strategic_tiff=args.strategic_tiff,
            output_tiff=args.output_tiff,
            nodata_value=args.nodata,
            parallel=args.parallel,
            num_cores=args.cores,
            chunk_size=args.chunk_size
        )
        
        print("Processing completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 