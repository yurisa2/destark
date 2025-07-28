#!/usr/bin/env python3
"""
Unified Raster Fuzzy Inference System using tifffile instead of rasterio.
Processes 3 input TIFF files (social, environmental, strategic) and outputs a single TIFF.
Supports both sequential and parallel processing with automatic fallback.
Compatible with AWS Glue environments where GDAL/rasterio may have issues.
"""

import tifffile
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
    A unified fuzzy inference system designed to process raster TIFF files using tifffile.
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
    
    def _read_tiff_with_tifffile(self, file_path: str) -> Tuple[np.ndarray, Dict]:
        """
        Read a TIFF file using tifffile and return data and metadata.
        
        Args:
            file_path: Path to the TIFF file
            
        Returns:
            Tuple of (data, metadata)
        """
        try:
            # Read the TIFF file
            tiff_data = tifffile.imread(file_path)
            
            # Handle different data formats
            if len(tiff_data.shape) == 3:
                # Multi-band image, take first band
                data = tiff_data[0]
            else:
                # Single band image
                data = tiff_data
            
            # Create metadata similar to rasterio profile
            metadata = {
                'driver': 'GTiff',
                'height': data.shape[0],
                'width': data.shape[1],
                'count': 1,
                'dtype': str(data.dtype),
                'nodata': None,  # tifffile doesn't provide nodata by default
                'crs': None,     # tifffile doesn't provide CRS by default
                'transform': None # tifffile doesn't provide transform by default
            }
            
            return data, metadata
            
        except Exception as e:
            raise ValueError(f"Error reading TIFF file {file_path}: {e}")
    
    def _write_tiff_with_tifffile(self, data: np.ndarray, file_path: str, 
                                  metadata: Dict = None) -> None:
        """
        Write a TIFF file using tifffile.
        
        Args:
            data: Array data to write
            file_path: Path to the output TIFF file
            metadata: Optional metadata dictionary
        """
        try:
            # Ensure the output directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Write the TIFF file
            tifffile.imwrite(file_path, data, 
                           photometric='minisblack',
                           planarconfig='contig')
            
        except Exception as e:
            raise ValueError(f"Error writing TIFF file {file_path}: {e}")
    
    def _process_rasters_sequential(self, 
                                   social_tiff: str, 
                                   environmental_tiff: str, 
                                   strategic_tiff: str, 
                                   output_tiff: str,
                                   nodata_value: float = 5.0) -> None:
        """
        Process three input TIFF files sequentially and output a single TIFF file.
        
        Args:
            social_tiff: Path to social factor TIFF file
            environmental_tiff: Path to environmental factor TIFF file
            strategic_tiff: Path to strategic factor TIFF file
            output_tiff: Path for output TIFF file
            nodata_value: Value to use for NoData pixels
        """
        # Read input rasters using tifffile
        print("Reading input rasters...")
        social_data, social_metadata = self._read_tiff_with_tifffile(social_tiff)
        environmental_data, env_metadata = self._read_tiff_with_tifffile(environmental_tiff)
        strategic_data, strat_metadata = self._read_tiff_with_tifffile(strategic_tiff)
        
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
                    
                    # Skip NoData pixels (check for NaN and common nodata values)
                    if (np.isnan(social_val) or np.isnan(env_val) or np.isnan(strat_val) or
                        social_val == nodata_value or env_val == nodata_value or strat_val == nodata_value):
                        pbar.update(1)
                        continue
                    
                    try:
                        # Set input values for fuzzy system
                        self.simulation.input[input_vars[0]] = float(social_val)
                        self.simulation.input[input_vars[1]] = float(env_val)
                        self.simulation.input[input_vars[2]] = float(strat_val)
                        
                        # Compute output
                        self.simulation.compute()
                        
                        # Store result
                        output_data[row, col] = self.simulation.output[self.config['output_variable']['name']]
                        
                    except Exception as e:
                        # If fuzzy computation fails, use nodata value
                        output_data[row, col] = nodata_value
                        print(f"Warning: Fuzzy computation failed at pixel ({row}, {col}): {e}")
                    
                    pbar.update(1)
        
        # Write output raster
        print("Writing output raster...")
        self._write_tiff_with_tifffile(output_data, output_tiff, social_metadata)
        
        end_time = time.time()
        processing_time = end_time - start_time
        print(f"Processing completed in {processing_time:.2f} seconds")
        print(f"Output saved to: {output_tiff}")
    
    @staticmethod
    def _create_fis_from_config(config: Dict) -> 'UnifiedRasterFuzzyInferenceSystem':
        """Create a FIS instance from a configuration dictionary."""
        # Create a temporary config file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config, f)
            temp_config_path = f.name
        
        try:
            return UnifiedRasterFuzzyInferenceSystem(temp_config_path)
        finally:
            # Clean up temporary file
            os.unlink(temp_config_path)
    
    def _process_chunk_parallel(self, chunk_data: Tuple[int, int, np.ndarray, np.ndarray, np.ndarray, 
                                                        Dict, List[str], float]) -> Tuple[int, int, np.ndarray]:
        """
        Process a chunk of raster data in parallel.
        
        Args:
            chunk_data: Tuple containing (start_row, end_row, social_data, env_data, strat_data, config, input_vars, nodata_value)
            
        Returns:
            Tuple of (start_row, end_row, output_data)
        """
        start_row, end_row, social_data, env_data, strat_data, config, input_vars, nodata_value = chunk_data
        
        # Create FIS instance for this process
        fis = self._create_fis_from_config(config)
        
        # Process the chunk
        rows = end_row - start_row
        cols = social_data.shape[1]
        output_chunk = np.full((rows, cols), nodata_value, dtype=np.float32)
        
        for row in range(rows):
            for col in range(cols):
                # Get pixel values
                social_val = social_data[row, col]
                env_val = env_data[row, col]
                strat_val = strat_data[row, col]
                
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
                    
                    # Store result
                    output_chunk[row, col] = fis.simulation.output[config['output_variable']['name']]
                    
                except Exception as e:
                    # If fuzzy computation fails, use nodata value
                    output_chunk[row, col] = nodata_value
        
        return start_row, end_row, output_chunk
    
    def _process_rasters_parallel(self, 
                                 social_tiff: str, 
                                 environmental_tiff: str, 
                                 strategic_tiff: str, 
                                 output_tiff: str,
                                 nodata_value: float = 5.0,
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
            num_cores: Number of CPU cores to use (None for auto-detection)
            chunk_size: Number of rows per chunk
        """
        # Read input rasters
        print("Reading input rasters...")
        social_data, social_metadata = self._read_tiff_with_tifffile(social_tiff)
        environmental_data, env_metadata = self._read_tiff_with_tifffile(environmental_tiff)
        strategic_data, strat_metadata = self._read_tiff_with_tifffile(strategic_tiff)
        
        # Check if all rasters have the same dimensions
        if (social_data.shape != environmental_data.shape or 
            social_data.shape != strategic_data.shape):
            raise ValueError("All input rasters must have the same dimensions")
        
        # Create output array
        output_data = np.full(social_data.shape, nodata_value, dtype=np.float32)
        
        # Get input variable names from config
        input_vars = list(self.config['input_variables'].keys())
        
        # Determine number of cores
        if num_cores is None:
            num_cores = mp.cpu_count()
        
        # Prepare chunks for parallel processing
        rows, cols = social_data.shape
        chunks = []
        
        for start_row in range(0, rows, chunk_size):
            end_row = min(start_row + chunk_size, rows)
            chunk_social = social_data[start_row:end_row, :]
            chunk_env = environmental_data[start_row:end_row, :]
            chunk_strat = strategic_data[start_row:end_row, :]
            
            chunks.append((start_row, end_row, chunk_social, chunk_env, chunk_strat, 
                         self.config, input_vars, nodata_value))
        
        print(f"Processing {len(chunks)} chunks with {num_cores} cores...")
        start_time = time.time()
        
        # Process chunks in parallel
        with mp.Pool(processes=num_cores) as pool:
            results = list(tqdm(
                pool.imap(self._process_chunk_parallel, chunks),
                total=len(chunks),
                desc="Processing chunks",
                unit="chunk"
            ))
        
        # Combine results
        for start_row, end_row, chunk_output in results:
            output_data[start_row:end_row, :] = chunk_output
        
        # Write output raster
        print("Writing output raster...")
        self._write_tiff_with_tifffile(output_data, output_tiff, social_metadata)
        
        end_time = time.time()
        processing_time = end_time - start_time
        print(f"Processing completed in {processing_time:.2f} seconds")
        print(f"Output saved to: {output_tiff}")
    
    def process_rasters(self, 
                       social_tiff: str, 
                       environmental_tiff: str, 
                       strategic_tiff: str, 
                       output_tiff: str,
                       nodata_value: float = 5.0,
                       parallel: bool = False,
                       num_cores: int = None,
                       chunk_size: int = 100) -> None:
        """
        Process three input TIFF files and output a single TIFF file.
        
        Args:
            social_tiff: Path to social factor TIFF file
            environmental_tiff: Path to environmental factor TIFF file
            strategic_tiff: Path to strategic factor TIFF file
            output_tiff: Path for output TIFF file
            nodata_value: Value to use for NoData pixels
            parallel: Whether to use parallel processing
            num_cores: Number of CPU cores to use (None for auto-detection)
            chunk_size: Number of rows per chunk (for parallel processing)
        """
        print("Starting raster fuzzy inference processing...")
        print(f"Input files:")
        print(f"  Social: {social_tiff}")
        print(f"  Environmental: {environmental_tiff}")
        print(f"  Strategic: {strategic_tiff}")
        print(f"Output file: {output_tiff}")
        print(f"Processing mode: {'Parallel' if parallel else 'Sequential'}")
        
        if parallel:
            self._process_rasters_parallel(
                social_tiff, environmental_tiff, strategic_tiff, output_tiff,
                nodata_value, num_cores, chunk_size
            )
        else:
            self._process_rasters_sequential(
                social_tiff, environmental_tiff, strategic_tiff, output_tiff,
                nodata_value
            )


def create_raster_config_template():
    """
    Create a template configuration for raster fuzzy inference system.
    
    Returns:
        Dictionary containing the configuration template
    """
    config = {
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
                        "type": "trimf",
                        "params": [3, 5, 7]
                    },
                    "high": {
                        "type": "trapmf",
                        "params": [6, 8, 10, 10]
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
                        "params": [0, 0, 2, 4]
                    },
                    "medium": {
                        "type": "trimf",
                        "params": [3, 5, 7]
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
                        "params": [0, 0, 2, 4]
                    },
                    "medium": {
                        "type": "trimf",
                        "params": [3, 5, 7]
                    },
                    "high": {
                        "type": "trapmf",
                        "params": [6, 8, 10, 10]
                    }
                }
            }
        },
        "output_variable": {
            "name": "suitability",
            "min": 0,
            "max": 10,
            "step": 0.1,
            "membership_functions": {
                "low": {
                    "type": "trapmf",
                    "params": [0, 0, 2, 4]
                },
                "medium": {
                    "type": "trimf",
                    "params": [3, 5, 7]
                },
                "high": {
                    "type": "trapmf",
                    "params": [6, 8, 10, 10]
                }
            }
        },
        "rules": [
            {
                "antecedent": [
                    {"variable": "social", "membership": "low"},
                    {"variable": "environmental", "membership": "low"},
                    {"variable": "strategic", "membership": "low"}
                ],
                "consequent": "low"
            },
            {
                "antecedent": [
                    {"variable": "social", "membership": "medium"},
                    {"variable": "environmental", "membership": "medium"},
                    {"variable": "strategic", "membership": "medium"}
                ],
                "consequent": "medium"
            },
            {
                "antecedent": [
                    {"variable": "social", "membership": "high"},
                    {"variable": "environmental", "membership": "high"},
                    {"variable": "strategic", "membership": "high"}
                ],
                "consequent": "high"
            }
        ]
    }
    
    return config


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Raster Fuzzy Inference System using tifffile')
    parser.add_argument('--social', required=True, help='Path to social factor TIFF file')
    parser.add_argument('--environmental', required=True, help='Path to environmental factor TIFF file')
    parser.add_argument('--strategic', required=True, help='Path to strategic factor TIFF file')
    parser.add_argument('--output', required=True, help='Path for output TIFF file')
    parser.add_argument('--config', required=True, help='Path to configuration JSON file')
    parser.add_argument('--parallel', action='store_true', help='Use parallel processing')
    parser.add_argument('--cores', type=int, help='Number of CPU cores to use')
    parser.add_argument('--chunk-size', type=int, default=100, help='Chunk size for parallel processing')
    parser.add_argument('--nodata', type=float, default=5.0, help='NoData value')
    
    args = parser.parse_args()
    
    # Create FIS instance
    fis = UnifiedRasterFuzzyInferenceSystem(args.config)
    
    # Process rasters
    fis.process_rasters(
        social_tiff=args.social,
        environmental_tiff=args.environmental,
        strategic_tiff=args.strategic,
        output_tiff=args.output,
        nodata_value=args.nodata,
        parallel=args.parallel,
        num_cores=args.cores,
        chunk_size=args.chunk_size
    )


if __name__ == "__main__":
    main() 