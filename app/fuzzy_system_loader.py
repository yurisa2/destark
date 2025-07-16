import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from generalized_fuzzy_system import FuzzyInferenceSystem


class FuzzySystemLoader:
    """
    Utility class for loading fuzzy inference system configurations from JSON files
    and creating systems dynamically.
    """
    
    def __init__(self, config_file_path: str):
        """
        Initialize the loader with a configuration file.
        
        Args:
            config_file_path: Path to the JSON configuration file
        """
        self.config_file_path = config_file_path
        self.configurations = self._load_configurations()
    
    def _load_configurations(self) -> Dict:
        """Load configurations from JSON file."""
        try:
            with open(self.config_file_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Configuration file {self.config_file_path} not found.")
            return {}
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON configuration: {e}")
            return {}
    
    def get_available_systems(self) -> List[str]:
        """Get list of available system configurations."""
        return list(self.configurations.keys())
    
    def get_system_info(self, system_name: str) -> Optional[Dict]:
        """Get information about a specific system configuration."""
        if system_name in self.configurations:
            config = self.configurations[system_name]
            return {
                'description': config.get('description', 'No description available'),
                'input_variables': list(config.get('input_variables', {}).keys()),
                'output_variable': config.get('output_variable', {}).get('name', 'unknown'),
                'rule_count': len(config.get('rules', []))
            }
        return None
    
    def create_system(self, system_name: str) -> Optional[FuzzyInferenceSystem]:
        """
        Create a fuzzy inference system from configuration.
        
        Args:
            system_name: Name of the system configuration to use
            
        Returns:
            FuzzyInferenceSystem instance or None if configuration not found
        """
        if system_name not in self.configurations:
            print(f"System '{system_name}' not found in configuration.")
            return None
        
        config = self.configurations[system_name]
        
        try:
            # Extract configuration components
            input_variables = config['input_variables']
            output_variable = config['output_variable']
            rules = config['rules']
            
            # Create and return the system
            return FuzzyInferenceSystem(input_variables, output_variable, rules)
            
        except KeyError as e:
            print(f"Missing required configuration key: {e}")
            return None
        except Exception as e:
            print(f"Error creating system '{system_name}': {e}")
            return None
    
    def process_data_with_system(self, 
                               system_name: str, 
                               data: pd.DataFrame, 
                               input_columns: List[str], 
                               output_column: str = 'output') -> Optional[pd.DataFrame]:
        """
        Process data using a specific system configuration.
        
        Args:
            system_name: Name of the system configuration to use
            data: Input DataFrame
            input_columns: List of column names corresponding to input variables
            output_column: Name for the output column
            
        Returns:
            DataFrame with results or None if processing failed
        """
        system = self.create_system(system_name)
        if system is None:
            return None
        
        try:
            return system.process_data(data, input_columns, output_column)
        except Exception as e:
            print(f"Error processing data with system '{system_name}': {e}")
            return None


def create_sample_data_for_system(system_name: str, n_samples: int = 100) -> pd.DataFrame:
    """
    Create sample data for testing a specific system configuration.
    
    Args:
        system_name: Name of the system to create sample data for
        output_column: Number of samples to generate
        
    Returns:
        DataFrame with sample data
    """
    loader = FuzzySystemLoader('fuzzy_config_example.json')
    config = loader.configurations.get(system_name)
    
    if config is None:
        print(f"System '{system_name}' not found in configuration.")
        return pd.DataFrame()
    
    np.random.seed(42)
    sample_data = {}
    
    # Generate sample data for each input variable
    for var_name, var_config in config['input_variables'].items():
        min_val = var_config['min']
        max_val = var_config['max']
        sample_data[var_name] = np.random.uniform(min_val, max_val, n_samples)
    
    return pd.DataFrame(sample_data)


def main():
    """
    Main function demonstrating how to use the fuzzy system loader.
    """
    # Initialize the loader
    loader = FuzzySystemLoader('fuzzy_config_example.json')
    
    # Show available systems
    available_systems = loader.get_available_systems()
    print("Available fuzzy systems:")
    for system_name in available_systems:
        info = loader.get_system_info(system_name)
        print(f"  - {system_name}: {info['description']}")
        print(f"    Input variables: {info['input_variables']}")
        print(f"    Output variable: {info['output_variable']}")
        print(f"    Number of rules: {info['rule_count']}")
        print()
    
    # Example: Process data with environmental assessment system
    if 'environmental_assessment' in available_systems:
        print("Processing data with environmental assessment system...")
        
        # Create sample data
        sample_data = create_sample_data_for_system('environmental_assessment', 50)
        print(f"Created sample data with {len(sample_data)} records")
        print(sample_data.head())
        
        # Process the data
        input_columns = ['environmental', 'socioeconomic', 'strategic']
        result = loader.process_data_with_system(
            'environmental_assessment', 
            sample_data, 
            input_columns, 
            'priority'
        )
        
        if result is not None:
            print("\nProcessing results:")
            print(f"Priority scores range: {result['priority'].min():.2f} to {result['priority'].max():.2f}")
            print(result.head())
            
            # Save results
            result.to_csv("python_move/environmental_assessment_results.csv", index=False)
            print("Results saved to 'python_move/environmental_assessment_results.csv'")
    
    # Example: Process data with temperature control system
    if 'simple_temperature_control' in available_systems:
        print("\nProcessing data with temperature control system...")
        
        # Create sample data
        sample_data = create_sample_data_for_system('simple_temperature_control', 30)
        print(f"Created sample data with {len(sample_data)} records")
        print(sample_data.head())
        
        # Process the data
        input_columns = ['temperature', 'humidity']
        result = loader.process_data_with_system(
            'simple_temperature_control', 
            sample_data, 
            input_columns, 
            'cooling_power'
        )
        
        if result is not None:
            print("\nProcessing results:")
            print(f"Cooling power range: {result['cooling_power'].min():.2f} to {result['cooling_power'].max():.2f}")
            print(result.head())
            
            # Save results
            result.to_csv("python_move/temperature_control_results.csv", index=False)
            print("Results saved to 'python_move/temperature_control_results.csv'")


if __name__ == "__main__":
    main() 