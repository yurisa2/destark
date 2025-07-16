import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import numpy as np
from typing import Dict, List, Tuple, Optional
import json


class FuzzyInferenceSystem:
    """
    A generalized fuzzy inference system that can handle multiple input variables
    with configurable membership functions and rules.
    """
    
    def __init__(self, input_variables: Dict[str, Dict], output_variable: Dict[str, Dict], rules: List[Dict]):
        """
        Initialize the fuzzy inference system.
        
        Args:
            input_variables: Dictionary defining input variables and their membership functions
            output_variable: Dictionary defining output variable and its membership functions
            rules: List of fuzzy rules
        """
        self.input_variables = input_variables
        self.output_variable = output_variable
        self.rules = rules
        
        # Create fuzzy variables
        self.fuzzy_inputs = {}
        self.fuzzy_output = None
        self.control_system = None
        self.simulation = None
        
        self._create_fuzzy_variables()
        self._create_rules()
        self._create_control_system()
    
    def _create_fuzzy_variables(self):
        """Create fuzzy antecedent and consequent variables."""
        # Create input variables
        for var_name, var_config in self.input_variables.items():
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
        output_config = self.output_variable
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
        
        for rule_config in self.rules:
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
    
    def process_data(self, data: pd.DataFrame, input_columns: List[str], output_column: str = 'output') -> pd.DataFrame:
        """
        Process data through the fuzzy inference system.
        
        Args:
            data: Input DataFrame
            input_columns: List of column names corresponding to input variables
            output_column: Name for the output column
            
        Returns:
            DataFrame with original data plus output column
        """
        result_data = data.copy()
        result_data[output_column] = 0
        
        for index, row in result_data.iterrows():
            try:
                # Set input values
                for i, col in enumerate(input_columns):
                    var_name = list(self.input_variables.keys())[i]
                    self.simulation.input[var_name] = row[col]
                
                # Compute output
                self.simulation.compute()
                result_data.loc[index, output_column] = float(self.simulation.output[self.output_variable['name']])
                
            except Exception as e:
                print(f"Error processing row {index}: {e}")
                print(f"Row data: {row}")
                result_data.loc[index, output_column] = 0
        
        return result_data


def create_environmental_system_config():
    """
    Create configuration for the environmental assessment system.
    This is the generalized version of the original code.
    """
    input_variables = {
        'environmental': {
            'min': 0,
            'max': 10,
            'step': 1,
            'membership_functions': {
                'low': {
                    'type': 'trapmf',
                    'params': [0, 0, 2, 4]
                },
                'medium': {
                    'type': 'trapmf',
                    'params': [2, 4, 6, 7]
                },
                'high': {
                    'type': 'trapmf',
                    'params': [6, 7, 10, 10]
                }
            }
        },
        'socioeconomic': {
            'min': 0,
            'max': 10,
            'step': 1,
            'membership_functions': {
                'low': {
                    'type': 'trapmf',
                    'params': [0, 0, 2, 5]
                },
                'medium': {
                    'type': 'trapmf',
                    'params': [2, 5, 6, 8]
                },
                'high': {
                    'type': 'trapmf',
                    'params': [6, 8, 10, 10]
                }
            }
        },
        'strategic': {
            'min': 0,
            'max': 10,
            'step': 1,
            'membership_functions': {
                'low': {
                    'type': 'trapmf',
                    'params': [0, 0, 3, 5]
                },
                'medium': {
                    'type': 'trapmf',
                    'params': [3, 5, 7, 8]
                },
                'high': {
                    'type': 'trapmf',
                    'params': [7, 8, 10, 10]
                }
            }
        }
    }
    
    output_variable = {
        'name': 'priority',
        'min': 0,
        'max': 10,
        'step': 1,
        'membership_functions': {
            'very_low': {
                'type': 'trimf',
                'params': [0, 0, 2.5]
            },
            'low': {
                'type': 'trimf',
                'params': [0, 2.5, 5]
            },
            'medium': {
                'type': 'trimf',
                'params': [2.5, 5, 7.5]
            },
            'high': {
                'type': 'trimf',
                'params': [5, 5.5, 10]
            },
            'very_high': {
                'type': 'trimf',
                'params': [7.5, 10, 10]
            }
        }
    }
    
    # Define rules (generalized from the original 27 rules)
    rules = [
        # Environmental low rules
        {'antecedent': [{'variable': 'environmental', 'membership': 'low'}, 
                       {'variable': 'socioeconomic', 'membership': 'low'}, 
                       {'variable': 'strategic', 'membership': 'high'}], 'consequent': 'very_low'},
        {'antecedent': [{'variable': 'environmental', 'membership': 'low'}, 
                       {'variable': 'socioeconomic', 'membership': 'low'}, 
                       {'variable': 'strategic', 'membership': 'medium'}], 'consequent': 'very_low'},
        {'antecedent': [{'variable': 'environmental', 'membership': 'low'}, 
                       {'variable': 'socioeconomic', 'membership': 'low'}, 
                       {'variable': 'strategic', 'membership': 'low'}], 'consequent': 'very_low'},
        {'antecedent': [{'variable': 'environmental', 'membership': 'low'}, 
                       {'variable': 'socioeconomic', 'membership': 'medium'}, 
                       {'variable': 'strategic', 'membership': 'high'}], 'consequent': 'medium'},
        {'antecedent': [{'variable': 'environmental', 'membership': 'low'}, 
                       {'variable': 'socioeconomic', 'membership': 'medium'}, 
                       {'variable': 'strategic', 'membership': 'medium'}], 'consequent': 'medium'},
        {'antecedent': [{'variable': 'environmental', 'membership': 'low'}, 
                       {'variable': 'socioeconomic', 'membership': 'medium'}, 
                       {'variable': 'strategic', 'membership': 'low'}], 'consequent': 'medium'},
        {'antecedent': [{'variable': 'environmental', 'membership': 'low'}, 
                       {'variable': 'socioeconomic', 'membership': 'high'}, 
                       {'variable': 'strategic', 'membership': 'high'}], 'consequent': 'low'},
        {'antecedent': [{'variable': 'environmental', 'membership': 'low'}, 
                       {'variable': 'socioeconomic', 'membership': 'high'}, 
                       {'variable': 'strategic', 'membership': 'medium'}], 'consequent': 'medium'},
        {'antecedent': [{'variable': 'environmental', 'membership': 'low'}, 
                       {'variable': 'socioeconomic', 'membership': 'high'}, 
                       {'variable': 'strategic', 'membership': 'low'}], 'consequent': 'high'},
        
        # Environmental medium rules
        {'antecedent': [{'variable': 'environmental', 'membership': 'medium'}, 
                       {'variable': 'socioeconomic', 'membership': 'low'}, 
                       {'variable': 'strategic', 'membership': 'high'}], 'consequent': 'low'},
        {'antecedent': [{'variable': 'environmental', 'membership': 'medium'}, 
                       {'variable': 'socioeconomic', 'membership': 'low'}, 
                       {'variable': 'strategic', 'membership': 'medium'}], 'consequent': 'medium'},
        {'antecedent': [{'variable': 'environmental', 'membership': 'medium'}, 
                       {'variable': 'socioeconomic', 'membership': 'low'}, 
                       {'variable': 'strategic', 'membership': 'low'}], 'consequent': 'medium'},
        {'antecedent': [{'variable': 'environmental', 'membership': 'medium'}, 
                       {'variable': 'socioeconomic', 'membership': 'medium'}, 
                       {'variable': 'strategic', 'membership': 'high'}], 'consequent': 'low'},
        {'antecedent': [{'variable': 'environmental', 'membership': 'medium'}, 
                       {'variable': 'socioeconomic', 'membership': 'medium'}, 
                       {'variable': 'strategic', 'membership': 'medium'}], 'consequent': 'medium'},
        {'antecedent': [{'variable': 'environmental', 'membership': 'medium'}, 
                       {'variable': 'socioeconomic', 'membership': 'medium'}, 
                       {'variable': 'strategic', 'membership': 'low'}], 'consequent': 'medium'},
        {'antecedent': [{'variable': 'environmental', 'membership': 'medium'}, 
                       {'variable': 'socioeconomic', 'membership': 'high'}, 
                       {'variable': 'strategic', 'membership': 'high'}], 'consequent': 'medium'},
        {'antecedent': [{'variable': 'environmental', 'membership': 'medium'}, 
                       {'variable': 'socioeconomic', 'membership': 'high'}, 
                       {'variable': 'strategic', 'membership': 'medium'}], 'consequent': 'medium'},
        {'antecedent': [{'variable': 'environmental', 'membership': 'medium'}, 
                       {'variable': 'socioeconomic', 'membership': 'high'}, 
                       {'variable': 'strategic', 'membership': 'low'}], 'consequent': 'high'},
        
        # Environmental high rules
        {'antecedent': [{'variable': 'environmental', 'membership': 'high'}, 
                       {'variable': 'socioeconomic', 'membership': 'low'}, 
                       {'variable': 'strategic', 'membership': 'high'}], 'consequent': 'low'},
        {'antecedent': [{'variable': 'environmental', 'membership': 'high'}, 
                       {'variable': 'socioeconomic', 'membership': 'low'}, 
                       {'variable': 'strategic', 'membership': 'medium'}], 'consequent': 'high'},
        {'antecedent': [{'variable': 'environmental', 'membership': 'high'}, 
                       {'variable': 'socioeconomic', 'membership': 'low'}, 
                       {'variable': 'strategic', 'membership': 'low'}], 'consequent': 'high'},
        {'antecedent': [{'variable': 'environmental', 'membership': 'high'}, 
                       {'variable': 'socioeconomic', 'membership': 'medium'}, 
                       {'variable': 'strategic', 'membership': 'high'}], 'consequent': 'medium'},
        {'antecedent': [{'variable': 'environmental', 'membership': 'high'}, 
                       {'variable': 'socioeconomic', 'membership': 'medium'}, 
                       {'variable': 'strategic', 'membership': 'medium'}], 'consequent': 'high'},
        {'antecedent': [{'variable': 'environmental', 'membership': 'high'}, 
                       {'variable': 'socioeconomic', 'membership': 'medium'}, 
                       {'variable': 'strategic', 'membership': 'low'}], 'consequent': 'very_high'},
        {'antecedent': [{'variable': 'environmental', 'membership': 'high'}, 
                       {'variable': 'socioeconomic', 'membership': 'high'}, 
                       {'variable': 'strategic', 'membership': 'high'}], 'consequent': 'very_high'},
        {'antecedent': [{'variable': 'environmental', 'membership': 'high'}, 
                       {'variable': 'socioeconomic', 'membership': 'high'}, 
                       {'variable': 'strategic', 'membership': 'medium'}], 'consequent': 'very_high'},
        {'antecedent': [{'variable': 'environmental', 'membership': 'high'}, 
                       {'variable': 'socioeconomic', 'membership': 'high'}, 
                       {'variable': 'strategic', 'membership': 'low'}], 'consequent': 'very_high'}
    ]
    
    return input_variables, output_variable, rules


def main():
    """
    Main function demonstrating how to use the generalized fuzzy inference system.
    """
    # Load data (assuming the same CSV files as in the original code)
    try:
        environmental_data = pd.read_csv("python_move/ambiental_1000m_normal.csv")
        socioeconomic_data = pd.read_csv("python_move/socioeconomico_1000m_normal.csv")
        strategic_data = pd.read_csv("python_move/estrategico_1000m_normal.csv")
        
        # Consolidate data
        consolidated = pd.concat([
            environmental_data, 
            socioeconomic_data["2"], 
            strategic_data["2"]
        ], axis=1)
        consolidated.columns = ["X", "Y", "environmental", "socioeconomic", "strategic"]
        
    except FileNotFoundError:
        print("Input files not found. Creating sample data for demonstration.")
        # Create sample data for demonstration
        np.random.seed(42)
        n_samples = 100
        consolidated = pd.DataFrame({
            'X': np.random.uniform(0, 100, n_samples),
            'Y': np.random.uniform(0, 100, n_samples),
            'environmental': np.random.uniform(0, 10, n_samples),
            'socioeconomic': np.random.uniform(0, 10, n_samples),
            'strategic': np.random.uniform(0, 10, n_samples)
        })
    
    # Create system configuration
    input_variables, output_variable, rules = create_environmental_system_config()
    
    # Initialize the fuzzy inference system
    fis = FuzzyInferenceSystem(input_variables, output_variable, rules)
    
    # Process the data
    input_columns = ['environmental', 'socioeconomic', 'strategic']
    result = fis.process_data(consolidated, input_columns, 'priority')
    
    # Save results
    result.to_csv("python_move/generalized_consolidated.csv", index=False)
    print(f"Processing complete. Results saved to 'python_move/generalized_consolidated.csv'")
    print(f"Processed {len(result)} records")
    print(f"Priority scores range: {result['priority'].min():.2f} to {result['priority'].max():.2f}")


if __name__ == "__main__":
    main() 