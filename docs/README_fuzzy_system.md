# Generalized Fuzzy Inference System

This is a generalized and modular implementation of a fuzzy inference system that can handle multiple input variables with configurable membership functions and rules. The system is designed to be flexible and reusable for various applications.

## Features

- **Modular Design**: The system is built as a class that can be easily extended and customized
- **Configurable Inputs**: Support for any number of input variables with custom membership functions
- **Flexible Rules**: Dynamic rule creation from configuration
- **JSON Configuration**: System configurations can be stored and loaded from JSON files
- **English Variable Names**: All variables and functions use clear English names
- **Error Handling**: Robust error handling for data processing
- **Type Hints**: Full type annotation for better code maintainability

## Files Overview

1. **`generalized_fuzzy_system.py`**: Main fuzzy inference system class
2. **`fuzzy_config_example.json`**: Example configurations for different systems
3. **`fuzzy_system_loader.py`**: Utility for loading configurations from JSON files
4. **`README_fuzzy_system.md`**: This documentation file

## Quick Start

### Basic Usage

```python
from generalized_fuzzy_system import FuzzyInferenceSystem, create_environmental_system_config

# Create system configuration
input_variables, output_variable, rules = create_environmental_system_config()

# Initialize the system
fis = FuzzyInferenceSystem(input_variables, output_variable, rules)

# Process data
import pandas as pd
data = pd.DataFrame({
    'environmental': [5, 8, 3],
    'socioeconomic': [6, 7, 4],
    'strategic': [4, 9, 6]
})

result = fis.process_data(data, ['environmental', 'socioeconomic', 'strategic'], 'priority')
print(result)
```

### Using JSON Configuration

```python
from fuzzy_system_loader import FuzzySystemLoader

# Load configurations from JSON file
loader = FuzzySystemLoader('fuzzy_config_example.json')

# List available systems
print(loader.get_available_systems())

# Process data with a specific system
sample_data = create_sample_data_for_system('environmental_assessment', 100)
result = loader.process_data_with_system(
    'environmental_assessment',
    sample_data,
    ['environmental', 'socioeconomic', 'strategic'],
    'priority'
)
```

## System Configuration

### Input Variables

Each input variable is defined with:
- `min`, `max`, `step`: Universe of discourse parameters
- `membership_functions`: Dictionary of membership functions

```json
{
  "environmental": {
    "min": 0,
    "max": 10,
    "step": 1,
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
  }
}
```

### Output Variable

Similar structure to input variables but with a `name` field:

```json
{
  "name": "priority",
  "min": 0,
  "max": 10,
  "step": 1,
  "membership_functions": {
    "very_low": {
      "type": "trimf",
      "params": [0, 0, 2.5]
    },
    "low": {
      "type": "trimf",
      "params": [0, 2.5, 5]
    }
  }
}
```

### Rules

Rules define the fuzzy logic relationships:

```json
{
  "antecedent": [
    {"variable": "environmental", "membership": "low"},
    {"variable": "socioeconomic", "membership": "low"},
    {"variable": "strategic", "membership": "high"}
  ],
  "consequent": "very_low"
}
```

## Membership Function Types

### Trapezoidal Membership Function (`trapmf`)
- Parameters: `[a, b, c, d]` where a ≤ b ≤ c ≤ d
- Creates a trapezoidal shape

### Triangular Membership Function (`trimf`)
- Parameters: `[a, b, c]` where a ≤ b ≤ c
- Creates a triangular shape

## Example Systems

### 1. Environmental Assessment System
- **Inputs**: Environmental, Socioeconomic, Strategic factors
- **Output**: Priority level (very_low to very_high)
- **Rules**: 27 rules covering all combinations
- **Application**: Environmental impact assessment and prioritization

### 2. Temperature Control System
- **Inputs**: Temperature, Humidity
- **Output**: Cooling power
- **Rules**: 9 rules for basic temperature control
- **Application**: HVAC system control

## Creating Custom Systems

### Step 1: Define Input Variables
```python
input_variables = {
    'temperature': {
        'min': 0,
        'max': 50,
        'step': 1,
        'membership_functions': {
            'cold': {'type': 'trapmf', 'params': [0, 0, 10, 20]},
            'warm': {'type': 'trapmf', 'params': [15, 25, 35, 40]},
            'hot': {'type': 'trapmf', 'params': [35, 45, 50, 50]}
        }
    }
}
```

### Step 2: Define Output Variable
```python
output_variable = {
    'name': 'cooling_power',
    'min': 0,
    'max': 100,
    'step': 1,
    'membership_functions': {
        'low': {'type': 'trimf', 'params': [0, 0, 30]},
        'medium': {'type': 'trimf', 'params': [20, 50, 80]},
        'high': {'type': 'trimf', 'params': [70, 100, 100]}
    }
}
```

### Step 3: Define Rules
```python
rules = [
    {
        'antecedent': [
            {'variable': 'temperature', 'membership': 'cold'},
            {'variable': 'humidity', 'membership': 'low'}
        ],
        'consequent': 'low'
    }
]
```

### Step 4: Create and Use System
```python
fis = FuzzyInferenceSystem(input_variables, output_variable, rules)
result = fis.process_data(data, ['temperature', 'humidity'], 'cooling_power')
```

## Error Handling

The system includes comprehensive error handling:
- Invalid membership function parameters
- Missing input variables
- Data type mismatches
- Rule configuration errors

Errors are logged with detailed information to help with debugging.

## Performance Considerations

- The system processes data row by row
- For large datasets, consider batch processing
- Membership function calculations are optimized using scikit-fuzzy
- Memory usage scales with the number of rules and variables

## Dependencies

- `pandas`: Data manipulation
- `numpy`: Numerical operations
- `skfuzzy`: Fuzzy logic implementation
- `typing`: Type hints (Python 3.5+)

## Running Examples

### Basic Example
```bash
python generalized_fuzzy_system.py
```

### Loader Example
```bash
python fuzzy_system_loader.py
```

## Migration from Original Code

The original code in `run_fis.py` has been generalized with these improvements:

1. **Variable Names**: Changed from Portuguese to English
   - `ambiental` → `environmental`
   - `socioeconomico` → `socioeconomic`
   - `estrategico` → `strategic`
   - `saida` → `priority`

2. **Modularity**: Created reusable `FuzzyInferenceSystem` class

3. **Configuration**: Added JSON-based configuration system

4. **Flexibility**: Support for any number of input variables and custom rules

5. **Error Handling**: Improved error handling and logging

6. **Documentation**: Comprehensive documentation and examples

## Future Enhancements

- Support for OR operators in rules
- Additional membership function types (Gaussian, sigmoid)
- Visualization tools for membership functions
- Parallel processing for large datasets
- Integration with machine learning pipelines 