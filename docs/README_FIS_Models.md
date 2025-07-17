# FIS Model Configurations

This directory contains 6 different Fuzzy Inference System (FIS) configurations based on the `models_fis.csv` file. Each configuration uses the same antecedents (environmental, social, strategic) but different consequent aggregation methods.

## Available Models

### 1. **Round Up Model** (`config_round_up.json`)
- **Aggregation Method**: Round Up
- **Characteristics**: Tends to produce higher priority values
- **Use Case**: Conservative approach, prioritizing areas that need attention

### 2. **Round Down Model** (`config_round_down.json`)
- **Aggregation Method**: Round Down
- **Characteristics**: Tends to produce lower priority values
- **Use Case**: Liberal approach, minimizing false positives

### 3. **Mode Model** (`config_mode.json`)
- **Aggregation Method**: Mode (most frequent value)
- **Characteristics**: Uses the most common value among multiple assessments
- **Use Case**: Consensus-based approach

### 4. **Minimum Model** (`config_minimum.json`)
- **Aggregation Method**: Minimum value
- **Characteristics**: Uses the lowest value among assessments
- **Use Case**: Most conservative approach, only flags areas with clear issues

### 5. **Maximum Model** (`config_max.json`)
- **Aggregation Method**: Maximum value
- **Characteristics**: Uses the highest value among assessments
- **Use Case**: Most sensitive approach, flags any potential issues

### 6. **Median Model** (`config_median.json`)
- **Aggregation Method**: Median value
- **Characteristics**: Uses the middle value among assessments
- **Use Case**: Balanced approach, reduces impact of outliers

## Input Variables

All models use the same input variables with Portuguese labels:

### Environmental (`environmental`)
- **Pouco vulnerável**: Low vulnerability (0-3.75)
- **Vulnerável**: Medium vulnerability (2.5-7.5)
- **Muito vulnerável**: High vulnerability (6.25-10)

### Social (`social`)
- **Pouco vulnerável**: Low vulnerability (0-3.75)
- **Vulnerável**: Medium vulnerability (2.5-7.5)
- **Muito vulnerável**: High vulnerability (6.25-10)

### Strategic (`strategic`)
- **Pouco esforço**: Low effort (0-3.75)
- **Esforço médio**: Medium effort (2.5-7.5)
- **Muito esforço**: High effort (6.25-10)

## Output Variable

All models produce a `priority` output with values 0-10:
- **0**: Very low priority (0-1.25)
- **1**: Low priority (0-3.75)
- **2**: Medium priority (2.5-6.25)
- **3**: High priority (5-8.75)
- **4**: Very high priority (7.5-10)

## Usage Examples

### Using Round Up Model
```bash
python run_raster_fis_parallel.py social.tif environmental.tif strategic.tif output_roundup.tif --config config_round_up.json
```

### Using Minimum Model
```bash
python run_raster_fis_parallel.py social.tif environmental.tif strategic.tif output_min.tif --config config_minimum.json
```

### Using Maximum Model
```bash
python run_raster_fis_parallel.py social.tif environmental.tif strategic.tif output_max.tif --config config_max.json
```

## Model Comparison

| Model | Aggregation | Tendency | Best For |
|-------|-------------|----------|----------|
| Round Up | Ceiling function | Higher values | Conservative assessment |
| Round Down | Floor function | Lower values | Liberal assessment |
| Mode | Most frequent | Consensus | Agreement-based |
| Minimum | Lowest value | Conservative | Risk-averse |
| Maximum | Highest value | Sensitive | Comprehensive |
| Median | Middle value | Balanced | Robust to outliers |

## Rule Structure

Each model contains 27 rules covering all combinations of:
- Environmental: 3 levels (Pouco vulnerável, Vulnerável, Muito vulnerável)
- Social: 3 levels (Pouco vulnerável, Vulnerável, Muito vulnerável)
- Strategic: 3 levels (Pouco esforço, Esforço médio, Muito esforço)

The rules are based on the `models_fis.csv` file and represent different aggregation strategies for the same input conditions.

## Recommendations

1. **Start with Median Model**: Most balanced approach
2. **Use Maximum Model**: If you want to be comprehensive and catch all potential issues
3. **Use Minimum Model**: If you want to be conservative and only flag clear problems
4. **Compare Results**: Run multiple models on the same data to understand the range of outcomes 