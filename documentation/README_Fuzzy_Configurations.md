# Fuzzy Inference System (FIS) Configurations

Your project includes several different fuzzy inference system configurations that use different aggregation methods and rule sets for environmental assessment.

## Available Configurations

### 1. **config_max.json** - Maximum Aggregation Model
- **Description**: Uses maximum aggregation for rule evaluation
- **Use Case**: Conservative approach, prioritizes highest risk factors
- **Output**: Higher priority values when any factor is high

### 2. **config_minimum.json** - Minimum Aggregation Model  
- **Description**: Uses minimum aggregation for rule evaluation
- **Use Case**: Optimistic approach, requires all factors to be high
- **Output**: Lower priority values, only high when all factors are high

### 3. **config_median.json** - Median Aggregation Model
- **Description**: Uses median aggregation for rule evaluation
- **Use Case**: Balanced approach between max and min
- **Output**: Moderate priority values

### 4. **config_mode.json** - Mode Aggregation Model
- **Description**: Uses mode aggregation for rule evaluation
- **Use Case**: Most common value approach
- **Output**: Based on most frequent membership function

### 5. **config_round_up.json** - Round Up Model
- **Description**: Rounds results up to nearest integer
- **Use Case**: Conservative rounding for priority assessment
- **Output**: Higher priority classifications

### 6. **config_round_down.json** - Round Down Model
- **Description**: Rounds results down to nearest integer
- **Use Case**: Conservative rounding for priority assessment
- **Output**: Lower priority classifications

### 7. **raster_fis_config.json** - Default Configuration
- **Description**: Standard fuzzy inference system configuration
- **Use Case**: General purpose environmental assessment
- **Output**: Standard priority values

### 8. **test_config.json** - Test Configuration
- **Description**: Configuration for testing and validation
- **Use Case**: Development and testing
- **Output**: Test results

## How to Use Different Configurations

### **On EMR Master Node:**

```bash
# Navigate to your code directory
cd /opt/raster-fuzzy

# Run with Maximum Aggregation
./scripts/run_on_emr_master.sh \
    s3://your-bucket/input/social.tif \
    s3://your-bucket/input/environmental.tif \
    s3://your-bucket/input/strategic.tif \
    s3://your-bucket/output/result_max.tif \
    app/config/config_max.json

# Run with Minimum Aggregation
./scripts/run_on_emr_master.sh \
    s3://your-bucket/input/social.tif \
    s3://your-bucket/input/environmental.tif \
    s3://your-bucket/input/strategic.tif \
    s3://your-bucket/output/result_min.tif \
    app/config/config_minimum.json

# Run with Median Aggregation
./scripts/run_on_emr_master.sh \
    s3://your-bucket/input/social.tif \
    s3://your-bucket/input/environmental.tif \
    s3://your-bucket/input/strategic.tif \
    s3://your-bucket/output/result_median.tif \
    app/config/config_median.json

# Run with Round Up
./scripts/run_on_emr_master.sh \
    s3://your-bucket/input/social.tif \
    s3://your-bucket/input/environmental.tif \
    s3://your-bucket/input/strategic.tif \
    s3://your-bucket/output/result_round_up.tif \
    app/config/config_round_up.json

# Run with Round Down
./scripts/run_on_emr_master.sh \
    s3://your-bucket/input/social.tif \
    s3://your-bucket/input/environmental.tif \
    s3://your-bucket/input/strategic.tif \
    s3://your-bucket/output/result_round_down.tif \
    app/config/config_round_down.json
```

### **Direct Python Commands:**

```bash
# Maximum Aggregation
python3 app/raster_fuzzy_spark_ultra_optimized.py \
    s3://your-bucket/input/social.tif \
    s3://your-bucket/input/environmental.tif \
    s3://your-bucket/input/strategic.tif \
    s3://your-bucket/output/result_max.tif \
    --config app/config/config_max.json \
    --verbose

# Minimum Aggregation
python3 app/raster_fuzzy_spark_ultra_optimized.py \
    s3://your-bucket/input/social.tif \
    s3://your-bucket/input/environmental.tif \
    s3://your-bucket/input/strategic.tif \
    s3://your-bucket/output/result_min.tif \
    --config app/config/config_minimum.json \
    --verbose

# Median Aggregation
python3 app/raster_fuzzy_spark_ultra_optimized.py \
    s3://your-bucket/input/social.tif \
    s3://your-bucket/input/environmental.tif \
    s3://your-bucket/input/strategic.tif \
    s3://your-bucket/output/result_median.tif \
    --config app/config/config_median.json \
    --verbose
```

## Configuration Comparison

| Configuration | Aggregation Method | Risk Assessment | Use Case |
|---------------|-------------------|-----------------|----------|
| `config_max.json` | Maximum | Conservative | High-risk scenarios |
| `config_minimum.json` | Minimum | Optimistic | Low-risk scenarios |
| `config_median.json` | Median | Balanced | General assessment |
| `config_mode.json` | Mode | Most common | Statistical analysis |
| `config_round_up.json` | Round Up | Conservative | Strict classification |
| `config_round_down.json` | Round Down | Conservative | Strict classification |
| `raster_fis_config.json` | Default | Standard | General purpose |

## Input Variables

All configurations use the same input variables:

- **Environmental**: Environmental vulnerability (0-10)
- **Social**: Social vulnerability (0-10)  
- **Strategic**: Strategic effort required (0-10)

## Output Variables

All configurations produce a **Priority** output (0-10) with 5 levels:
- **0**: No priority
- **1**: Low priority
- **2**: Medium priority
- **3**: High priority
- **4**: Very high priority

## Membership Functions

Each input variable has 3 membership functions:
- **Environmental/Social**: Pouco vulnerável, Vulnerável, Muito vulnerável
- **Strategic**: Pouco esforço, Esforço médio, Muito esforço

## Running Multiple Configurations

To compare different approaches, run multiple configurations:

```bash
# Create a script to run all configurations
cat > run_all_configs.sh << 'EOF'
#!/bin/bash

INPUT_SOCIAL="s3://your-bucket/input/social.tif"
INPUT_ENV="s3://your-bucket/input/environmental.tif"
INPUT_STRAT="s3://your-bucket/input/strategic.tif"
OUTPUT_PREFIX="s3://your-bucket/output/result"

# Run all configurations
./scripts/run_on_emr_master.sh "$INPUT_SOCIAL" "$INPUT_ENV" "$INPUT_STRAT" "${OUTPUT_PREFIX}_max.tif" app/config/config_max.json
./scripts/run_on_emr_master.sh "$INPUT_SOCIAL" "$INPUT_ENV" "$INPUT_STRAT" "${OUTPUT_PREFIX}_min.tif" app/config/config_minimum.json
./scripts/run_on_emr_master.sh "$INPUT_SOCIAL" "$INPUT_ENV" "$INPUT_STRAT" "${OUTPUT_PREFIX}_median.tif" app/config/config_median.json
./scripts/run_on_emr_master.sh "$INPUT_SOCIAL" "$INPUT_ENV" "$INPUT_STRAT" "${OUTPUT_PREFIX}_mode.tif" app/config/config_mode.json
./scripts/run_on_emr_master.sh "$INPUT_SOCIAL" "$INPUT_ENV" "$INPUT_STRAT" "${OUTPUT_PREFIX}_round_up.tif" app/config/config_round_up.json
./scripts/run_on_emr_master.sh "$INPUT_SOCIAL" "$INPUT_ENV" "$INPUT_STRAT" "${OUTPUT_PREFIX}_round_down.tif" app/config/config_round_down.json

echo "All configurations completed!"
EOF

chmod +x run_all_configs.sh
./run_all_configs.sh
```

## Recommendations

- **Start with `raster_fis_config.json`** for general assessment
- **Use `config_max.json`** for conservative risk assessment
- **Use `config_minimum.json`** for optimistic scenarios
- **Compare multiple configurations** to understand sensitivity
- **Use `test_config.json`** for validation and testing 