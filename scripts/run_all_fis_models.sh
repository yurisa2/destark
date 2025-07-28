#!/bin/bash

# Run All FIS Models Script
# This script runs all fuzzy inference system configurations sequentially
# and outputs one file for each model

set -e

# Parse arguments
SOCIAL_TIFF="$1"
ENVIRONMENTAL_TIFF="$2"
STRATEGIC_TIFF="$3"
OUTPUT_PREFIX="$4"
CONFIG_DIR="${5:-app/config}"

# Validate inputs
if [[ -z "$SOCIAL_TIFF" || -z "$ENVIRONMENTAL_TIFF" || -z "$STRATEGIC_TIFF" || -z "$OUTPUT_PREFIX" ]]; then
    echo "Usage: $0 <social_tiff> <environmental_tiff> <strategic_tiff> <output_prefix> [config_dir]"
    echo ""
    echo "Examples:"
    echo "  # Run all models with S3 files"
    echo "  $0 s3://bucket/input/social.tif s3://bucket/input/env.tif s3://bucket/input/strat.tif s3://bucket/output/result"
    echo ""
    echo "  # Run all models with local files"
    echo "  $0 social.tif env.tif strat.tif ./output/result"
    echo ""
    echo "  # Run with custom config directory"
    echo "  $0 social.tif env.tif strat.tif ./output/result /path/to/configs"
    exit 1
fi

echo "=== Running All FIS Models ==="
echo "Input files:"
echo "  Social: $SOCIAL_TIFF"
echo "  Environmental: $ENVIRONMENTAL_TIFF"
echo "  Strategic: $STRATEGIC_TIFF"
echo "Output prefix: $OUTPUT_PREFIX"
echo "Config directory: $CONFIG_DIR"
echo ""

# Find the code directory
CODE_DIR=""
POSSIBLE_DIRS=(
    "/opt/raster-fuzzy"
    "/home/hadoop/raster-fuzzy"
    "/home/hadoop/destark"
    "/home/ssm-user/destark"
    "/tmp/raster-fuzzy"
    "/usr/local/raster-fuzzy"
    "/opt/destark"
    "/home/hadoop"
    "/tmp"
)

echo "Looking for code directory..."
for dir in "${POSSIBLE_DIRS[@]}"; do
    if [[ -d "$dir" ]] && [[ -f "$dir/app/raster_fuzzy_spark_ultra_optimized.py" ]]; then
        CODE_DIR="$dir"
        echo "Found code in: $CODE_DIR"
        break
    fi
done

if [[ -z "$CODE_DIR" ]]; then
    echo "Error: Could not find the raster fuzzy code directory."
    echo "Please check where your code is located and update the script."
    exit 1
fi

# Set up environment
export PYTHONPATH="$CODE_DIR:$PYTHONPATH"
cd "$CODE_DIR"

echo "Working directory: $(pwd)"
echo "Python path: $PYTHONPATH"

# Define all FIS configurations
declare -A FIS_CONFIGS=(
    ["max"]="config_max.json"
    ["minimum"]="config_minimum.json"
    ["median"]="config_median.json"
    ["mode"]="config_mode.json"
    ["round_up"]="config_round_up.json"
    ["round_down"]="config_round_down.json"
    ["default"]="raster_fis_config.json"
)

# Create output directory if it's local
if [[ "$OUTPUT_PREFIX" != s3://* ]]; then
    OUTPUT_DIR=$(dirname "$OUTPUT_PREFIX")
    if [[ ! -d "$OUTPUT_DIR" ]]; then
        mkdir -p "$OUTPUT_DIR"
        echo "Created output directory: $OUTPUT_DIR"
    fi
fi

# Create logs directory
mkdir -p logs

# Generate timestamp for log files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/all_fis_models_${TIMESTAMP}.log"
ERROR_LOG="logs/all_fis_models_${TIMESTAMP}_error.log"

echo "Logs will be saved to: $LOG_FILE"
echo "Errors will be saved to: $ERROR_LOG"
echo ""

# Function to run a single FIS model
run_fis_model() {
    local model_name="$1"
    local config_file="$2"
    local output_file="$3"
    
    echo "=== Running $model_name model ==="
    echo "Config: $config_file"
    echo "Output: $output_file"
    echo ""
    
    # Build the command
    CMD="python3 $CODE_DIR/app/raster_fuzzy_spark_ultra_optimized.py"
    CMD="$CMD \"$SOCIAL_TIFF\" \"$ENVIRONMENTAL_TIFF\" \"$STRATEGIC_TIFF\" \"$output_file\""
    CMD="$CMD --config \"$config_file\""
    CMD="$CMD --block-size 1000"
    CMD="$CMD --partitions 8"
    CMD="$CMD --verbose"
    
    echo "Command: $CMD"
    echo ""
    
    # Execute the command
    if eval $CMD 2>> "$ERROR_LOG"; then
        echo "✓ $model_name model completed successfully!"
        echo "Output saved to: $output_file"
        
        # Show output file info if it's local
        if [[ "$output_file" != s3://* ]] && [[ -f "$output_file" ]]; then
            echo "Output file size: $(ls -lh "$output_file" | awk '{print $5}')"
        fi
    else
        echo "✗ $model_name model failed!"
        echo "Check error log: $ERROR_LOG"
        return 1
    fi
    
    echo ""
}

# Run all FIS models
echo "Starting sequential processing of all FIS models..."
echo ""

for model_name in "${!FIS_CONFIGS[@]}"; do
    config_file="$CONFIG_DIR/${FIS_CONFIGS[$model_name]}"
    output_file="${OUTPUT_PREFIX}_${model_name}.tif"
    
    # Check if config file exists
    if [[ ! -f "$config_file" ]]; then
        echo "Warning: Config file not found: $config_file"
        echo "Skipping $model_name model..."
        echo ""
        continue
    fi
    
    # Run the model
    if ! run_fis_model "$model_name" "$config_file" "$output_file"; then
        echo "Error: Failed to run $model_name model"
        echo "Continuing with next model..."
        echo ""
    fi
    
    echo "---"
done

echo "=== All FIS Models Processing Complete ==="
echo ""
echo "Summary:"
echo "Log file: $LOG_FILE"
echo "Error log: $ERROR_LOG"
echo ""

# List all output files
echo "Generated output files:"
for model_name in "${!FIS_CONFIGS[@]}"; do
    output_file="${OUTPUT_PREFIX}_${model_name}.tif"
    if [[ "$output_file" == s3://* ]]; then
        echo "  $model_name: $output_file"
    elif [[ -f "$output_file" ]]; then
        echo "  $model_name: $output_file ($(ls -lh "$output_file" | awk '{print $5}'))"
    else
        echo "  $model_name: $output_file (not found)"
    fi
done

echo ""
echo "✓ All FIS models processing completed!" 