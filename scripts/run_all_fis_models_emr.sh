#!/bin/bash

# Run All FIS Models on EMR Script
# This script runs all fuzzy inference system configurations sequentially on EMR
# and outputs one file for each model

set -e

# Parse arguments
SOCIAL_TIFF="$1"
ENVIRONMENTAL_TIFF="$2"
STRATEGIC_TIFF="$3"
OUTPUT_PREFIX="$4"
S3_CONFIG_PREFIX="${5:-s3://adveng-pipeline/unifile_test}"

# Validate inputs
if [[ -z "$SOCIAL_TIFF" || -z "$ENVIRONMENTAL_TIFF" || -z "$STRATEGIC_TIFF" || -z "$OUTPUT_PREFIX" ]]; then
    echo "Usage: $0 <social_tiff> <environmental_tiff> <strategic_tiff> <output_prefix> [s3_config_prefix]"
    echo ""
    echo "Examples:"
    echo "  # Run all models with S3 files"
    echo "  $0 s3://bucket/input/social.tif s3://bucket/input/env.tif s3://bucket/input/strat.tif s3://bucket/output/result"
    echo ""
    echo "  # Run with custom S3 config prefix"
    echo "  $0 s3://bucket/input/social.tif s3://bucket/input/env.tif s3://bucket/input/strat.tif s3://bucket/output/result s3://bucket/configs"
    exit 1
fi

echo "=== Running All FIS Models on EMR ==="
echo "Input files:"
echo "  Social: $SOCIAL_TIFF"
echo "  Environmental: $ENVIRONMENTAL_TIFF"
echo "  Strategic: $STRATEGIC_TIFF"
echo "Output prefix: $OUTPUT_PREFIX"
echo "S3 Config prefix: $S3_CONFIG_PREFIX"
echo ""

# Find the code directory
CODE_DIR=""
POSSIBLE_DIRS=(
    "/mnt/destark"
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

# Create working directory
mkdir -p /tmp/raster_processing
cd /tmp/raster_processing

# Define all FIS configurations with S3 paths
declare -A FIS_CONFIGS=(
    ["max"]="${S3_CONFIG_PREFIX}/config_max.json"
    ["minimum"]="${S3_CONFIG_PREFIX}/config_minimum.json"
    ["median"]="${S3_CONFIG_PREFIX}/config_median.json"
    ["mode"]="${S3_CONFIG_PREFIX}/config_mode.json"
    ["round_up"]="${S3_CONFIG_PREFIX}/config_round_up.json"
    ["round_down"]="${S3_CONFIG_PREFIX}/config_round_down.json"
    ["default"]="${S3_CONFIG_PREFIX}/raster_fis_config.json"
)

# Create logs directory
mkdir -p logs

# Generate timestamp for log files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/all_fis_models_emr_${TIMESTAMP}.log"
ERROR_LOG="logs/all_fis_models_emr_${TIMESTAMP}_error.log"

echo "Logs will be saved to: $LOG_FILE"
echo "Errors will be saved to: $ERROR_LOG"
echo ""

# Function to run a single FIS model
run_fis_model() {
    local model_name="$1"
    local s3_config_file="$2"
    local output_file="$3"
    
    echo "=== Running $model_name model ==="
    echo "S3 Config: $s3_config_file"
    echo "Output: $output_file"
    echo ""
    
    # Download config file from S3
    local local_config_file="./config_${model_name}.json"
    echo "Downloading config file from S3..."
    if aws s3 cp "$s3_config_file" "$local_config_file" 2>/dev/null; then
        echo "✓ Config file downloaded successfully"
    else
        echo "✗ Failed to download config file: $s3_config_file"
        echo "Skipping $model_name model..."
        return 1
    fi
    
    # Build the command
    CMD="python3 $CODE_DIR/app/raster_fuzzy_spark_ultra_optimized.py"
    CMD="$CMD \"$SOCIAL_TIFF\" \"$ENVIRONMENTAL_TIFF\" \"$STRATEGIC_TIFF\" \"$output_file\""
    CMD="$CMD --config \"$local_config_file\""
    CMD="$CMD --block-size 1000"
    CMD="$CMD --partitions 8"
    CMD="$CMD --verbose"
    
    echo "Command: $CMD"
    echo ""
    
    # Execute the command
    if eval $CMD 2>> "$ERROR_LOG"; then
        echo "✓ $model_name model completed successfully!"
        echo "Output saved to: $output_file"
    else
        echo "✗ $model_name model failed!"
        echo "Check error log: $ERROR_LOG"
        return 1
    fi
    
    # Clean up local config file
    rm -f "$local_config_file"
    
    echo ""
}

# Run all FIS models
echo "Starting sequential processing of all FIS models..."
echo ""

for model_name in "${!FIS_CONFIGS[@]}"; do
    s3_config_file="${FIS_CONFIGS[$model_name]}"
    output_file="${OUTPUT_PREFIX}_${model_name}.tif"
    
    # Run the model
    if ! run_fis_model "$model_name" "$s3_config_file" "$output_file"; then
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
    echo "  $model_name: $output_file"
done

echo ""
echo "✓ All FIS models processing completed!" 