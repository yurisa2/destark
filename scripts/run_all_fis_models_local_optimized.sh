#!/bin/bash

# Run All FIS Models using Local Mode with Full Resources
# This script runs all available FIS configurations using local Spark mode
# with maximum CPU cores and memory utilization

set -e  # Exit on any error

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

for dir in "${POSSIBLE_DIRS[@]}"; do
    if [ -d "$dir" ] && [ -f "$dir/app/raster_fuzzy_spark_ultra_optimized.py" ]; then
        CODE_DIR="$dir"
        echo "Found code directory: $CODE_DIR"
        break
    fi
done

if [ -z "$CODE_DIR" ]; then
    echo "Error: Could not find the code directory"
    echo "Searched in: ${POSSIBLE_DIRS[*]}"
    exit 1
fi

# Set up environment
export PYTHONPATH="$CODE_DIR:$PYTHONPATH"
cd "$CODE_DIR"

# Check if we have the required arguments
if [ $# -ne 4 ]; then
    echo "Usage: $0 <social_tiff> <environmental_tiff> <strategic_tiff> <output_prefix>"
    echo ""
    echo "Example:"
    echo "  $0 s3://bucket/input/social.tif s3://bucket/input/env.tif s3://bucket/input/strat.tif s3://bucket/output/result"
    echo ""
    echo "This will create:"
    echo "  s3://bucket/output/result_max.tif"
    echo "  s3://bucket/output/result_median.tif"
    echo "  s3://bucket/output/result_minimum.tif"
    echo "  s3://bucket/output/result_mode.tif"
    echo "  s3://bucket/output/result_round_up.tif"
    echo "  s3://bucket/output/result_round_down.tif"
    exit 1
fi

SOCIAL_TIFF="$1"
ENVIRONMENTAL_TIFF="$2"
STRATEGIC_TIFF="$3"
OUTPUT_PREFIX="$4"

echo "=== Running All FIS Models with Local Mode (Full Resources) ==="
echo "Code directory: $CODE_DIR"
echo "Social: $SOCIAL_TIFF"
echo "Environmental: $ENVIRONMENTAL_TIFF"
echo "Strategic: $STRATEGIC_TIFF"
echo "Output prefix: $OUTPUT_PREFIX"
echo ""

# Get system information
CPU_CORES=$(nproc)
TOTAL_MEMORY=$(free -g | awk '/^Mem:/{print $2}')
AVAILABLE_MEMORY=$((TOTAL_MEMORY - 2))  # Leave 2GB for system

echo "System Information:"
echo "  CPU Cores: $CPU_CORES"
echo "  Total Memory: ${TOTAL_MEMORY}GB"
echo "  Available Memory: ${AVAILABLE_MEMORY}GB"
echo ""

# Define all FIS configurations (S3 paths)
declare -A FIS_CONFIGS=(
    ["max"]="s3://adveng-pipeline/unifile_test/config_max.json"
    ["median"]="s3://adveng-pipeline/unifile_test/config_median.json"
    ["minimum"]="s3://adveng-pipeline/unifile_test/config_minimum.json"
    ["mode"]="s3://adveng-pipeline/unifile_test/config_mode.json"
    ["round_up"]="s3://adveng-pipeline/unifile_test/config_round_up.json"
    ["round_down"]="s3://adveng-pipeline/unifile_test/config_round_down.json"
)

# Create temporary directory for config files
TEMP_DIR=$(mktemp -d)
echo "Using temporary directory: $TEMP_DIR"

# Function to download config from S3 if needed
download_config() {
    local config_path="$1"
    local temp_path="$2"
    
    if [[ "$config_path" == s3://* ]]; then
        echo "Downloading config from S3: $config_path"
        aws s3 cp "$config_path" "$temp_path"
        echo "$temp_path"
    else
        echo "$config_path"
    fi
}

# Function to run local processing for a single model
run_local_model() {
    local model_name="$1"
    local config_path="$2"
    local output_file="$3"
    
    echo "=== Processing $model_name model with Local Mode ==="
    echo "Config: $config_path"
    echo "Output: $output_file"
    echo "Using $CPU_CORES cores and ${AVAILABLE_MEMORY}GB memory"
    
    # Download config if it's on S3
    local_config=$(download_config "$config_path" "$TEMP_DIR/config_${model_name}.json")
    
    # Run the processing with local mode
    echo "Starting local processing..."
    start_time=$(date +%s)
    
    python3 app/raster_fuzzy_spark_ultra_optimized.py \
        "$SOCIAL_TIFF" \
        "$ENVIRONMENTAL_TIFF" \
        "$STRATEGIC_TIFF" \
        "$output_file" \
        --config "$local_config" \
        --local \
        --verbose
    
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    
    echo "✓ $model_name model completed in ${duration} seconds"
    echo "  Output: $output_file"
    echo ""
}

# Process each FIS configuration
for model_name in "${!FIS_CONFIGS[@]}"; do
    config_path="${FIS_CONFIGS[$model_name]}"
    output_file="${OUTPUT_PREFIX}_${model_name}.tif"
    
    # Run the model
    if ! run_local_model "$model_name" "$config_path" "$output_file"; then
        echo "✗ Failed to process $model_name model"
        echo "Continuing with next model..."
        echo ""
    fi
done

# Clean up
rm -rf "$TEMP_DIR"
echo "=== All FIS Models Completed ==="
echo "Output files:"
for model_name in "${!FIS_CONFIGS[@]}"; do
    echo "  ${OUTPUT_PREFIX}_${model_name}.tif"
done 