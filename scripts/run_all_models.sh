#!/bin/bash

# Script to run all FIS models with the same input files
# Usage: ./run_all_models.sh [--nodata VALUE] [--cores N] [--chunk-size N]

# Default parameters
NODATA_VALUE="--nodata 5"
CORES="--cores 8"
CHUNK_SIZE="--chunk-size 10000"
INPUT_DIR="app/files/input/base"
OUTPUT_DIR="app/files/output"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --nodata)
            NODATA_VALUE="--nodata $2"
            shift 2
            ;;
        --cores)
            CORES="--cores $2"
            shift 2
            ;;
        --chunk-size)
            CHUNK_SIZE="--chunk-size $2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--nodata VALUE] [--cores N] [--chunk-size N]"
            exit 1
            ;;
    esac
done

# Input files
ENVIRONMENTAL="$INPUT_DIR/ambiental_1000m.tif"
SOCIAL="$INPUT_DIR/socioeconomico_1000m.tif"
STRATEGIC="$INPUT_DIR/estrategico_1000m.tif"

# Check if input files exist
if [[ ! -f "$ENVIRONMENTAL" ]]; then
    echo "Error: Environmental file not found: $ENVIRONMENTAL"
    exit 1
fi

if [[ ! -f "$SOCIAL" ]]; then
    echo "Error: Social file not found: $SOCIAL"
    exit 1
fi

if [[ ! -f "$STRATEGIC" ]]; then
    echo "Error: Strategic file not found: $STRATEGIC"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Models to run (using arrays for compatibility)
MODEL_NAMES=("round_up" "round_down" "mode" "minimum" "maximum" "median")
MODEL_CONFIGS=("config_round_up.json" "config_round_down.json" "config_mode.json" "config_minimum.json" "config_max.json" "config_median.json")

echo "=== Running All FIS Models ==="
echo "Input files:"
echo "  Environmental: $ENVIRONMENTAL"
echo "  Social: $SOCIAL"
echo "  Strategic: $STRATEGIC"
echo "Parameters: $NODATA_VALUE $CORES $CHUNK_SIZE"
echo ""

# Function to run a single model
run_model() {
    local model_name=$1
    local config_file=$2
    local output_file="output_${model_name}.tif"
    
    echo "Running $model_name model..."
    echo "  Config: $config_file"
    echo "  Output: $output_file"
    
    python app/run_raster_fis_parallel.py "$SOCIAL" "$ENVIRONMENTAL" "$STRATEGIC" "$output_file" \
        --config "app/config/$config_file" $NODATA_VALUE $CORES $CHUNK_SIZE
    
    if [[ $? -eq 0 ]]; then
        echo "  ✓ $model_name completed successfully"
    else
        echo "  ✗ $model_name failed"
        return 1
    fi
    echo ""
}

# Run all models
success_count=0
total_count=0

for i in "${!MODEL_NAMES[@]}"; do
    model_name="${MODEL_NAMES[$i]}"
    config_file="${MODEL_CONFIGS[$i]}"
    total_count=$((total_count + 1))
    if run_model "$model_name" "$config_file"; then
        success_count=$((success_count + 1))
    fi
done

echo "=== Summary ==="
echo "Successfully completed: $success_count/$total_count models"

if [[ $success_count -eq $total_count ]]; then
    echo "✓ All models completed successfully!"
    echo ""
    echo "Output files:"
    for model_name in "${MODEL_NAMES[@]}"; do
        echo "  output_${model_name}.tif"
    done
else
    echo "✗ Some models failed. Check the output above for errors."
    exit 1
fi 