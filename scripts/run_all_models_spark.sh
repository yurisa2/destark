#!/bin/bash

# Batch script to run all raster fuzzy inference models using Spark
# This script processes multiple configurations and outputs using the Spark cluster

set -e  # Exit on any error

# Configuration
INPUT_DIR="app/files/input/base"
OUTPUT_DIR="app/files/output/base"
CONFIG_DIR="app/config"

# Check if Spark cluster is running
echo "Checking Spark cluster status..."
if ! docker compose ps | grep -q "Up"; then
    echo "Error: Spark cluster is not running. Please start it first:"
    echo "sh scripts/setup_spark.sh"
    exit 1
fi

# Check if input files exist
SOCIAL_TIFF="$INPUT_DIR/socioeconomico_300m.tif"
ENVIRONMENTAL_TIFF="$INPUT_DIR/ambiental_300m.tif"
STRATEGIC_TIFF="$INPUT_DIR/estrategico_300m.tif"

for file in "$SOCIAL_TIFF" "$ENVIRONMENTAL_TIFF" "$STRATEGIC_TIFF"; do
    if [[ ! -f "$file" ]]; then
        echo "Error: Input file not found: $file"
        echo "Please ensure the following files exist:"
        echo "  $SOCIAL_TIFF"
        echo "  $ENVIRONMENTAL_TIFF"
        echo "  $STRATEGIC_TIFF"
        exit 1
    fi
done

echo "✓ All input files found"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Get list of config files
CONFIG_FILES=($(ls "$CONFIG_DIR"/*.json 2>/dev/null || echo ""))

if [[ ${#CONFIG_FILES[@]} -eq 0 ]]; then
    echo "Error: No configuration files found in $CONFIG_DIR"
    exit 1
fi

echo "Found ${#CONFIG_FILES[@]} configuration files:"
for config in "${CONFIG_FILES[@]}"; do
    echo "  - $(basename "$config")"
done
echo ""

# Process each configuration
TOTAL_CONFIGS=${#CONFIG_FILES[@]}
CURRENT=0
SUCCESSFUL=0
FAILED=0

for config_file in "${CONFIG_FILES[@]}"; do
    CURRENT=$((CURRENT + 1))
    CONFIG_NAME=$(basename "$config_file" .json)
    OUTPUT_FILE="$OUTPUT_DIR/output_${CONFIG_NAME}.tif"
    
    echo "=== Processing Configuration $CURRENT/$TOTAL_CONFIGS ==="
    echo "Config: $CONFIG_NAME"
    echo "Output: $OUTPUT_FILE"
    echo ""
    
    # Run the Spark job
    if scripts/run_spark_job.sh \
        "$SOCIAL_TIFF" \
        "$ENVIRONMENTAL_TIFF" \
        "$STRATEGIC_TIFF" \
        "$OUTPUT_FILE" \
        --config "$(basename "$config_file")" \
        --chunk-size 1000 \
        --partitions 4 \
        --verbose; then
        
        echo "✓ Successfully processed $CONFIG_NAME"
        SUCCESSFUL=$((SUCCESSFUL + 1))
    else
        echo "✗ Failed to process $CONFIG_NAME"
        FAILED=$((FAILED + 1))
    fi
    
    echo ""
    echo "---"
    echo ""
done

# Summary
echo "=== Batch Processing Summary ==="
echo "Total configurations: $TOTAL_CONFIGS"
echo "Successful: $SUCCESSFUL"
echo "Failed: $FAILED"
echo ""

if [[ $FAILED -eq 0 ]]; then
    echo "✓ All configurations processed successfully!"
    echo ""
    echo "Output files:"
    for config_file in "${CONFIG_FILES[@]}"; do
        CONFIG_NAME=$(basename "$config_file" .json)
        OUTPUT_FILE="$OUTPUT_DIR/output_${CONFIG_NAME}.tif"
        if [[ -f "$OUTPUT_FILE" ]]; then
            echo "  - $OUTPUT_FILE ($(ls -lh "$OUTPUT_FILE" | awk '{print $5}')"
        fi
    done
else
    echo "✗ Some configurations failed. Check the logs above for details."
    exit 1
fi 