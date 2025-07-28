#!/bin/bash

# Test script to debug config file issue

echo "=== Testing Config File Access ==="

# Test the exact same command that's failing
SOCIAL_TIFF="s3://adveng-pipeline/unifile_test/so300m.in"
ENVIRONMENTAL_TIFF="s3://adveng-pipeline/unifile_test/e300m.in"
STRATEGIC_TIFF="s3://adveng-pipeline/unifile_test/s300m.in"
OUTPUT_FILE="s3://adveng-pipeline/unifile_test/result_300m_round_up.tif"

# Create temporary directory
TEMP_DIR=$(mktemp -d)
echo "Using temporary directory: $TEMP_DIR"

# Download config file
echo "Downloading config file..."
aws s3 cp "s3://adveng-pipeline/unifile_test/config_round_up.json" "$TEMP_DIR/config_round_up.json"

# Check if file was downloaded
if [ -f "$TEMP_DIR/config_round_up.json" ]; then
    echo "✓ Config file downloaded successfully"
    echo "File size: $(ls -lh "$TEMP_DIR/config_round_up.json" | awk '{print $5}')"
    echo "File content (first 5 lines):"
    head -5 "$TEMP_DIR/config_round_up.json"
else
    echo "✗ Config file download failed"
    exit 1
fi

# Test Python script with the downloaded config
echo ""
echo "=== Testing Python Script ==="
echo "Current directory: $(pwd)"
echo "Config file path: $TEMP_DIR/config_round_up.json"

python3 app/raster_fuzzy_spark_ultra_optimized.py \
    "$SOCIAL_TIFF" \
    "$ENVIRONMENTAL_TIFF" \
    "$STRATEGIC_TIFF" \
    "$OUTPUT_FILE" \
    --config "$TEMP_DIR/config_round_up.json" \
    --local \
    --verbose

# Clean up
rm -rf "$TEMP_DIR" 