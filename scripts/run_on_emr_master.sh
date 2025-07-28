#!/bin/bash

# Run Raster Fuzzy Inference on EMR Master Node
# This script runs the processing directly on the EMR master node

set -e

# Parse arguments
SOCIAL_TIFF="$1"
ENVIRONMENTAL_TIFF="$2"
STRATEGIC_TIFF="$3"
OUTPUT_TIFF="$4"
CONFIG_FILE="${5:-raster_fis_config.json}"
CHUNK_SIZE="${6:-1000}"
PARTITIONS="${7:-8}"
LOCAL_MODE="${8:-false}"

# Validate inputs
if [[ -z "$SOCIAL_TIFF" || -z "$ENVIRONMENTAL_TIFF" || -z "$STRATEGIC_TIFF" || -z "$OUTPUT_TIFF" ]]; then
    echo "Usage: $0 <social_tiff> <environmental_tiff> <strategic_tiff> <output_tiff> [config_file] [chunk_size] [partitions] [local_mode]"
    echo ""
    echo "Examples:"
    echo "  # Basic usage with S3 files"
    echo "  $0 s3://bucket/social.tif s3://bucket/env.tif s3://bucket/strat.tif s3://bucket/output.tif"
    echo ""
    echo "  # With custom parameters"
    echo "  $0 s3://bucket/social.tif s3://bucket/env.tif s3://bucket/strat.tif s3://bucket/output.tif my_config.json 2000 16"
    echo ""
    echo "  # Local mode for testing"
    echo "  $0 local_social.tif local_env.tif local_strat.tif local_output.tif config.json 500 4 true"
    exit 1
fi

echo "=== Running Raster Fuzzy Inference on EMR ==="
echo "Input files:"
echo "  Social: $SOCIAL_TIFF"
echo "  Environmental: $ENVIRONMENTAL_TIFF"
echo "  Strategic: $STRATEGIC_TIFF"
echo "Output: $OUTPUT_TIFF"
echo "Config: $CONFIG_FILE"
echo "Chunk size: $CHUNK_SIZE"
echo "Partitions: $PARTITIONS"
echo "Local mode: $LOCAL_MODE"
echo ""

# Set up environment
export PYTHONPATH="/opt/raster-fuzzy:$PYTHONPATH"
cd /opt/raster-fuzzy

# Create working directory
mkdir -p /tmp/raster_processing
cd /tmp/raster_processing

# Download config file if it's an S3 path
if [[ "$CONFIG_FILE" == s3://* ]]; then
    echo "Downloading config file from S3..."
    aws s3 cp "$CONFIG_FILE" ./config.json
    CONFIG_FILE="./config.json"
fi

# Build the command
CMD="python3 /opt/raster-fuzzy/app/raster_fuzzy_spark_ultra_optimized.py"
CMD="$CMD \"$SOCIAL_TIFF\" \"$ENVIRONMENTAL_TIFF\" \"$STRATEGIC_TIFF\" \"$OUTPUT_TIFF\""
CMD="$CMD --config \"$CONFIG_FILE\""
CMD="$CMD --block-size $CHUNK_SIZE"
CMD="$CMD --partitions $PARTITIONS"
CMD="$CMD --verbose"

# Add local mode if specified
if [[ "$LOCAL_MODE" == "true" ]]; then
    CMD="$CMD --local"
fi

# Run the job
echo "Starting raster fuzzy inference..."
echo "Command: $CMD"
echo ""

# Execute the command
if eval $CMD; then
    echo ""
    echo "✓ Raster fuzzy inference completed successfully!"
    echo "Output saved to: $OUTPUT_TIFF"
    
    # Show output file info if it's local
    if [[ "$OUTPUT_TIFF" != s3://* ]] && [[ -f "$OUTPUT_TIFF" ]]; then
        echo "Output file size: $(ls -lh "$OUTPUT_TIFF" | awk '{print $5}')"
    fi
else
    echo ""
    echo "✗ Raster fuzzy inference failed!"
    exit 1
fi 