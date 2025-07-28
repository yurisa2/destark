#!/bin/bash

# Ultra-Optimized Spark-based Raster Fuzzy Inference Job Submission Script
# This script submits jobs to the Spark cluster using the ultra-optimized version
# with 1.76x performance improvement over the original version

set -e  # Exit on any error

# Default values
NODATA_VALUE=""
PARALLEL_FLAG=""
CORES=""
CHUNK_SIZE=""
CONFIG_FILE="raster_fis_config.json"

# Function to display usage
show_usage() {
    cat << EOF
Usage: $0 <social_tiff> <environmental_tiff> <strategic_tiff> <output_tiff> [options]

Required arguments:
  social_tiff        Path to social factor TIFF file
  environmental_tiff Path to environmental factor TIFF file
  strategic_tiff     Path to strategic factor TIFF file
  output_tiff        Path for output TIFF file

Optional arguments:
  --config FILE      Configuration JSON file (default: raster_fis_config.json)
  --nodata VALUE     NoData value for output raster (default: 5.0)
  --chunk-size SIZE  Number of rows per block (default: 500)
  --partitions NUM   Number of Spark partitions (default: 8 for 8 cores)
  --local            Run in local mode for testing
  --verbose          Enable verbose output
  --help             Show this help message

Examples:
  # Basic usage
  $0 social.tif env.tif strat.tif output.tif

  # With custom parameters
  $0 social.tif env.tif strat.tif output.tif \
    --config my_config.json \
    --chunk-size 500 \
    --partitions 8

  # Local mode for testing
  $0 social.tif env.tif strat.tif output.tif --local

EOF
}

# Parse command line arguments
SOCIAL_TIFF=""
ENVIRONMENTAL_TIFF=""
STRATEGIC_TIFF=""
OUTPUT_TIFF=""
LOCAL_MODE=""
VERBOSE=""
CHUNK_SIZE=""
PARTITIONS=""
NODATA_VALUE=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --nodata)
            NODATA_VALUE="--nodata $2"
            shift 2
            ;;
        --chunk-size)
            CHUNK_SIZE="--block-size $2"
            shift 2
            ;;
        --partitions)
            PARTITIONS="--partitions $2"
            shift 2
            ;;
        --local)
            LOCAL_MODE="--local"
            shift
            ;;
        --verbose)
            VERBOSE="--verbose"
            shift
            ;;
        --help)
            show_usage
            exit 0
            ;;
        -*)
            echo "Error: Unknown option $1"
            show_usage
            exit 1
            ;;
        *)
            # Positional arguments
            if [[ -z "$SOCIAL_TIFF" ]]; then
                SOCIAL_TIFF="$1"
            elif [[ -z "$ENVIRONMENTAL_TIFF" ]]; then
                ENVIRONMENTAL_TIFF="$1"
            elif [[ -z "$STRATEGIC_TIFF" ]]; then
                STRATEGIC_TIFF="$1"
            elif [[ -z "$OUTPUT_TIFF" ]]; then
                OUTPUT_TIFF="$1"
            else
                echo "Error: Too many positional arguments"
                show_usage
                exit 1
            fi
            shift
            ;;
    esac
done

# Check required arguments
if [[ -z "$SOCIAL_TIFF" || -z "$ENVIRONMENTAL_TIFF" || -z "$STRATEGIC_TIFF" || -z "$OUTPUT_TIFF" ]]; then
    echo "Error: Missing required arguments"
    show_usage
    exit 1
fi

# Check if input files exist
for file in "$SOCIAL_TIFF" "$ENVIRONMENTAL_TIFF" "$STRATEGIC_TIFF"; do
    if [[ ! -f "$file" ]]; then
        echo "Error: Input file not found: $file"
        exit 1
    fi
done

# Check if config file exists
if [[ ! -f "app/config/$CONFIG_FILE" ]]; then
    echo "Error: Configuration file not found: app/config/$CONFIG_FILE"
    echo "Available config files:"
    ls -la app/config/*.json 2>/dev/null || echo "No config files found in app/config/"
    exit 1
fi

# Check if Spark cluster is running
echo "Checking Spark cluster status..."
if ! docker compose ps | grep -q "Up"; then
    echo "Error: Spark cluster is not running. Please start it first:"
    echo "sh scripts/setup_spark.sh"
    exit 1
fi

# Create output directory if it doesn't exist
OUTPUT_DIR=$(dirname "$OUTPUT_TIFF")
if [[ ! -d "$OUTPUT_DIR" ]]; then
    mkdir -p "$OUTPUT_DIR"
    echo "Created output directory: $OUTPUT_DIR"
fi

# Build the command (using ultra-optimized version)
CMD="python /home/jovyan/app/raster_fuzzy_spark_ultra_optimized.py"
CMD="$CMD \"$SOCIAL_TIFF\" \"$ENVIRONMENTAL_TIFF\" \"$STRATEGIC_TIFF\" \"$OUTPUT_TIFF\""
CMD="$CMD --config \"/home/jovyan/app/config/$CONFIG_FILE\""

# Add optional parameters
if [[ -n "$NODATA_VALUE" ]]; then
    CMD="$CMD $NODATA_VALUE"
fi
if [[ -n "$CHUNK_SIZE" ]]; then
    CMD="$CMD $CHUNK_SIZE"
fi
if [[ -n "$PARTITIONS" ]]; then
    CMD="$CMD $PARTITIONS"
fi
if [[ -n "$LOCAL_MODE" ]]; then
    CMD="$CMD $LOCAL_MODE"
fi
if [[ -n "$VERBOSE" ]]; then
    CMD="$CMD $VERBOSE"
fi

# Display job information
echo "=== Ultra-Optimized Spark Job Submission ==="
echo "Input files:"
echo "  Social: $SOCIAL_TIFF"
echo "  Environmental: $ENVIRONMENTAL_TIFF"
echo "  Strategic: $STRATEGIC_TIFF"
echo "Output: $OUTPUT_TIFF"
echo "Config: app/config/$CONFIG_FILE"
echo "Mode: $([ -n "$LOCAL_MODE" ] && echo "Local" || echo "Cluster")"
echo ""

# Create logs directory if it doesn't exist
mkdir -p logs

# Generate timestamp for log files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/spark_job_${TIMESTAMP}.log"
ERROR_LOG="logs/spark_job_${TIMESTAMP}_error.log"

# Submit the job to the Spark cluster
echo "Submitting job to Spark cluster..."
echo "Command: $CMD"
echo "Logs will be saved to: $LOG_FILE"
echo "Errors will be saved to: $ERROR_LOG"
echo ""

# Use the Jupyter container to run the job and capture logs
if docker compose exec jupyter bash -c "$CMD" 2> "$ERROR_LOG" | tee "$LOG_FILE"; then
    echo ""
    echo "✓ Job completed successfully!"
    echo "Output saved to: $OUTPUT_TIFF"
    
    # Check if output file exists and show info
    if [[ -f "$OUTPUT_TIFF" ]]; then
        echo "Output file size: $(ls -lh "$OUTPUT_TIFF" | awk '{print $5}')"
        echo "Output file created: $(stat -f "%Sm" "$OUTPUT_TIFF" 2>/dev/null || stat -c "%y" "$OUTPUT_TIFF" 2>/dev/null || echo "unknown")"
    else
        echo "Warning: Output file not found at expected location"
    fi
else
    echo ""
    echo "✗ Job failed!"
    echo "Check the logs for error details:"
    echo "  Main log: $LOG_FILE"
    echo "  Error log: $ERROR_LOG"
    echo ""
    echo "Recent error log content:"
    if [[ -f "$ERROR_LOG" ]]; then
        tail -20 "$ERROR_LOG"
    fi
    echo ""
    echo "You can also check Spark Web UI at: http://localhost:8080"
    exit 1
fi 