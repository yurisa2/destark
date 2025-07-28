#!/bin/bash

# Simplified version that directly uses S3 config paths
# This avoids the complex download/upload logic that might be causing issues

set -e

# Fix Java version compatibility
echo "=== Fixing Java Version Compatibility ==="
export JAVA_HOME="/usr/lib/jvm/jre-17"
export PATH="$JAVA_HOME/bin:$PATH"
echo "JAVA_HOME: $JAVA_HOME"

# Set EMR/Hadoop configuration
echo "=== Setting EMR/Hadoop Configuration ==="
export HADOOP_CONF_DIR="/etc/hadoop/conf"
export YARN_CONF_DIR="/etc/hadoop/conf"
echo "HADOOP_CONF_DIR: $HADOOP_CONF_DIR"

# Find code directory
CODE_DIR="/mnt/destark"
cd "$CODE_DIR"

# Check arguments
if [ $# -ne 4 ]; then
    echo "Usage: $0 <social_tiff> <environmental_tiff> <strategic_tiff> <output_prefix>"
    exit 1
fi

SOCIAL_TIFF="$1"
ENVIRONMENTAL_TIFF="$2"
STRATEGIC_TIFF="$3"
OUTPUT_PREFIX="$4"

echo "=== Running All FIS Models with Spark Submit (Simple) ==="
echo "Social: $SOCIAL_TIFF"
echo "Environmental: $ENVIRONMENTAL_TIFF"
echo "Strategic: $STRATEGIC_TIFF"
echo "Output prefix: $OUTPUT_PREFIX"

# Define configs directly as S3 paths
declare -A FIS_CONFIGS=(
    ["max"]="s3://adveng-pipeline/unifile_test/config_max.json"
    ["median"]="s3://adveng-pipeline/unifile_test/config_median.json"
    ["minimum"]="s3://adveng-pipeline/unifile_test/config_minimum.json"
    ["mode"]="s3://adveng-pipeline/unifile_test/config_mode.json"
    ["round_up"]="s3://adveng-pipeline/unifile_test/config_round_up.json"
    ["round_down"]="s3://adveng-pipeline/unifile_test/config_round_down.json"
)

# Function to run a single model
run_model() {
    local model_name="$1"
    local config_path="$2"
    local output_file="$3"
    
    echo "=== Processing $model_name model ==="
    echo "Config: $config_path"
    echo "Output: $output_file"
    
    start_time=$(date +%s)
    
    spark-submit \
        --master yarn \
        --deploy-mode cluster \
        --conf spark.yarn.appMasterEnv.PYTHONPATH="$CODE_DIR" \
        --conf spark.executorEnv.PYTHONPATH="$CODE_DIR" \
        --conf spark.pyspark.python=/usr/bin/python3.9 \
        --conf spark.pyspark.driver.python=/usr/bin/python3.9 \
        --conf spark.executor.memory=8g \
        --conf spark.driver.memory=8g \
        --conf spark.executor.cores=4 \
        --conf spark.yarn.maxAppAttempts=1 \
        --conf spark.yarn.submit.waitAppCompletion=true \
        app/raster_fuzzy_spark_ultra_optimized.py \
        "$SOCIAL_TIFF" \
        "$ENVIRONMENTAL_TIFF" \
        "$STRATEGIC_TIFF" \
        "$output_file" \
        --config "$config_path" \
        --verbose
    
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    
    echo "✓ $model_name completed in ${duration} seconds"
    echo ""
}

# Process each model
for model_name in "${!FIS_CONFIGS[@]}"; do
    config_path="${FIS_CONFIGS[$model_name]}"
    output_file="${OUTPUT_PREFIX}_${model_name}.tif"
    
    if ! run_model "$model_name" "$config_path" "$output_file"; then
        echo "✗ Failed to process $model_name"
        echo "Continuing with next model..."
        echo ""
    fi
done

echo "=== All Models Completed ===" 