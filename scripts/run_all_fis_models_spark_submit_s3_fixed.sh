#!/bin/bash

# Fixed version that uses correct S3 endpoints to avoid PermanentRedirect errors
# This resolves the S3 bucket endpoint issue

set -e

# Fix Java version compatibility for master node
echo "=== Fixing Java Version Compatibility ==="
export JAVA_HOME="/usr/lib/jvm/jre-17"
export PATH="$JAVA_HOME/bin:$PATH"
echo "JAVA_HOME: $JAVA_HOME"
echo "Java version:"
java -version

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

# Fix S3 endpoints for input files
SOCIAL_TIFF_FIXED=$(echo "$SOCIAL_TIFF" | sed 's|s3://adveng-pipeline/|s3://adveng-pipeline.s3.amazonaws.com/|')
ENVIRONMENTAL_TIFF_FIXED=$(echo "$ENVIRONMENTAL_TIFF" | sed 's|s3://adveng-pipeline/|s3://adveng-pipeline.s3.amazonaws.com/|')
STRATEGIC_TIFF_FIXED=$(echo "$STRATEGIC_TIFF" | sed 's|s3://adveng-pipeline/|s3://adveng-pipeline.s3.amazonaws.com/|')
OUTPUT_PREFIX_FIXED=$(echo "$OUTPUT_PREFIX" | sed 's|s3://adveng-pipeline/|s3://adveng-pipeline.s3.amazonaws.com/|')

echo "=== Running All FIS Models with Spark Submit (S3 Fixed) ==="
echo "Social: $SOCIAL_TIFF_FIXED"
echo "Environmental: $ENVIRONMENTAL_TIFF_FIXED"
echo "Strategic: $STRATEGIC_TIFF_FIXED"
echo "Output prefix: $OUTPUT_PREFIX_FIXED"

# Define configs with correct S3 endpoints (using s3.amazonaws.com)
declare -A FIS_CONFIGS=(
    ["max"]="s3://adveng-pipeline.s3.amazonaws.com/unifile_test/config_max.json"
    ["median"]="s3://adveng-pipeline.s3.amazonaws.com/unifile_test/config_median.json"
    ["minimum"]="s3://adveng-pipeline.s3.amazonaws.com/unifile_test/config_minimum.json"
    ["mode"]="s3://adveng-pipeline.s3.amazonaws.com/unifile_test/config_mode.json"
    ["round_up"]="s3://adveng-pipeline.s3.amazonaws.com/unifile_test/config_round_up.json"
    ["round_down"]="s3://adveng-pipeline.s3.amazonaws.com/unifile_test/config_round_down.json"
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
    
    # Set Java 17 for YARN containers via environment variables
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
        --conf spark.yarn.appMasterEnv.JAVA_HOME="$JAVA_HOME" \
        --conf spark.executorEnv.JAVA_HOME="$JAVA_HOME" \
        --conf spark.yarn.appMasterEnv.PATH="$JAVA_HOME/bin:$PATH" \
        --conf spark.executorEnv.PATH="$JAVA_HOME/bin:$PATH" \
        --conf spark.yarn.appMasterEnv.LD_LIBRARY_PATH="$JAVA_HOME/lib:$LD_LIBRARY_PATH" \
        --conf spark.executorEnv.LD_LIBRARY_PATH="$JAVA_HOME/lib:$LD_LIBRARY_PATH" \
        --conf spark.yarn.appMasterEnv.JAVA_OPTS="-Djava.library.path=$JAVA_HOME/lib" \
        --conf spark.executorEnv.JAVA_OPTS="-Djava.library.path=$JAVA_HOME/lib" \
        app/raster_fuzzy_spark_ultra_optimized.py \
        "$SOCIAL_TIFF_FIXED" \
        "$ENVIRONMENTAL_TIFF_FIXED" \
        "$STRATEGIC_TIFF_FIXED" \
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
    output_file="${OUTPUT_PREFIX_FIXED}_${model_name}.tif"
    
    if ! run_model "$model_name" "$config_path" "$output_file"; then
        echo "✗ Failed to process $model_name"
        echo "Continuing with next model..."
        echo ""
    fi
done

echo "=== All Models Completed ===" 