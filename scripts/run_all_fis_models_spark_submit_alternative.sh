#!/bin/bash

# Alternative approach that tries different strategies to avoid Java version issues
# This script attempts multiple approaches to get Spark working

set -e

echo "=== Alternative Spark Submit Strategy ==="

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

echo "=== Running All FIS Models with Alternative Strategy ==="
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

# Function to run a single model with different strategies
run_model_alternative() {
    local model_name="$1"
    local config_path="$2"
    local output_file="$3"
    
    echo "=== Processing $model_name model (Alternative Strategy) ==="
    echo "Config: $config_path"
    echo "Output: $output_file"
    
    start_time=$(date +%s)
    
    # Strategy 1: Try with local mode first (avoids YARN Java issues)
    echo "Trying local mode first..."
    if spark-submit \
        --master local[*] \
        --conf spark.pyspark.python=/usr/bin/python3.9 \
        --conf spark.pyspark.driver.python=/usr/bin/python3.9 \
        --conf spark.executor.memory=8g \
        --conf spark.driver.memory=8g \
        app/raster_fuzzy_spark_ultra_optimized.py \
        "$SOCIAL_TIFF" \
        "$ENVIRONMENTAL_TIFF" \
        "$STRATEGIC_TIFF" \
        "$output_file" \
        --config "$config_path" \
        --verbose; then
        
        echo "✓ $model_name completed in local mode"
        return 0
    fi
    
    echo "Local mode failed, trying YARN with Java 17..."
    
    # Strategy 2: Try YARN with explicit Java 17 configuration
    if spark-submit \
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
        app/raster_fuzzy_spark_ultra_optimized.py \
        "$SOCIAL_TIFF" \
        "$ENVIRONMENTAL_TIFF" \
        "$STRATEGIC_TIFF" \
        "$output_file" \
        --config "$config_path" \
        --verbose; then
        
        echo "✓ $model_name completed in YARN mode"
        return 0
    fi
    
    echo "YARN mode failed, trying client mode..."
    
    # Strategy 3: Try client mode (runs driver on master node)
    if spark-submit \
        --master yarn \
        --deploy-mode client \
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
        --verbose; then
        
        echo "✓ $model_name completed in client mode"
        return 0
    fi
    
    echo "✗ All strategies failed for $model_name"
    return 1
}

# Process each model
for model_name in "${!FIS_CONFIGS[@]}"; do
    config_path="${FIS_CONFIGS[$model_name]}"
    output_file="${OUTPUT_PREFIX}_${model_name}.tif"
    
    if ! run_model_alternative "$model_name" "$config_path" "$output_file"; then
        echo "✗ Failed to process $model_name with all strategies"
        echo "Continuing with next model..."
        echo ""
    fi
    
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    echo "Total time for $model_name: ${duration} seconds"
    echo ""
done

echo "=== All Models Completed ===" 