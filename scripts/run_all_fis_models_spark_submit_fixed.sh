#!/bin/bash

# Run All FIS Models using Spark Submit on EMR Cluster
# This script runs all available FIS configurations using spark-submit
# to utilize the full power of the EMR cluster
# FIXED VERSION: Handles Java version compatibility issues and EMR configuration

set -e  # Exit on any error

# Fix Java version compatibility issue
echo "=== Fixing Java Version Compatibility ==="
echo "Current Java version:"
java -version 2>&1 || echo "Java not found"

# Set JAVA_HOME to Java 17 for Spark compatibility
if [ -d "/usr/lib/jvm/java-17-amazon-corretto" ]; then
    export JAVA_HOME="/usr/lib/jvm/java-17-amazon-corretto"
    export PATH="$JAVA_HOME/bin:$PATH"
    echo "Set JAVA_HOME to Amazon Corretto 17"
elif [ -d "/usr/lib/jvm/jre-17" ]; then
    export JAVA_HOME="/usr/lib/jvm/jre-17"
    export PATH="$JAVA_HOME/bin:$PATH"
    echo "Set JAVA_HOME to JRE 17"
elif [ -d "/usr/lib/jvm/java-17-openjdk" ]; then
    export JAVA_HOME="/usr/lib/jvm/java-17-openjdk"
    export PATH="$JAVA_HOME/bin:$PATH"
    echo "Set JAVA_HOME to OpenJDK 17"
else
    echo "Warning: Java 17 not found in standard locations"
    echo "Available Java installations:"
    ls -la /usr/lib/jvm/ 2>/dev/null || echo "No /usr/lib/jvm/ directory found"
    echo ""
    echo "Trying to find Java 17..."
    find /usr -name "java-17*" -type d 2>/dev/null | head -5
    echo ""
fi

echo "Updated Java version:"
java -version 2>&1 || echo "Java still not working"
echo "JAVA_HOME: $JAVA_HOME"
echo ""

# Set EMR/Hadoop configuration directories
echo "=== Setting EMR/Hadoop Configuration ==="
if [ -d "/etc/hadoop/conf" ]; then
    export HADOOP_CONF_DIR="/etc/hadoop/conf"
    echo "Set HADOOP_CONF_DIR to /etc/hadoop/conf"
elif [ -d "/etc/emr/conf" ]; then
    export HADOOP_CONF_DIR="/etc/emr/conf"
    echo "Set HADOOP_CONF_DIR to /etc/emr/conf"
elif [ -d "/usr/lib/hadoop/etc/hadoop" ]; then
    export HADOOP_CONF_DIR="/usr/lib/hadoop/etc/hadoop"
    echo "Set HADOOP_CONF_DIR to /usr/lib/hadoop/etc/hadoop"
else
    echo "Warning: HADOOP_CONF_DIR not found in standard locations"
    echo "Available Hadoop config directories:"
    find /etc -name "*hadoop*" -type d 2>/dev/null | head -5
    find /usr -name "*hadoop*" -type d 2>/dev/null | head -5
    echo ""
fi

if [ -d "/etc/hadoop/conf" ]; then
    export YARN_CONF_DIR="/etc/hadoop/conf"
    echo "Set YARN_CONF_DIR to /etc/hadoop/conf"
elif [ -d "/etc/emr/conf" ]; then
    export YARN_CONF_DIR="/etc/emr/conf"
    echo "Set YARN_CONF_DIR to /etc/emr/conf"
elif [ -d "/usr/lib/hadoop/etc/hadoop" ]; then
    export YARN_CONF_DIR="/usr/lib/hadoop/etc/hadoop"
    echo "Set YARN_CONF_DIR to /usr/lib/hadoop/etc/hadoop"
else
    echo "Warning: YARN_CONF_DIR not found in standard locations"
fi

echo "HADOOP_CONF_DIR: $HADOOP_CONF_DIR"
echo "YARN_CONF_DIR: $YARN_CONF_DIR"
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

echo "=== Running All FIS Models with Spark Submit ==="
echo "Code directory: $CODE_DIR"
echo "Social: $SOCIAL_TIFF"
echo "Environmental: $ENVIRONMENTAL_TIFF"
echo "Strategic: $STRATEGIC_TIFF"
echo "Output prefix: $OUTPUT_PREFIX"
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

# Function to run spark-submit for a single model
run_spark_submit() {
    local model_name="$1"
    local config_path="$2"
    local output_file="$3"
    
    echo "=== Processing $model_name model with Spark Submit ==="
    echo "Config: $config_path"
    echo "Output: $output_file"
    
    # Download config if it's on S3
    local_config=$(download_config "$config_path" "$TEMP_DIR/config_${model_name}.json")
    
    # Test Spark before running
    echo "Testing Spark configuration..."
    if ! spark-submit --version >/dev/null 2>&1; then
        echo "Warning: spark-submit version check failed, but continuing..."
    fi
    
    # Run the processing with spark-submit
    echo "Starting Spark Submit processing..."
    start_time=$(date +%s)
    
    # Ensure JAVA_HOME is set for spark-submit
    export JAVA_HOME="${JAVA_HOME:-/usr/lib/jvm/java-17-amazon-corretto}"
    
    # Ensure Hadoop/YARN config directories are set
    if [ -z "$HADOOP_CONF_DIR" ] && [ -d "/etc/hadoop/conf" ]; then
        export HADOOP_CONF_DIR="/etc/hadoop/conf"
    fi
    if [ -z "$YARN_CONF_DIR" ] && [ -d "/etc/hadoop/conf" ]; then
        export YARN_CONF_DIR="/etc/hadoop/conf"
    fi
    
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
        --conf spark.sql.shuffle.partitions=20 \
        --conf spark.serializer=org.apache.spark.serializer.KryoSerializer \
        --conf spark.sql.files.maxPartitionBytes=128m \
        --conf spark.sql.broadcastTimeout=300 \
        --conf spark.sql.autoBroadcastJoinThreshold=10485760 \
        --conf spark.executor.extraJavaOptions="-XX:+UseG1GC -XX:MaxGCPauseMillis=200" \
        --conf spark.driver.extraJavaOptions="-XX:+UseG1GC -XX:MaxGCPauseMillis=200" \
        --conf spark.yarn.maxAppAttempts=1 \
        --conf spark.yarn.submit.waitAppCompletion=true \
        --conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=/usr/bin/python3.9 \
        --conf spark.executorEnv.PYSPARK_PYTHON=/usr/bin/python3.9 \
        --conf spark.yarn.appMasterEnv.JAVA_HOME="$JAVA_HOME" \
        --conf spark.executorEnv.JAVA_HOME="$JAVA_HOME" \
        app/raster_fuzzy_spark_ultra_optimized.py \
        "$SOCIAL_TIFF" \
        "$ENVIRONMENTAL_TIFF" \
        "$STRATEGIC_TIFF" \
        "$output_file" \
        --config "$local_config" \
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
    if ! run_spark_submit "$model_name" "$config_path" "$output_file"; then
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