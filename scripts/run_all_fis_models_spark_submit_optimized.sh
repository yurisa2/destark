#!/bin/bash

# Optimized version for EMR cluster: 1 master, 2 core, 1 task (all m5.xlarge, 4 vCPUs, 16GB RAM)
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

# Install missing Python dependencies
echo "=== Installing Missing Python Dependencies ==="
pip3 install python-dateutil boto3

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

echo "=== Running All FIS Models with Spark Submit (EMR-optimized, nohup background) ==="
echo "Social: $SOCIAL_TIFF"
echo "Environmental: $ENVIRONMENTAL_TIFF"
echo "Strategic: $STRATEGIC_TIFF"
echo "Output prefix: $OUTPUT_PREFIX"

declare -A FIS_CONFIGS=(
    ["max"]="s3://adveng-pipeline/unifile_test/config_max.json"
    ["median"]="s3://adveng-pipeline/unifile_test/config_median.json"
    ["minimum"]="s3://adveng-pipeline/unifile_test/config_minimum.json"
    ["mode"]="s3://adveng-pipeline/unifile_test/config_mode.json"
    ["round_up"]="s3://adveng-pipeline/unifile_test/config_round_up.json"
    ["round_down"]="s3://adveng-pipeline/unifile_test/config_round_down.json"
)

run_model() {
    local model_name="$1"
    local config_path="$2"
    local output_file="$3"
    echo "=== Submitting $model_name model (nohup background) ==="
    echo "Config: $config_path"
    echo "Output: $output_file"
    nohup spark-submit \
        --master yarn \
        --deploy-mode cluster \
        --conf spark.yarn.appMasterEnv.PYTHONPATH="$CODE_DIR" \
        --conf spark.executorEnv.PYTHONPATH="$CODE_DIR" \
        --conf spark.pyspark.python=/usr/bin/python3.9 \
        --conf spark.pyspark.driver.python=/usr/bin/python3.9 \
        --conf spark.executor.memory=8g \
        --conf spark.driver.memory=2g \
        --conf spark.executor.cores=2 \
        --conf spark.driver.cores=1 \
        --conf spark.dynamicAllocation.enabled=true \
        --conf spark.dynamicAllocation.maxExecutors=4 \
        --conf spark.dynamicAllocation.minExecutors=1 \
        --conf spark.sql.shuffle.partitions=500 \
        --conf spark.default.parallelism=500 \
        --conf spark.yarn.maxAppAttempts=1 \
        --conf spark.yarn.appMasterEnv.JAVA_HOME="$JAVA_HOME" \
        --conf spark.executorEnv.JAVA_HOME="$JAVA_HOME" \
        --conf spark.yarn.appMasterEnv.PATH="$JAVA_HOME/bin:$PATH" \
        --conf spark.executorEnv.PATH="$JAVA_HOME/bin:$PATH" \
        --conf spark.yarn.appMasterEnv.LD_LIBRARY_PATH="$JAVA_HOME/lib:$LD_LIBRARY_PATH" \
        --conf spark.executorEnv.LD_LIBRARY_PATH="$JAVA_HOME/lib:$LD_LIBRARY_PATH" \
        --conf spark.yarn.appMasterEnv.JAVA_OPTS="-Djava.library.path=$JAVA_HOME/lib" \
        --conf spark.executorEnv.JAVA_OPTS="-Djava.library.path=$JAVA_HOME/lib" \
        --conf spark.yarn.appMasterEnv.PYTHONPATH="/usr/local/lib/python3.9/site-packages:$PYTHONPATH" \
        --conf spark.executorEnv.PYTHONPATH="/usr/local/lib/python3.9/site-packages:$PYTHONPATH" \
        --conf spark.yarn.appMasterEnv.AWS_S3_ENDPOINT="s3.amazonaws.com" \
        --conf spark.executorEnv.AWS_S3_ENDPOINT="s3.amazonaws.com" \
        --conf spark.yarn.appMasterEnv.GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR" \
        --conf spark.executorEnv.GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR" \
        --conf spark.rpc.message.maxSize=512 \
        --conf spark.sql.adaptive.enabled=true \
        --conf spark.sql.adaptive.coalescePartitions.enabled=true \
        --conf spark.sql.adaptive.skewJoin.enabled=true \
        --conf spark.sql.adaptive.localShuffleReader.enabled=true \
        --conf spark.sql.adaptive.advisoryPartitionSizeInBytes=128m \
        --conf spark.sql.adaptive.maxShuffledHashJoinLocalMapThreshold=0 \
        --conf spark.sql.adaptive.forceApply=true \
        --conf spark.sql.adaptive.logLevel=DEBUG \
        --conf spark.serializer=org.apache.spark.serializer.KryoSerializer \
        --conf spark.kryoserializer.buffer.max=512m \
        --conf spark.kryoserializer.buffer=128m \
        --conf spark.sql.adaptive.maxShuffledHashJoinLocalMapThreshold=0 \
        --conf spark.executor.extraJavaOptions="-XX:+UseG1GC -XX:MaxGCPauseMillis=200" \
        --conf spark.driver.extraJavaOptions="-XX:+UseG1GC -XX:MaxGCPauseMillis=200" \
        --conf spark.sql.adaptive.maxShuffledHashJoinLocalMapThreshold=0 \
        --conf spark.sql.adaptive.optimizeSkewedJoin.enabled=true \
        --conf spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes=256m \
        --conf spark.sql.adaptive.skewJoin.skewedPartitionFactor=5 \
        --conf spark.yarn.queue=default \
        app/raster_fuzzy_spark_ultra_optimized.py \
        "$SOCIAL_TIFF" \
        "$ENVIRONMENTAL_TIFF" \
        "$STRATEGIC_TIFF" \
        "$output_file" \
        --config "$config_path" \
        --verbose \
        --block-size 100 \
        > logs/${model_name}_nohup.log 2>&1 &
    echo "✓ $model_name submitted (nohup, PID: $!)"
    echo ""
}

for model_name in "${!FIS_CONFIGS[@]}"; do
    config_path="${FIS_CONFIGS[$model_name]}"
    output_file="${OUTPUT_PREFIX}_${model_name}.tif"
    run_model "$model_name" "$config_path" "$output_file"
done

echo "=== All Models Submitted in Background (nohup) ==="
echo "Jobs will continue even if you disconnect the terminal."
echo "Monitor logs in the logs/ directory or with: tail -f logs/<model>_nohup.log"
echo "Check YARN: yarn application -list"
echo "View logs: yarn logs -applicationId <app_id>" 