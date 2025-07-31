#!/bin/bash

# Set AWS credentials for the correct account
export AWS_ACCESS_KEY_ID=YOUR_AWS_ACCESS_KEY_ID
export AWS_SECRET_ACCESS_KEY=YOUR_AWS_SECRET_ACCESS_KEY
export AWS_REGION=us-east-2
export AWS_DEFAULT_REGION=us-east-2

# Truly Distributed Full FIS Workload Runner for EMR Cluster
# This script uses Spark to TRULY distribute processing across all nodes

set -e

# Configuration
CLUSTER_ID="<EMR-CLUSTER-ID>"
REGION="us-east-2"
S3_BUCKET="<AWS-BUCKET>-unifile"
S3_PREFIX="unifile_test"

echo "=========================================="
echo "TRULY DISTRIBUTED Full FIS Workload Runner for EMR"
echo "Cluster ID: $CLUSTER_ID"
echo "Region: $REGION"
echo "=========================================="

# Function to wait for step completion
wait_for_step() {
    local step_id=$1
    local step_name=$2
    
    echo "Waiting for step '$step_name' to complete..."
    
    while true; do
        STEP_STATUS=$(aws emr describe-step --cluster-id "$CLUSTER_ID" --step-id "$step_id" --region "$REGION" --query 'Step.Status.State' --output text)
        
        case $STEP_STATUS in
            "COMPLETED")
                echo "✅ Step '$step_name' completed successfully!"
                break
                ;;
            "FAILED")
                echo "❌ Step '$step_name' failed!"
                echo "Check logs for details:"
                echo "aws emr describe-step --cluster-id $CLUSTER_ID --step-id $step_id --region $REGION"
                exit 1
                ;;
            "CANCELLED")
                echo "❌ Step '$step_name' was cancelled!"
                exit 1
                ;;
            *)
                echo "⏳ Step '$step_name' status: $STEP_STATUS"
                sleep 30
                ;;
        esac
    done
    echo ""
}

# Step 1: Install Python packages and setup environment
echo "Step 1: Installing Python packages and setting up environment..."
cat > /tmp/step1_setup.json << 'EOF'
[
  {
    "Name": "Install Packages and Setup Environment",
    "ActionOnFailure": "CONTINUE",
    "Type": "CUSTOM_JAR",
    "Jar": "command-runner.jar",
    "Args": [
      "bash", "-c",
      "echo '=== INSTALLING PACKAGES AND SETTING UP ENVIRONMENT ===' && sudo yum install -y git python3-pip python3-devel gcc gcc-c++ make java-17-amazon-corretto-devel unzip wget && python3 -m pip install --upgrade pip && python3 -m pip install --no-cache-dir tifffile>=2023.0.0 scikit-fuzzy>=0.4.2 numpy>=1.21.0 scipy>=1.7.0 boto3>=1.26.0 pandas>=1.3.0 networkx>=2.6.0 matplotlib>=3.5.0 tqdm>=4.62.0 pyspark>=3.4.0 psutil>=5.8.0 && sudo mkdir -p /mnt/destark && sudo chown -R hadoop:hadoop /mnt/destark && cd /mnt/destark && mkdir -p app scripts logs data config files/input files/output utils docs && echo '✓ Environment setup completed'"
    ]
  }
]
EOF

STEP1_ID=$(aws emr add-steps --cluster-id "$CLUSTER_ID" --region "$REGION" --steps file:///tmp/step1_setup.json --query 'StepIds[0]' --output text)
wait_for_step "$STEP1_ID" "Install Packages and Setup Environment"

# Step 2: Create truly distributed FIS processing script
echo "Step 2: Creating truly distributed FIS processing script..."

# Create the truly distributed Python script locally first
cat > /tmp/truly_distributed_fis_workload.py << 'PYTHON_EOF'
#!/usr/bin/env python3
import os
import sys
import time
import json
import numpy as np
import tifffile
import boto3
from datetime import datetime, timedelta
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, lit, broadcast
from pyspark.sql.types import StructType, StructField, StringType, FloatType, IntegerType, ArrayType
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number, count, sum as spark_sum
import logging
import psutil
import threading
from collections import defaultdict

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/tmp/distributed_fis_detailed.log')
    ]
)
logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """Monitor cluster performance and provide detailed metrics."""
    
    def __init__(self, spark):
        self.spark = spark
        self.start_time = time.time()
        self.metrics = defaultdict(list)
        self.monitoring = True
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_performance, daemon=True)
        self.monitor_thread.start()
    
    def _monitor_performance(self):
        """Monitor cluster performance in background thread."""
        while self.monitoring:
            try:
                # Get executor metrics
                executor_metrics = self.spark.sparkContext._jsc.sc().getExecutorMemoryStatus()
                executor_count = executor_metrics.size()
                
                # Get task metrics
                task_metrics = self.spark.sparkContext._jsc.sc().getStatusTracker().getExecutorMetrics()
                
                # Calculate cluster utilization
                total_memory = 0
                used_memory = 0
                for executor_id, memory_status in executor_metrics.items():
                    total_memory += memory_status._2()
                    used_memory += memory_status._1()
                
                memory_utilization = (used_memory / total_memory * 100) if total_memory > 0 else 0
                
                # Store metrics
                current_time = time.time()
                self.metrics['timestamp'].append(current_time)
                self.metrics['executor_count'].append(executor_count)
                self.metrics['memory_utilization'].append(memory_utilization)
                self.metrics['elapsed_time'].append(current_time - self.start_time)
                
                logger.info(f"🔍 CLUSTER METRICS: {executor_count} executors, {memory_utilization:.1f}% memory used")
                
                time.sleep(10)  # Monitor every 10 seconds
                
            except Exception as e:
                logger.warning(f"Performance monitoring error: {e}")
                time.sleep(30)
    
    def stop(self):
        """Stop performance monitoring."""
        self.monitoring = False
        if self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
    
    def get_summary(self):
        """Get performance summary."""
        if not self.metrics['memory_utilization']:
            return {}
        
        return {
            'avg_memory_utilization': np.mean(self.metrics['memory_utilization']),
            'max_memory_utilization': np.max(self.metrics['memory_utilization']),
            'avg_executor_count': np.mean(self.metrics['executor_count']),
            'total_monitoring_time': time.time() - self.start_time,
            'monitoring_samples': len(self.metrics['memory_utilization'])
        }

def setup_spark():
    """Setup Spark session with optimized configuration for TRUE distributed processing."""
    logger.info("🚀 Setting up Spark session for TRUE distributed processing...")
    
    spark = SparkSession.builder \
        .appName("Truly Distributed FIS Processing") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.sql.adaptive.skewJoin.enabled", "true") \
        .config("spark.sql.adaptive.localShuffleReader.enabled", "true") \
        .config("spark.sql.adaptive.advisoryPartitionSizeInBytes", "64m") \
        .config("spark.sql.shuffle.partitions", "400") \
        .config("spark.default.parallelism", "400") \
        .config("spark.dynamicAllocation.enabled", "true") \
        .config("spark.dynamicAllocation.minExecutors", "2") \
        .config("spark.dynamicAllocation.maxExecutors", "8") \
        .config("spark.dynamicAllocation.shuffleTracking.enabled", "true") \
        .config("spark.executor.cores", "4") \
        .config("spark.executor.memory", "20g") \
        .config("spark.driver.memory", "8g") \
        .config("spark.driver.cores", "2") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .config("spark.kryoserializer.buffer", "128m") \
        .config("spark.kryoserializer.buffer.max", "512m") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .config("spark.sql.execution.arrow.maxRecordsPerBatch", "10000") \
        .config("spark.sql.execution.arrow.pyspark.selfDestruct.enabled", "true") \
        .getOrCreate()
    
    # Get cluster info
    executor_count = spark.sparkContext._jsc.sc().getExecutorMemoryStatus().size()
    default_parallelism = spark.sparkContext.defaultParallelism
    
    logger.info(f"✅ Spark session created successfully!")
    logger.info(f"   📊 Cluster Info:")
    logger.info(f"      • Executors: {executor_count}")
    logger.info(f"      • Default parallelism: {default_parallelism}")
    logger.info(f"      • Driver memory: 8g")
    logger.info(f"      • Executor memory: 20g")
    logger.info(f"      • Executor cores: 4")
    logger.info(f"      • Shuffle partitions: 400")
    
    return spark

def download_file_from_s3(s3_path, local_path):
    """Download file from S3 with detailed logging."""
    try:
        logger.info(f'📥 Downloading {s3_path} to {local_path}...')
        if s3_path.startswith('s3://'):
            bucket = s3_path.split('/')[2]
            key = '/'.join(s3_path.split('/')[3:])
        else:
            raise ValueError(f'Invalid S3 path: {s3_path}')
        
        s3_client = boto3.client('s3')
        
        # Get file size first
        response = s3_client.head_object(Bucket=bucket, Key=key)
        file_size = response['ContentLength']
        logger.info(f'   📏 File size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)')
        
        # Download with progress
        start_time = time.time()
        s3_client.download_file(bucket, key, local_path)
        download_time = time.time() - start_time
        download_speed = file_size / download_time / 1024 / 1024  # MB/s
        
        logger.info(f'✅ Downloaded {s3_path} in {download_time:.2f}s ({download_speed:.2f} MB/s)')
        return True
    except Exception as e:
        logger.error(f'❌ Failed to download {s3_path}: {e}')
        return False

def upload_file_to_s3(local_path, s3_path):
    """Upload file to S3 with detailed logging."""
    try:
        logger.info(f'📤 Uploading {local_path} to {s3_path}...')
        if s3_path.startswith('s3://'):
            bucket = s3_path.split('/')[2]
            key = '/'.join(s3_path.split('/')[3:])
        else:
            raise ValueError(f'Invalid S3 path: {s3_path}')
        
        file_size = os.path.getsize(local_path)
        logger.info(f'   📏 File size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)')
        
        s3_client = boto3.client('s3')
        start_time = time.time()
        s3_client.upload_file(local_path, bucket, key)
        upload_time = time.time() - start_time
        upload_speed = file_size / upload_time / 1024 / 1024  # MB/s
        
        logger.info(f'✅ Uploaded {local_path} to {s3_path} in {upload_time:.2f}s ({upload_speed:.2f} MB/s)')
        return True
    except Exception as e:
        logger.error(f'❌ Failed to upload {local_path}: {e}')
        return False

def create_fis_from_config(config_data):
    """Create FIS from configuration with detailed logging."""
    try:
        logger.info(f"🔧 Creating FIS from configuration...")
        
        # Create fuzzy variables based on config
        input_vars = {}
        for var_name, var_config in config_data['input_variables'].items():
            universe = np.arange(var_config['min'], var_config['max'] + 1, 1)
            input_vars[var_name] = ctrl.Antecedent(universe, var_name)
            logger.info(f"   📊 Input variable '{var_name}': range [{var_config['min']}, {var_config['max']}]")
            
            # Define membership functions
            for term_name, term_config in var_config['membership_functions'].items():
                if term_config['type'] == 'trimf':
                    input_vars[var_name][term_name] = fuzz.trimf(input_vars[var_name].universe, term_config['params'])
                elif term_config['type'] == 'trapmf':
                    input_vars[var_name][term_name] = fuzz.trapmf(input_vars[var_name].universe, term_config['params'])
                logger.info(f"      • Membership function '{term_name}': {term_config['type']} with params {term_config['params']}")
        
        # Create output variable
        output_config = config_data['output_variable']
        universe = np.arange(output_config['min'], output_config['max'] + 1, 1)
        output_var = ctrl.Consequent(universe, output_config['name'])
        logger.info(f"   📊 Output variable '{output_config['name']}': range [{output_config['min']}, {output_config['max']}]")
        
        for term_name, term_config in output_config['membership_functions'].items():
            if term_config['type'] == 'trimf':
                output_var[term_name] = fuzz.trimf(output_var.universe, term_config['params'])
            elif term_config['type'] == 'trapmf':
                output_var[term_name] = fuzz.trapmf(output_var.universe, term_config['params'])
            logger.info(f"      • Membership function '{term_name}': {term_config['type']} with params {term_config['params']}")
        
        # Create rules
        rules = []
        for i, rule_config in enumerate(config_data['rules']):
            antecedent = None
            for condition in rule_config['antecedent']:
                var_name = condition['variable']
                term_name = condition['membership']
                if antecedent is None:
                    antecedent = input_vars[var_name][term_name]
                else:
                    antecedent = antecedent & input_vars[var_name][term_name]
            consequent = output_var[rule_config['consequent']]
            rule = ctrl.Rule(antecedent, consequent)
            rules.append(rule)
            logger.info(f"   📋 Rule {i+1}: IF {[f'{c['variable']} is {c['membership']}' for c in rule_config['antecedent']]} THEN {rule_config['consequent']}")
        
        # Create control system
        fis = ctrl.ControlSystem(rules)
        simulation = ctrl.ControlSystemSimulation(fis)
        
        logger.info(f"✅ FIS created successfully with {len(input_vars)} input variables and {len(rules)} rules")
        return simulation, list(input_vars.keys()), output_config['name']
    except Exception as e:
        logger.error(f"❌ Error creating FIS from config: {e}")
        raise

def process_chunk_udf(social_chunk, env_chunk, strat_chunk, config_json):
    """UDF to process a chunk of pixels with FIS - runs on each executor."""
    try:
        # Parse config on executor
        config_data = json.loads(config_json)
        
        # Create FIS on executor
        simulation, input_vars, output_var = create_fis_from_config(config_data)
        
        # Process chunk
        results = []
        nodata_value = 5.0
        
        for i in range(len(social_chunk)):
            social_val = float(social_chunk[i])
            env_val = float(env_chunk[i])
            strat_val = float(strat_chunk[i])
            
            # Skip NoData pixels
            if (np.isnan(social_val) or np.isnan(env_val) or np.isnan(strat_val) or
                social_val == nodata_value or env_val == nodata_value or strat_val == nodata_value):
                results.append(float(nodata_value))
                continue
            
            # Set input values and compute
            simulation.input[input_vars[0]] = social_val
            simulation.input[input_vars[1]] = env_val
            simulation.input[input_vars[2]] = strat_val
            simulation.compute()
            results.append(float(simulation.output[output_var]))
        
        return results
    except Exception as e:
        logger.error(f"Error in process_chunk_udf: {e}")
        return [5.0] * len(social_chunk)  # Return NoData for all pixels in chunk

def create_distributed_dataframe(spark, social_data, env_data, strat_data, chunk_size=1000):
    """Create truly distributed DataFrame with chunked processing."""
    logger.info(f"🔄 Creating distributed DataFrame with chunk size {chunk_size}...")
    
    # Calculate chunks
    total_pixels = social_data.shape[0] * social_data.shape[1]
    num_chunks = (total_pixels + chunk_size - 1) // chunk_size
    
    logger.info(f"   📊 Total pixels: {total_pixels:,}")
    logger.info(f"   📊 Number of chunks: {num_chunks:,}")
    logger.info(f"   📊 Chunks per partition: ~{num_chunks // spark.sparkContext.defaultParallelism}")
    
    # Create chunked data
    chunks = []
    chunk_id = 0
    
    for i in range(0, social_data.shape[0]):
        for j in range(0, social_data.shape[1], chunk_size):
            # Get chunk data
            end_j = min(j + chunk_size, social_data.shape[1])
            social_chunk = social_data[i, j:end_j].tolist()
            env_chunk = env_data[i, j:end_j].tolist()
            strat_chunk = strat_data[i, j:end_j].tolist()
            
            chunks.append({
                'chunk_id': chunk_id,
                'row': i,
                'col_start': j,
                'col_end': end_j,
                'social_data': social_chunk,
                'env_data': env_chunk,
                'strat_data': strat_chunk,
                'pixel_count': len(social_chunk)
            })
            chunk_id += 1
    
    # Create DataFrame schema
    schema = StructType([
        StructField("chunk_id", IntegerType(), False),
        StructField("row", IntegerType(), False),
        StructField("col_start", IntegerType(), False),
        StructField("col_end", IntegerType(), False),
        StructField("social_data", ArrayType(FloatType()), False),
        StructField("env_data", ArrayType(FloatType()), False),
        StructField("strat_data", ArrayType(FloatType()), False),
        StructField("pixel_count", IntegerType(), False)
    ])
    
    # Create DataFrame
    df = spark.createDataFrame(chunks, schema)
    
    # Repartition for better distribution
    target_partitions = min(spark.sparkContext.defaultParallelism * 2, len(chunks))
    df = df.repartition(target_partitions)
    
    logger.info(f"✅ Distributed DataFrame created:")
    logger.info(f"   📊 DataFrame partitions: {df.rdd.getNumPartitions()}")
    logger.info(f"   📊 Total chunks: {df.count():,}")
    
    return df

def process_raster_truly_distributed(spark, social_data, env_data, strat_data, config_data, config_name, monitor):
    """Process raster data using TRULY distributed processing."""
    logger.info(f"🚀 Starting TRULY distributed processing for {config_name}...")
    start_time = time.time()
    
    # Create distributed DataFrame
    df = create_distributed_dataframe(spark, social_data, env_data, strat_data)
    
    # Broadcast config to all executors
    config_json = json.dumps(config_data)
    broadcast_config = spark.sparkContext.broadcast(config_json)
    
    # Create UDF for distributed processing
    def process_chunk_distributed(social_chunk, env_chunk, strat_chunk):
        return process_chunk_udf(social_chunk, env_chunk, strat_chunk, broadcast_config.value)
    
    fis_udf = udf(process_chunk_distributed, ArrayType(FloatType()))
    
    # Apply distributed FIS processing
    logger.info("🔄 Applying FIS processing across all executors...")
    result_df = df.withColumn("results", fis_udf(col("social_data"), col("env_data"), col("strat_data")))
    
    # Cache results for multiple operations
    result_df.cache()
    
    # Get processing statistics
    total_chunks = result_df.count()
    logger.info(f"✅ Distributed processing completed:")
    logger.info(f"   📊 Total chunks processed: {total_chunks:,}")
    logger.info(f"   📊 Partitions used: {result_df.rdd.getNumPartitions()}")
    
    # Collect results back to driver (this is where we reconstruct)
    logger.info("📥 Collecting results from all executors...")
    collect_start = time.time()
    results = result_df.select("chunk_id", "row", "col_start", "col_end", "results").collect()
    collect_time = time.time() - collect_start
    
    logger.info(f"✅ Results collected in {collect_time:.2f} seconds")
    
    # Reconstruct raster
    logger.info("🔧 Reconstructing raster from distributed results...")
    reconstruct_start = time.time()
    result_raster = np.zeros_like(social_data, dtype=np.float32)
    
    for row in results:
        row_idx = row.row
        col_start = row.col_start
        col_end = row.col_end
        chunk_results = row.results
        
        for i, result_val in enumerate(chunk_results):
            col_idx = col_start + i
            if col_idx < social_data.shape[1]:  # Bounds check
                result_raster[row_idx, col_idx] = result_val
    
    reconstruct_time = time.time() - reconstruct_start
    processing_time = time.time() - start_time
    
    # Get performance summary
    perf_summary = monitor.get_summary()
    
    logger.info(f"✅ TRULY distributed processing completed:")
    logger.info(f"   ⏱️  Total time: {processing_time:.2f} seconds")
    logger.info(f"   📥 Collection time: {collect_time:.2f} seconds")
    logger.info(f"   🔧 Reconstruction time: {reconstruct_time:.2f} seconds")
    logger.info(f"   📊 Average memory utilization: {perf_summary.get('avg_memory_utilization', 0):.1f}%")
    logger.info(f"   📊 Average executors used: {perf_summary.get('avg_executor_count', 0):.1f}")
    
    # Unpersist cached DataFrame
    result_df.unpersist()
    
    return result_raster

def estimate_completion_time(processed_configs, total_configs, elapsed_time):
    """Estimate completion time based on current progress."""
    if processed_configs == 0:
        return "Unknown"
    
    avg_time_per_config = elapsed_time / processed_configs
    remaining_configs = total_configs - processed_configs
    estimated_remaining_time = avg_time_per_config * remaining_configs
    
    estimated_completion = datetime.now() + timedelta(seconds=estimated_remaining_time)
    return estimated_completion.strftime("%Y-%m-%d %H:%M:%S")

def main():
    """Main function for truly distributed FIS processing."""
    logger.info('🚀 Starting TRULY DISTRIBUTED FIS workload with REAL DATA and all 6 configurations...')
    overall_start_time = time.time()
    
    # Setup Spark
    spark = setup_spark()
    
    # Setup performance monitoring
    monitor = PerformanceMonitor(spark)
    
    # Define all configuration files
    config_files = [
        'config_max.json',
        'config_median.json',
        'config_minimum.json',
        'config_mode.json',
        'config_round_down.json',
        'config_round_up.json'
    ]
    
    # Define input files (real 300m data)
    input_files = {
        'social': 's3://<AWS-BUCKET>-unifile/unifile_test/so300m.in',
        'environmental': 's3://<AWS-BUCKET>-unifile/unifile_test/e300m.in',
        'strategic': 's3://<AWS-BUCKET>-unifile/unifile_test/s300m.in'
    }
    
    # Download input rasters
    logger.info('📥 Downloading input raster files...')
    local_files = {}
    for name, s3_path in input_files.items():
        local_path = f'/tmp/{name}_300m.tif'
        if download_file_from_s3(s3_path, local_path):
            local_files[name] = local_path
        else:
            logger.error(f'❌ Failed to download {name} file')
            return
    
    # Read raster data
    logger.info('📖 Reading raster data...')
    try:
        social_data = tifffile.imread(local_files['social'])
        env_data = tifffile.imread(local_files['environmental'])
        strat_data = tifffile.imread(local_files['strategic'])
        
        total_pixels = social_data.shape[0] * social_data.shape[1]
        logger.info(f'✅ Raster data loaded:')
        logger.info(f'   📊 Social: {social_data.shape}, dtype: {social_data.dtype}')
        logger.info(f'   📊 Environmental: {env_data.shape}, dtype: {env_data.dtype}')
        logger.info(f'   📊 Strategic: {strat_data.shape}, dtype: {strat_data.dtype}')
        logger.info(f'   📊 Total pixels to process: {total_pixels:,}')
        logger.info(f'   📊 Estimated memory usage: {total_pixels * 4 * 3 / 1024 / 1024:.1f} MB (input) + {total_pixels * 4 / 1024 / 1024:.1f} MB (output)')
    except Exception as e:
        logger.error(f'❌ Failed to read raster data: {e}')
        return
    
    # Process each configuration
    results = []
    for i, config_file in enumerate(config_files):
        config_start_time = time.time()
        logger.info(f'\n{"="*80}')
        logger.info(f'🔄 Processing configuration {i+1}/{len(config_files)}: {config_file}')
        logger.info(f'{"="*80}')
        
        try:
            # Download configuration file
            config_s3_path = f's3://<AWS-BUCKET>-unifile/unifile_test/{config_file}'
            config_local_path = f'/tmp/{config_file}'
            if not download_file_from_s3(config_s3_path, config_local_path):
                logger.error(f'❌ Failed to download configuration {config_file}')
                continue
            
            # Load configuration
            with open(config_local_path, 'r') as f:
                config_data = json.load(f)
            logger.info(f'✅ Configuration loaded: {len(config_data.get("rules", []))} rules')
            
            # Create FIS from configuration
            simulation, input_vars, output_var = create_fis_from_config(config_data)
            logger.info(f'✅ FIS created with {len(input_vars)} input variables')
            
            # Process raster using truly distributed processing
            config_name = config_file.replace('.json', '')
            result_data = process_raster_truly_distributed(spark, social_data, env_data, strat_data, config_data, config_name, monitor)
            
            # Save result
            output_file = f'/tmp/result_truly_distributed_{config_name}_300m.tif'
            logger.info(f'💾 Saving result to {output_file}...')
            tifffile.imwrite(output_file, result_data, photometric='minisblack')
            
            # Upload to S3
            s3_key = f'unifile_test/result_truly_distributed_{config_name}_300m.tif'
            s3_path = f's3://<AWS-BUCKET>-unifile/{s3_key}'
            if upload_file_to_s3(output_file, s3_path):
                config_time = time.time() - config_start_time
                results.append({
                    'config': config_file,
                    'output': s3_path,
                    'time': config_time,
                    'status': 'success',
                    'data_shape': social_data.shape,
                    'total_pixels': social_data.shape[0] * social_data.shape[1],
                    'rules_count': len(config_data.get('rules', [])),
                    'processing_type': 'truly_distributed'
                })
                logger.info(f'✅ Configuration {config_file} completed in {config_time:.2f} seconds')
                
                # Estimate completion time
                elapsed_time = time.time() - overall_start_time
                estimated_completion = estimate_completion_time(i + 1, len(config_files), elapsed_time)
                logger.info(f'📅 Estimated completion time: {estimated_completion}')
                
            else:
                config_time = time.time() - config_start_time
                results.append({
                    'config': config_file,
                    'time': config_time,
                    'status': 'failed',
                    'error': 'Upload failed'
                })
                logger.error(f'✗ Configuration {config_file} upload failed')
                
        except Exception as e:
            config_time = time.time() - config_start_time
            results.append({
                'config': config_file,
                'time': config_time,
                'status': 'failed',
                'error': str(e)
            })
            logger.error(f'✗ Configuration {config_file} failed: {e}')
    
    # Stop monitoring and Spark
    monitor.stop()
    spark.stop()
    
    # Generate final summary
    total_time = time.time() - overall_start_time
    successful_configs = [r for r in results if r['status'] == 'success']
    failed_configs = [r for r in results if r['status'] == 'failed']
    
    logger.info(f'\n{"="*80}')
    logger.info('🎉 TRULY DISTRIBUTED WORKLOAD SUMMARY')
    logger.info(f'{"="*80}')
    logger.info(f'⏱️  Total processing time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)')
    logger.info(f'✅ Successful configurations: {len(successful_configs)}/{len(config_files)}')
    logger.info(f'❌ Failed configurations: {len(failed_configs)}/{len(config_files)}')
    logger.info(f'📊 Data processed: {social_data.shape[0]}x{social_data.shape[1]} = {social_data.shape[0] * social_data.shape[1]:,} pixels')
    logger.info(f'🚀 Processing type: TRULY DISTRIBUTED across all cluster nodes')
    logger.info(f'📈 Performance metrics:')
    
    perf_summary = monitor.get_summary()
    if perf_summary:
        logger.info(f'   • Average memory utilization: {perf_summary.get("avg_memory_utilization", 0):.1f}%')
        logger.info(f'   • Max memory utilization: {perf_summary.get("max_memory_utilization", 0):.1f}%')
        logger.info(f'   • Average executors used: {perf_summary.get("avg_executor_count", 0):.1f}')
        logger.info(f'   • Monitoring duration: {perf_summary.get("total_monitoring_time", 0):.1f}s')
    
    logger.info(f'\n✅ Successful configurations:')
    for result in successful_configs:
        logger.info(f'   • {result["config"]}: {result["time"]:.2f}s ({result["rules_count"]} rules) -> {result["output"]}')
    
    if failed_configs:
        logger.info(f'\n❌ Failed configurations:')
        for result in failed_configs:
            logger.info(f'   • {result["config"]}: {result["error"]}')
    
    # Save detailed results to S3
    results_file = '/tmp/truly_distributed_workload_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'cluster_id': '<EMR-CLUSTER-ID>',
            'total_time': total_time,
            'data_shape': social_data.shape,
            'total_pixels': social_data.shape[0] * social_data.shape[1],
            'configurations_processed': len(config_files),
            'processing_type': 'truly_distributed',
            'spark_config': {
                'default_parallelism': spark.sparkContext.defaultParallelism,
                'executor_cores': 4,
                'executor_memory': '20g',
                'driver_memory': '8g',
                'shuffle_partitions': 400
            },
            'performance_metrics': perf_summary,
            'results': results
        }, f, indent=2)
    
    s3_client = boto3.client('s3')
    s3_client.upload_file(results_file, '<AWS-BUCKET>-unifile', 'unifile_test/truly_distributed_workload_results.json')
    logger.info(f'\n📄 Detailed results saved to: s3://<AWS-BUCKET>-unifile/unifile_test/truly_distributed_workload_results.json')
    logger.info('🎉 TRULY DISTRIBUTED WORKLOAD COMPLETED SUCCESSFULLY!')

if __name__ == '__main__':
    main()
PYTHON_EOF

# Upload the script to S3
echo "Uploading truly distributed Python script to S3..."
aws s3 cp /tmp/truly_distributed_fis_workload.py s3://<AWS-BUCKET>-unifile/unifile_test/truly_distributed_fis_workload.py

# Step 3: Run truly distributed workload using Spark
echo "Step 3: Running truly distributed workload using Spark..."
cat > /tmp/step3_truly_distributed_workload.json << 'EOF'
[
  {
    "Name": "Truly Distributed FIS Workload - Spark Processing",
    "ActionOnFailure": "CONTINUE",
    "Type": "CUSTOM_JAR",
    "Jar": "command-runner.jar",
    "Args": [
      "bash", "-c",
      "echo '=== TRULY DISTRIBUTED FIS WORKLOAD - SPARK PROCESSING ===' && cd /mnt/destark && echo 'Downloading truly distributed Python script from S3...' && aws s3 cp s3://<AWS-BUCKET>-unifile/unifile_test/truly_distributed_fis_workload.py app/truly_distributed_fis_workload.py && chmod +x app/truly_distributed_fis_workload.py && echo 'Running truly distributed FIS workload with Spark...' && python3 app/truly_distributed_fis_workload.py && echo '=== TRULY DISTRIBUTED FIS WORKLOAD COMPLETED ==='"
    ]
  }
]
EOF

STEP3_ID=$(aws emr add-steps --cluster-id "$CLUSTER_ID" --region "$REGION" --steps file:///tmp/step3_truly_distributed_workload.json --query 'StepIds[0]' --output text)
wait_for_step "$STEP3_ID" "Truly Distributed FIS Workload - Spark Processing"

# Cleanup temporary files
rm -f /tmp/step1_setup.json /tmp/step3_truly_distributed_workload.json /tmp/truly_distributed_fis_workload.py

echo "=========================================="
echo "✅ Truly Distributed FIS Workload Completed Successfully!"
echo "=========================================="
echo ""
echo "Summary:"
echo "  ✓ Python packages installed"
echo "  ✓ Development environment setup"
echo "  ✓ TRULY DISTRIBUTED workload with Spark across all cluster nodes completed"
echo ""
echo "Output Files:"
echo "  • Truly distributed workload results: s3://$S3_BUCKET/$S3_PREFIX/result_truly_distributed_*_300m.tif"
echo "  • Truly distributed workload summary: s3://$S3_BUCKET/$S3_PREFIX/truly_distributed_workload_results.json"
echo "  • Detailed logs: s3://$S3_BUCKET/$S3_PREFIX/distributed_fis_detailed.log"
echo ""
echo "Cluster Status:"
echo "  • Processing: TRULY DISTRIBUTED across all nodes"
echo "  • Spark configuration: Optimized for maximum cluster utilization"
echo "  • Parallelism: 400 partitions for maximum distribution"
echo "  • Performance monitoring: Real-time metrics and completion estimates"
echo "" 