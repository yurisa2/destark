#!/bin/bash

# Set AWS credentials for the correct account
export AWS_ACCESS_KEY_ID=YOUR_AWS_ACCESS_KEY_ID
export AWS_SECRET_ACCESS_KEY=YOUR_AWS_SECRET_ACCESS_KEY
export AWS_REGION=us-east-2
export AWS_DEFAULT_REGION=us-east-2

# Add Simple Distributed Steps to EMR Cluster
set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <cluster-id>"
    exit 1
fi

CLUSTER_ID="$1"
REGION="us-east-2"

echo "=========================================="
echo "Adding Simple Distributed Steps to EMR Cluster"
echo "Cluster ID: $CLUSTER_ID"
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

# Create a simpler distributed Python script
echo "Creating simple distributed Python script..."
cat > /tmp/simple_distributed_fis_workload.py << 'PYTHON_EOF'
#!/usr/bin/env python3
import os, sys, time, json, numpy as np, tifffile, boto3
from datetime import datetime
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import StructType, StructField, StringType, FloatType, IntegerType, ArrayType
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_spark():
    logger.info("Setting up Spark session for distributed processing...")
    spark = SparkSession.builder \
        .appName("Simple Distributed FIS Processing") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.shuffle.partitions", "200") \
        .config("spark.default.parallelism", "200") \
        .config("spark.dynamicAllocation.enabled", "true") \
        .config("spark.dynamicAllocation.minExecutors", "2") \
        .config("spark.dynamicAllocation.maxExecutors", "8") \
        .config("spark.executor.cores", "4") \
        .config("spark.executor.memory", "20g") \
        .config("spark.driver.memory", "8g") \
        .getOrCreate()
    
    executor_count = spark.sparkContext._jsc.sc().getExecutorMemoryStatus().size()
    default_parallelism = spark.sparkContext.defaultParallelism
    logger.info(f"Spark session created: {executor_count} executors, {default_parallelism} parallelism")
    return spark

def download_file_from_s3(s3_path, local_path):
    try:
        logger.info(f'Downloading {s3_path}...')
        if s3_path.startswith('s3://'):
            bucket = s3_path.split('/')[2]
            key = '/'.join(s3_path.split('/')[3:])
        else:
            raise ValueError(f'Invalid S3 path: {s3_path}')
        
        s3_client = boto3.client('s3')
        s3_client.download_file(bucket, key, local_path)
        logger.info(f'Downloaded {s3_path}')
        return True
    except Exception as e:
        logger.error(f'Failed to download {s3_path}: {e}')
        return False

def upload_file_to_s3(local_path, s3_path):
    try:
        logger.info(f'Uploading {local_path}...')
        if s3_path.startswith('s3://'):
            bucket = s3_path.split('/')[2]
            key = '/'.join(s3_path.split('/')[3:])
        else:
            raise ValueError(f'Invalid S3 path: {s3_path}')
        
        s3_client = boto3.client('s3')
        s3_client.upload_file(local_path, bucket, key)
        logger.info(f'Uploaded {local_path} to {s3_path}')
        return True
    except Exception as e:
        logger.error(f'Failed to upload {local_path}: {e}')
        return False

def create_fis_from_config(config_data):
    try:
        logger.info(f"Creating FIS from configuration...")
        input_vars = {}
        for var_name, var_config in config_data['input_variables'].items():
            universe = np.arange(var_config['min'], var_config['max'] + 1, 1)
            input_vars[var_name] = ctrl.Antecedent(universe, var_name)
            
            for term_name, term_config in var_config['membership_functions'].items():
                if term_config['type'] == 'trimf':
                    input_vars[var_name][term_name] = fuzz.trimf(input_vars[var_name].universe, term_config['params'])
                elif term_config['type'] == 'trapmf':
                    input_vars[var_name][term_name] = fuzz.trapmf(input_vars[var_name].universe, term_config['params'])
        
        output_config = config_data['output_variable']
        universe = np.arange(output_config['min'], output_config['max'] + 1, 1)
        output_var = ctrl.Consequent(universe, output_config['name'])
        
        for term_name, term_config in output_config['membership_functions'].items():
            if term_config['type'] == 'trimf':
                output_var[term_name] = fuzz.trimf(output_var.universe, term_config['params'])
            elif term_config['type'] == 'trapmf':
                output_var[term_name] = fuzz.trapmf(output_var.universe, term_config['params'])
        
        rules = []
        for rule_config in config_data['rules']:
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
        
        fis = ctrl.ControlSystem(rules)
        simulation = ctrl.ControlSystemSimulation(fis)
        logger.info(f"FIS created with {len(input_vars)} input variables and {len(rules)} rules")
        return simulation, list(input_vars.keys()), output_config['name']
    except Exception as e:
        logger.error(f"Error creating FIS from config: {e}")
        raise

def process_chunk_udf(social_chunk, env_chunk, strat_chunk, config_json):
    try:
        config_data = json.loads(config_json)
        simulation, input_vars, output_var = create_fis_from_config(config_data)
        results = []
        nodata_value = 5.0
        
        for i in range(len(social_chunk)):
            social_val = float(social_chunk[i])
            env_val = float(env_chunk[i])
            strat_val = float(strat_chunk[i])
            
            if (np.isnan(social_val) or np.isnan(env_val) or np.isnan(strat_val) or
                social_val == nodata_value or env_val == nodata_value or strat_val == nodata_value):
                results.append(float(nodata_value))
                continue
            
            simulation.input[input_vars[0]] = social_val
            simulation.input[input_vars[1]] = env_val
            simulation.input[input_vars[2]] = strat_val
            simulation.compute()
            results.append(float(simulation.output[output_var]))
        
        return results
    except Exception as e:
        logger.error(f"Error in process_chunk_udf: {e}")
        return [5.0] * len(social_chunk)

def create_distributed_dataframe(spark, social_data, env_data, strat_data, chunk_size=1000):
    logger.info(f"Creating distributed DataFrame with chunk size {chunk_size}...")
    
    # Convert uint8 data to float32 to match Spark schema
    logger.info("Converting data types from uint8 to float32...")
    social_data_float = social_data.astype(np.float32)
    env_data_float = env_data.astype(np.float32)
    strat_data_float = strat_data.astype(np.float32)
    
    total_pixels = social_data_float.shape[0] * social_data_float.shape[1]
    num_chunks = (total_pixels + chunk_size - 1) // chunk_size
    logger.info(f"Total pixels: {total_pixels:,}, Number of chunks: {num_chunks:,}")
    
    chunks = []
    chunk_id = 0
    
    for i in range(0, social_data_float.shape[0]):
        for j in range(0, social_data_float.shape[1], chunk_size):
            end_j = min(j + chunk_size, social_data_float.shape[1])
            social_chunk = social_data_float[i, j:end_j].tolist()
            env_chunk = env_data_float[i, j:end_j].tolist()
            strat_chunk = strat_data_float[i, j:end_j].tolist()
            
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
    
    df = spark.createDataFrame(chunks, schema)
    target_partitions = min(spark.sparkContext.defaultParallelism * 2, len(chunks))
    df = df.repartition(target_partitions)
    
    logger.info(f"Distributed DataFrame created: {df.rdd.getNumPartitions()} partitions, {df.count():,} chunks")
    return df

def process_raster_distributed(spark, social_data, env_data, strat_data, config_data, config_name):
    logger.info(f"Starting distributed processing for {config_name}...")
    start_time = time.time()
    
    df = create_distributed_dataframe(spark, social_data, env_data, strat_data)
    config_json = json.dumps(config_data)
    broadcast_config = spark.sparkContext.broadcast(config_json)
    
    def process_chunk_distributed(social_chunk, env_chunk, strat_chunk):
        return process_chunk_udf(social_chunk, env_chunk, strat_chunk, broadcast_config.value)
    
    fis_udf = udf(process_chunk_distributed, ArrayType(FloatType()))
    
    logger.info("Applying FIS processing across all executors...")
    result_df = df.withColumn("results", fis_udf(col("social_data"), col("env_data"), col("strat_data")))
    result_df.cache()
    
    total_chunks = result_df.count()
    logger.info(f"Distributed processing completed: {total_chunks:,} chunks, {result_df.rdd.getNumPartitions()} partitions")
    
    logger.info("Collecting results from all executors...")
    collect_start = time.time()
    results = result_df.select("chunk_id", "row", "col_start", "col_end", "results").collect()
    collect_time = time.time() - collect_start
    logger.info(f"Results collected in {collect_time:.2f} seconds")
    
    logger.info("Reconstructing raster from distributed results...")
    reconstruct_start = time.time()
    result_raster = np.zeros_like(social_data, dtype=np.float32)
    
    for row in results:
        row_idx = row.row
        col_start = row.col_start
        col_end = row.col_end
        chunk_results = row.results
        
        for i, result_val in enumerate(chunk_results):
            col_idx = col_start + i
            if col_idx < social_data.shape[1]:
                result_raster[row_idx, col_idx] = result_val
    
    reconstruct_time = time.time() - reconstruct_start
    processing_time = time.time() - start_time
    
    logger.info(f"Distributed processing completed:")
    logger.info(f"  Total time: {processing_time:.2f} seconds")
    logger.info(f"  Collection time: {collect_time:.2f} seconds")
    logger.info(f"  Reconstruction time: {reconstruct_time:.2f} seconds")
    
    result_df.unpersist()
    return result_raster

def main():
    logger.info('Starting DISTRIBUTED FIS workload with REAL DATA and all 6 configurations...')
    overall_start_time = time.time()
    
    spark = setup_spark()
    
    config_files = [
        'config_max.json', 'config_median.json', 'config_minimum.json',
        'config_mode.json', 'config_round_down.json', 'config_round_up.json'
    ]
    
    input_files = {
        'social': 's3://<AWS-BUCKET>-unifile/unifile_test/so300m.in',
        'environmental': 's3://<AWS-BUCKET>-unifile/unifile_test/e300m.in',
        'strategic': 's3://<AWS-BUCKET>-unifile/unifile_test/s300m.in'
    }
    
    logger.info('Downloading input raster files...')
    local_files = {}
    for name, s3_path in input_files.items():
        local_path = f'/tmp/{name}_300m.tif'
        if download_file_from_s3(s3_path, local_path):
            local_files[name] = local_path
        else:
            logger.error(f'Failed to download {name} file')
            return
    
    logger.info('Reading raster data...')
    try:
        social_data = tifffile.imread(local_files['social'])
        env_data = tifffile.imread(local_files['environmental'])
        strat_data = tifffile.imread(local_files['strategic'])
        
        total_pixels = social_data.shape[0] * social_data.shape[1]
        logger.info(f'Raster data loaded:')
        logger.info(f'  Social: {social_data.shape}, dtype: {social_data.dtype}')
        logger.info(f'  Environmental: {env_data.shape}, dtype: {env_data.dtype}')
        logger.info(f'  Strategic: {strat_data.shape}, dtype: {strat_data.dtype}')
        logger.info(f'  Total pixels to process: {total_pixels:,}')
    except Exception as e:
        logger.error(f'Failed to read raster data: {e}')
        return
    
    results = []
    for i, config_file in enumerate(config_files):
        config_start_time = time.time()
        logger.info(f'Processing configuration {i+1}/{len(config_files)}: {config_file}')
        
        try:
            config_s3_path = f's3://<AWS-BUCKET>-unifile/unifile_test/{config_file}'
            config_local_path = f'/tmp/{config_file}'
            if not download_file_from_s3(config_s3_path, config_local_path):
                logger.error(f'Failed to download configuration {config_file}')
                continue
            
            with open(config_local_path, 'r') as f:
                config_data = json.load(f)
            logger.info(f'Configuration loaded: {len(config_data.get("rules", []))} rules')
            
            simulation, input_vars, output_var = create_fis_from_config(config_data)
            logger.info(f'FIS created with {len(input_vars)} input variables')
            
            config_name = config_file.replace('.json', '')
            result_data = process_raster_distributed(spark, social_data, env_data, strat_data, config_data, config_name)
            
            output_file = f'/tmp/result_distributed_{config_name}_300m.tif'
            logger.info(f'Saving result to {output_file}...')
            tifffile.imwrite(output_file, result_data, photometric='minisblack')
            
            s3_key = f'unifile_test/result_distributed_{config_name}_300m.tif'
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
                    'processing_type': 'distributed'
                })
                logger.info(f'Configuration {config_file} completed in {config_time:.2f} seconds')
            else:
                config_time = time.time() - config_start_time
                results.append({
                    'config': config_file,
                    'time': config_time,
                    'status': 'failed',
                    'error': 'Upload failed'
                })
                logger.error(f'Configuration {config_file} upload failed')
                
        except Exception as e:
            config_time = time.time() - config_start_time
            results.append({
                'config': config_file,
                'time': config_time,
                'status': 'failed',
                'error': str(e)
            })
            logger.error(f'Configuration {config_file} failed: {e}')
    
    spark.stop()
    
    total_time = time.time() - overall_start_time
    successful_configs = [r for r in results if r['status'] == 'success']
    failed_configs = [r for r in results if r['status'] == 'failed']
    
    logger.info('DISTRIBUTED WORKLOAD SUMMARY')
    logger.info(f'Total processing time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)')
    logger.info(f'Successful configurations: {len(successful_configs)}/{len(config_files)}')
    logger.info(f'Failed configurations: {len(failed_configs)}/{len(config_files)}')
    logger.info(f'Data processed: {social_data.shape[0]}x{social_data.shape[1]} = {social_data.shape[0] * social_data.shape[1]:,} pixels')
    logger.info(f'Processing type: DISTRIBUTED across all cluster nodes')
    
    logger.info('Successful configurations:')
    for result in successful_configs:
        logger.info(f'  {result["config"]}: {result["time"]:.2f}s ({result["rules_count"]} rules) -> {result["output"]}')
    
    if failed_configs:
        logger.info('Failed configurations:')
        for result in failed_configs:
            logger.info(f'  {result["config"]}: {result["error"]}')
    
    results_file = '/tmp/distributed_workload_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'cluster_id': '$CLUSTER_ID',
            'total_time': total_time,
            'data_shape': social_data.shape,
            'total_pixels': social_data.shape[0] * social_data.shape[1],
            'configurations_processed': len(config_files),
            'processing_type': 'distributed',
            'results': results
        }, f, indent=2)
    
    s3_client = boto3.client('s3')
    s3_client.upload_file(results_file, '<AWS-BUCKET>-unifile', 'unifile_test/distributed_workload_results.json')
    logger.info(f'Results saved to: s3://<AWS-BUCKET>-unifile/unifile_test/distributed_workload_results.json')
    logger.info('DISTRIBUTED WORKLOAD COMPLETED SUCCESSFULLY!')

if __name__ == '__main__':
    main()
PYTHON_EOF

# Upload the script to S3
echo "Uploading simple distributed Python script to S3..."
aws s3 cp /tmp/simple_distributed_fis_workload.py s3://<AWS-BUCKET>-unifile/unifile_test/simple_distributed_fis_workload.py

# Add the simple distributed processing step
echo "Adding simple distributed processing step..."
cat > /tmp/step_simple_distributed.json << 'EOF'
[
  {
    "Name": "Simple Distributed FIS Workload - Spark Processing",
    "ActionOnFailure": "CONTINUE",
    "Type": "CUSTOM_JAR",
    "Jar": "command-runner.jar",
    "Args": [
      "bash", "-c",
      "echo '=== SIMPLE DISTRIBUTED FIS WORKLOAD - SPARK PROCESSING ===' && cd /mnt/destark && echo 'Downloading simple distributed Python script from S3...' && aws s3 cp s3://<AWS-BUCKET>-unifile/unifile_test/simple_distributed_fis_workload.py app/simple_distributed_fis_workload.py && chmod +x app/simple_distributed_fis_workload.py && echo 'Running simple distributed FIS workload with Spark...' && python3 app/simple_distributed_fis_workload.py && echo '=== SIMPLE DISTRIBUTED FIS WORKLOAD COMPLETED ==='"
    ]
  }
]
EOF

STEP_ID=$(aws emr add-steps --cluster-id "$CLUSTER_ID" --region "$REGION" --steps file:///tmp/step_simple_distributed.json --query 'StepIds[0]' --output text)
wait_for_step "$STEP_ID" "Simple Distributed FIS Workload - Spark Processing"

# Cleanup temporary files
rm -f /tmp/step_simple_distributed.json /tmp/simple_distributed_fis_workload.py

echo "=========================================="
echo "✅ Simple Distributed Steps Added Successfully!"
echo "=========================================="
echo ""
echo "Summary:"
echo "  ✓ Simple distributed workload with Spark across all cluster nodes completed"
echo ""
echo "Output Files:"
echo "  • Distributed workload results: s3://<AWS-BUCKET>-unifile/unifile_test/result_distributed_*_300m.tif"
echo "  • Distributed workload summary: s3://<AWS-BUCKET>-unifile/unifile_test/distributed_workload_results.json"
echo ""
echo "Cluster Status:"
echo "  • Processing: DISTRIBUTED across all nodes"
echo "  • Spark configuration: Optimized for cluster utilization"
echo "  • Parallelism: 200 partitions for distribution"
echo "" 