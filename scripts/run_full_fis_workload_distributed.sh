#!/bin/bash

# Set AWS credentials for the correct account
export AWS_ACCESS_KEY_ID=YOUR_AWS_ACCESS_KEY_ID
export AWS_SECRET_ACCESS_KEY=YOUR_AWS_SECRET_ACCESS_KEY
export AWS_REGION=us-east-2
export AWS_DEFAULT_REGION=us-east-2

# Distributed Full FIS Workload Runner for EMR Cluster
# This script uses Spark to distribute processing across all nodes

set -e

# Configuration
CLUSTER_ID="<EMR-CLUSTER-ID>"
REGION="us-east-2"
S3_BUCKET="<AWS-BUCKET>-unifile"
S3_PREFIX="unifile_test"

echo "=========================================="
echo "Distributed Full FIS Workload Runner for EMR"
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

# Step 2: Create distributed FIS processing script
echo "Step 2: Creating distributed FIS processing script..."

# Create the distributed Python script locally first
cat > /tmp/distributed_fis_workload.py << 'PYTHON_EOF'
#!/usr/bin/env python3
import os
import sys
import time
import json
import numpy as np
import tifffile
import boto3
from datetime import datetime
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, lit
from pyspark.sql.types import StructType, StructField, StringType, FloatType, IntegerType
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_spark():
    """Setup Spark session with optimized configuration for distributed processing."""
    spark = SparkSession.builder \
        .appName("Distributed FIS Processing") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.sql.adaptive.skewJoin.enabled", "true") \
        .config("spark.sql.adaptive.localShuffleReader.enabled", "true") \
        .config("spark.sql.adaptive.advisoryPartitionSizeInBytes", "128m") \
        .config("spark.sql.shuffle.partitions", "200") \
        .config("spark.default.parallelism", "200") \
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
        .getOrCreate()
    
    logger.info(f"Spark session created with {spark.sparkContext.defaultParallelism} partitions")
    logger.info(f"Available executors: {spark.sparkContext._jsc.sc().getExecutorMemoryStatus().size()}")
    return spark

def download_file_from_s3(s3_path, local_path):
    """Download file from S3 with logging."""
    try:
        logger.info(f'Downloading {s3_path} to {local_path}...')
        if s3_path.startswith('s3://'):
            bucket = s3_path.split('/')[2]
            key = '/'.join(s3_path.split('/')[3:])
        else:
            raise ValueError(f'Invalid S3 path: {s3_path}')
        s3_client = boto3.client('s3')
        s3_client.download_file(bucket, key, local_path)
        file_size = os.path.getsize(local_path)
        logger.info(f'✓ Downloaded {s3_path} ({file_size:,} bytes)')
        return True
    except Exception as e:
        logger.error(f'❌ Failed to download {s3_path}: {e}')
        return False

def upload_file_to_s3(local_path, s3_path):
    """Upload file to S3 with logging."""
    try:
        logger.info(f'Uploading {local_path} to {s3_path}...')
        if s3_path.startswith('s3://'):
            bucket = s3_path.split('/')[2]
            key = '/'.join(s3_path.split('/')[3:])
        else:
            raise ValueError(f'Invalid S3 path: {s3_path}')
        s3_client = boto3.client('s3')
        s3_client.upload_file(local_path, bucket, key)
        file_size = os.path.getsize(local_path)
        logger.info(f'✓ Uploaded {local_path} to {s3_path} ({file_size:,} bytes)')
        return True
    except Exception as e:
        logger.error(f'❌ Failed to upload {local_path}: {e}')
        return False

def create_fis_from_config(config_data):
    """Create FIS from configuration with proper error handling."""
    try:
        # Create fuzzy variables based on config
        input_vars = {}
        for var_name, var_config in config_data['input_variables'].items():
            universe = np.arange(var_config['min'], var_config['max'] + 1, 1)
            input_vars[var_name] = ctrl.Antecedent(universe, var_name)
            # Define membership functions
            for term_name, term_config in var_config['membership_functions'].items():
                if term_config['type'] == 'trimf':
                    input_vars[var_name][term_name] = fuzz.trimf(input_vars[var_name].universe, term_config['params'])
                elif term_config['type'] == 'trapmf':
                    input_vars[var_name][term_name] = fuzz.trapmf(input_vars[var_name].universe, term_config['params'])
        
        # Create output variable
        output_config = config_data['output_variable']
        universe = np.arange(output_config['min'], output_config['max'] + 1, 1)
        output_var = ctrl.Consequent(universe, output_config['name'])
        for term_name, term_config in output_config['membership_functions'].items():
            if term_config['type'] == 'trimf':
                output_var[term_name] = fuzz.trimf(output_var.universe, term_config['params'])
            elif term_config['type'] == 'trapmf':
                output_var[term_name] = fuzz.trapmf(output_var.universe, term_config['params'])
        
        # Create rules
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
        
        # Create control system
        fis = ctrl.ControlSystem(rules)
        simulation = ctrl.ControlSystemSimulation(fis)
        return simulation, list(input_vars.keys()), output_config['name']
    except Exception as e:
        logger.error(f"Error creating FIS from config: {e}")
        raise

def process_pixel_udf(social_val, env_val, strat_val, simulation, input_vars, output_var, nodata_value=5.0):
    """UDF to process a single pixel with FIS."""
    try:
        # Skip NoData pixels
        if (np.isnan(social_val) or np.isnan(env_val) or np.isnan(strat_val) or
            social_val == nodata_value or env_val == nodata_value or strat_val == nodata_value):
            return float(nodata_value)
        
        # Set input values
        simulation.input[input_vars[0]] = float(social_val)
        simulation.input[input_vars[1]] = float(env_val)
        simulation.input[input_vars[2]] = float(strat_val)
        simulation.compute()
        return float(simulation.output[output_var])
    except Exception as e:
        logger.warning(f"Error processing pixel: {e}")
        return float(nodata_value)

def create_pixel_dataframe(spark, social_data, env_data, strat_data):
    """Create Spark DataFrame from raster data for distributed processing."""
    logger.info("Creating distributed DataFrame from raster data...")
    
    # Convert raster data to list of tuples (row, col, social, env, strat)
    pixels = []
    for i in range(social_data.shape[0]):
        for j in range(social_data.shape[1]):
            pixels.append((i, j, float(social_data[i, j]), float(env_data[i, j]), float(strat_data[i, j])))
    
    # Create DataFrame schema
    schema = StructType([
        StructField("row", IntegerType(), False),
        StructField("col", IntegerType(), False),
        StructField("social", FloatType(), False),
        StructField("environmental", FloatType(), False),
        StructField("strategic", FloatType(), False)
    ])
    
    # Create DataFrame
    df = spark.createDataFrame(pixels, schema)
    logger.info(f"Created DataFrame with {df.count():,} pixels")
    logger.info(f"DataFrame partitions: {df.rdd.getNumPartitions()}")
    
    return df

def process_raster_distributed(spark, social_data, env_data, strat_data, simulation, input_vars, output_var, config_name):
    """Process raster data using Spark distributed processing."""
    logger.info(f"Starting distributed processing for {config_name}...")
    start_time = time.time()
    
    # Create DataFrame
    df = create_pixel_dataframe(spark, social_data, env_data, strat_data)
    
    # Create UDF for FIS processing
    def fis_process(social, env, strat):
        return process_pixel_udf(social, env, strat, simulation, input_vars, output_var)
    
    fis_udf = udf(fis_process, FloatType())
    
    # Apply FIS processing
    logger.info("Applying FIS processing across all partitions...")
    result_df = df.withColumn("result", fis_udf(col("social"), col("environmental"), col("strategic")))
    
    # Collect results back to driver
    logger.info("Collecting results from all executors...")
    results = result_df.select("row", "col", "result").collect()
    
    # Reconstruct raster
    logger.info("Reconstructing raster from distributed results...")
    result_raster = np.zeros_like(social_data, dtype=np.float32)
    for row in results:
        result_raster[row.row, row.col] = row.result
    
    processing_time = time.time() - start_time
    logger.info(f"✓ Distributed processing completed in {processing_time:.2f} seconds")
    
    return result_raster

def main():
    """Main function for distributed FIS processing."""
    logger.info('Starting DISTRIBUTED FIS workload with REAL DATA and all 6 configurations...')
    start_time = time.time()
    
    # Setup Spark
    spark = setup_spark()
    
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
    logger.info('Downloading input raster files...')
    local_files = {}
    for name, s3_path in input_files.items():
        local_path = f'/tmp/{name}_300m.tif'
        if download_file_from_s3(s3_path, local_path):
            local_files[name] = local_path
        else:
            logger.error(f'❌ Failed to download {name} file')
            return
    
    # Read raster data
    logger.info('Reading raster data...')
    try:
        social_data = tifffile.imread(local_files['social'])
        env_data = tifffile.imread(local_files['environmental'])
        strat_data = tifffile.imread(local_files['strategic'])
        logger.info(f'✓ Raster data loaded:')
        logger.info(f'  Social: {social_data.shape}, dtype: {social_data.dtype}')
        logger.info(f'  Environmental: {env_data.shape}, dtype: {env_data.dtype}')
        logger.info(f'  Strategic: {strat_data.shape}, dtype: {strat_data.dtype}')
    except Exception as e:
        logger.error(f'❌ Failed to read raster data: {e}')
        return
    
    # Process each configuration
    results = []
    for i, config_file in enumerate(config_files):
        logger.info(f'\nProcessing configuration {i+1}/{len(config_files)}: {config_file}')
        config_start_time = time.time()
        
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
            logger.info(f'✓ Configuration loaded: {len(config_data.get("rules", []))} rules')
            
            # Create FIS from configuration
            simulation, input_vars, output_var = create_fis_from_config(config_data)
            logger.info(f'✓ FIS created with {len(input_vars)} input variables')
            
            # Process raster using distributed processing
            config_name = config_file.replace('.json', '')
            result_data = process_raster_distributed(spark, social_data, env_data, strat_data, simulation, input_vars, output_var, config_name)
            
            # Save result
            output_file = f'/tmp/result_distributed_{config_name}_300m.tif'
            logger.info(f'Saving result to {output_file}...')
            tifffile.imwrite(output_file, result_data, photometric='minisblack')
            
            # Upload to S3
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
                logger.info(f'✓ Configuration {config_file} completed in {config_time:.2f} seconds')
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
    
    # Stop Spark
    spark.stop()
    
    # Generate summary
    total_time = time.time() - start_time
    successful_configs = [r for r in results if r['status'] == 'success']
    failed_configs = [r for r in results if r['status'] == 'failed']
    
    logger.info('\n=== DISTRIBUTED WORKLOAD SUMMARY ===')
    logger.info(f'Total processing time: {total_time:.2f} seconds')
    logger.info(f'Successful configurations: {len(successful_configs)}/{len(config_files)}')
    logger.info(f'Failed configurations: {len(failed_configs)}/{len(config_files)}')
    logger.info(f'Data processed: {social_data.shape[0]}x{social_data.shape[1]} = {social_data.shape[0] * social_data.shape[1]:,} pixels')
    logger.info(f'Processing type: DISTRIBUTED across all cluster nodes')
    
    logger.info('\nSuccessful configurations:')
    for result in successful_configs:
        logger.info(f'  ✓ {result["config"]}: {result["time"]:.2f}s ({result["rules_count"]} rules) -> {result["output"]}')
    
    if failed_configs:
        logger.info('\nFailed configurations:')
        for result in failed_configs:
            logger.info(f'  ✗ {result["config"]}: {result["error"]}')
    
    # Save results to S3
    results_file = '/tmp/distributed_workload_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'cluster_id': '<EMR-CLUSTER-ID>',
            'total_time': total_time,
            'data_shape': social_data.shape,
            'total_pixels': social_data.shape[0] * social_data.shape[1],
            'configurations_processed': len(config_files),
            'processing_type': 'distributed',
            'spark_config': {
                'default_parallelism': spark.sparkContext.defaultParallelism,
                'executor_cores': 4,
                'executor_memory': '20g',
                'driver_memory': '8g'
            },
            'results': results
        }, f, indent=2)
    
    s3_client = boto3.client('s3')
    s3_client.upload_file(results_file, '<AWS-BUCKET>-unifile', 'unifile_test/distributed_workload_results.json')
    logger.info(f'\nResults saved to: s3://<AWS-BUCKET>-unifile/unifile_test/distributed_workload_results.json')
    logger.info('=== DISTRIBUTED WORKLOAD COMPLETED ===')

if __name__ == '__main__':
    main()
PYTHON_EOF

# Upload the script to S3
echo "Uploading distributed Python script to S3..."
aws s3 cp /tmp/distributed_fis_workload.py s3://<AWS-BUCKET>-unifile/unifile_test/distributed_fis_workload.py

# Step 3: Run distributed workload using Spark
echo "Step 3: Running distributed workload using Spark..."
cat > /tmp/step3_distributed_workload.json << 'EOF'
[
  {
    "Name": "Distributed FIS Workload - Spark Processing",
    "ActionOnFailure": "CONTINUE",
    "Type": "CUSTOM_JAR",
    "Jar": "command-runner.jar",
    "Args": [
      "bash", "-c",
      "echo '=== DISTRIBUTED FIS WORKLOAD - SPARK PROCESSING ===' && cd /mnt/destark && echo 'Downloading distributed Python script from S3...' && aws s3 cp s3://<AWS-BUCKET>-unifile/unifile_test/distributed_fis_workload.py app/distributed_fis_workload.py && chmod +x app/distributed_fis_workload.py && echo 'Running distributed FIS workload with Spark...' && python3 app/distributed_fis_workload.py && echo '=== DISTRIBUTED FIS WORKLOAD COMPLETED ==='"
    ]
  }
]
EOF

STEP3_ID=$(aws emr add-steps --cluster-id "$CLUSTER_ID" --region "$REGION" --steps file:///tmp/step3_distributed_workload.json --query 'StepIds[0]' --output text)
wait_for_step "$STEP3_ID" "Distributed FIS Workload - Spark Processing"

# Cleanup temporary files
rm -f /tmp/step1_setup.json /tmp/step3_distributed_workload.json /tmp/distributed_fis_workload.py

echo "=========================================="
echo "✅ Distributed FIS Workload Completed Successfully!"
echo "=========================================="
echo ""
echo "Summary:"
echo "  ✓ Python packages installed"
echo "  ✓ Development environment setup"
echo "  ✓ DISTRIBUTED workload with Spark across all cluster nodes completed"
echo ""
echo "Output Files:"
echo "  • Distributed workload results: s3://$S3_BUCKET/$S3_PREFIX/result_distributed_*_300m.tif"
echo "  • Distributed workload summary: s3://$S3_BUCKET/$S3_PREFIX/distributed_workload_results.json"
echo ""
echo "Cluster Status:"
echo "  • Processing: DISTRIBUTED across all nodes"
echo "  • Spark configuration: Optimized for cluster utilization"
echo "  • Parallelism: Maximum cluster utilization"
echo "" 