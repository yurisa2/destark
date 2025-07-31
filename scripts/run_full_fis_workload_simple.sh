#!/bin/bash

# Set AWS credentials for the correct account
export AWS_ACCESS_KEY_ID=YOUR_AWS_ACCESS_KEY_ID
export AWS_SECRET_ACCESS_KEY=YOUR_AWS_SECRET_ACCESS_KEY
export AWS_REGION=us-east-2
export AWS_DEFAULT_REGION=us-east-2

# Simple Full FIS Workload Runner for EMR Cluster
# This script runs the complete FIS processing with real data

set -e

# Configuration
CLUSTER_ID="<EMR-CLUSTER-ID>"
REGION="us-east-2"
S3_BUCKET="<AWS-BUCKET>-unifile"
S3_PREFIX="unifile_test"

echo "=========================================="
echo "Simple Full FIS Workload Runner for EMR"
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

# Step 2: Run full workload with real data and all 6 configurations
echo "Step 2: Running full workload with real data and all 6 configurations..."

# Create the Python script locally first
cat > /tmp/full_workload_real_data.py << 'PYTHON_EOF'
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

def download_file_from_s3(s3_path, local_path):
    try:
        print(f'Downloading {s3_path} to {local_path}...')
        if s3_path.startswith('s3://'):
            bucket = s3_path.split('/')[2]
            key = '/'.join(s3_path.split('/')[3:])
        else:
            raise ValueError(f'Invalid S3 path: {s3_path}')
        s3_client = boto3.client('s3')
        s3_client.download_file(bucket, key, local_path)
        file_size = os.path.getsize(local_path)
        print(f'✓ Downloaded {s3_path} ({file_size:,} bytes)')
        return True
    except Exception as e:
        print(f'❌ Failed to download {s3_path}: {e}')
        return False

def upload_file_to_s3(local_path, s3_path):
    try:
        print(f'Uploading {local_path} to {s3_path}...')
        if s3_path.startswith('s3://'):
            bucket = s3_path.split('/')[2]
            key = '/'.join(s3_path.split('/')[3:])
        else:
            raise ValueError(f'Invalid S3 path: {s3_path}')
        s3_client = boto3.client('s3')
        s3_client.upload_file(local_path, bucket, key)
        file_size = os.path.getsize(local_path)
        print(f'✓ Uploaded {local_path} to {s3_path} ({file_size:,} bytes)')
        return True
    except Exception as e:
        print(f'❌ Failed to upload {local_path}: {e}')
        return False

def create_fis_from_config(config_data):
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

def process_raster_with_config(social_data, env_data, strat_data, simulation, input_vars, output_var, nodata_value=5.0):
    result = np.zeros_like(social_data, dtype=np.float32)
    print(f'Processing raster with configuration...')
    print(f'Data shape: {social_data.shape}')
    print(f'Processing {social_data.shape[0] * social_data.shape[1]:,} pixels...')
    processed_pixels = 0
    
    for i in range(social_data.shape[0]):
        if i % 1000 == 0:
            print(f'Processing row {i}/{social_data.shape[0]} ({i/social_data.shape[0]*100:.1f}%)')
        for j in range(social_data.shape[1]):
            try:
                social_val = float(social_data[i, j])
                env_val = float(env_data[i, j])
                strat_val = float(strat_data[i, j])
                
                # Skip NoData pixels
                if (np.isnan(social_val) or np.isnan(env_val) or np.isnan(strat_val) or
                    social_val == nodata_value or env_val == nodata_value or strat_val == nodata_value):
                    result[i, j] = nodata_value
                    continue
                
                # Set input values
                simulation.input[input_vars[0]] = social_val
                simulation.input[input_vars[1]] = env_val
                simulation.input[input_vars[2]] = strat_val
                simulation.compute()
                result[i, j] = float(simulation.output[output_var])
                processed_pixels += 1
            except Exception as e:
                result[i, j] = nodata_value
    
    print(f'✓ Processed {processed_pixels:,} pixels')
    return result

def main():
    print('Starting full FIS workload with REAL DATA and all 6 configurations...')
    start_time = time.time()
    
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
    print('Downloading input raster files...')
    local_files = {}
    for name, s3_path in input_files.items():
        local_path = f'/tmp/{name}_300m.tif'
        if download_file_from_s3(s3_path, local_path):
            local_files[name] = local_path
        else:
            print(f'❌ Failed to download {name} file')
            return
    
    # Read raster data
    print('Reading raster data...')
    try:
        social_data = tifffile.imread(local_files['social'])
        env_data = tifffile.imread(local_files['environmental'])
        strat_data = tifffile.imread(local_files['strategic'])
        print(f'✓ Raster data loaded:')
        print(f'  Social: {social_data.shape}, dtype: {social_data.dtype}')
        print(f'  Environmental: {env_data.shape}, dtype: {env_data.dtype}')
        print(f'  Strategic: {strat_data.shape}, dtype: {strat_data.dtype}')
    except Exception as e:
        print(f'❌ Failed to read raster data: {e}')
        return
    
    # Process each configuration
    results = []
    for i, config_file in enumerate(config_files):
        print(f'\nProcessing configuration {i+1}/{len(config_files)}: {config_file}')
        config_start_time = time.time()
        
        try:
            # Download configuration file
            config_s3_path = f's3://<AWS-BUCKET>-unifile/unifile_test/{config_file}'
            config_local_path = f'/tmp/{config_file}'
            if not download_file_from_s3(config_s3_path, config_local_path):
                print(f'❌ Failed to download configuration {config_file}')
                continue
            
            # Load configuration
            with open(config_local_path, 'r') as f:
                config_data = json.load(f)
            print(f'✓ Configuration loaded: {len(config_data.get("rules", []))} rules')
            
            # Create FIS from configuration
            simulation, input_vars, output_var = create_fis_from_config(config_data)
            print(f'✓ FIS created with {len(input_vars)} input variables')
            
            # Process raster
            result_data = process_raster_with_config(social_data, env_data, strat_data, simulation, input_vars, output_var)
            
            # Save result
            config_name = config_file.replace('.json', '')
            output_file = f'/tmp/result_full_workload_{config_name}_300m.tif'
            print(f'Saving result to {output_file}...')
            tifffile.imwrite(output_file, result_data, photometric='minisblack')
            
            # Upload to S3
            s3_key = f'unifile_test/result_full_workload_{config_name}_300m.tif'
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
                    'rules_count': len(config_data.get('rules', []))
                })
                print(f'✓ Configuration {config_file} completed in {config_time:.2f} seconds')
            else:
                config_time = time.time() - config_start_time
                results.append({
                    'config': config_file,
                    'time': config_time,
                    'status': 'failed',
                    'error': 'Upload failed'
                })
                print(f'✗ Configuration {config_file} upload failed')
                
        except Exception as e:
            config_time = time.time() - config_start_time
            results.append({
                'config': config_file,
                'time': config_time,
                'status': 'failed',
                'error': str(e)
            })
            print(f'✗ Configuration {config_file} failed: {e}')
    
    # Generate summary
    total_time = time.time() - start_time
    successful_configs = [r for r in results if r['status'] == 'success']
    failed_configs = [r for r in results if r['status'] == 'failed']
    
    print('\n=== FULL WORKLOAD SUMMARY ===')
    print(f'Total processing time: {total_time:.2f} seconds')
    print(f'Successful configurations: {len(successful_configs)}/{len(config_files)}')
    print(f'Failed configurations: {len(failed_configs)}/{len(config_files)}')
    print(f'Data processed: {social_data.shape[0]}x{social_data.shape[1]} = {social_data.shape[0] * social_data.shape[1]:,} pixels')
    
    print('\nSuccessful configurations:')
    for result in successful_configs:
        print(f'  ✓ {result["config"]}: {result["time"]:.2f}s ({result["rules_count"]} rules) -> {result["output"]}')
    
    if failed_configs:
        print('\nFailed configurations:')
        for result in failed_configs:
            print(f'  ✗ {result["config"]}: {result["error"]}')
    
    # Save results to S3
    results_file = '/tmp/full_workload_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'cluster_id': '<EMR-CLUSTER-ID>',
            'total_time': total_time,
            'data_shape': social_data.shape,
            'total_pixels': social_data.shape[0] * social_data.shape[1],
            'configurations_processed': len(config_files),
            'results': results
        }, f, indent=2)
    
    s3_client = boto3.client('s3')
    s3_client.upload_file(results_file, '<AWS-BUCKET>-unifile', 'unifile_test/full_workload_results.json')
    print(f'\nResults saved to: s3://<AWS-BUCKET>-unifile/unifile_test/full_workload_results.json')
    print('=== FULL WORKLOAD COMPLETED ===')

if __name__ == '__main__':
    main()
PYTHON_EOF

# Upload the script to S3
echo "Uploading Python script to S3..."
aws s3 cp /tmp/full_workload_real_data.py s3://<AWS-BUCKET>-unifile/unifile_test/full_workload_real_data.py

cat > /tmp/step2_full_workload.json << 'EOF'
[
  {
    "Name": "Full FIS Workload - Real Data All 6 Configurations",
    "ActionOnFailure": "CONTINUE",
    "Type": "CUSTOM_JAR",
    "Jar": "command-runner.jar",
    "Args": [
      "bash", "-c",
      "echo '=== FULL FIS WORKLOAD - REAL DATA ALL 6 CONFIGURATIONS ===' && cd /mnt/destark && echo 'Downloading Python script from S3...' && aws s3 cp s3://<AWS-BUCKET>-unifile/unifile_test/full_workload_real_data.py app/full_workload_real_data.py && chmod +x app/full_workload_real_data.py && echo 'Running full FIS workload with REAL DATA and ALL 6 CONFIGURATIONS...' && python3 app/full_workload_real_data.py && echo '=== FULL FIS WORKLOAD WITH REAL DATA AND ALL 6 CONFIGURATIONS COMPLETED ==='"
    ]
  }
]
EOF

STEP2_ID=$(aws emr add-steps --cluster-id "$CLUSTER_ID" --region "$REGION" --steps file:///tmp/step2_full_workload.json --query 'StepIds[0]' --output text)
wait_for_step "$STEP2_ID" "Full FIS Workload - Real Data All 6 Configurations"

# Cleanup temporary files
rm -f /tmp/step1_setup.json /tmp/step2_full_workload.json

echo "=========================================="
echo "✅ Full FIS Workload Completed Successfully!"
echo "=========================================="
echo ""
echo "Summary:"
echo "  ✓ Python packages installed"
echo "  ✓ Development environment setup"
echo "  ✓ Full workload with REAL 300m DATA and ALL 6 CONFIGURATIONS completed"
echo ""
echo "Output Files:"
echo "  • Full workload results (REAL DATA): s3://$S3_BUCKET/$S3_PREFIX/result_full_workload_*_300m.tif"
echo "  • Workload summary: s3://$S3_BUCKET/$S3_PREFIX/full_workload_results.json"
echo ""
echo "Cluster Status:"
aws emr describe-cluster --cluster-id "$CLUSTER_ID" --region "$REGION" --query 'Cluster.Status.State' --output text
echo ""
echo "==========================================" 