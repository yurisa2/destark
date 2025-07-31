#!/bin/bash

# Full FIS Workload Runner for EMR Cluster
# This script tests everything and runs complete FIS processing with all configurations

set -e

# Configuration
CLUSTER_ID="<EMR-CLUSTER-ID>"
REGION="us-east-2"
S3_BUCKET="<AWS-BUCKET>-unifile"
S3_PREFIX="unifile_test"

echo "=========================================="
echo "Full FIS Workload Runner for EMR"
echo "Cluster ID: $CLUSTER_ID"
echo "Region: $REGION"
echo "=========================================="

# Function to check cluster status
check_cluster_status() {
    echo "Checking cluster status..."
    CLUSTER_STATUS=$(aws emr describe-cluster --cluster-id "$CLUSTER_ID" --region "$REGION" --query 'Cluster.Status.State' --output text)
    
    if [ "$CLUSTER_STATUS" != "WAITING" ] && [ "$CLUSTER_STATUS" != "RUNNING" ]; then
        echo "❌ Cluster is not ready. Status: $CLUSTER_STATUS"
        echo "Please wait for cluster to be in WAITING or RUNNING state"
        exit 1
    fi
    
    echo "✅ Cluster is ready. Status: $CLUSTER_STATUS"
    echo ""
}

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

# Step 1: Basic Environment Test
echo "Step 1: Testing Basic Environment..."
cat > /tmp/step1_basic_test.json << 'EOF'
[
  {
    "Name": "Basic Environment Test",
    "ActionOnFailure": "CONTINUE",
    "Type": "CUSTOM_JAR",
    "Jar": "command-runner.jar",
    "Args": [
      "bash", "-c",
      "echo '=== BASIC ENVIRONMENT TEST ===' && echo 'Testing Python environment...' && python3 --version && echo 'Testing package imports...' && python3 -c \"import tifffile, numpy, skfuzzy, boto3, pyspark, psutil, pandas, networkx, matplotlib, tqdm; print('✓ ALL packages imported successfully')\" && echo 'Testing tifffile operations...' && python3 -c \"import tifffile; import numpy as np; data = np.random.rand(100, 100).astype(np.float32); tifffile.imwrite('/tmp/test.tif', data); print('✓ Tifffile write test passed')\" && python3 -c \"import tifffile; data = tifffile.imread('/tmp/test.tif'); print(f'✓ Tifffile read test passed, shape: {data.shape}')\" && echo 'Testing S3 connectivity...' && python3 -c \"import boto3; s3 = boto3.client('s3'); print('✓ S3 client created successfully')\" && echo 'Testing Spark environment...' && python3 -c \"from pyspark.sql import SparkSession; print('✓ PySpark imported successfully')\" && echo '=== BASIC ENVIRONMENT TEST PASSED ==='"
    ]
  }
]
EOF

STEP1_ID=$(aws emr add-steps --cluster-id "$CLUSTER_ID" --region "$REGION" --steps file:///tmp/step1_basic_test.json --query 'StepIds[0]' --output text)
wait_for_step "$STEP1_ID" "Basic Environment Test"

# Step 2: S3 Integration Test
echo "Step 2: Testing S3 Integration..."
cat > /tmp/step2_s3_test.json << 'EOF'
[
  {
    "Name": "S3 Integration Test",
    "ActionOnFailure": "CONTINUE",
    "Type": "CUSTOM_JAR",
    "Jar": "command-runner.jar",
    "Args": [
      "bash", "-c",
      "echo '=== S3 INTEGRATION TEST ===' && export S3_BUCKET=<AWS-BUCKET>-unifile && export S3_PREFIX=unifile_test && echo 'Testing S3 download...' && python3 -c \"import boto3; import os; s3 = boto3.client('s3'); s3.download_file('$S3_BUCKET', '$S3_PREFIX/config_max.json', '/tmp/config_test.json'); print('✓ S3 download test passed')\" && echo 'Testing S3 upload...' && python3 -c \"import boto3; import numpy as np; import tifffile; data = np.random.rand(50, 50).astype(np.float32); tifffile.imwrite('/tmp/test_upload.tif', data); s3 = boto3.client('s3'); s3.upload_file('/tmp/test_upload.tif', '$S3_BUCKET', '$S3_PREFIX/test_upload.tif'); print('✓ S3 upload test passed')\" && echo '=== S3 INTEGRATION TEST PASSED ==='"
    ]
  }
]
EOF

STEP2_ID=$(aws emr add-steps --cluster-id "$CLUSTER_ID" --region "$REGION" --steps file:///tmp/step2_s3_test.json --query 'StepIds[0]' --output text)
wait_for_step "$STEP2_ID" "S3 Integration Test"

# Step 3: Deploy FIS Application
echo "Step 3: Deploying FIS Application..."
cat > /tmp/step3_deploy_app.json << 'EOF'
[
  {
    "Name": "Deploy FIS Application",
    "ActionOnFailure": "CONTINUE",
    "Type": "CUSTOM_JAR",
    "Jar": "command-runner.jar",
    "Args": [
      "bash", "-c",
      "echo '=== DEPLOYING FIS APPLICATION ===' && cd /mnt/destark && echo 'Downloading application files from S3...' && aws s3 cp s3://<AWS-BUCKET>-unifile/unifile_test/raster_fuzzy_spark_s3_tifffile.py app/ && aws s3 cp s3://<AWS-BUCKET>-unifile/unifile_test/raster_fuzzy_lib_tifffile.py app/ && aws s3 cp s3://<AWS-BUCKET>-unifile/unifile_test/raster_fuzzy_cli_tifffile.py app/ && echo 'Downloading configuration files...' && aws s3 cp s3://<AWS-BUCKET>-unifile/unifile_test/config_max.json config/ && aws s3 cp s3://<AWS-BUCKET>-unifile/unifile_test/config_median.json config/ && aws s3 cp s3://<AWS-BUCKET>-unifile/unifile_test/config_minimum.json config/ && aws s3 cp s3://<AWS-BUCKET>-unifile/unifile_test/config_mode.json config/ && aws s3 cp s3://<AWS-BUCKET>-unifile/unifile_test/config_round_down.json config/ && aws s3 cp s3://<AWS-BUCKET>-unifile/unifile_test/config_round_up.json config/ && echo 'Setting up development environment...' && chmod +x app/*.py && echo 'Application files deployed:' && ls -la app/ && ls -la config/ && echo '=== FIS APPLICATION DEPLOYED ==='"
    ]
  }
]
EOF

STEP3_ID=$(aws emr add-steps --cluster-id "$CLUSTER_ID" --region "$REGION" --steps file:///tmp/step3_deploy_app.json --query 'StepIds[0]' --output text)
wait_for_step "$STEP3_ID" "Deploy FIS Application"

# Step 4: Test Individual FIS Processing
echo "Step 4: Testing Individual FIS Processing..."
cat > /tmp/step4_fis_test.json << 'EOF'
[
  {
    "Name": "Individual FIS Processing Test",
    "ActionOnFailure": "CONTINUE",
    "Type": "CUSTOM_JAR",
    "Jar": "command-runner.jar",
    "Args": [
      "bash", "-c",
      "echo '=== INDIVIDUAL FIS PROCESSING TEST ===' && cd /mnt/destark && export S3_BUCKET=<AWS-BUCKET>-unifile && export S3_PREFIX=unifile_test && echo 'Testing FIS library import...' && python3 -c \"import sys; sys.path.append('/mnt/destark/app'); from raster_fuzzy_lib_tifffile import UnifiedRasterFuzzyInferenceSystem; print('✓ FIS library imported successfully')\" && echo 'Testing FIS configuration loading...' && python3 -c \"import sys; sys.path.append('/mnt/destark/app'); from raster_fuzzy_lib_tifffile import UnifiedRasterFuzzyInferenceSystem; fis = UnifiedRasterFuzzyInferenceSystem('/mnt/destark/config/config_max.json'); print('✓ FIS configuration loaded successfully')\" && echo 'Testing small raster processing...' && python3 -c \"import sys; sys.path.append('/mnt/destark/app'); import numpy as np; import tifffile; data = np.random.rand(100, 100).astype(np.float32); tifffile.imwrite('/tmp/test_social.tif', data); tifffile.imwrite('/tmp/test_env.tif', data); tifffile.imwrite('/tmp/test_strat.tif', data); from raster_fuzzy_lib_tifffile import UnifiedRasterFuzzyInferenceSystem; fis = UnifiedRasterFuzzyInferenceSystem('/mnt/destark/config/config_max.json'); fis.process_rasters('/tmp/test_social.tif', '/tmp/test_env.tif', '/tmp/test_strat.tif', '/tmp/test_output.tif', parallel=False); print('✓ Individual FIS processing test passed')\" && echo '=== INDIVIDUAL FIS PROCESSING TEST PASSED ==='"
    ]
  }
]
EOF

STEP4_ID=$(aws emr add-steps --cluster-id "$CLUSTER_ID" --region "$REGION" --steps file:///tmp/step4_fis_test.json --query 'StepIds[0]' --output text)
wait_for_step "$STEP4_ID" "Individual FIS Processing Test"

# Step 5: Full Spark FIS Workload with All Configurations
echo "Step 5: Running Full Spark FIS Workload with All Configurations..."
cat > /tmp/step5_full_workload.json << 'EOF'
[
  {
    "Name": "Full Spark FIS Workload - All Configurations",
    "ActionOnFailure": "CONTINUE",
    "Type": "CUSTOM_JAR",
    "Jar": "command-runner.jar",
    "Args": [
      "bash", "-c",
      "echo '=== FULL SPARK FIS WORKLOAD - ALL CONFIGURATIONS ===' && cd /mnt/destark && export S3_BUCKET=<AWS-BUCKET>-unifile && export S3_PREFIX=unifile_test && echo 'Creating comprehensive FIS workload script...' && cat > /tmp/full_fis_workload.py << 'PYTHON_EOF' && #!/usr/bin/env python3 && import os && import sys && import time && import json && import numpy as np && import tifffile && import boto3 && from datetime import datetime && sys.path.append('/mnt/destark/app') && from raster_fuzzy_spark_s3_tifffile import process_rasters_spark_s3_tifffile, create_spark_session_s3 && def run_full_workload(): &&     print('Starting full FIS workload with all configurations...') &&     start_time = time.time() &&     configs = ['config_max.json', 'config_median.json', 'config_minimum.json', 'config_mode.json', 'config_round_down.json', 'config_round_up.json'] &&     input_files = { &&         'social': 's3://<AWS-BUCKET>-unifile/unifile_test/so300m.in', &&         'environmental': 's3://<AWS-BUCKET>-unifile/unifile_test/e300m.in', &&         'strategic': 's3://<AWS-BUCKET>-unifile/unifile_test/s300m.in' &&     } &&     # Create Spark session &&     print('Creating Spark session...') &&     spark = create_spark_session_s3('FullFISWorkload', local_mode=False) &&     print(f'Spark session created: {spark.sparkContext.appName}') &&     results = [] &&     for i, config_file in enumerate(configs): &&         config_path = f'/mnt/destark/config/{config_file}' &&         output_file = f's3://<AWS-BUCKET>-unifile/unifile_test/result_full_workload_{config_file.replace(\".json\", \"\")}.tif' &&         print(f'\\nProcessing configuration {i+1}/{len(configs)}: {config_file}') &&         config_start_time = time.time() &&         try: &&             process_rasters_spark_s3_tifffile( &&                 spark=spark, &&                 social_tiff_s3=input_files['social'], &&                 environmental_tiff_s3=input_files['environmental'], &&                 strategic_tiff_s3=input_files['strategic'], &&                 output_tiff_s3=output_file, &&                 config_file=config_path, &&                 nodata_value=5.0, &&                 block_size=200, &&                 num_partitions=20 &&             ) &&             config_time = time.time() - config_start_time &&             results.append({ &&                 'config': config_file, &&                 'output': output_file, &&                 'time': config_time, &&                 'status': 'success' &&             }) &&             print(f'✓ Configuration {config_file} completed in {config_time:.2f} seconds') &&         except Exception as e: &&             config_time = time.time() - config_start_time &&             results.append({ &&                 'config': config_file, &&                 'output': output_file, &&                 'time': config_time, &&                 'status': 'failed', &&                 'error': str(e) &&             }) &&             print(f'✗ Configuration {config_file} failed: {e}') &&     # Stop Spark session &&     spark.stop() &&     # Generate summary &&     total_time = time.time() - start_time &&     successful_configs = [r for r in results if r['status'] == 'success'] &&     failed_configs = [r for r in results if r['status'] == 'failed'] &&     print('\\n=== FULL WORKLOAD SUMMARY ===') && print(f'Total processing time: {total_time:.2f} seconds') && print(f'Successful configurations: {len(successful_configs)}/{len(configs)}') && print(f'Failed configurations: {len(failed_configs)}/{len(configs)}') && print('\\nSuccessful configurations:') && for result in successful_configs: &&     print(f'  ✓ {result[\"config\"]}: {result[\"time\"]:.2f}s -> {result[\"output\"]}') && if failed_configs: &&     print('\\nFailed configurations:') &&     for result in failed_configs: &&         print(f'  ✗ {result[\"config\"]}: {result[\"error\"]}') && # Save results to S3 && results_file = '/tmp/workload_results.json' && with open(results_file, 'w') as f: &&     json.dump({ &&         'timestamp': datetime.now().isoformat(), &&         'cluster_id': '<EMR-CLUSTER-ID>', &&         'total_time': total_time, &&         'results': results &&     }, f, indent=2) && s3_client = boto3.client('s3') && s3_client.upload_file(results_file, '<AWS-BUCKET>-unifile', 'unifile_test/full_workload_results.json') && print(f'\\nResults saved to: s3://<AWS-BUCKET>-unifile/unifile_test/full_workload_results.json') && print('=== FULL WORKLOAD COMPLETED ===') && if __name__ == '__main__': &&     run_full_workload() && PYTHON_EOF && echo 'Running full FIS workload...' && python3 /tmp/full_fis_workload.py && echo '=== FULL SPARK FIS WORKLOAD COMPLETED ==='"
    ]
  }
]
EOF

STEP5_ID=$(aws emr add-steps --cluster-id "$CLUSTER_ID" --region "$REGION" --steps file:///tmp/step5_full_workload.json --query 'StepIds[0]' --output text)
wait_for_step "$STEP5_ID" "Full Spark FIS Workload - All Configurations"

# Step 6: Performance Analysis and Cleanup
echo "Step 6: Performance Analysis and Cleanup..."
cat > /tmp/step6_analysis.json << 'EOF'
[
  {
    "Name": "Performance Analysis and Cleanup",
    "ActionOnFailure": "CONTINUE",
    "Type": "CUSTOM_JAR",
    "Jar": "command-runner.jar",
    "Args": [
      "bash", "-c",
      "echo '=== PERFORMANCE ANALYSIS AND CLEANUP ===' && echo 'Downloading workload results...' && aws s3 cp s3://<AWS-BUCKET>-unifile/unifile_test/full_workload_results.json /tmp/ && echo 'Analyzing results...' && python3 -c \"import json; import sys; data = json.load(open('/tmp/full_workload_results.json')); print('=== WORKLOAD ANALYSIS ==='); print(f'Total processing time: {data[\"total_time\"]:.2f} seconds'); print(f'Successful configurations: {len([r for r in data[\"results\"] if r[\"status\"] == \"success\"])}/{len(data[\"results\"])}'); print('\\nConfiguration Performance:'); for result in data['results']: print(f'  {result[\"config\"]}: {result[\"time\"]:.2f}s ({result[\"status\"]})')\" && echo 'Cleaning up temporary files...' && rm -f /tmp/test*.tif /tmp/full_fis_workload.py /tmp/workload_results.json && echo '=== PERFORMANCE ANALYSIS COMPLETED ==='"
    ]
  }
]
EOF

STEP6_ID=$(aws emr add-steps --cluster-id "$CLUSTER_ID" --region "$REGION" --steps file:///tmp/step6_analysis.json --query 'StepIds[0]' --output text)
wait_for_step "$STEP6_ID" "Performance Analysis and Cleanup"

# Cleanup temporary files
rm -f /tmp/step1_basic_test.json /tmp/step2_s3_test.json /tmp/step3_deploy_app.json /tmp/step4_fis_test.json /tmp/step5_full_workload.json /tmp/step6_analysis.json

echo "=========================================="
echo "✅ Full FIS Workload Completed Successfully!"
echo "=========================================="
echo ""
echo "Summary:"
echo "  ✓ Basic environment test passed"
echo "  ✓ S3 integration test passed"
echo "  ✓ FIS application deployed"
echo "  ✓ Individual FIS processing test passed"
echo "  ✓ Full Spark workload with all configurations completed"
echo "  ✓ Performance analysis completed"
echo ""
echo "Output Files:"
echo "  • Individual results: s3://$S3_BUCKET/$S3_PREFIX/result_full_workload_*.tif"
echo "  • Workload summary: s3://$S3_BUCKET/$S3_PREFIX/full_workload_results.json"
echo ""
echo "Cluster Status:"
aws emr describe-cluster --cluster-id "$CLUSTER_ID" --region "$REGION" --query 'Cluster.Status.State' --output text
echo ""
echo "==========================================" 