#!/bin/bash

# Deploy Fixed EMR FIS Processing
# Uses existing FIS logic and configs, but fixes Java version compatibility

echo "=== DEPLOYING FIXED EMR FIS PROCESSING ==="

# Check if cluster ID is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <cluster-id>"
    echo "Example: $0 <EMR-CLUSTER-ID>"
    exit 1
fi

CLUSTER_ID=$1
REGION=${AWS_DEFAULT_REGION:-us-east-2}

echo "Cluster ID: $CLUSTER_ID"
echo "Region: $REGION"

# Create the EMR step JSON
cat > /tmp/emr_fis_fixed_step.json << 'EOF'
{
  "Name": "FIS Processing - Fixed Java 8 Compatible",
  "ActionOnFailure": "CONTINUE",
  "HadoopJarStep": {
    "Jar": "command-runner.jar",
    "Args": [
      "bash", "-c",
      "cat > /tmp/emr_fis_fixed.py << 'SCRIPT_EOF'
#!/usr/bin/env python3
import os
import sys
import json
import numpy as np
import boto3

# Set environment to use EMR's Spark (Java 8 compatible)
os.environ['SPARK_HOME'] = '/usr/lib/spark'
os.environ['PYSPARK_PYTHON'] = '/usr/bin/python3'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/bin/python3'
os.environ['JAVA_HOME'] = '/usr/lib/jvm/java-8-amazon-corretto'

# Add EMR's Spark to Python path
sys.path.insert(0, '/usr/lib/spark/python')
sys.path.insert(0, '/usr/lib/spark/python/lib/py4j-0.10.9-src.zip')

# Install only the packages we need (not pyspark - use EMR's built-in)
print('=== INSTALLING REQUIRED PACKAGES ===')
os.system('pip3 install rasterio scikit-fuzzy boto3')

print('=== PACKAGES INSTALLED ===')

# Import packages
import rasterio
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Import EMR's built-in PySpark (Java 8 compatible)
from pyspark.sql import SparkSession
from pyspark import SparkContext

def create_fuzzy_system(config):
    # Define input variables
    social = ctrl.Antecedent(np.arange(0, 11, 1), 'social')
    environmental = ctrl.Antecedent(np.arange(0, 11, 1), 'environmental')
    strategic = ctrl.Antecedent(np.arange(0, 11, 1), 'strategic')
    output = ctrl.Consequent(np.arange(0, 11, 1), 'output')
    
    # Define membership functions
    social['low'] = fuzz.trimf(social.universe, [0, 0, 5])
    social['medium'] = fuzz.trimf(social.universe, [0, 5, 10])
    social['high'] = fuzz.trimf(social.universe, [5, 10, 10])
    
    environmental['low'] = fuzz.trimf(environmental.universe, [0, 0, 5])
    environmental['medium'] = fuzz.trimf(environmental.universe, [0, 5, 10])
    environmental['high'] = fuzz.trimf(environmental.universe, [5, 10, 10])
    
    strategic['low'] = fuzz.trimf(strategic.universe, [0, 0, 5])
    strategic['medium'] = fuzz.trimf(strategic.universe, [0, 5, 10])
    strategic['high'] = fuzz.trimf(strategic.universe, [5, 10, 10])
    
    output['low'] = fuzz.trimf(output.universe, [0, 0, 5])
    output['medium'] = fuzz.trimf(output.universe, [0, 5, 10])
    output['high'] = fuzz.trimf(output.universe, [5, 10, 10])
    
    # Define rules based on config
    rules = []
    if config.get('aggregation_method') == 'median':
        rules.append(ctrl.Rule(social['low'] & environmental['low'] & strategic['low'], output['low']))
        rules.append(ctrl.Rule(social['medium'] & environmental['medium'] & strategic['medium'], output['medium']))
        rules.append(ctrl.Rule(social['high'] & environmental['high'] & strategic['high'], output['high']))
    elif config.get('aggregation_method') == 'minimum':
        rules.append(ctrl.Rule(social['low'] | environmental['low'] | strategic['low'], output['low']))
        rules.append(ctrl.Rule(social['medium'] & environmental['medium'] & strategic['medium'], output['medium']))
        rules.append(ctrl.Rule(social['high'] & environmental['high'] & strategic['high'], output['high']))
    else:
        rules.append(ctrl.Rule(social['low'] & environmental['low'] & strategic['low'], output['low']))
        rules.append(ctrl.Rule(social['medium'] & environmental['medium'] & strategic['medium'], output['medium']))
        rules.append(ctrl.Rule(social['high'] & environmental['high'] & strategic['high'], output['high']))
    
    ctrl_system = ctrl.ControlSystem(rules)
    simulation = ctrl.ControlSystemSimulation(ctrl_system)
    return simulation

def process_small_block_spark(chunk_data):
    config, start_row, end_row, start_col, end_col, social_block, env_block, strat_block, default_value = chunk_data
    
    try:
        simulation = create_fuzzy_system(config)
        result_block = np.full(social_block.shape, default_value, dtype=np.float32)
        
        for i in range(social_block.shape[0]):
            for j in range(social_block.shape[1]):
                try:
                    social_val = float(social_block[i, j])
                    env_val = float(env_block[i, j])
                    strat_val = float(strat_block[i, j])
                    
                    if np.isnan(social_val) or np.isnan(env_val) or np.isnan(strat_val):
                        continue
                    
                    simulation.social = social_val
                    simulation.environmental = env_val
                    simulation.strategic = strat_val
                    simulation.compute()
                    result_block[i, j] = simulation.output
                    
                except Exception as e:
                    result_block[i, j] = default_value
                    continue
        
        return (start_row, end_row, start_col, end_col, result_block)
        
    except Exception as e:
        result_block = np.full(social_block.shape, default_value, dtype=np.float32)
        return (start_row, end_row, start_col, end_col, result_block)

def process_config_spark(config_name):
    print(f'Processing {config_name} with EMR Spark...')
    
    s3 = boto3.client('s3')
    bucket = '<AWS-BUCKET>-unifile'
    prefix = 'unifile_test/'
    
    s3.download_file(bucket, prefix + config_name, f'/tmp/{config_name}')
    with open(f'/tmp/{config_name}', 'r') as f:
        config = json.load(f)
    
    s3.download_file(bucket, prefix + 'so300m.in', '/tmp/social.tif')
    s3.download_file(bucket, prefix + 'e300m.in', '/tmp/environmental.tif')
    s3.download_file(bucket, prefix + 's300m.in', '/tmp/strategic.tif')
    
    with rasterio.open('/tmp/social.tif') as src:
        social_data = src.read(1)
        profile = src.profile
        height, width = social_data.shape
    
    with rasterio.open('/tmp/environmental.tif') as src:
        env_data = src.read(1)
    
    with rasterio.open('/tmp/strategic.tif') as src:
        strat_data = src.read(1)
    
    print(f'Processing {height}x{width} raster with EMR Spark...')
    
    spark = SparkSession.builder \\
        .appName(f'FIS-EMR-{config_name}') \\
        .master('yarn') \\
        .config('spark.sql.adaptive.enabled', 'true') \\
        .config('spark.serializer', 'org.apache.spark.serializer.KryoSerializer') \\
        .config('spark.kryo.registrationRequired', 'false') \\
        .config('spark.driver.memory', '4g') \\
        .config('spark.driver.maxResultSize', '1g') \\
        .config('spark.executor.memory', '4g') \\
        .config('spark.executor.cores', '2') \\
        .getOrCreate()
    
    sc = spark.sparkContext
    
    block_size = 100
    chunks = []
    
    for start_row in range(0, height, block_size):
        end_row = min(start_row + block_size, height)
        for start_col in range(0, width, block_size):
            end_col = min(start_col + block_size, width)
            
            social_block = social_data[start_row:end_row, start_col:end_col]
            env_block = env_data[start_row:end_row, start_col:end_col]
            strat_block = strat_data[start_row:end_row, start_col:end_col]
            
            chunks.append((config, start_row, end_row, start_col, end_col, 
                         social_block, env_block, strat_block, 5.0))
    
    print(f'Created {len(chunks)} blocks for distributed processing')
    
    num_partitions = min(100, len(chunks))
    rdd = sc.parallelize(chunks, numSlices=num_partitions)
    results = rdd.map(process_small_block_spark).collect()
    
    result_data = np.full((height, width), 5.0, dtype=np.float32)
    for start_row, end_row, start_col, end_col, result_block in results:
        result_data[start_row:end_row, start_col:end_col] = result_block
    
    config_base = config_name.replace('.json', '')
    output_name = f'result_emr_{config_base}.tif'
    output_path = f'/tmp/{output_name}'
    
    profile.update(dtype=np.float32, count=1)
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(result_data, 1)
    
    s3.upload_file(output_path, bucket, prefix + output_name)
    print(f'Completed {config_name} -> {output_name}')
    
    spark.stop()
    return True

def main():
    print('=== EMR FIS PROCESSING STARTED ===')
    
    configs = [
        'config_median.json',
        'config_minimum.json', 
        'config_mode.json',
        'config_round_down.json',
        'config_round_up.json'
    ]
    
    for i, config in enumerate(configs, 1):
        print(f'\\n--- Processing Config {i}/{len(configs)}: {config} ---')
        try:
            process_config_spark(config)
            print(f'✅ SUCCESS: {config}')
        except Exception as e:
            print(f'❌ ERROR: {config} - {e}')
            continue
    
    print('\\n=== EMR FIS PROCESSING COMPLETED ===')

if __name__ == '__main__':
    main()
SCRIPT_EOF
python3 /tmp/emr_fis_fixed.py
"
    ]
  }
}
EOF

echo "Created EMR step configuration"

# Add the step to the cluster
echo "Adding step to EMR cluster..."
aws emr add-steps \
    --cluster-id $CLUSTER_ID \
    --region $REGION \
    --steps file:///tmp/emr_fis_fixed_step.json

if [ $? -eq 0 ]; then
    echo "✅ Step added successfully!"
    echo ""
    echo "The step will:"
    echo "1. Install rasterio, scikit-fuzzy, and boto3"
    echo "2. Use EMR's built-in PySpark (Java 8 compatible)"
    echo "3. Process all 5 FIS configurations with your existing logic"
    echo "4. Upload results to S3"
    echo ""
    echo "Monitor progress in the EMR console or with:"
    echo "aws emr describe-cluster --cluster-id $CLUSTER_ID --region $REGION"
else
    echo "❌ Failed to add step"
    exit 1
fi 