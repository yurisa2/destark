#!/bin/bash

# Test Python Dependencies on EMR Cluster
echo "=== Testing Python Dependencies ==="

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
    exit 1
fi

cd "$CODE_DIR"

echo ""
echo "=== Testing Python Environment ==="
echo "Python version:"
python3 --version

echo ""
echo "=== Testing Required Packages ==="

# Test each required package
PACKAGES=("numpy" "rasterio" "skfuzzy" "pyspark" "boto3" "s3fs")

for package in "${PACKAGES[@]}"; do
    echo -n "Testing $package... "
    if python3 -c "import $package; print('✓ OK')" 2>/dev/null; then
        echo "✓ $package is available"
    else
        echo "✗ $package is MISSING"
    fi
done

echo ""
echo "=== Testing Basic S3 Access ==="
python3 -c "
import boto3
try:
    s3 = boto3.client('s3')
    print('✓ S3 client created successfully')
except Exception as e:
    print(f'✗ S3 client failed: {e}')
"

echo ""
echo "=== Testing Basic Rasterio ==="
python3 -c "
import rasterio
try:
    print(f'✓ Rasterio version: {rasterio.__version__}')
except Exception as e:
    print(f'✗ Rasterio test failed: {e}')
"

echo ""
echo "=== Testing Basic Spark ==="
python3 -c "
from pyspark.sql import SparkSession
try:
    spark = SparkSession.builder.appName('Test').master('local[1]').getOrCreate()
    print('✓ Spark session created successfully')
    spark.stop()
except Exception as e:
    print(f'✗ Spark test failed: {e}')
"

echo ""
echo "=== Testing Import of Main Application ==="
python3 -c "
import sys
sys.path.insert(0, '$CODE_DIR')
try:
    from app.raster_fuzzy_spark_ultra_optimized import create_ultra_optimized_spark_session
    print('✓ Main application imports successfully')
except Exception as e:
    print(f'✗ Main application import failed: {e}')
    import traceback
    traceback.print_exc()
"

echo ""
echo "=== Testing Config File Access ==="
CONFIG_FILES=(
    "app/config/config_max.json"
    "app/config/config_median.json"
    "app/config/config_minimum.json"
)

for config in "${CONFIG_FILES[@]}"; do
    if [ -f "$config" ]; then
        echo "✓ $config exists"
        # Test if it's valid JSON
        if python3 -c "import json; json.load(open('$config')); print('  ✓ Valid JSON')" 2>/dev/null; then
            echo "  ✓ Valid JSON"
        else
            echo "  ✗ Invalid JSON"
        fi
    else
        echo "✗ $config missing"
    fi
done

echo ""
echo "=== Testing S3 File Access ==="
S3_FILES=(
    "s3://adveng-pipeline/unifile_test/so300m.in"
    "s3://adveng-pipeline/unifile_test/e300m.in"
    "s3://adveng-pipeline/unifile_test/s300m.in"
)

for s3_file in "${S3_FILES[@]}"; do
    echo -n "Testing $s3_file... "
    if aws s3 ls "$s3_file" >/dev/null 2>&1; then
        echo "✓ Accessible"
    else
        echo "✗ Not accessible"
    fi
done

echo ""
echo "=== Test Complete ===" 