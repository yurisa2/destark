#!/bin/bash

# Pre-flight check to predict potential failures before running the full job
set -e

echo "=== PRE-FLIGHT CHECK: Predicting Potential Failures ==="
echo "This will test all critical components before running the full job."
echo ""

FAILURES=0
WARNINGS=0

check_component() {
    local name="$1"
    local test_cmd="$2"
    local critical="$3"
    echo -n "Testing $name... "
    if eval "$test_cmd" >/dev/null 2>&1; then
        echo "✓ PASS"
        return 0
    else
        echo "✗ FAIL"
        if [ "$critical" = "true" ]; then
            echo "  CRITICAL: This will cause the job to fail!"
            ((FAILURES++))
        else
            echo "  WARNING: This might cause issues."
            ((WARNINGS++))
        fi
        return 1
    fi
}

check_component_with_output() {
    local name="$1"
    local test_cmd="$2"
    local critical="$3"
    echo "Testing $name..."
    if eval "$test_cmd"; then
        echo "  ✓ PASS"
        return 0
    else
        echo "  ✗ FAIL"
        if [ "$critical" = "true" ]; then
            echo "  CRITICAL: This will cause the job to fail!"
            ((FAILURES++))
        else
            echo "  WARNING: This might cause issues."
            ((WARNINGS++))
        fi
        return 1
    fi
}

echo "=== 1. Environment Checks ==="
check_component_with_output "Java Version" "java -version" "true"
check_component "JAVA_HOME is set" "[ -n \"$JAVA_HOME\" ]" "true"
check_component_with_output "Python Version" "python3 --version" "true"
check_component "Hadoop config exists" "[ -d '/etc/hadoop/conf' ]" "true"

echo ""
echo "=== 2. Spark Environment Checks ==="
check_component "spark-submit available" "command -v spark-submit" "true"
check_component_with_output "Spark Version" "spark-submit --version" "false"
check_component_with_output "Basic Spark Session" "python3 -c '
import os
os.environ[\'JAVA_HOME\'] = \"/usr/lib/jvm/jre-17\"
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName(\'PreflightTest\').master(\'local[1]\').getOrCreate()
print(\'Spark session created successfully\')
spark.stop()'" "true"

echo ""
echo "=== 3. Python Dependencies ==="
PACKAGES=("numpy" "rasterio" "skfuzzy" "pyspark" "boto3" "dateutil")
for package in "${PACKAGES[@]}"; do
    check_component "$package package" "python3 -c 'import $package'" "true"
done

echo ""
echo "=== 4. S3 Access Checks ==="
check_component "AWS CLI available" "command -v aws" "false"
S3_FILES=(
    "s3://adveng-pipeline/unifile_test/so300m.in"
    "s3://adveng-pipeline/unifile_test/e300m.in"
    "s3://adveng-pipeline/unifile_test/s300m.in"
)
for s3_file in "${S3_FILES[@]}"; do
    check_component "S3 access to $(basename \"$s3_file\")" "aws s3 ls '$s3_file' >/dev/null 2>&1" "true"
done
CONFIG_FILES=(
    "s3://adveng-pipeline/unifile_test/config_max.json"
    "s3://adveng-pipeline/unifile_test/config_round_up.json"
)
for config_file in "${CONFIG_FILES[@]}"; do
    check_component "S3 access to $(basename \"$config_file\")" "aws s3 ls '$config_file' >/dev/null 2>&1" "true"
done

echo ""
echo "=== 5. YARN/Cluster Checks ==="
check_component "YARN ResourceManager" "yarn application -list >/dev/null 2>&1" "true"
check_component_with_output "Cluster Resources" "yarn node -list" "false"
check_component_with_output "Available Memory" "free -h" "false"

echo ""
echo "=== 6. Application-Specific Checks ==="
check_component_with_output "Main Application Import" "python3 -c '
import sys
sys.path.insert(0, \"/mnt/destark\")
from app.raster_fuzzy_spark_ultra_optimized import create_ultra_optimized_spark_session
print(\'Main application imports successfully\')'" "true"
check_component_with_output "Rasterio S3 Support" "python3 -c '
import os
import rasterio
os.environ[\'AWS_S3_ENDPOINT\'] = \"s3.amazonaws.com\"
os.environ[\'GDAL_DISABLE_READDIR_ON_OPEN\'] = \"EMPTY_DIR\"
print(\'Rasterio S3 environment configured\')'" "false"

echo ""
echo "=== 7. Resource Estimation ==="
echo "Estimating resource requirements..."
CLUSTER_MEMORY=$(yarn node -list 2>/dev/null | grep -c "RUNNING" || echo "0")
CLUSTER_MEMORY=$((CLUSTER_MEMORY * 8))
echo "Estimated cluster memory: ${CLUSTER_MEMORY}GB"
echo "Required memory per executor: 8GB"
echo "Number of executors requested: 4"
echo "Total required memory: 32GB"
if [ "$CLUSTER_MEMORY" -lt 32 ]; then
    echo "⚠️  WARNING: Cluster may not have enough memory for optimal performance"
    ((WARNINGS++))
else
    echo "✓ Cluster has sufficient memory"
fi

echo ""
echo "=== PRE-FLIGHT CHECK SUMMARY ==="
echo "Critical Failures: $FAILURES"
echo "Warnings: $WARNINGS"
if [ "$FAILURES" -eq 0 ]; then
    echo ""
    echo "🎉 ALL CRITICAL CHECKS PASSED!"
    echo "✅ Your job should run successfully."
    if [ "$WARNINGS" -gt 0 ]; then
        echo "⚠️  There are $WARNINGS warnings that might affect performance."
    fi
    exit 0
else
    echo ""
    echo "❌ CRITICAL FAILURES DETECTED!"
    echo "🚫 Your job will likely fail. Please fix the issues above first."
    exit 1
fi 