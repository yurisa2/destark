#!/bin/bash

# Test Python paths on EMR cluster
echo "=== Testing Python Paths on EMR Cluster ==="

# Test common Python paths
PYTHON_PATHS=(
    "/usr/bin/python3"
    "/usr/bin/python"
    "/opt/conda/bin/python"
    "/opt/conda/bin/python3"
    "/usr/local/bin/python3"
    "/usr/local/bin/python"
    "/opt/bitnami/python/bin/python"
    "/opt/bitnami/python/bin/python3"
)

echo "Testing Python paths:"
for path in "${PYTHON_PATHS[@]}"; do
    if [ -f "$path" ]; then
        echo "✓ $path exists"
        if "$path" --version 2>/dev/null; then
            echo "  ✓ $path works"
        else
            echo "  ✗ $path doesn't work"
        fi
    else
        echo "✗ $path doesn't exist"
    fi
done

echo ""
echo "=== Testing spark-submit with simple Python job ==="

# Test spark-submit with a simple Python job
echo "Testing spark-submit with /usr/bin/python3..."

spark-submit \
    --master yarn \
    --deploy-mode cluster \
    --conf spark.pyspark.python=/usr/bin/python3 \
    --conf spark.pyspark.driver.python=/usr/bin/python3 \
    --conf spark.executor.memory=2g \
    --conf spark.driver.memory=2g \
    --conf spark.yarn.maxAppAttempts=1 \
    --conf spark.yarn.submit.waitAppCompletion=true \
    --py-files /dev/null \
    - << 'EOF'
import sys
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print("✓ Python job completed successfully!")
EOF

echo "Test completed!" 