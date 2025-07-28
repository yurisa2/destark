#!/bin/bash

# Simple test for spark-submit
echo "=== Testing Spark Submit with Simple Python Code ==="

# Create a simple Python script
cat > /tmp/test_spark.py << 'EOF'
#!/usr/bin/env python3
import sys
import os

print("=== Simple Spark Test ===")
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print(f"Current working directory: {os.getcwd()}")
print(f"Environment variables:")
for key, value in os.environ.items():
    if 'SPARK' in key or 'HADOOP' in key or 'YARN' in key:
        print(f"  {key}: {value}")

try:
    from pyspark.sql import SparkSession
    print("✓ PySpark import successful")
    
    # Create a simple Spark session
    spark = SparkSession.builder.appName("SimpleTest").getOrCreate()
    print("✓ Spark session created successfully")
    
    # Test basic Spark functionality
    data = [("test", 1), ("data", 2)]
    df = spark.createDataFrame(data, ["word", "count"])
    result = df.collect()
    print(f"✓ Spark DataFrame test successful: {result}")
    
    spark.stop()
    print("✓ Spark session stopped successfully")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("=== Test completed successfully! ===")
EOF

# Run the test with spark-submit
echo "Running spark-submit test..."
spark-submit \
    --master yarn \
    --deploy-mode cluster \
    --conf spark.pyspark.python=/usr/bin/python3 \
    --conf spark.pyspark.driver.python=/usr/bin/python3 \
    --conf spark.executor.memory=2g \
    --conf spark.driver.memory=2g \
    --conf spark.yarn.maxAppAttempts=1 \
    --conf spark.yarn.submit.waitAppCompletion=true \
    /tmp/test_spark.py

echo "Test completed!" 