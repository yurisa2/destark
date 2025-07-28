#!/usr/bin/env python3
"""
Test script to verify Spark connectivity and diagnose issues.
"""

import os
import sys
import time

def test_spark_connection():
    """Test Spark connection with detailed diagnostics."""
    print("=== Spark Connection Test ===")
    
    # Check environment variables
    print("\n1. Environment Variables:")
    env_vars = ['JAVA_HOME', 'SPARK_HOME', 'PYSPARK_PYTHON', 'PYSPARK_DRIVER_PYTHON']
    for var in env_vars:
        value = os.environ.get(var, 'NOT SET')
        print(f"  {var}: {value}")
    
    # Check Python path
    print(f"\n2. Python Path: {sys.executable}")
    
    # Check if findspark is available
    try:
        import findspark
        print("✓ findspark imported successfully")
    except ImportError as e:
        print(f"✗ findspark import failed: {e}")
        return False
    
    # Initialize findspark
    try:
        findspark.init()
        print("✓ findspark initialized successfully")
    except Exception as e:
        print(f"✗ findspark initialization failed: {e}")
        return False
    
    # Test local mode first
    print("\n3. Testing Local Mode:")
    try:
        from pyspark.sql import SparkSession
        
        spark_local = SparkSession.builder \
            .appName("ConnectionTest") \
            .master("local[2]") \
            .config("spark.driver.memory", "2g") \
            .config("spark.executor.memory", "2g") \
            .getOrCreate()
        
        print(f"✓ Local Spark session created successfully!")
        print(f"  Spark version: {spark_local.version}")
        try:
            executors = spark_local.sparkContext.getExecutorMemoryStatus()
            print(f"  Available executors: {len(executors)}")
        except AttributeError:
            print(f"  Available executors: Local mode (no executors) - getExecutorMemoryStatus not available in this Spark version")
        
        spark_local.stop()
        print("✓ Local Spark session stopped successfully")
        
    except Exception as e:
        print(f"✗ Local Spark session failed: {e}")
        return False
    
    # Test cluster mode
    print("\n4. Testing Cluster Mode:")
    try:
        spark_cluster = SparkSession.builder \
            .appName("ConnectionTest") \
            .master("spark://spark-master:7077") \
            .config("spark.driver.memory", "2g") \
            .config("spark.executor.memory", "2g") \
            .config("spark.driver.host", "jupyter-spark") \
            .config("spark.driver.bindAddress", "0.0.0.0") \
            .config("spark.driver.port", "0") \
            .config("spark.driver.blockManager.port", "0") \
            .getOrCreate()
        
        print(f"✓ Cluster Spark session created successfully!")
        print(f"  Spark version: {spark_cluster.version}")
        
        # Test executor connectivity
        try:
            executors = spark_cluster.sparkContext.getExecutorMemoryStatus()
            print(f"  Available executors: {len(executors)}")
            for executor_id, memory_info in executors.items():
                print(f"    {executor_id}: {memory_info}")
        except AttributeError:
            # Alternative way to check executors for Spark versions without getExecutorMemoryStatus
            try:
                print(f"  Available executors: getExecutorMemoryStatus not available in this Spark version")
                # Create a simple RDD to trigger executor allocation and test connectivity
                test_rdd = spark_cluster.sparkContext.parallelize([1, 2, 3, 4, 5])
                count = test_rdd.count()
                print(f"  Test RDD count: {count}")
                print(f"  Cluster mode active - executors should be available")
            except Exception as e:
                print(f"  Cluster connectivity issue: {e}")
        
        spark_cluster.stop()
        print("✓ Cluster Spark session stopped successfully")
        
    except Exception as e:
        print(f"✗ Cluster Spark session failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n✓ All Spark connection tests passed!")
    return True

if __name__ == "__main__":
    success = test_spark_connection()
    sys.exit(0 if success else 1) 