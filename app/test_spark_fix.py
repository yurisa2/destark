#!/usr/bin/env python3
"""
Test script to verify Spark session creation fix.
"""

import sys
import os

def test_spark_session_creation():
    """Test that Spark session can be created without JavaPackage errors."""
    try:
        print("Testing Spark session creation...")
        
        # Import the fixed function
        from raster_fuzzy_spark_simple import create_spark_session
        
        # Test local mode
        print("Creating Spark session in local mode...")
        spark_local = create_spark_session(
            app_name="TestLocal",
            local_mode=True
        )
        
        print(f"✓ Local Spark session created successfully!")
        print(f"  Spark version: {spark_local.version}")
        print(f"  Master URL: {spark_local.conf.get('spark.master')}")
        
        # Test basic operations
        print("Testing basic DataFrame operations...")
        data = [(1, "test1"), (2, "test2"), (3, "test3")]
        df = spark_local.createDataFrame(data, ["id", "value"])
        
        print(f"✓ DataFrame created with {df.count()} rows")
        print("DataFrame content:")
        df.show()
        
        # Test RDD operations
        print("Testing RDD operations...")
        rdd = spark_local.sparkContext.parallelize([1, 2, 3, 4, 5])
        result = rdd.map(lambda x: x * 2).collect()
        print(f"✓ RDD operation successful: {result}")
        
        spark_local.stop()
        print("✓ Local Spark session stopped successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Spark session creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_spark_cluster_connection():
    """Test connection to Spark cluster (if available)."""
    try:
        print("\nTesting Spark cluster connection...")
        
        from raster_fuzzy_spark_simple import create_spark_session
        
        # Test cluster mode
        print("Creating Spark session in cluster mode...")
        spark_cluster = create_spark_session(
            app_name="TestCluster",
            master_url="spark://spark-master:7077",
            local_mode=False
        )
        
        print(f"✓ Cluster Spark session created successfully!")
        print(f"  Spark version: {spark_cluster.version}")
        print(f"  Master URL: {spark_cluster.conf.get('spark.master')}")
        
        # Test basic operations
        print("Testing cluster DataFrame operations...")
        data = [(1, "cluster_test1"), (2, "cluster_test2")]
        df = spark_cluster.createDataFrame(data, ["id", "value"])
        
        print(f"✓ Cluster DataFrame created with {df.count()} rows")
        print("Cluster DataFrame content:")
        df.show()
        
        spark_cluster.stop()
        print("✓ Cluster Spark session stopped successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Spark cluster connection failed: {e}")
        print("This is expected if the Spark cluster is not running.")
        return False

def main():
    """Run all Spark tests."""
    print("=== Spark Session Creation Test ===")
    
    # Test local mode
    local_success = test_spark_session_creation()
    
    # Test cluster mode
    cluster_success = test_spark_cluster_connection()
    
    print("\n=== Test Results ===")
    print(f"Local mode: {'✓ PASSED' if local_success else '✗ FAILED'}")
    print(f"Cluster mode: {'✓ PASSED' if cluster_success else '✗ FAILED'}")
    
    if local_success:
        print("\n✓ Spark session creation fix is working!")
        print("The 'JavaPackage' object is not callable error should be resolved.")
    else:
        print("\n✗ Spark session creation still has issues.")
        print("Please check the error messages above.")

if __name__ == "__main__":
    main() 