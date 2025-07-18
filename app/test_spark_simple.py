#!/usr/bin/env python3
"""
Simple test script to verify Spark is working.
"""

import sys
import os

def test_spark_basic():
    """Test basic Spark functionality."""
    try:
        from pyspark.sql import SparkSession
        
        print("Creating Spark session...")
        spark = SparkSession.builder \
            .appName("TestSpark") \
            .master("local[2]") \
            .config("spark.driver.memory", "512m") \
            .config("spark.executor.memory", "512m") \
            .getOrCreate()
        
        print("Spark session created successfully!")
        
        # Test basic DataFrame operations
        print("Testing DataFrame operations...")
        data = [(1, "a"), (2, "b"), (3, "c")]
        df = spark.createDataFrame(data, ["id", "value"])
        
        print(f"DataFrame count: {df.count()}")
        print("DataFrame content:")
        df.show()
        
        # Test RDD operations
        print("Testing RDD operations...")
        rdd = spark.sparkContext.parallelize([1, 2, 3, 4, 5])
        result = rdd.map(lambda x: x * 2).collect()
        print(f"RDD result: {result}")
        
        print("All Spark tests passed!")
        spark.stop()
        return True
        
    except Exception as e:
        print(f"Spark test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_rasterio():
    """Test rasterio functionality."""
    try:
        import rasterio
        print("Rasterio imported successfully!")
        
        # Test reading a small raster
        test_file = "/home/jovyan/app/files/input/base/socioeconomico_300m.tif"
        if os.path.exists(test_file):
            with rasterio.open(test_file) as src:
                print(f"Raster shape: {src.shape}")
                print(f"Raster dtype: {src.dtypes}")
                print(f"Raster nodata: {src.nodata}")
            print("Rasterio test passed!")
            return True
        else:
            print(f"Test file not found: {test_file}")
            return False
            
    except Exception as e:
        print(f"Rasterio test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_skfuzzy():
    """Test skfuzzy functionality."""
    try:
        import skfuzzy as fuzz
        from skfuzzy import control as ctrl
        import numpy as np
        
        print("Skfuzzy imported successfully!")
        
        # Test basic fuzzy system creation
        x = np.linspace(0, 10, 11)
        var = ctrl.Antecedent(x, 'var')
        var['low'] = fuzz.trimf(var.universe, [0, 0, 5])
        var['high'] = fuzz.trimf(var.universe, [5, 10, 10])
        
        output = ctrl.Consequent(x, 'output')
        output['low'] = fuzz.trimf(output.universe, [0, 0, 5])
        output['high'] = fuzz.trimf(output.universe, [5, 10, 10])
        
        rule = ctrl.Rule(var['low'], output['low'])
        control_system = ctrl.ControlSystem([rule])
        simulation = ctrl.ControlSystemSimulation(control_system)
        
        simulation.input['var'] = 2.0
        simulation.compute()
        result = simulation.output['output']
        
        print(f"Fuzzy system test result: {result}")
        print("Skfuzzy test passed!")
        return True
        
    except Exception as e:
        print(f"Skfuzzy test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=== Spark Test Suite ===")
    
    tests = [
        ("Spark Basic", test_spark_basic),
        ("Rasterio", test_rasterio),
        ("Skfuzzy", test_skfuzzy),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n--- Testing {test_name} ---")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"Test {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n=== Test Results ===")
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
    
    all_passed = all(result for _, result in results)
    if all_passed:
        print("\nAll tests passed! Spark environment is working correctly.")
        return 0
    else:
        print("\nSome tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 