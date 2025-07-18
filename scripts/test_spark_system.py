#!/usr/bin/env python3
"""
Test script for the Spark-based raster fuzzy inference system.
Generates sample raster data and tests the system in both local and cluster modes.
"""

import os
import sys
import numpy as np
import rasterio
from pathlib import Path
import subprocess
import time

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def create_test_rasters():
    """Create small test raster files for testing."""
    print("Creating test raster files...")
    
    # Create test directory
    test_dir = Path("data/test")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Create small test rasters (100x100 pixels)
    size = (100, 100)
    
    # Social raster: values 0-10
    social_data = np.random.uniform(0, 10, size).astype(np.float32)
    
    # Environmental raster: values 0-10
    env_data = np.random.uniform(0, 10, size).astype(np.float32)
    
    # Strategic raster: values 0-10
    strat_data = np.random.uniform(0, 10, size).astype(np.float32)
    
    # Profile for writing rasters
    profile = {
        'driver': 'GTiff',
        'height': size[0],
        'width': size[1],
        'count': 1,
        'dtype': 'float32',
        'crs': 'EPSG:4326',
        'transform': rasterio.transform.from_bounds(0, 0, 1, 1, size[1], size[0]),
        'nodata': 5.0
    }
    
    # Write test rasters
    social_path = test_dir / "test_social.tif"
    env_path = test_dir / "test_environmental.tif"
    strat_path = test_dir / "test_strategic.tif"
    
    with rasterio.open(social_path, 'w', **profile) as dst:
        dst.write(social_data, 1)
    
    with rasterio.open(env_path, 'w', **profile) as dst:
        dst.write(env_data, 1)
    
    with rasterio.open(strat_path, 'w', **profile) as dst:
        dst.write(strat_data, 1)
    
    print(f"Created test rasters:")
    print(f"  Social: {social_path}")
    print(f"  Environmental: {env_path}")
    print(f"  Strategic: {strat_path}")
    
    return str(social_path), str(env_path), str(strat_path)

def test_local_mode():
    """Test the Spark system in local mode."""
    print("\n=== Testing Local Mode ===")
    
    # Create test rasters
    social_tiff, env_tiff, strat_tiff = create_test_rasters()
    output_tiff = "data/test/output_local.tif"
    
    # Create config if it doesn't exist
    config_path = "app/config/test_config.json"
    if not os.path.exists(config_path):
        from app.raster_fuzzy_lib import create_raster_config_template
        import json
        
        config = create_raster_config_template()
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Created test configuration: {config_path}")
    
    # Run local mode test
    cmd = [
        "python", "app/raster_fuzzy_spark.py",
        social_tiff, env_tiff, strat_tiff, output_tiff,
        "--config", config_path,
        "--local",
        "--chunk-size", "100",
        "--verbose"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✓ Local mode test passed!")
            print(f"Processing time: {time.time() - start_time:.2f} seconds")
            print(f"Output: {output_tiff}")
            
            # Verify output exists
            if os.path.exists(output_tiff):
                with rasterio.open(output_tiff) as src:
                    data = src.read(1)
                    print(f"Output value range: {np.nanmin(data):.2f} to {np.nanmax(data):.2f}")
                return True
            else:
                print("✗ Output file not found")
                return False
        else:
            print("✗ Local mode test failed!")
            print(f"Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ Local mode test timed out!")
        return False
    except Exception as e:
        print(f"✗ Local mode test failed with exception: {e}")
        return False

def test_cluster_mode():
    """Test the Spark system in cluster mode."""
    print("\n=== Testing Cluster Mode ===")
    
    # Create test rasters
    social_tiff, env_tiff, strat_tiff = create_test_rasters()
    output_tiff = "data/test/output_cluster.tif"
    
    # Create config if it doesn't exist
    config_path = "app/config/test_config.json"
    if not os.path.exists(config_path):
        from app.raster_fuzzy_lib import create_raster_config_template
        import json
        
        config = create_raster_config_template()
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Created test configuration: {config_path}")
    
    # Run cluster mode test
    cmd = [
        "python", "app/raster_fuzzy_spark.py",
        social_tiff, env_tiff, strat_tiff, output_tiff,
        "--config", config_path,
        "--chunk-size", "100",
        "--partitions", "4",
        "--verbose"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            print("✓ Cluster mode test passed!")
            print(f"Processing time: {time.time() - start_time:.2f} seconds")
            print(f"Output: {output_tiff}")
            
            # Verify output exists
            if os.path.exists(output_tiff):
                with rasterio.open(output_tiff) as src:
                    data = src.read(1)
                    print(f"Output value range: {np.nanmin(data):.2f} to {np.nanmax(data):.2f}")
                return True
            else:
                print("✗ Output file not found")
                return False
        else:
            print("✗ Cluster mode test failed!")
            print(f"Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ Cluster mode test timed out!")
        return False
    except Exception as e:
        print(f"✗ Cluster mode test failed with exception: {e}")
        return False

def check_spark_cluster():
    """Check if Spark cluster is running."""
    print("Checking Spark cluster status...")
    
    try:
        # Check if containers are running
        result = subprocess.run(["docker", "compose", "ps"], capture_output=True, text=True)
        
        if result.returncode == 0:
            if "Up" in result.stdout:
                print("✓ Spark cluster is running")
                return True
            else:
                print("✗ Spark cluster is not running")
                return False
        else:
            print("✗ Could not check cluster status")
            return False
            
    except Exception as e:
        print(f"✗ Error checking cluster status: {e}")
        return False

def main():
    """Run all tests."""
    print("=== Testing Spark-based Raster Fuzzy Inference System ===\n")
    
    # Check if Spark cluster is running
    if not check_spark_cluster():
        print("Please start the Spark cluster first:")
        print("sh scripts/setup_spark.sh")
        return False
    
    # Run tests
    tests = [
        test_local_mode,
        test_cluster_mode
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"Test failed with exception: {e}")
        print()
    
    # Summary
    print("=== Test Summary ===")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed!")
        print("\n=== System is ready for use ===")
        print("You can now use the Spark-based system for large-scale raster processing.")
        return True
    else:
        print("✗ Some tests failed")
        print("Check the logs above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 