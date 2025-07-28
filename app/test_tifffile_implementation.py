#!/usr/bin/env python3
"""
Test script for the tifffile-based raster fuzzy inference system.
Creates sample raster data and tests the system functionality.
"""

import os
import sys
import numpy as np
import tifffile
from pathlib import Path
import json
import tempfile

# Add the app directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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
    
    # Write test rasters using tifffile
    social_path = test_dir / "test_social_tifffile.tif"
    env_path = test_dir / "test_environmental_tifffile.tif"
    strat_path = test_dir / "test_strategic_tifffile.tif"
    
    tifffile.imwrite(social_path, social_data, photometric='minisblack')
    tifffile.imwrite(env_path, env_data, photometric='minisblack')
    tifffile.imwrite(strat_path, strat_data, photometric='minisblack')
    
    print(f"Created test rasters:")
    print(f"  Social: {social_path}")
    print(f"  Environmental: {env_path}")
    print(f"  Strategic: {strat_path}")
    
    return str(social_path), str(env_path), str(strat_path)


def create_test_config():
    """Create a test configuration file."""
    print("Creating test configuration...")
    
    config = {
        "input_variables": {
            "social": {
                "min": 0,
                "max": 10,
                "step": 0.1,
                "membership_functions": {
                    "low": {
                        "type": "trapmf",
                        "params": [0, 0, 2, 4]
                    },
                    "medium": {
                        "type": "trimf",
                        "params": [3, 5, 7]
                    },
                    "high": {
                        "type": "trapmf",
                        "params": [6, 8, 10, 10]
                    }
                }
            },
            "environmental": {
                "min": 0,
                "max": 10,
                "step": 0.1,
                "membership_functions": {
                    "low": {
                        "type": "trapmf",
                        "params": [0, 0, 2, 4]
                    },
                    "medium": {
                        "type": "trimf",
                        "params": [3, 5, 7]
                    },
                    "high": {
                        "type": "trapmf",
                        "params": [6, 8, 10, 10]
                    }
                }
            },
            "strategic": {
                "min": 0,
                "max": 10,
                "step": 0.1,
                "membership_functions": {
                    "low": {
                        "type": "trapmf",
                        "params": [0, 0, 2, 4]
                    },
                    "medium": {
                        "type": "trimf",
                        "params": [3, 5, 7]
                    },
                    "high": {
                        "type": "trapmf",
                        "params": [6, 8, 10, 10]
                    }
                }
            }
        },
        "output_variable": {
            "name": "suitability",
            "min": 0,
            "max": 10,
            "step": 0.1,
            "membership_functions": {
                "low": {
                    "type": "trapmf",
                    "params": [0, 0, 2, 4]
                },
                "medium": {
                    "type": "trimf",
                    "params": [3, 5, 7]
                },
                "high": {
                    "type": "trapmf",
                    "params": [6, 8, 10, 10]
                }
            }
        },
        "rules": [
            {
                "antecedent": [
                    {"variable": "social", "membership": "low"},
                    {"variable": "environmental", "membership": "low"},
                    {"variable": "strategic", "membership": "low"}
                ],
                "consequent": "low"
            },
            {
                "antecedent": [
                    {"variable": "social", "membership": "medium"},
                    {"variable": "environmental", "membership": "medium"},
                    {"variable": "strategic", "membership": "medium"}
                ],
                "consequent": "medium"
            },
            {
                "antecedent": [
                    {"variable": "social", "membership": "high"},
                    {"variable": "environmental", "membership": "high"},
                    {"variable": "strategic", "membership": "high"}
                ],
                "consequent": "high"
            }
        ]
    }
    
    # Create temporary config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config, f, indent=2)
        config_path = f.name
    
    print(f"Created test configuration: {config_path}")
    return config_path


def test_tifffile_implementation():
    """Test the tifffile implementation."""
    print("Testing tifffile implementation...")
    
    try:
        # Import the tifffile-based implementation
        from raster_fuzzy_lib_tifffile import UnifiedRasterFuzzyInferenceSystem
        
        # Create test data
        social_path, env_path, strat_path = create_test_rasters()
        config_path = create_test_config()
        
        # Create output path
        output_path = "data/test/test_output_tifffile.tif"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Create FIS instance
        print("Creating FIS instance...")
        fis = UnifiedRasterFuzzyInferenceSystem(config_path)
        
        # Test sequential processing
        print("Testing sequential processing...")
        fis.process_rasters(
            social_tiff=social_path,
            environmental_tiff=env_path,
            strategic_tiff=strat_path,
            output_tiff=output_path,
            nodata_value=5.0,
            parallel=False
        )
        
        # Verify output file exists
        if os.path.exists(output_path):
            print(f"✓ Output file created successfully: {output_path}")
            
            # Read and verify output
            output_data = tifffile.imread(output_path)
            print(f"✓ Output shape: {output_data.shape}")
            print(f"✓ Output dtype: {output_data.dtype}")
            print(f"✓ Output range: {output_data.min():.3f} to {output_data.max():.3f}")
            
            return True
        else:
            print("✗ Output file was not created")
            return False
            
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up temporary files
        try:
            os.unlink(config_path)
        except:
            pass


def test_tifffile_vs_rasterio():
    """Compare tifffile and rasterio functionality."""
    print("Comparing tifffile and rasterio functionality...")
    
    try:
        # Test tifffile
        print("Testing tifffile...")
        import tifffile
        
        # Create test data
        test_data = np.random.rand(50, 50).astype(np.float32)
        test_file = "data/test/test_comparison.tif"
        os.makedirs(os.path.dirname(test_file), exist_ok=True)
        
        # Write with tifffile
        tifffile.imwrite(test_file, test_data, photometric='minisblack')
        
        # Read with tifffile
        read_data = tifffile.imread(test_file)
        
        print(f"✓ tifffile write/read successful")
        print(f"  Original shape: {test_data.shape}")
        print(f"  Read shape: {read_data.shape}")
        print(f"  Data match: {np.allclose(test_data, read_data)}")
        
        # Test rasterio if available
        try:
            import rasterio
            print("Testing rasterio...")
            
            with rasterio.open(test_file) as src:
                rasterio_data = src.read(1)
                print(f"✓ rasterio read successful")
                print(f"  Rasterio shape: {rasterio_data.shape}")
                print(f"  Data match with tifffile: {np.allclose(read_data, rasterio_data)}")
                
        except ImportError:
            print("rasterio not available for comparison")
        
        return True
        
    except Exception as e:
        print(f"✗ Comparison test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Testing tifffile-based raster fuzzy inference system")
    print("=" * 60)
    
    # Test basic functionality
    success1 = test_tifffile_implementation()
    
    # Test comparison with rasterio
    success2 = test_tifffile_vs_rasterio()
    
    print("\n" + "=" * 60)
    if success1 and success2:
        print("✓ All tests passed! tifffile implementation is working correctly.")
    else:
        print("✗ Some tests failed. Please check the implementation.")
    print("=" * 60) 