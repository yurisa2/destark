#!/usr/bin/env python3
"""
Local testing script for AWS Glue-compatible raster fuzzy processing.
Simulates the AWS Glue environment for testing.
"""

import os
import sys
import subprocess
import tempfile
import shutil
from pathlib import Path

def setup_test_environment():
    """Set up test environment with sample data."""
    print("=== Setting up Local Glue Test Environment ===")
    
    # Create test directory
    test_dir = Path("test_glue_local")
    test_dir.mkdir(exist_ok=True)
    
    # Copy sample files
    sample_files = [
        "app/files/input/base/socioeconomico_1000m.tif",
        "app/files/input/base/ambiental_1000m.tif", 
        "app/files/input/base/estratégico_1000m.tif",
        "app/config/config_round_down.json"
    ]
    
    for file_path in sample_files:
        if os.path.exists(file_path):
            dest_path = test_dir / Path(file_path).name
            shutil.copy2(file_path, dest_path)
            print(f"✓ Copied {file_path} to {dest_path}")
        else:
            print(f"⚠ Sample file not found: {file_path}")
    
    return test_dir

def test_local_glue_script():
    """Test the Glue-compatible script locally."""
    print("\n=== Testing Glue Script Locally ===")
    
    test_dir = setup_test_environment()
    
    # Test with local files
    social_tiff = test_dir / "socioeconomico_1000m.tif"
    env_tiff = test_dir / "ambiental_1000m.tif"
    strat_tiff = test_dir / "estratégico_1000m.tif"
    config_file = test_dir / "config_round_down.json"
    output_tiff = test_dir / "output_glue_test.tif"
    
    if not all(f.exists() for f in [social_tiff, env_tiff, strat_tiff, config_file]):
        print("✗ Missing required test files")
        return False
    
    # Run the Glue script
    cmd = [
        sys.executable, "app/raster_fuzzy_glue.py",
        str(social_tiff),
        str(env_tiff), 
        str(strat_tiff),
        str(output_tiff),
        "--config", str(config_file),
        "--block-size", "500",  # Smaller blocks for testing
        "--partitions", "2"     # Fewer partitions for testing
    ]
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        print("=== STDOUT ===")
        print(result.stdout)
        
        if result.stderr:
            print("=== STDERR ===")
            print(result.stderr)
        
        if result.returncode == 0:
            print("✓ Local Glue test completed successfully!")
            if output_tiff.exists():
                size_mb = output_tiff.stat().st_size / (1024 * 1024)
                print(f"✓ Output file created: {output_tiff} ({size_mb:.1f} MB)")
            return True
        else:
            print(f"✗ Local Glue test failed with return code: {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ Local Glue test timed out")
        return False
    except Exception as e:
        print(f"✗ Local Glue test error: {e}")
        return False

def test_glue_environment_variables():
    """Test the Glue script with environment variables (AWS Glue mode)."""
    print("\n=== Testing Glue Script with Environment Variables ===")
    
    test_dir = setup_test_environment()
    
    # Set up environment variables like AWS Glue
    env_vars = {
        'SOCIAL_TIFF': str(test_dir / "socioeconomico_1000m.tif"),
        'ENVIRONMENTAL_TIFF': str(test_dir / "ambiental_1000m.tif"),
        'STRATEGIC_TIFF': str(test_dir / "estratégico_1000m.tif"),
        'OUTPUT_TIFF': str(test_dir / "output_glue_env_test.tif"),
        'CONFIG_FILE': str(test_dir / "config_round_down.json"),
        'AWS_EXECUTION_ENV': 'AWS_Glue'  # Simulate AWS Glue environment
    }
    
    # Run the Glue script without arguments (AWS Glue mode)
    cmd = [sys.executable, "app/raster_fuzzy_glue.py"]
    
    print(f"Running with environment variables: {env_vars}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env_vars)
        
        print("=== STDOUT ===")
        print(result.stdout)
        
        if result.stderr:
            print("=== STDERR ===")
            print(result.stderr)
        
        if result.returncode == 0:
            print("✓ Environment variables test completed successfully!")
            output_file = test_dir / "output_glue_env_test.tif"
            if output_file.exists():
                size_mb = output_file.stat().st_size / (1024 * 1024)
                print(f"✓ Output file created: {output_file} ({size_mb:.1f} MB)")
            return True
        else:
            print(f"✗ Environment variables test failed with return code: {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ Environment variables test timed out")
        return False
    except Exception as e:
        print(f"✗ Environment variables test error: {e}")
        return False

def create_glue_job_template():
    """Create a template for AWS Glue job parameters."""
    print("\n=== AWS Glue Job Template ===")
    
    template = """
# AWS Glue Job Parameters Template
# Copy these parameters when creating your Glue job

Job Parameters:
- SOCIAL_TIFF: s3://your-bucket/input/social.tif
- ENVIRONMENTAL_TIFF: s3://your-bucket/input/environmental.tif  
- STRATEGIC_TIFF: s3://your-bucket/input/strategic.tif
- OUTPUT_TIFF: s3://your-bucket/output/result.tif
- CONFIG_FILE: s3://your-bucket/config/raster_fis_config.json

Additional Parameters (optional):
- NODATA_VALUE: 5.0
- BLOCK_SIZE: 1000
- NUM_PARTITIONS: 8

# Python script to paste in Glue console:
"""
    
    # Read the Glue script
    with open("app/raster_fuzzy_glue.py", "r") as f:
        script_content = f.read()
    
    template += f"""
{script_content}

# Instructions:
# 1. Create a new Glue job
# 2. Set the job parameters above
# 3. Paste the script content into the job
# 4. Set the job type to "Spark"
# 5. Configure worker type and number based on your data size
# 6. Set timeout to at least 30 minutes for large datasets
"""
    
    # Save template
    template_file = "glue_job_template.txt"
    with open(template_file, "w") as f:
        f.write(template)
    
    print(f"✓ Glue job template saved to: {template_file}")
    print("\nTemplate content:")
    print(template)

def main():
    """Run all local tests."""
    print("=== AWS Glue Local Testing Suite ===")
    
    # Check if required files exist
    if not os.path.exists("app/raster_fuzzy_glue.py"):
        print("✗ Glue script not found: app/raster_fuzzy_glue.py")
        return False
    
    # Run tests
    test1_success = test_local_glue_script()
    test2_success = test_glue_environment_variables()
    
    # Create template
    create_glue_job_template()
    
    if test1_success and test2_success:
        print("\n✓ All tests passed! Your script is ready for AWS Glue deployment.")
        print("\nNext steps:")
        print("1. Upload your input files to S3")
        print("2. Upload your config file to S3") 
        print("3. Create a new AWS Glue job")
        print("4. Use the template parameters and script")
        print("5. Deploy and run!")
        return True
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 