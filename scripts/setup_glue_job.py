#!/usr/bin/env python3
"""
AWS Glue Job Setup Helper
Generates job configuration and parameters for AWS Glue deployment.
"""

import json
import os

def generate_glue_job_config():
    """Generate AWS Glue job configuration."""
    
    config = {
        "JobName": "RasterFuzzyProcessing",
        "Role": "AWSGlueServiceRole",
        "Command": {
            "Name": "glueetl",
            "ScriptLocation": "s3://your-bucket/scripts/raster_fuzzy_glue.py",
            "PythonVersion": "3"
        },
        "DefaultArguments": {
            "--job-language": "python",
            "--job-bookmark-option": "job-bookmark-disable"
        },
        "ExecutionProperty": {
            "MaxConcurrentRuns": 1
        },
        "MaxRetries": 0,
        "Timeout": 3600,
        "WorkerType": "G.2X",
        "NumberOfWorkers": 4,
        "GlueVersion": "4.0"
    }
    
    return config

def generate_job_parameters():
    """Generate job parameters for AWS Glue."""
    
    parameters = {
        "SOCIAL_TIFF": "s3://your-bucket/input/socioeconomico_1000m.tif",
        "ENVIRONMENTAL_TIFF": "s3://your-bucket/input/ambiental_1000m.tif",
        "STRATEGIC_TIFF": "s3://your-bucket/input/estratégico_1000m.tif",
        "OUTPUT_TIFF": "s3://your-bucket/output/result.tif",
        "CONFIG_FILE": "s3://your-bucket/config/config_round_down.json"
    }
    
    return parameters

def generate_python_libraries():
    """Generate Python library list."""
    
    libraries = [
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "rasterio>=1.3.0",
        "scikit-fuzzy>=0.4.2",
        "boto3>=1.26.0"
    ]
    
    return libraries

def main():
    """Generate all configuration files."""
    
    print("=== AWS Glue Job Configuration Generator ===\n")
    
    # Generate configurations
    job_config = generate_glue_job_config()
    job_params = generate_job_parameters()
    python_libs = generate_python_libraries()
    
    # Save job configuration
    with open("glue_job_config.json", "w") as f:
        json.dump(job_config, f, indent=2)
    
    # Save job parameters
    with open("glue_job_parameters.json", "w") as f:
        json.dump(job_params, f, indent=2)
    
    # Save Python libraries
    with open("glue_python_libraries.txt", "w") as f:
        f.write("\n".join(python_libs))
    
    print("✓ Generated configuration files:")
    print("  - glue_job_config.json")
    print("  - glue_job_parameters.json")
    print("  - glue_python_libraries.txt")
    
    print("\n=== AWS Glue Console Instructions ===\n")
    
    print("1. Go to AWS Glue Console")
    print("2. Click 'Jobs' → 'Add job'")
    print("3. Fill in job details:")
    print("   - Job name: RasterFuzzyProcessing")
    print("   - Job type: Spark")
    print("   - Glue version: 4.0")
    
    print("\n4. Add Job Parameters:")
    for key, value in job_params.items():
        print(f"   - {key}: {value}")
    
    print("\n5. Configure Job Settings:")
    print(f"   - Worker type: {job_config['WorkerType']}")
    print(f"   - Number of workers: {job_config['NumberOfWorkers']}")
    print(f"   - Job timeout: {job_config['Timeout']} seconds")
    
    print("\n6. Add Python Libraries:")
    for lib in python_libs:
        print(f"   - {lib}")
    
    print("\n7. Paste the script content from app/raster_fuzzy_glue.py")
    print("8. Save and run the job")
    
    print("\n=== S3 Upload Commands ===\n")
    
    print("# Upload input files")
    print("aws s3 cp app/files/input/base/socioeconomico_1000m.tif s3://your-bucket/input/")
    print("aws s3 cp app/files/input/base/ambiental_1000m.tif s3://your-bucket/input/")
    print("aws s3 cp app/files/input/base/estratégico_1000m.tif s3://your-bucket/input/")
    print("")
    print("# Upload config file")
    print("aws s3 cp app/config/config_round_down.json s3://your-bucket/config/")
    print("")
    print("# Upload script (optional)")
    print("aws s3 cp app/raster_fuzzy_glue.py s3://your-bucket/scripts/")

if __name__ == "__main__":
    main() 