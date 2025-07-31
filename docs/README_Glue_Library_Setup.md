# AWS Glue Python Library Setup Guide

## Problem
Your AWS Glue job failed with:
```
ModuleNotFoundError: No module named 'rasterio'
```

This happens because AWS Glue doesn't include all Python libraries by default.

## Solution Options

### Option 1: Use Glue's Built-in Python Library Path (Recommended)

1. **Go to your AWS Glue Job**
   - Navigate to AWS Glue Console
   - Find your job "Test Adveng" or "adveng_full_pipeline"
   - Click "Edit job"

2. **Add Python Libraries**
   - In the job configuration, find "Python library path"
   - Add these libraries (one per line):
   ```
   numpy
   scipy
   rasterio
   scikit-fuzzy
   boto3
   ```

3. **Save and Run**
   - Save the job configuration
   - Run the job again

### Option 2: Use Additional Python Libraries (S3 Paths)

If Option 1 doesn't work, you can specify S3 paths to wheel files:

1. **Upload Libraries to S3**
   ```bash
   # Create a directory for libraries
   aws s3 mb s3://aws-glue-assets-<AWS-ACCOUNT-ID>-us-east-1/python-libs
   
   # Download and upload wheel files
   pip download --platform manylinux2014_x86_64 --only-binary=all \
       numpy>=1.21.0 scipy>=1.7.0 rasterio>=1.3.0 scikit-fuzzy>=0.4.2 boto3>=1.26.0
   
   # Upload to S3
   aws s3 cp *.whl s3://aws-glue-assets-<AWS-ACCOUNT-ID>-us-east-1/python-libs/
   ```

2. **Add to Glue Job**
   - In "Additional Python libraries" section, add S3 paths:
   ```
   s3://aws-glue-assets-<AWS-ACCOUNT-ID>-us-east-1/python-libs/numpy-*.whl
s3://aws-glue-assets-<AWS-ACCOUNT-ID>-us-east-1/python-libs/scipy-*.whl
s3://aws-glue-assets-<AWS-ACCOUNT-ID>-us-east-1/python-libs/rasterio-*.whl
s3://aws-glue-assets-<AWS-ACCOUNT-ID>-us-east-1/python-libs/scikit_fuzzy-*.whl
s3://aws-glue-assets-<AWS-ACCOUNT-ID>-us-east-1/python-libs/boto3-*.whl
   ```

### Option 3: Use Glue Version with More Libraries

1. **Change Glue Version**
   - Edit your job
   - Change "Glue version" to "4.0" (if not already)
   - This version includes more pre-installed libraries

2. **Check Available Libraries**
   - Glue 4.0 includes `numpy`, `scipy`, `boto3` by default
   - You may only need to add `rasterio` and `scikit-fuzzy`

## Quick Fix Steps

1. **Edit your Glue job**
2. **Add to "Python library path":**
   ```
   rasterio
   scikit-fuzzy
   ```
3. **Save and run**

## Verification

After adding libraries, your job should start without the `ModuleNotFoundError`. The script will then proceed to download files from S3 and process them.

## Troubleshooting

- **Still getting errors?** Try adding `fiona` and `shapely` to the library path as well
- **Permission issues?** Make sure your Glue job has S3 read/write permissions
- **Timeout issues?** Increase the job timeout to 3600 seconds

## Alternative: Use Glue Studio

If console configuration is problematic, you can also:
1. Use AWS Glue Studio (visual editor)
2. Create a new job with the same script
3. Configure libraries through the visual interface 