#!/bin/bash

# Test script to check file formats and rasterio compatibility

echo "=== Testing File Formats ==="

# Test with gdalinfo to see what format the files are
echo "=== Testing with gdalinfo ==="
echo "Testing social file:"
gdalinfo "s3://adveng-pipeline/unifile_test/so300m.in" 2>&1 || echo "❌ gdalinfo failed"

echo ""
echo "=== Testing with rasterio ==="
python3 -c "
import rasterio
import os

# Set S3 configuration
os.environ['AWS_S3_ENDPOINT'] = 's3.amazonaws.com'
os.environ['GDAL_DISABLE_READDIR_ON_OPEN'] = 'EMPTY_DIR'

try:
    with rasterio.open('s3://adveng-pipeline/unifile_test/so300m.in') as src:
        print(f'✅ Rasterio can open: {src.name}')
        print(f'   Driver: {src.driver}')
        print(f'   Shape: {src.shape}')
        print(f'   CRS: {src.crs}')
        print(f'   Dtype: {src.dtypes[0]}')
        print(f'   Bounds: {src.bounds}')
except Exception as e:
    print(f'❌ Rasterio failed: {e}')
    print(f'   Error type: {type(e).__name__}')
"

echo ""
echo "=== Testing with explicit driver ==="
python3 -c "
import rasterio
import os

# Set S3 configuration
os.environ['AWS_S3_ENDPOINT'] = 's3.amazonaws.com'
os.environ['GDAL_DISABLE_READDIR_ON_OPEN'] = 'EMPTY_DIR'

# Try different drivers
drivers = ['GTiff', 'HFA', 'ENVI', 'EHdr', 'ISIS3', 'VRT']

for driver in drivers:
    try:
        with rasterio.open('s3://adveng-pipeline/unifile_test/so300m.in', driver=driver) as src:
            print(f'✅ Driver {driver} works!')
            print(f'   Shape: {src.shape}')
            break
    except Exception as e:
        print(f'❌ Driver {driver} failed: {str(e)[:100]}...')
"

echo ""
echo "=== Testing file download and inspection ==="
python3 -c "
import boto3
import tempfile
import os
import subprocess

# Download the file locally
s3_client = boto3.client('s3')
with tempfile.NamedTemporaryFile(suffix='.in', delete=False) as tmp_file:
    temp_path = tmp_file.name

try:
    s3_client.download_file('adveng-pipeline', 'unifile_test/so300m.in', temp_path)
    print(f'✅ Downloaded to: {temp_path}')
    
    # Check file size
    size = os.path.getsize(temp_path)
    print(f'   File size: {size} bytes')
    
    # Check first few bytes
    with open(temp_path, 'rb') as f:
        header = f.read(100)
    print(f'   Header (hex): {header.hex()[:50]}...')
    
    # Try gdalinfo locally
    try:
        result = subprocess.run(['gdalinfo', temp_path], capture_output=True, text=True)
        if result.returncode == 0:
            print('✅ gdalinfo works locally')
            print(f'   Format: {result.stdout[:200]}...')
        else:
            print(f'❌ gdalinfo failed: {result.stderr}')
    except Exception as e:
        print(f'❌ gdalinfo error: {e}')
        
finally:
    if os.path.exists(temp_path):
        os.unlink(temp_path)
" 