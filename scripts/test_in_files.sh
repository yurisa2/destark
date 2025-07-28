#!/bin/bash

# Test script to verify .in files can be opened as TIFF

echo "=== Testing .in files as TIFF ==="

python3 -c "
import rasterio
import os

# Set S3 configuration
os.environ['AWS_S3_ENDPOINT'] = 's3.amazonaws.com'
os.environ['GDAL_DISABLE_READDIR_ON_OPEN'] = 'EMPTY_DIR'

# Test opening .in file with explicit GTiff driver
try:
    with rasterio.open('s3://adveng-pipeline/unifile_test/so300m.in', driver='GTiff') as src:
        print(f'✅ Successfully opened .in file as TIFF!')
        print(f'   Driver: {src.driver}')
        print(f'   Shape: {src.shape}')
        print(f'   CRS: {src.crs}')
        print(f'   Dtype: {src.dtypes[0]}')
        print(f'   Bounds: {src.bounds}')
        
        # Test reading a small sample
        sample = src.read(1, window=((0, 10), (0, 10)))
        print(f'   Sample data shape: {sample.shape}')
        print(f'   Sample data type: {sample.dtype}')
        
except Exception as e:
    print(f'❌ Failed to open .in file as TIFF: {e}')
" 