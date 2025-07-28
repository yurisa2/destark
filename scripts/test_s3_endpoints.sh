#!/bin/bash

# Test script to verify S3 endpoint fixes

echo "=== Testing S3 Endpoint Fixes ==="

# Test the sed replacement
ORIGINAL="s3://adveng-pipeline/unifile_test/so300m.in"
FIXED=$(echo "$ORIGINAL" | sed 's|s3://adveng-pipeline/|s3://adveng-pipeline.s3.amazonaws.com/|')

echo "Original: $ORIGINAL"
echo "Fixed: $FIXED"

# Test AWS CLI access to both formats
echo ""
echo "=== Testing AWS CLI Access ==="

echo "Testing original format:"
aws s3 ls "s3://adveng-pipeline/unifile_test/so300m.in" || echo "❌ Original format failed"

echo "Testing fixed format:"
aws s3 ls "s3://adveng-pipeline.s3.amazonaws.com/unifile_test/so300m.in" || echo "❌ Fixed format failed"

# Test boto3 access
echo ""
echo "=== Testing Boto3 Access ==="
python3 -c "
import boto3
import tempfile
import os

# Test original format
try:
    s3_client = boto3.client('s3')
    s3_client.head_object(Bucket='adveng-pipeline', Key='unifile_test/so300m.in')
    print('✅ Original format works with boto3')
except Exception as e:
    print(f'❌ Original format failed with boto3: {e}')

# Test fixed format (should be same bucket)
try:
    s3_client = boto3.client('s3')
    s3_client.head_object(Bucket='adveng-pipeline', Key='unifile_test/so300m.in')
    print('✅ Fixed format works with boto3')
except Exception as e:
    print(f'❌ Fixed format failed with boto3: {e}')
"

echo ""
echo "=== Testing Rasterio Access ==="
python3 -c "
import rasterio
import os

# Test with rasterio
try:
    with rasterio.open('s3://adveng-pipeline.s3.amazonaws.com/unifile_test/so300m.in') as src:
        print(f'✅ Rasterio can open: {src.name}')
        print(f'   Shape: {src.shape}')
        print(f'   CRS: {src.crs}')
except Exception as e:
    print(f'❌ Rasterio failed: {e}')
" 