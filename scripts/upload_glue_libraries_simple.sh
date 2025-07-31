#!/bin/bash

# Set your S3 bucket name (using the one from your error log)
S3_BUCKET="aws-glue-assets-<AWS-ACCOUNT-ID>-us-east-1"
S3_PREFIX="python-libs"

echo "=== Uploading Python Libraries to S3 for AWS Glue ==="

# Create a temporary directory for downloads
TEMP_DIR="/tmp/glue_libs_$(date +%s)"
mkdir -p $TEMP_DIR
cd $TEMP_DIR

echo "Downloading required libraries..."

# Download the required libraries for AWS Glue
pip download --platform manylinux2014_x86_64 --only-binary=all \
    numpy>=1.21.0 \
    scipy>=1.7.0 \
    rasterio>=1.3.0 \
    scikit-fuzzy>=0.4.2 \
    boto3>=1.26.0

echo "Uploading to S3..."

# Upload each library to S3
for file in *.whl; do
    if [ -f "$file" ]; then
        echo "Uploading $file..."
        aws s3 cp "$file" "s3://$S3_BUCKET/$S3_PREFIX/$file"
    fi
done

echo "Cleaning up..."
cd /
rm -rf $TEMP_DIR

echo ""
echo "✓ Libraries uploaded to s3://$S3_BUCKET/$S3_PREFIX/"
echo ""
echo "Now add these to your AWS Glue job in 'Additional Python libraries':"
echo "s3://$S3_BUCKET/$S3_PREFIX/numpy-*.whl"
echo "s3://$S3_BUCKET/$S3_PREFIX/scipy-*.whl"
echo "s3://$S3_BUCKET/$S3_PREFIX/rasterio-*.whl"
echo "s3://$S3_BUCKET/$S3_PREFIX/scikit_fuzzy-*.whl"
echo "s3://$S3_BUCKET/$S3_PREFIX/boto3-*.whl"
echo ""
echo "Or simply add these to 'Python library path':"
echo "rasterio"
echo "scikit-fuzzy" 