#!/bin/bash

# Upload Python libraries for AWS Glue 5 to S3
# This script downloads and uploads the required libraries to S3 for use with AWS Glue 5

set -e

# Configuration
S3_BUCKET=${S3_BUCKET:-"aws-glue-assets-475136118191-us-east-1"}
S3_PREFIX=${S3_PREFIX:-"python-libs-glue5"}
AWS_REGION=${AWS_REGION:-"us-east-1"}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Uploading Python Libraries for AWS Glue 5 to S3 ===${NC}"

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo -e "${RED}Error: AWS CLI is not installed. Please install it first.${NC}"
    exit 1
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo -e "${RED}Error: pip3 is not installed. Please install it first.${NC}"
    exit 1
fi

echo -e "${GREEN}S3 Bucket: ${S3_BUCKET}${NC}"
echo -e "${GREEN}S3 Prefix: ${S3_PREFIX}${NC}"
echo -e "${GREEN}AWS Region: ${AWS_REGION}${NC}"

# Create a temporary directory for downloads
TEMP_DIR="/tmp/glue5_libs_$(date +%s)"
mkdir -p $TEMP_DIR
cd $TEMP_DIR

echo -e "${YELLOW}Downloading libraries for AWS Glue 5...${NC}"

# Download the required libraries for AWS Glue 5
# Using manylinux2014_x86_64 platform for compatibility with AWS Glue
pip3 download --platform manylinux2014_x86_64 --only-binary=all \
    numpy>=1.21.0 \
    scipy>=1.7.0 \
    rasterio>=1.3.0 \
    scikit-fuzzy>=0.4.2 \
    fiona>=1.8.0 \
    shapely>=1.8.0 \
    boto3>=1.26.0 \
    numba>=0.56.0 \
    click>=8.0.0 \
    cligj>=0.5.0 \
    attrs>=21.0.0 \
    certifi>=2021.0.0 \
    affine>=2.3.0 \
    pyparsing>=3.0.0

echo -e "${YELLOW}Uploading to S3...${NC}"

# Upload each library to S3
for file in *.whl; do
    if [ -f "$file" ]; then
        echo -e "${YELLOW}Uploading $file...${NC}"
        aws s3 cp "$file" "s3://$S3_BUCKET/$S3_PREFIX/$file" --region $AWS_REGION
    fi
done

echo -e "${GREEN}Cleaning up...${NC}"
cd /
rm -rf $TEMP_DIR

echo -e "${GREEN}✓ Libraries uploaded to s3://$S3_BUCKET/$S3_PREFIX/${NC}"
echo ""
echo -e "${YELLOW}Now add these to your AWS Glue 5 job:${NC}"
echo -e "s3://$S3_BUCKET/$S3_PREFIX/numpy-*.whl"
echo -e "s3://$S3_BUCKET/$S3_PREFIX/scipy-*.whl"
echo -e "s3://$S3_BUCKET/$S3_PREFIX/rasterio-*.whl"
echo -e "s3://$S3_BUCKET/$S3_PREFIX/scikit_fuzzy-*.whl"
echo -e "s3://$S3_BUCKET/$S3_PREFIX/fiona-*.whl"
echo -e "s3://$S3_BUCKET/$S3_PREFIX/shapely-*.whl"
echo -e "s3://$S3_BUCKET/$S3_PREFIX/boto3-*.whl"
echo -e "s3://$S3_BUCKET/$S3_PREFIX/numba-*.whl"
echo ""
echo -e "${YELLOW}Instructions for AWS Glue 5:${NC}"
echo -e "1. Go to AWS Glue Console"
echo -e "2. Create a new Glue job"
echo -e "3. In the job parameters, add the S3 paths above to the 'Python library path' field"
echo -e "4. Make sure your Glue job has the necessary IAM permissions to access S3" 