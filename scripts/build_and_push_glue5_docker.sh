#!/bin/bash

# Build and push AWS Glue 5 Docker image with rasterio to ECR
# This script builds a custom Docker image based on AWS Glue 5 and pushes it to ECR

set -e

# Configuration
AWS_REGION=${AWS_REGION:-"us-east-1"}
AWS_ACCOUNT_ID=${AWS_ACCOUNT_ID:-""}
ECR_REPOSITORY_NAME=${ECR_REPOSITORY_NAME:-"glue5-rasterio"}
IMAGE_TAG=${IMAGE_TAG:-"latest"}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Building and Pushing AWS Glue 5 Docker Image with Rasterio ===${NC}"

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo -e "${RED}Error: AWS CLI is not installed. Please install it first.${NC}"
    exit 1
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed. Please install it first.${NC}"
    exit 1
fi

# Get AWS account ID if not provided
if [ -z "$AWS_ACCOUNT_ID" ]; then
    echo -e "${YELLOW}Getting AWS account ID...${NC}"
    AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    if [ $? -ne 0 ]; then
        echo -e "${RED}Error: Could not get AWS account ID. Please check your AWS credentials.${NC}"
        exit 1
    fi
fi

echo -e "${GREEN}AWS Account ID: ${AWS_ACCOUNT_ID}${NC}"
echo -e "${GREEN}AWS Region: ${AWS_REGION}${NC}"
echo -e "${GREEN}ECR Repository: ${ECR_REPOSITORY_NAME}${NC}"
echo -e "${GREEN}Image Tag: ${IMAGE_TAG}${NC}"

# Create ECR repository if it doesn't exist
echo -e "${YELLOW}Creating ECR repository if it doesn't exist...${NC}"
aws ecr describe-repositories --repository-names ${ECR_REPOSITORY_NAME} --region ${AWS_REGION} 2>/dev/null || \
aws ecr create-repository --repository-name ${ECR_REPOSITORY_NAME} --region ${AWS_REGION}

# Get ECR login token
echo -e "${YELLOW}Logging in to ECR...${NC}"
aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com

# Build Docker image
echo -e "${YELLOW}Building Docker image...${NC}"
docker build -f Dockerfile.glue5 -t ${ECR_REPOSITORY_NAME}:${IMAGE_TAG} .

# Tag the image for ECR
ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY_NAME}"
docker tag ${ECR_REPOSITORY_NAME}:${IMAGE_TAG} ${ECR_URI}:${IMAGE_TAG}

# Push to ECR
echo -e "${YELLOW}Pushing image to ECR...${NC}"
docker push ${ECR_URI}:${IMAGE_TAG}

echo -e "${GREEN}✓ Successfully built and pushed Docker image!${NC}"
echo -e "${GREEN}✓ ECR Image URI: ${ECR_URI}:${IMAGE_TAG}${NC}"
echo ""
echo -e "${YELLOW}To use this image in AWS Glue:${NC}"
echo -e "1. Go to AWS Glue Console"
echo -e "2. Create a new Glue job"
echo -e "3. In the job parameters, set:"
echo -e "   - Image URI: ${ECR_URI}:${IMAGE_TAG}"
echo -e "4. Make sure your Glue job has the necessary IAM permissions to pull from ECR"
echo ""
echo -e "${YELLOW}To test the image locally:${NC}"
echo -e "docker run -it ${ECR_URI}:${IMAGE_TAG} /opt/amazon/glue/test_environment.sh" 