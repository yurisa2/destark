#!/bin/bash

# AWS Setup Verification Script for Glue FIS Deployment
# This script checks and helps configure AWS credentials

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

echo "🔧 AWS Setup Verification for Glue FIS Deployment"
echo "=================================================="

# Check 1: AWS CLI Installation
print_status "Checking AWS CLI installation..."
if command -v aws &> /dev/null; then
    AWS_VERSION=$(aws --version)
    print_success "AWS CLI found: $AWS_VERSION"
else
    print_error "AWS CLI not found!"
    echo ""
    echo "Please install AWS CLI:"
    echo "  macOS: brew install awscli"
    echo "  Ubuntu: sudo apt install awscli"
    echo "  Windows: Download from https://aws.amazon.com/cli/"
    echo ""
    exit 1
fi

# Check 2: Docker Installation
print_status "Checking Docker installation..."
if command -v docker &> /dev/null; then
    DOCKER_VERSION=$(docker --version)
    print_success "Docker found: $DOCKER_VERSION"
    
    # Check if Docker daemon is running
    if docker info &> /dev/null; then
        print_success "Docker daemon is running"
    else
        print_error "Docker daemon is not running!"
        echo "Please start Docker Desktop or Docker service"
        exit 1
    fi
else
    print_error "Docker not found!"
    echo ""
    echo "Please install Docker:"
    echo "  macOS: brew install --cask docker"
    echo "  Ubuntu: sudo apt install docker.io"
    echo ""
    exit 1
fi

# Check 3: AWS Credentials
print_status "Checking AWS credentials..."
if aws sts get-caller-identity &> /dev/null; then
    print_success "AWS credentials are configured!"
    
    # Get account information
    ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    USER_ARN=$(aws sts get-caller-identity --query Arn --output text)
    REGION=$(aws configure get region || echo "us-east-2")
    
    echo "  Account ID: $ACCOUNT_ID"
    echo "  User ARN: $USER_ARN"
    echo "  Region: $REGION"
    
else
    print_error "AWS credentials not configured!"
    echo ""
    echo "Please configure AWS credentials using one of these methods:"
    echo ""
    echo "Method 1: Interactive configuration"
    echo "  aws configure"
    echo ""
    echo "Method 2: Environment variables"
    echo "  export AWS_ACCESS_KEY_ID=your_access_key_id"
    echo "  export AWS_SECRET_ACCESS_KEY=your_secret_access_key"
    echo "  export AWS_DEFAULT_REGION=us-east-2"
    echo ""
    echo "Method 3: Credentials file"
    echo "  Create ~/.aws/credentials with your credentials"
    echo ""
    exit 1
fi

# Check 4: Test AWS Services
print_status "Testing AWS service access..."

# Test S3
if aws s3 ls &> /dev/null; then
    print_success "S3 access: OK"
else
    print_warning "S3 access: Limited or no access"
fi

# Test ECR
if aws ecr describe-repositories --region "$REGION" &> /dev/null; then
    print_success "ECR access: OK"
else
    print_warning "ECR access: Limited or no access"
fi

# Test Glue
if aws glue get-jobs --region "$REGION" &> /dev/null; then
    print_success "Glue access: OK"
else
    print_warning "Glue access: Limited or no access"
fi

# Check 5: Required IAM Role
print_status "Checking for AWSGlueServiceRole..."
if aws iam get-role --role-name AWSGlueServiceRole &> /dev/null; then
    print_success "AWSGlueServiceRole exists"
else
    print_warning "AWSGlueServiceRole not found"
    echo "  This role is required for Glue jobs. You may need to create it or use a different role."
fi

echo ""
echo "=================================================="
print_success "AWS Setup Verification Complete!"
echo ""

# Summary
echo "📋 Summary:"
echo "  ✅ AWS CLI: Installed"
echo "  ✅ Docker: Installed and running"
echo "  ✅ AWS Credentials: Configured"
echo "  ✅ Account ID: $ACCOUNT_ID"
echo "  ✅ Region: $REGION"
echo ""

# Next steps
echo "🚀 Next Steps:"
echo "  1. Upload input files: ./upload_inputs_to_s3.sh $REGION your-bucket your-prefix 1000m"
echo "  2. Deploy Glue job: ./deploy_glue_tifffile.sh $REGION"
echo "  3. Run the job: aws glue start-job-run --job-name fis-tifffile-processor --region $REGION"
echo ""

# Check if scripts exist
print_status "Checking deployment scripts..."
if [ -f "upload_inputs_to_s3.sh" ] && [ -f "deploy_glue_tifffile.sh" ]; then
    print_success "Deployment scripts found"
    
    # Make scripts executable
    chmod +x upload_inputs_to_s3.sh deploy_glue_tifffile.sh
    print_success "Made deployment scripts executable"
else
    print_warning "Deployment scripts not found in current directory"
    echo "  Make sure you're in the correct directory with the deployment files"
fi

echo ""
print_success "You're ready to deploy the Glue FIS job! 🎉" 