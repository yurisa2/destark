#!/bin/bash

# AWS Glue Tifffile FIS Deployment Script
# This script builds and deploys the Glue job with tifffile FIS implementation

set -e

# Configuration
AWS_REGION="${1:-us-east-2}"
AWS_ACCOUNT_ID="${2:-}"
ECR_REPOSITORY_NAME="glue-fis-tifffile"
GLUE_JOB_NAME="fis-tifffile-processor"
S3_BUCKET="${3:-<AWS-BUCKET>-unifile}"
S3_PREFIX="${4:-unifile_test}"

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

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    print_error "AWS CLI is not installed. Please install it first."
    exit 1
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install it first."
    exit 1
fi

# Get AWS account ID if not provided
if [ -z "$AWS_ACCOUNT_ID" ]; then
    print_status "Getting AWS account ID..."
    AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text --region "$AWS_REGION")
    if [ $? -ne 0 ]; then
        print_error "Failed to get AWS account ID. Please check your AWS credentials."
        exit 1
    fi
    print_success "AWS Account ID: $AWS_ACCOUNT_ID"
fi

# Set ECR repository URI
ECR_REPOSITORY_URI="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY_NAME"

print_status "Deployment Configuration:"
echo "  AWS Region: $AWS_REGION"
echo "  AWS Account ID: $AWS_ACCOUNT_ID"
echo "  ECR Repository: $ECR_REPOSITORY_URI"
echo "  Glue Job Name: $GLUE_JOB_NAME"
echo "  S3 Bucket: $S3_BUCKET"
echo "  S3 Prefix: $S3_PREFIX"

# Step 1: Create ECR repository
print_status "Step 1: Creating ECR repository..."
aws ecr describe-repositories --repository-names "$ECR_REPOSITORY_NAME" --region "$AWS_REGION" > /dev/null 2>&1 || {
    print_status "Creating ECR repository: $ECR_REPOSITORY_NAME"
    aws ecr create-repository \
        --repository-name "$ECR_REPOSITORY_NAME" \
        --region "$AWS_REGION" \
        --image-scanning-configuration scanOnPush=true \
        --encryption-configuration encryptionType=AES256
    print_success "ECR repository created"
}

# Step 2: Get ECR login token
print_status "Step 2: Authenticating with ECR..."
aws ecr get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin "$ECR_REPOSITORY_URI"
print_success "ECR authentication successful"

# Step 3: Build Docker image
print_status "Step 3: Building Docker image..."
IMAGE_TAG="latest"
FULL_IMAGE_URI="$ECR_REPOSITORY_URI:$IMAGE_TAG"

print_status "Building image: $FULL_IMAGE_URI"
docker build -f Dockerfile.glue-tifffile -t "$FULL_IMAGE_URI" .

if [ $? -ne 0 ]; then
    print_error "Docker build failed"
    exit 1
fi
print_success "Docker image built successfully"

# Step 4: Push Docker image to ECR
print_status "Step 4: Pushing Docker image to ECR..."
docker push "$FULL_IMAGE_URI"

if [ $? -ne 0 ]; then
    print_error "Failed to push Docker image to ECR"
    exit 1
fi
print_success "Docker image pushed to ECR"

# Step 5: Create Glue job
print_status "Step 5: Creating/updating Glue job..."

# Create job parameters JSON
cat > /tmp/glue_job_parameters.json << EOF
{
    "S3_BUCKET": "$S3_BUCKET",
    "S3_PREFIX": "$S3_PREFIX",
    "CONFIG_NAME": "config_max",
    "RESOLUTION": "1000m"
}
EOF

# Check if Glue job exists
JOB_EXISTS=$(aws glue get-job --job-name "$GLUE_JOB_NAME" --region "$AWS_REGION" 2>/dev/null || echo "NOT_FOUND")

if [ "$JOB_EXISTS" = "NOT_FOUND" ]; then
    print_status "Creating new Glue job: $GLUE_JOB_NAME"
    
    # Create Glue job
    aws glue create-job \
        --name "$GLUE_JOB_NAME" \
        --role "AWSGlueServiceRole" \
        --command "Name=glueetl,ScriptLocation=s3://$S3_BUCKET/$S3_PREFIX/glue_fis_tifffile_job.py" \
        --default-arguments '{"--job-language":"python","--job-bookmark-option":"job-bookmark-enable"}' \
        --parameters file:///tmp/glue_job_parameters.json \
        --max-retries 0 \
        --timeout 2880 \
        --max-capacity 2 \
        --region "$AWS_REGION"
    
    print_success "Glue job created"
else
    print_status "Updating existing Glue job: $GLUE_JOB_NAME"
    
    # Update Glue job
    aws glue update-job \
        --job-name "$GLUE_JOB_NAME" \
        --job-update "Command={Name=glueetl,ScriptLocation=s3://$S3_BUCKET/$S3_PREFIX/glue_fis_tifffile_job.py},DefaultArguments={--job-language=python,--job-bookmark-option=job-bookmark-enable},Parameters={S3_BUCKET=$S3_BUCKET,S3_PREFIX=$S3_PREFIX,CONFIG_NAME=config_max,RESOLUTION=1000m},MaxRetries=0,Timeout=2880,MaxCapacity=2" \
        --region "$AWS_REGION"
    
    print_success "Glue job updated"
fi

# Step 6: Upload job script to S3
print_status "Step 6: Uploading job script to S3..."
aws s3 cp glue_fis_tifffile_job.py "s3://$S3_BUCKET/$S3_PREFIX/glue_fis_tifffile_job.py" --region "$AWS_REGION"

if [ $? -ne 0 ]; then
    print_error "Failed to upload job script to S3"
    exit 1
fi
print_success "Job script uploaded to S3"

# Step 7: Upload FIS library to S3
print_status "Step 7: Uploading FIS library to S3..."
aws s3 cp app/raster_fuzzy_lib_tifffile.py "s3://$S3_BUCKET/$S3_PREFIX/raster_fuzzy_lib_tifffile.py" --region "$AWS_REGION"

if [ $? -ne 0 ]; then
    print_error "Failed to upload FIS library to S3"
    exit 1
fi
print_success "FIS library uploaded to S3"

# Step 8: Upload config files to S3
print_status "Step 8: Uploading config files to S3..."
aws s3 cp app/config/ "s3://$S3_BUCKET/$S3_PREFIX/config/" --recursive --region "$AWS_REGION"

if [ $? -ne 0 ]; then
    print_error "Failed to upload config files to S3"
    exit 1
fi
print_success "Config files uploaded to S3"

# Step 9: Create deployment summary
print_status "Step 9: Creating deployment summary..."

cat > deployment_summary.md << EOF
# AWS Glue Tifffile FIS Deployment Summary

## Deployment Information
- **Date**: $(date)
- **AWS Region**: $AWS_REGION
- **AWS Account ID**: $AWS_ACCOUNT_ID
- **ECR Repository**: $ECR_REPOSITORY_URI
- **Glue Job Name**: $GLUE_JOB_NAME
- **S3 Bucket**: $S3_BUCKET
- **S3 Prefix**: $S3_PREFIX

## Resources Created/Updated

### 1. ECR Repository
- **Name**: $ECR_REPOSITORY_NAME
- **URI**: $ECR_REPOSITORY_URI
- **Image Tag**: $IMAGE_TAG

### 2. Glue Job
- **Name**: $GLUE_JOB_NAME
- **Type**: ETL Job
- **Language**: Python
- **Max Capacity**: 2 DPU
- **Timeout**: 2880 minutes (48 hours)
- **Max Retries**: 0

### 3. S3 Files
- **Job Script**: s3://$S3_BUCKET/$S3_PREFIX/glue_fis_tifffile_job.py
- **FIS Library**: s3://$S3_BUCKET/$S3_PREFIX/raster_fuzzy_lib_tifffile.py
- **Config Files**: s3://$S3_BUCKET/$S3_PREFIX/config/

## Job Parameters
- **S3_BUCKET**: $S3_BUCKET
- **S3_PREFIX**: $S3_PREFIX
- **CONFIG_NAME**: config_max
- **RESOLUTION**: 1000m

## Expected Input/Output Structure

### Input Files (S3)
- s3://$S3_BUCKET/$S3_PREFIX/config/config_max.json
- s3://$S3_BUCKET/$S3_PREFIX/input/1000m/socioeconomico_1000m.tif
- s3://$S3_BUCKET/$S3_PREFIX/input/1000m/ambiental_1000m.tif
- s3://$S3_BUCKET/$S3_PREFIX/input/1000m/estrategico_1000m.tif

### Output Files (S3)
- s3://$S3_BUCKET/$S3_PREFIX/output/output_1000m_config_max_glue.tif

## Next Steps

1. **Upload Input Data**: Ensure all input files are uploaded to S3 in the correct structure
2. **Run Glue Job**: Execute the job from AWS Glue console or CLI
3. **Monitor**: Check CloudWatch logs for job execution status
4. **Verify Output**: Download and verify the output file from S3

## Commands to Run Job

### Via AWS CLI
\`\`\`bash
aws glue start-job-run --job-name "$GLUE_JOB_NAME" --region "$AWS_REGION"
\`\`\`

### Via AWS Console
1. Go to AWS Glue Console
2. Navigate to Jobs
3. Select "$GLUE_JOB_NAME"
4. Click "Run job"

## Monitoring

### CloudWatch Logs
- **Log Group**: /aws-glue/jobs/$GLUE_JOB_NAME
- **Log Stream**: Will be created automatically for each job run

### S3 Output
- **Location**: s3://$S3_BUCKET/$S3_PREFIX/output/
- **File Pattern**: output_{resolution}_{config}_glue.tif

## Troubleshooting

1. **Check IAM Permissions**: Ensure Glue service role has S3 read/write permissions
2. **Verify Input Files**: Confirm all input files exist in S3
3. **Check Logs**: Review CloudWatch logs for detailed error messages
4. **Resource Limits**: Monitor DPU usage and timeout settings

EOF

print_success "Deployment summary created: deployment_summary.md"

# Step 10: Cleanup
rm -f /tmp/glue_job_parameters.json

print_success "🎉 AWS Glue Tifffile FIS deployment completed successfully!"
print_status "Next steps:"
echo "  1. Upload your input files to S3:"
echo "     s3://$S3_BUCKET/$S3_PREFIX/input/1000m/"
echo "  2. Run the Glue job:"
echo "     aws glue start-job-run --job-name $GLUE_JOB_NAME --region $AWS_REGION"
echo "  3. Monitor the job in AWS Glue console"
echo "  4. Check output at: s3://$S3_BUCKET/$S3_PREFIX/output/"

print_status "Deployment summary saved to: deployment_summary.md" 