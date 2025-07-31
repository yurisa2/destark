# AWS Setup Guide for Glue FIS Deployment

This guide will help you set up AWS credentials and tools needed for the Glue FIS deployment.

## 🔧 Prerequisites Check

### 1. Check if AWS CLI is installed
```bash
aws --version
```

If not installed, install it:

**macOS (using Homebrew):**
```bash
brew install awscli
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install awscli
```

**Windows:**
Download from: https://aws.amazon.com/cli/

### 2. Check if Docker is installed
```bash
docker --version
```

If not installed, install it:

**macOS:**
```bash
brew install --cask docker
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install docker.io
sudo usermod -aG docker $USER
```

## 🔐 AWS Credentials Setup

### Option 1: AWS CLI Configuration (Recommended)

1. **Get your AWS credentials:**
   - Access Key ID
   - Secret Access Key
   - Default region (e.g., us-east-2)

2. **Configure AWS CLI:**
```bash
aws configure
```

You'll be prompted for:
```
AWS Access Key ID [None]: YOUR_ACCESS_KEY_ID
AWS Secret Access Key [None]: YOUR_SECRET_ACCESS_KEY
Default region name [None]: us-east-2
Default output format [None]: json
```

3. **Verify configuration:**
```bash
aws sts get-caller-identity
```

Expected output:
```json
{
    "UserId": "AIDA...",
    "Account": "123456789012",
    "Arn": "arn:aws:iam::123456789012:user/your-username"
}
```

### Option 2: Environment Variables

Set environment variables:
```bash
export AWS_ACCESS_KEY_ID=your_access_key_id
export AWS_SECRET_ACCESS_KEY=your_secret_access_key
export AWS_DEFAULT_REGION=us-east-2
```

### Option 3: AWS Credentials File

Create `~/.aws/credentials`:
```ini
[default]
aws_access_key_id = YOUR_ACCESS_KEY_ID
aws_secret_access_key = YOUR_SECRET_ACCESS_KEY
```

Create `~/.aws/config`:
```ini
[default]
region = us-east-2
output = json
```

## 🔑 Required AWS Permissions

Your AWS user/role needs these permissions:

### IAM Permissions
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "iam:GetRole",
                "iam:PassRole"
            ],
            "Resource": "arn:aws:iam::*:role/AWSGlueServiceRole"
        }
    ]
}
```

### ECR Permissions
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "ecr:CreateRepository",
                "ecr:DescribeRepositories",
                "ecr:GetAuthorizationToken",
                "ecr:BatchCheckLayerAvailability",
                "ecr:GetDownloadUrlForLayer",
                "ecr:BatchGetImage",
                "ecr:PutImage",
                "ecr:InitiateLayerUpload",
                "ecr:UploadLayerPart",
                "ecr:CompleteLayerUpload"
            ],
            "Resource": "*"
        }
    ]
}
```

### Glue Permissions
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "glue:CreateJob",
                "glue:UpdateJob",
                "glue:GetJob",
                "glue:StartJobRun",
                "glue:GetJobRun",
                "glue:GetJobRuns",
                "glue:DeleteJob"
            ],
            "Resource": "*"
        }
    ]
}
```

### S3 Permissions
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:DeleteObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::your-bucket-name",
                "arn:aws:s3:::your-bucket-name/*"
            ]
        }
    ]
}
```

## 🧪 Testing AWS Setup

### 1. Test AWS CLI
```bash
# Test basic connectivity
aws sts get-caller-identity

# Test S3 access
aws s3 ls

# Test ECR access
aws ecr describe-repositories --region us-east-2
```

### 2. Test Docker
```bash
# Test Docker daemon
docker info

# Test Docker build
docker build --help
```

### 3. Test ECR Login
```bash
# Get ECR login token
aws ecr get-login-password --region us-east-2 | docker login --username AWS --password-stdin 123456789012.dkr.ecr.us-east-2.amazonaws.com
```

## 🚀 Quick Setup Script

Create a setup script to automate the process:

```bash
#!/bin/bash
# setup_aws.sh

echo "🔧 AWS Setup for Glue FIS Deployment"

# Check AWS CLI
if ! command -v aws &> /dev/null; then
    echo "❌ AWS CLI not found. Please install it first."
    exit 1
fi

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install it first."
    exit 1
fi

# Check AWS credentials
if ! aws sts get-caller-identity &> /dev/null; then
    echo "❌ AWS credentials not configured. Please run 'aws configure' first."
    exit 1
fi

echo "✅ AWS CLI and Docker are ready!"
echo "✅ AWS credentials are configured!"

# Get account info
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=$(aws configure get region)

echo "📋 Account Information:"
echo "  Account ID: $ACCOUNT_ID"
echo "  Region: $REGION"

echo "🚀 Ready to deploy Glue FIS job!"
```

## 🔍 Troubleshooting

### Common Issues

#### 1. "aws: command not found"
- Install AWS CLI using the instructions above
- Add AWS CLI to your PATH

#### 2. "Unable to locate credentials"
- Run `aws configure` to set up credentials
- Check if credentials file exists: `ls ~/.aws/`
- Verify environment variables: `echo $AWS_ACCESS_KEY_ID`

#### 3. "Access Denied" errors
- Check IAM permissions
- Verify the user/role has required permissions
- Contact AWS administrator if needed

#### 4. Docker permission errors
```bash
# Add user to docker group
sudo usermod -aG docker $USER

# Restart Docker service
sudo systemctl restart docker

# Log out and back in, or run:
newgrp docker
```

#### 5. ECR login failures
```bash
# Clear Docker credentials
docker logout

# Re-authenticate with ECR
aws ecr get-login-password --region us-east-2 | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com
```

## 📋 Pre-deployment Checklist

Before running the deployment scripts, ensure:

- [ ] AWS CLI installed and configured
- [ ] Docker installed and running
- [ ] AWS credentials working (`aws sts get-caller-identity`)
- [ ] S3 bucket exists and accessible
- [ ] IAM permissions configured
- [ ] Input files ready for upload

## 🎯 Next Steps

Once AWS is properly configured:

1. **Upload input files:**
```bash
./upload_inputs_to_s3.sh us-east-2 your-bucket your-prefix 1000m
```

2. **Deploy Glue job:**
```bash
./deploy_glue_tifffile.sh us-east-2
```

3. **Run the job:**
```bash
aws glue start-job-run --job-name fis-tifffile-processor --region us-east-2
```

## 📞 Support

If you encounter issues:

1. Check AWS CLI documentation: https://docs.aws.amazon.com/cli/
2. Verify IAM permissions in AWS Console
3. Check CloudTrail logs for permission errors
4. Contact AWS support if needed

---

**Note:** Never commit AWS credentials to version control. Use IAM roles, environment variables, or AWS credentials file for secure credential management. 