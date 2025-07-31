# AWS Glue Tifffile FIS Deployment Guide

This guide provides a complete solution for deploying the Fuzzy Inference System (FIS) using the tifffile library to AWS Glue. The solution processes raster data from S3 and outputs results back to S3.

## 🎯 Overview

The deployment includes:
- **AWS Glue Job**: Processes FIS models using tifffile library
- **Docker Container**: Custom environment with all dependencies
- **S3 Integration**: Input/output file handling
- **Automated Deployment**: Scripts for easy deployment and management

## 📁 Project Structure

```
destark/
├── glue_fis_tifffile_job.py          # Main Glue job script
├── Dockerfile.glue-tifffile          # Docker image for Glue
├── requirements-glue-tifffile.txt    # Python dependencies
├── deploy_glue_tifffile.sh           # Deployment script
├── upload_inputs_to_s3.sh            # S3 upload script
├── app/
│   ├── raster_fuzzy_lib_tifffile.py  # FIS implementation
│   ├── config/                       # FIS configuration files
│   └── files/input/                  # Input raster files
└── README_AWS_Glue_Tifffile_Deployment.md
```

## 🚀 Quick Start

### Prerequisites

1. **AWS CLI** installed and configured
2. **Docker** installed and running
3. **AWS Permissions** for:
   - ECR (Elastic Container Registry)
   - Glue
   - S3
   - IAM (for service roles)

### Step 1: Upload Input Files to S3

```bash
# Upload input files for 1000m resolution
./upload_inputs_to_s3.sh us-east-2 <AWS-BUCKET>-unifile unifile_test 1000m

# Or for 300m resolution
./upload_inputs_to_s3.sh us-east-2 <AWS-BUCKET>-unifile unifile_test 300m
```

### Step 2: Deploy Glue Job

```bash
# Deploy the complete solution
./deploy_glue_tifffile.sh us-east-2
```

### Step 3: Run the Job

```bash
# Start the Glue job
aws glue start-job-run --job-name fis-tifffile-processor --region us-east-2
```

## 📋 Detailed Deployment Steps

### 1. Input File Upload

The `upload_inputs_to_s3.sh` script uploads:
- **Config files**: FIS configuration JSON files
- **Input rasters**: Social, Environmental, and Strategic TIFF files
- **Creates output directory**: Ready for job results

**Usage:**
```bash
./upload_inputs_to_s3.sh [AWS_REGION] [S3_BUCKET] [S3_PREFIX] [RESOLUTION]
```

**Example:**
```bash
./upload_inputs_to_s3.sh us-east-2 my-bucket fis-test 1000m
```

### 2. Glue Job Deployment

The `deploy_glue_tifffile.sh` script:
- Creates ECR repository
- Builds and pushes Docker image
- Creates/updates Glue job
- Uploads job scripts to S3

**Usage:**
```bash
./deploy_glue_tifffile.sh [AWS_REGION] [AWS_ACCOUNT_ID] [S3_BUCKET] [S3_PREFIX]
```

**Example:**
```bash
./deploy_glue_tifffile.sh us-east-2 123456789012 my-bucket fis-test
```

## 🔧 Configuration

### Environment Variables

The Glue job uses these environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `S3_BUCKET` | `<AWS-BUCKET>-unifile` | S3 bucket name |
| `S3_PREFIX` | `unifile_test` | S3 prefix for files |
| `CONFIG_NAME` | `config_max` | FIS configuration file |
| `RESOLUTION` | `1000m` | Input resolution |

### S3 File Structure

```
s3://your-bucket/your-prefix/
├── config/
│   ├── config_max.json
│   ├── config_median.json
│   └── config_minimum.json
├── input/
│   ├── 1000m/
│   │   ├── socioeconomico_1000m.tif
│   │   ├── ambiental_1000m.tif
│   │   └── estrategico_1000m.tif
│   └── 300m/
│       ├── socioeconomico_300m.tif
│       ├── ambiental_300m.tif
│       └── estrategico_300m.tif
└── output/
    └── (job outputs)
```

## 🐳 Docker Image

### Base Image
- **AWS Glue 4.0**: Based on `public.ecr.aws/glue/aws-glue-libs:glue_libs_4.0.0_image_01`
- **Python 3.9**: Compatible with Glue environment

### Dependencies
- **tifffile**: For raster file processing
- **scikit-fuzzy**: For fuzzy logic operations
- **numpy/scipy**: For scientific computing
- **boto3**: For AWS services integration
- **pyspark**: For Glue compatibility

### Build Process
```bash
# Build the image
docker build -f Dockerfile.glue-tifffile -t glue-fis-tifffile .

# Test locally (optional)
docker run --rm glue-fis-tifffile
```

## 🔍 Monitoring and Logging

### CloudWatch Logs
- **Log Group**: `/aws-glue/jobs/fis-tifffile-processor`
- **Log Stream**: Created automatically for each job run

### Job Status
```bash
# Check job status
aws glue get-job-run --job-name fis-tifffile-processor --run-id <RUN_ID> --region us-east-2

# List recent runs
aws glue get-job-runs --job-name fis-tifffile-processor --region us-east-2
```

### S3 Output Verification
```bash
# List output files
aws s3 ls s3://your-bucket/your-prefix/output/ --region us-east-2

# Download output file
aws s3 cp s3://your-bucket/your-prefix/output/output_1000m_config_max_glue.tif ./ --region us-east-2
```

## 🛠️ Troubleshooting

### Common Issues

#### 1. Docker Build Failures
```bash
# Check Docker daemon
docker info

# Clean Docker cache
docker system prune -a
```

#### 2. ECR Authentication Issues
```bash
# Re-authenticate with ECR
aws ecr get-login-password --region us-east-2 | docker login --username AWS --password-stdin <ACCOUNT_ID>.dkr.ecr.us-east-2.amazonaws.com
```

#### 3. Glue Job Failures
```bash
# Check job logs
aws logs describe-log-groups --log-group-name-prefix "/aws-glue/jobs/fis-tifffile-processor" --region us-east-2

# Get specific log stream
aws logs get-log-events --log-group-name "/aws-glue/jobs/fis-tifffile-processor" --log-stream-name <STREAM_NAME> --region us-east-2
```

#### 4. S3 Permission Issues
Ensure the Glue service role has these permissions:
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
                "arn:aws:s3:::your-bucket",
                "arn:aws:s3:::your-bucket/*"
            ]
        }
    ]
}
```

### Performance Optimization

#### 1. DPU Configuration
- **Default**: 2 DPU
- **Large datasets**: Increase to 4-8 DPU
- **Timeout**: 2880 minutes (48 hours)

#### 2. Parallel Processing
- **Cores**: Automatically uses 75% of available cores
- **Chunk size**: 100 rows per chunk
- **Memory**: Optimized for Glue environment

## 📊 Expected Output

### File Format
- **Format**: TIFF (Tagged Image File Format)
- **Data Type**: float32 (preserves precision)
- **Compression**: None (for maximum compatibility)

### File Naming
```
output_{resolution}_{config}_glue.tif
```

**Examples:**
- `output_1000m_config_max_glue.tif`
- `output_300m_config_median_glue.tif`

### File Size
- **1000m resolution**: ~77.5 MB
- **300m resolution**: ~700 MB (estimated)

## 🔄 Updating the Deployment

### Update Job Script
```bash
# Upload new job script
aws s3 cp glue_fis_tifffile_job.py s3://your-bucket/your-prefix/glue_fis_tifffile_job.py --region us-east-2

# Update Glue job
aws glue update-job --job-name fis-tifffile-processor --job-update "Command={Name=glueetl,ScriptLocation=s3://your-bucket/your-prefix/glue_fis_tifffile_job.py}" --region us-east-2
```

### Update Docker Image
```bash
# Rebuild and push
./deploy_glue_tifffile.sh us-east-2
```

## 🧪 Testing

### Local Testing
```bash
# Test FIS processing locally
python3 glue_fis_tifffile_job.py

# Test with different parameters
S3_BUCKET=test-bucket S3_PREFIX=test CONFIG_NAME=config_max RESOLUTION=1000m python3 glue_fis_tifffile_job.py
```

### S3 Testing
```bash
# Test S3 connectivity
aws s3 ls s3://your-bucket/your-prefix/ --region us-east-2

# Test file upload/download
aws s3 cp test.txt s3://your-bucket/your-prefix/test.txt --region us-east-2
aws s3 cp s3://your-bucket/your-prefix/test.txt test_download.txt --region us-east-2
```

## 📈 Cost Optimization

### Glue Costs
- **DPU-hour**: $0.44 per DPU-hour
- **Job duration**: Typically 5-10 minutes for 1000m data
- **Estimated cost**: $0.07-$0.15 per job run

### S3 Costs
- **Storage**: $0.023 per GB per month
- **Data transfer**: $0.09 per GB (outbound)

### ECR Costs
- **Storage**: $0.10 per GB per month
- **Data transfer**: $0.09 per GB (outbound)

## 🔐 Security

### IAM Roles
- **Glue Service Role**: `AWSGlueServiceRole`
- **S3 Permissions**: Read/write access to specific bucket
- **ECR Permissions**: Pull access to repository

### Data Encryption
- **S3**: Server-side encryption (SSE-S3)
- **ECR**: Repository encryption enabled
- **Glue**: Data encrypted in transit and at rest

## 📞 Support

### Documentation
- [AWS Glue Developer Guide](https://docs.aws.amazon.com/glue/)
- [AWS ECR User Guide](https://docs.aws.amazon.com/ecr/)
- [Tifffile Documentation](https://tifffile.readthedocs.io/)

### Monitoring
- **CloudWatch**: Job metrics and logs
- **Glue Console**: Job status and history
- **S3 Console**: File management and access

### Troubleshooting Resources
- [AWS Glue Troubleshooting](https://docs.aws.amazon.com/glue/latest/dg/monitor-debug-capacity.html)
- [ECR Troubleshooting](https://docs.aws.amazon.com/ecr/latest/userguide/troubleshooting.html)

---

## 🎉 Success!

Once deployed, your FIS processing pipeline will:
1. ✅ Read input files from S3
2. ✅ Process with tifffile FIS implementation
3. ✅ Preserve original precision (float32)
4. ✅ Output results back to S3
5. ✅ Provide comprehensive logging and monitoring

The solution is production-ready and scalable for processing large raster datasets in AWS. 