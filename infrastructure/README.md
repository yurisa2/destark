# Computing Infrastructure

This directory contains all infrastructure configurations and deployment scripts for our three-phase research journey, from local development to cloud-based distributed processing.

## 🏗️ Infrastructure Evolution

### Phase 1: Local Development
- **Environment**: Single machine, single-threaded processing
- **Requirements**: Python environment with scientific computing libraries
- **Deployment**: Local installation and configuration

### Phase 2: Local Multiprocessing
- **Environment**: Multi-core local machine
- **Requirements**: Multiprocessing-capable Python environment
- **Deployment**: Local cluster configuration

### Phase 3: Cloud Distributed Computing
- **Environment**: AWS cloud infrastructure
- **Requirements**: EMR clusters, Glue jobs, S3 storage
- **Deployment**: Automated cloud deployment

## 📁 Directory Structure

```
infrastructure/
├── local/                      # Local processing infrastructure
│   └── scripts/               # Local execution scripts
│       ├── run.sh             # Main execution script
│       ├── run_local.sh       # Local execution script
│       ├── run_spark_local_demo.sh # Local Spark demo
│       └── SPARK_LOCAL_GUIDE.md # Local Spark guide
├── aws_emr/                   # AWS EMR infrastructure
│   └── scripts/               # EMR execution scripts
│       ├── setup_aws.sh       # AWS setup script
│       └── upload_inputs_to_s3.sh # S3 upload script
├── aws_glue/                  # AWS Glue infrastructure
│   ├── scripts/               # Glue execution scripts
│   │   ├── deploy_glue_tifffile.sh # Glue deployment
│   │   └── glue_fis_tifffile_job.py # Glue job script
│   ├── glue_job_config.json   # Glue job configuration
│   ├── glue_job_parameters.json # Glue job parameters
│   ├── glue_python_libraries.txt # Glue dependencies
│   └── glue_job_template.txt  # Glue job template
├── requirements.txt           # Main Python dependencies
├── requirements-glue.txt      # Glue-specific dependencies
├── requirements-glue-tifffile.txt # Glue tifffile dependencies
├── requirements-emr.txt       # EMR-specific dependencies
├── requirements-spark.txt     # Spark-specific dependencies
├── Dockerfile                 # Main Docker configuration
├── Dockerfile.glue-tifffile   # Glue tifffile Docker
├── Dockerfile.glue5           # Glue 5 Docker
├── Dockerfile.glue5-tifffile  # Glue 5 tifffile Docker
├── Dockerfile.emr-local       # EMR local Docker
├── Dockerfile.emr-simple      # EMR simple Docker
├── Dockerfile.emr-official    # EMR official Docker
├── Dockerfile.emr-tifffile    # EMR tifffile Docker
├── Dockerfile.jupyter         # Jupyter Docker
├── Dockerfile.extract         # Extract Docker
├── docker-compose.yml         # Main Docker Compose
└── docker-compose.emr-local.yml # EMR local Docker Compose
```

## 🚀 Infrastructure Components

### 1. Local Infrastructure

#### Environment Setup
```bash
# Setup local environment
pip install -r infrastructure/requirements.txt

# Run local processing
./infrastructure/local/scripts/run.sh
```

#### Local Spark Setup
```bash
# Setup local Spark cluster
./infrastructure/local/scripts/run_spark_local_demo.sh
```

### 2. AWS EMR Infrastructure

#### AWS Setup
```bash
# Setup AWS environment
./infrastructure/aws_emr/scripts/setup_aws.sh

# Upload data to S3
./infrastructure/aws_emr/scripts/upload_inputs_to_s3.sh
```

### 3. AWS Glue Infrastructure

#### Glue Job Deployment
```bash
# Deploy Glue job
./infrastructure/aws_glue/scripts/deploy_glue_tifffile.sh
```

#### Glue Job Configuration
```json
{
  "job_name": "FIS-Processing-Job",
  "script_location": "s3://bucket/scripts/glue_fis_tifffile_job.py",
  "python_version": "3",
  "worker_type": "G.1X",
  "number_of_workers": 10,
  "timeout": 2880
}
```

## 📊 Infrastructure Comparison

### Performance Characteristics

| Infrastructure | Setup Time | Processing Time | Cost | Scalability | Complexity |
|----------------|------------|-----------------|------|-------------|------------|
| **Local** | 5 min | 9.7 min (1000m) | $0 | Low | Low |
| **Local Multiprocessing** | 5 min | 17.5 min (300m) | $0 | Medium | Low |
| **Local Spark** | 10 min | 45 min (30m) | $0 | Medium | Medium |
| **AWS EMR** | 15 min | 30 min (30m) | $50-100 | High | High |
| **AWS Glue** | 2 min | 40 min (30m) | $30-60 | Medium | Low |

### Resource Requirements

| Infrastructure | CPU Cores | Memory | Storage | Network |
|----------------|-----------|--------|---------|---------|
| **Local** | 1 | 4GB | 100GB | N/A |
| **Local Multiprocessing** | 8 | 16GB | 500GB | N/A |
| **Local Spark** | 8 | 32GB | 1TB | Local |
| **AWS EMR** | 16+ | 64GB+ | S3 | High |
| **AWS Glue** | 10+ | 40GB+ | S3 | High |

## 🔧 Configuration Management

### Environment Variables
```bash
# Local environment
export FIS_DATA_DIR=/path/to/data
export FIS_OUTPUT_DIR=/path/to/output
export FIS_NUM_CORES=8

# AWS environment
export AWS_REGION=us-east-2
export AWS_PROFILE=research
export S3_BUCKET=<AWS-BUCKET>
```

### Docker Configurations
```dockerfile
# Example: Tifffile Dockerfile
FROM python:3.9-slim

# Install minimal dependencies
RUN pip install -r requirements-glue-tifffile.txt

# Copy application code
COPY app/ /app/
WORKDIR /app

CMD ["python", "raster_fuzzy_lib_tifffile.py"]
```

## 🔒 Security and Access

### AWS IAM Configuration
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "emr:*",
        "glue:*"
      ],
      "Resource": "*"
    }
  ]
}
```

## 📈 Monitoring and Logging

### Performance Monitoring
```bash
# Monitor local performance
python infrastructure/local/scripts/monitor_local.py

# Monitor EMR performance
python infrastructure/aws_emr/scripts/monitor_emr.py

# Monitor Glue performance
python infrastructure/aws_glue/scripts/monitor_glue.py
```

## 🚀 Deployment Workflows

### Automated Deployment
```bash
# Deploy complete infrastructure
./infrastructure/deploy_all.sh \
  --environment production \
  --region us-east-2 \
  --config infrastructure_config.json
```

## 📖 Documentation

### Infrastructure Guides
- [Local Setup Guide](local/setup_guide.md)
- [EMR Deployment Guide](aws_emr/deployment_guide.md)
- [Glue Deployment Guide](aws_glue/deployment_guide.md)

### Troubleshooting
- [Common Issues](troubleshooting/common_issues.md)
- [Performance Tuning](troubleshooting/performance_tuning.md)
- [Cost Optimization](troubleshooting/cost_optimization.md) 