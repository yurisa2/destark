# EMR Tifffile Testing Guide

## Overview
This guide provides step-by-step instructions for testing tifffile-based raster processing on AWS EMR clusters. The setup is optimized for cost-effectiveness and performance while maintaining compatibility with your existing infrastructure.

## 🚀 Quick Start

### 1. Create Optimized EMR Cluster
```bash
./scripts/create_emr_tifffile_cluster.sh
```

### 2. Test Basic Functionality
```bash
# Once cluster is ready, get the cluster ID and test
./scripts/test_emr_tifffile_simple.sh <CLUSTER_ID> us-east-2
```

### 3. Test Full S3 Integration
```bash
# After basic test passes, run full S3 test
./scripts/test_emr_tifffile_s3.sh <CLUSTER_ID> us-east-2
```

## 📋 Cluster Configuration

### Optimized Specifications
- **Instance Types**: m5.xlarge (4 vCPU, 16 GB RAM)
- **Instance Count**: 1 Master + 1 Core + 1 Task
- **EBS Configuration**: 1 x 50GB gp3 per instance
- **Total Cost**: ~$2-3/hour (vs ~$6-8/hour for original config)

### Key Optimizations
1. **Smaller Instances**: m5.xlarge instead of m5.2xlarge
2. **Reduced Count**: 3 instances total instead of 4
3. **Smaller Storage**: 50GB instead of 100GB
4. **No GDAL**: Faster installation, no compilation needed
5. **Optimized Spark**: Configured for smaller instances

### Preserved Configuration
- ✅ Subnet: subnet-<SUBNET-ID>
- ✅ Security Groups: sg-<SECURITY-GROUP-ID>, sg-<SERVICE-SECURITY-GROUP-ID>
- ✅ IAM Roles: <EMR-ROLE-NAME>, <EMR-EC2-ROLE-NAME>
- ✅ Key Pair: <KEY-PAIR-NAME>
- ✅ Region: us-east-2
- ✅ Release Label: emr-7.9.0

## 🔧 Installation Steps

The cluster automatically installs:

### System Packages
- git, python3-pip, python3-devel
- gcc, gcc-c++, make
- java-17-amazon-corretto-devel
- unzip, wget, curl

### Python Packages (Tifffile-based)
- tifffile>=2023.0.0 (no GDAL dependency)
- scikit-fuzzy>=0.4.2
- numpy>=1.21.0
- scipy>=1.7.0
- boto3>=1.26.0
- pandas>=1.3.0
- networkx>=2.6.0
- matplotlib>=3.5.0
- tqdm>=4.62.0
- pyspark>=3.4.0

### Development Environment
- Creates `/mnt/destark` directory structure
- Sets up app, scripts, logs, data, config directories
- Configures proper permissions

## 🧪 Testing Workflow

### Step 1: Basic Tifffile Test
```bash
./scripts/test_emr_tifffile_simple.sh <CLUSTER_ID> us-east-2
```

**What it tests:**
- ✅ tifffile installation and import
- ✅ Basic read/write operations
- ✅ NumPy integration
- ✅ File system operations

### Step 2: S3 Integration Test
```bash
./scripts/test_emr_tifffile_s3.sh <CLUSTER_ID> us-east-2
```

**What it tests:**
- ✅ S3 download/upload operations
- ✅ Large raster file processing (1% crop)
- ✅ Weighted combination processing
- ✅ Result validation and cleanup

### Step 3: Full Processing Test
```bash
# Upload your processing script and run
aws s3 cp app/raster_fuzzy_spark_s3_tifffile.py s3://<AWS-BUCKET>-unifile/unifile_test/
# Then add EMR step to run the full processing
```

## 📊 Performance Comparison

| Metric | Original Config | Optimized Config | Improvement |
|--------|----------------|------------------|-------------|
| **Instance Type** | m5.2xlarge | m5.xlarge | 50% cost reduction |
| **Instance Count** | 4 (1M+2C+1T) | 3 (1M+1C+1T) | 25% cost reduction |
| **EBS Storage** | 800GB total | 150GB total | 81% cost reduction |
| **Installation Time** | ~15-20 min | ~8-10 min | 50% faster |
| **Memory per Node** | 32GB | 16GB | Sufficient for testing |
| **Processing Speed** | Full speed | 70-80% of full | Good for development |

## 🔍 Monitoring and Debugging

### Check Cluster Status
```bash
aws emr describe-cluster --cluster-id <CLUSTER_ID> --region us-east-2
```

### List Steps
```bash
aws emr list-steps --cluster-id <CLUSTER_ID> --region us-east-2
```

### Check Step Status
```bash
aws emr describe-step --cluster-id <CLUSTER_ID> --step-id <STEP_ID> --region us-east-2
```

### Connect to Master Node
```bash
aws emr ssh --cluster-id <CLUSTER_ID> --key-pair-file <KEY_FILE> --region us-east-2
```

### View Step Logs
```bash
# After SSH connection
tail -f /mnt/var/log/hadoop/steps/<STEP_ID>/stdout
tail -f /mnt/var/log/hadoop/steps/<STEP_ID>/stderr
```

## 🛠️ Troubleshooting

### Common Issues

#### 1. Permission Denied
```bash
# Check your AWS credentials and permissions
aws sts get-caller-identity
aws emr list-clusters --region us-east-2
```

#### 2. Cluster Not Ready
```bash
# Wait for cluster to be in WAITING or RUNNING state
aws emr describe-cluster --cluster-id <CLUSTER_ID> --region us-east-2
```

#### 3. Step Failed
```bash
# Check step logs for specific errors
aws emr describe-step --cluster-id <CLUSTER_ID> --step-id <STEP_ID> --region us-east-2
```

#### 4. Package Installation Issues
```bash
# SSH to master node and check manually
python3 -c "import tifffile; print('tifffile works')"
python3 -c "import skfuzzy; print('skfuzzy works')"
```

### Cost Optimization Tips

1. **Terminate when done**: Use `TERMINATE_AT_TASK_COMPLETION`
2. **Monitor usage**: Check AWS Cost Explorer
3. **Use spot instances**: For non-critical workloads
4. **Right-size instances**: Start small, scale up if needed

## 📁 Expected Output Files

### S3 Locations
- **Simple Test**: No output files (just verification)
- **S3 Test**: `s3://<AWS-BUCKET>-unifile/unifile_test/result_emr_tifffile_1pct.tif`
- **Full Processing**: `s3://<AWS-BUCKET>-unifile/unifile_test/result_emr_tifffile_full.tif`

### File Characteristics
- **Format**: TIFF (using tifffile)
- **Data Type**: float32
- **Compression**: None (raw data)
- **Size**: ~8.8MB for 1% test, ~880MB for full processing

## 🎯 Next Steps

### For Development
1. ✅ Use the simple test for quick validation
2. ✅ Use the S3 test for integration testing
3. ✅ Scale up instances if needed for production

### For Production
1. 🔄 Increase instance sizes (m5.2xlarge or larger)
2. 🔄 Add more core/task instances
3. 🔄 Implement proper error handling and monitoring
4. 🔄 Add data validation and quality checks
5. 🔄 Consider using spot instances for cost optimization

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review step logs for specific error messages
3. Verify AWS permissions and credentials
4. Contact your AWS administrator if needed

---

**Note**: This setup is optimized for testing and development. For production workloads, consider scaling up the instance types and counts based on your specific requirements. 