# EMR Development Cluster Setup

This guide explains how to create and connect to an EMR cluster optimized for development work on the adveng processing project.

## Prerequisites

1. AWS CLI configured with appropriate permissions
2. SSH key pair named `<KEY-PAIR-NAME>` in your AWS account
3. The `.pem` file for the key pair on your local machine

## Creating the Development Cluster

### Option 1: Using the Script (Recommended)

```bash
# Make the script executable (if not already done)
chmod +x scripts/create_development_cluster.sh

# Create the cluster
./scripts/create_development_cluster.sh
```

### Option 2: Manual Command

You can also run the AWS CLI command directly from the script file.

## What the Cluster Includes

### System Tools Installed
- **Git** - Version control
- **Python 3** with pip - Python development
- **GCC/G++** - C/C++ compilation
- **GDAL** - Geospatial data abstraction library
- **PROJ** - Cartographic projections library
- **GEOS** - Geometry engine
- **HDF5** - Hierarchical data format
- **NetCDF** - Network Common Data Form
- **Java 17** - Java runtime environment
- **Unzip, wget, curl** - Utility tools

### Python Packages Installed
- **NumPy** (≥1.21.0) - Numerical computing
- **SciPy** (≥1.7.0) - Scientific computing
- **Pandas** (≥1.3.0) - Data manipulation
- **Rasterio** (≥1.3.0) - Geospatial I/O
- **Fiona** (≥1.8.0) - Vector I/O
- **Shapely** (≥1.8.0) - Geometric operations
- **PyProj** (≥3.2.0) - Cartographic projections
- **Scikit-fuzzy** (≥0.4.2) - Fuzzy logic
- **Boto3** (≥1.26.0) - AWS SDK
- **S3fs** (≥2022.11.0) - S3 file system
- **Matplotlib** (≥3.5.0) - Plotting
- **Seaborn** (≥0.11.0) - Statistical plotting
- **PySpark** (≥3.4.0) - Spark Python API
- **NetworkX** (≥2.8.0) - Network analysis
- **TQDM** (≥4.62.0) - Progress bars

### Cluster Configuration
- **EMR Release**: 7.9.0
- **Instance Types**: m5.2xlarge (Master, Core, Task nodes)
- **Storage**: 100GB GP3 EBS volumes (2 per instance)
- **Applications**: Hadoop, Hive, JupyterEnterpriseGateway, Livy, Spark
- **Security**: Uses existing security groups and IAM roles

## Connecting to the Cluster

### Step 1: Get the Cluster ID

After running the creation script, note the cluster ID from the output. It will look like `j-XXXXXXXXX`.

### Step 2: Connect Using the Helper Script

```bash
# Make the script executable (if not already done)
chmod +x scripts/connect_to_cluster.sh

# Connect to the cluster (replace with your actual cluster ID)
./scripts/connect_to_cluster.sh j-XXXXXXXXX
```

### Step 3: Manual Connection

If you prefer to connect manually:

```bash
# Get cluster details
aws emr describe-cluster --cluster-id j-XXXXXXXXX --region us-east-2

# Extract the master node DNS and connect
ssh -i /path/to/<KEY-PAIR-NAME>.pem hadoop@<master-public-dns>
```

## Development Environment

Once connected to the cluster, you'll find:

- **Development Directory**: `/mnt/destark/`
- **Structure**:
  ```
  /mnt/destark/
  ├── app/          # Application code
  ├── scripts/      # Scripts
  ├── logs/         # Log files
  ├── data/         # Data files
  ├── config/       # Configuration files
  ├── files/        # Input/output files
  │   ├── input/
  │   └── output/
  ├── utils/        # Utility functions
  └── docs/         # Documentation
  ```

## Working on the Cluster

### 1. Clone Your Repository
```bash
cd /mnt/destark
git clone https://github.com/your-username/destark.git .
```

### 2. Install Additional Dependencies
```bash
# If you have requirements files
pip3 install -r requirements-emr.txt
pip3 install -r requirements-spark.txt
```

### 3. Test Your Environment
```bash
# Test Python packages
python3 -c "import rasterio, skfuzzy, boto3, pyspark; print('All packages working!')"

# Test system tools
gdal-config --version
proj --version
```

### 4. Run Your Code
```bash
# Example: Run a Python script
python3 app/adveng_processing.py

# Example: Submit a Spark job
spark-submit app/adveng_processing.py
```

## Monitoring and Management

### Check Cluster Status
```bash
aws emr describe-cluster --cluster-id j-XXXXXXXXX --region us-east-2
```

### View Step Logs
```bash
aws emr describe-step --cluster-id j-XXXXXXXXX --step-id s-XXXXXXXXX --region us-east-2
```

### Terminate Cluster
```bash
aws emr terminate-clusters --cluster-ids j-XXXXXXXXX --region us-east-2
```

## Troubleshooting

### Common Issues

1. **SSH Connection Failed**
   - Verify the key pair name matches `<KEY-PAIR-NAME>`
   - Ensure the `.pem` file has correct permissions (chmod 400)
   - Check security group allows SSH access

2. **Cluster Not Ready**
   - Wait for cluster to reach "WAITING" or "RUNNING" state
   - Check step logs for any installation failures

3. **Package Installation Issues**
   - Check the step logs for specific error messages
   - Some packages might need to be installed manually

### Getting Help

- Check EMR step logs in the AWS console
- Review CloudWatch logs for detailed error information
- Verify all required IAM permissions are in place

## Cost Optimization

- The cluster uses managed scaling to optimize costs
- Consider terminating the cluster when not in use
- Monitor usage through AWS Cost Explorer
- Use spot instances for non-critical workloads if needed 