# EMR Cluster Optimization Guide for Raster Processing

## Overview

This guide provides an optimized EMR cluster configuration specifically designed for raster processing workloads using Spark. The optimized configuration addresses the HDFS issues you encountered and provides better performance for raster data processing.

## 🚀 Quick Start

```bash
# Create the optimized cluster
./scripts/create_optimized_emr_cluster.sh

# After cluster is ready, run your raster processing
./scripts/run_all_fis_models_spark_submit_archive.sh \
  s3://adveng-pipeline/unifile_test/so300m.in \
  s3://adveng-pipeline/unifile_test/e300m.in \
  s3://adveng-pipeline/unifile_test/s300m.in \
  s3://adveng-pipeline/unifile_test/result_300m
```

## 📊 Configuration Comparison

### Instance Types & Resources

| Component | Original | Optimized | Improvement |
|-----------|----------|-----------|-------------|
| **Master** | m5.xlarge (4 vCPU, 16GB) | m5.2xlarge (8 vCPU, 32GB) | +100% CPU, +100% RAM |
| **Core** | m5.xlarge (4 vCPU, 16GB) | m5.2xlarge (8 vCPU, 32GB) | +100% CPU, +100% RAM |
| **Task** | m5.xlarge (4 vCPU, 16GB) | m5.2xlarge (8 vCPU, 32GB) | +100% CPU, +100% RAM |
| **Total Nodes** | 6 (1+2+3) | 6 (1+2+3) | Same count |
| **Total vCPUs** | 24 | 48 | +100% |
| **Total RAM** | 96GB | 192GB | +100% |

### EBS Storage

| Component | Original | Optimized | Improvement |
|-----------|----------|-----------|-------------|
| **Volume Type** | gp2 | gp3 | +20% IOPS, +125% throughput |
| **Volume Size** | 32GB | 100GB | +213% storage |
| **Volumes/Instance** | 2 | 2 | Same |
| **Total Storage** | 384GB | 1,200GB | +213% |

### Spark Configuration

| Setting | Original | Optimized | Improvement |
|---------|----------|-----------|-------------|
| **Driver Memory** | 1g | 8g | +700% |
| **Driver Cores** | 1 | 2 | +100% |
| **Executor Memory** | 12g | 20g | +67% |
| **Executor Cores** | 4 | 4 | Same |
| **Max Executors** | 200 | 8 | Optimized for workload |
| **Shuffle Partitions** | 1000 | 200 | Optimized for data size |
| **RPC Message Size** | Default | 512MB | Solves serialization issues |
| **Kryo Buffer** | Default | 512MB | Better serialization |

## 🔧 Key Optimizations

### 1. **HDFS Bypass Solution**
- **Problem**: Original cluster had HDFS DataNode issues
- **Solution**: Uses `spark.yarn.archive` with local JAR distribution
- **Benefit**: Eliminates HDFS dependency for Spark library distribution

### 2. **Java 17 Configuration**
- **Problem**: Java version conflicts in YARN containers
- **Solution**: Explicit Java 17 installation and configuration
- **Benefit**: Consistent Java environment across all nodes

### 3. **Raster Processing Dependencies**
- **Problem**: Missing geospatial libraries
- **Solution**: Comprehensive Python package installation including GDAL
- **Benefit**: All required libraries pre-installed and verified

### 4. **Memory Optimization**
- **Problem**: OutOfMemoryError in Spark executors
- **Solution**: Increased memory allocation and optimized GC settings
- **Benefit**: Better memory management for large raster processing

### 5. **Serialization Optimization**
- **Problem**: Large task serialization failures
- **Solution**: Kryo serializer with increased buffer sizes
- **Benefit**: Efficient data transfer between Spark components

## 📦 Python Package Installation

The optimized cluster automatically installs all required packages:

### Core Scientific Libraries
- `numpy>=1.21.0` - Numerical computing
- `scipy>=1.7.0` - Scientific computing
- `pandas>=1.3.0` - Data manipulation

### Geospatial Libraries
- `rasterio>=1.3.0` - Raster processing
- `fiona>=1.8.0` - Vector data access
- `shapely>=1.8.0` - Geometric operations
- `pyproj>=3.2.0` - Projection handling

### Fuzzy Logic & AWS
- `scikit-fuzzy>=0.4.2` - Fuzzy logic processing
- `boto3>=1.26.0` - AWS SDK
- `s3fs>=2022.11.0` - S3 file system

### System Dependencies
- GDAL development libraries
- PROJ projection library
- GEOS geometry library
- HDF5 and NetCDF libraries

## 🎯 Performance Benefits

### Expected Performance Improvements
- **Processing Speed**: 2-3x faster due to increased CPU and memory
- **Data Handling**: 3x more storage capacity for large rasters
- **Reliability**: Eliminated HDFS and Java version issues
- **Scalability**: Better resource utilization with optimized Spark settings

### Cost Optimization
- **Efficient Resource Usage**: Optimized Spark configurations reduce waste
- **Faster Processing**: Reduced cluster runtime costs
- **Better Reliability**: Fewer failed jobs and retries

## 🔍 Monitoring & Troubleshooting

### Cluster Health Check
```bash
# Check cluster status
aws emr describe-cluster --cluster-id <CLUSTER_ID> --region us-east-2

# Check node health
yarn node -list

# Check application status
yarn application -list
```

### Log Monitoring
```bash
# Monitor Spark application logs
tail -f logs/<model>_archive.log

# Check YARN application logs
yarn logs -applicationId <APP_ID>
```

### Performance Monitoring
```bash
# Check resource utilization
yarn application -status <APP_ID>

# Monitor Spark UI
# Access via: http://<MASTER_IP>:8080
```

## 🚨 Troubleshooting Common Issues

### 1. **HDFS Issues (Resolved)**
- **Symptom**: "0 datanode(s) running"
- **Solution**: Use `spark.yarn.archive` approach
- **Prevention**: Built into optimized configuration

### 2. **Java Version Issues (Resolved)**
- **Symptom**: "UnsupportedClassVersionError"
- **Solution**: Java 17 explicitly configured
- **Prevention**: Built into optimized configuration

### 3. **Memory Issues (Resolved)**
- **Symptom**: "OutOfMemoryError"
- **Solution**: Increased memory allocation and GC optimization
- **Prevention**: Built into optimized configuration

### 4. **Serialization Issues (Resolved)**
- **Symptom**: "Serialized task exceeds max allowed"
- **Solution**: Kryo serializer with large buffers
- **Prevention**: Built into optimized configuration

## 📋 Deployment Checklist

- [ ] Run `./scripts/create_optimized_emr_cluster.sh`
- [ ] Wait for cluster to reach "WAITING" state
- [ ] Verify Python packages installation in step logs
- [ ] Upload your code to the cluster
- [ ] Run `./scripts/run_all_fis_models_spark_submit_archive.sh`
- [ ] Monitor job progress and logs
- [ ] Verify output files in S3

## 💡 Best Practices

1. **Use the Archive Script**: Always use `run_all_fis_models_spark_submit_archive.sh` to avoid HDFS issues
2. **Monitor Resources**: Keep an eye on YARN resource utilization
3. **Scale Appropriately**: Adjust instance counts based on your data size
4. **Use Background Execution**: Jobs continue even if you disconnect
5. **Check Logs Regularly**: Monitor both Spark and YARN logs for issues

## 🔄 Scaling Options

### For Larger Datasets
- Increase `TASK_COUNT` in the script
- Use larger instance types (m5.4xlarge, m5.8xlarge)
- Increase EBS volume sizes

### For Faster Processing
- Use compute-optimized instances (c5.2xlarge, c5.4xlarge)
- Increase executor memory and cores
- Optimize block sizes in your Spark application

### For Cost Optimization
- Use spot instances for task nodes
- Implement auto-scaling policies
- Monitor and terminate unused clusters promptly 