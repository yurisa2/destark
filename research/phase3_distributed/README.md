# Phase 3: Distributed Computing - 30m Resolution Preparation

## Research Context

After successfully scaling to 300m resolution using multiprocessing, we faced the ultimate challenge: preparing for 30m resolution processing. This represents a 100x increase in pixel count compared to our original 1000m data, making even multiprocessing insufficient. We turned to distributed computing with Apache Spark and cloud platforms.

## 🎯 Research Objectives

1. **Distributed Processing**: Implement Apache Spark-based distributed FIS processing
2. **Cloud Platform Evaluation**: Compare AWS EMR vs AWS Glue performance
3. **Scalability Testing**: Validate processing capabilities for 30m resolution
4. **Cost-Benefit Analysis**: Evaluate cloud computing costs vs performance gains

## 📊 Dataset Characteristics

- **Target Resolution**: 30m (high resolution for production)
- **Estimated Dimensions**: ~144,790 × 151,870 pixels
- **Estimated Pixels**: ~22 billion (100x increase from 1000m)
- **Estimated File Size**: ~87GB per output
- **Processing Time**: 45+ minutes (distributed)

## 🔬 Technical Challenges

### Challenge 1: Memory Limitations
- **Problem**: 30m data exceeds available RAM on single machines
- **Solution**: Distributed memory across multiple nodes
- **Implementation**: Apache Spark RDDs and DataFrames

### Challenge 2: Processing Distribution
- **Problem**: Need to distribute computation across multiple machines
- **Solution**: Spark cluster with master/worker architecture
- **Optimization**: Efficient data partitioning and load balancing

### Challenge 3: Cloud Platform Selection
- **Problem**: Choose optimal cloud platform for distributed processing
- **Evaluation**: AWS EMR vs AWS Glue comparison
- **Decision**: Platform-specific optimizations

## 🚀 Core Components

### Apache Spark Implementation

```python
def process_fis_spark(social_rdd, environmental_rdd, strategic_rdd, config):
    """
    Process FIS using Apache Spark distributed computing.
    """
    # Create Spark session
    spark = SparkSession.builder \
        .appName("FIS-Distributed-Processing") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.dynamicAllocation.enabled", "true") \
        .getOrCreate()
    
    # Process data in distributed manner
    result_rdd = social_rdd.zip(environmental_rdd).zip(strategic_rdd) \
        .map(lambda x: process_fis_pixel(x[0][0], x[0][1], x[1], config))
    
    return result_rdd
```

### Platform Configurations

#### AWS EMR Configuration
```bash
# EMR Cluster Setup
aws emr create-cluster \
  --name "FIS-Processing-Cluster" \
  --release-label "emr-7.9.0" \
  --applications Name=Spark \
  --instance-groups InstanceGroupType=MASTER,InstanceCount=1,InstanceType=m5.xlarge \
  --instance-groups InstanceGroupType=CORE,InstanceCount=4,InstanceType=m5.2xlarge
```

#### AWS Glue Configuration
```python
# Glue Job Configuration
glue_job_config = {
    "MaxConcurrentRuns": 1,
    "Timeout": 2880,  # 48 hours
    "MaxRetries": 0,
    "WorkerType": "G.1X",
    "NumberOfWorkers": 10
}
```

## 📈 Performance Results

### Platform Comparison

| Platform | Setup Time | Processing Time | Cost | Scalability | Ease of Use |
|----------|------------|-----------------|------|-------------|-------------|
| **Local Spark** | 5 min | 45 min | $0 | Limited | Medium |
| **AWS EMR** | 15 min | 30 min | $50-100 | High | High |
| **AWS Glue** | 2 min | 40 min | $30-60 | Medium | Very High |

### Scaling Analysis

| Resolution | Pixels | Processing | Time | Platform |
|------------|--------|------------|------|----------|
| **1000m** | 20M | Single-threaded | 9.7 min | Local |
| **300m** | 220M | Multiprocessing | 17.5 min | Local |
| **30m** | 22B | Distributed Spark | 45 min | Cloud |

## 🔍 Key Files

### Spark Implementations
- `raster_fuzzy_spark.py` - Basic Spark implementation
- `raster_fuzzy_spark_optimized.py` - Optimized Spark processing
- `raster_fuzzy_spark_ultra_optimized.py` - Ultra-optimized version

### Cloud Platform Files
- `glue_fis_tifffile_job.py` - AWS Glue implementation
- `emr_fis_distributed.py` - AWS EMR implementation
- `create_emr_tifffile_cluster.sh` - EMR cluster setup

### Infrastructure Scripts
- `scripts/create_*_cluster.sh` - Cluster creation scripts
- `scripts/run_*_emr.sh` - EMR execution scripts
- `Dockerfile.*` - Container configurations

## 🚀 Usage Examples

### Local Spark Processing
```bash
# Start local Spark cluster
./infrastructure/local/start_spark_local.sh

# Run FIS processing
python raster_fuzzy_spark.py \
  data/30m/social.tif \
  data/30m/environmental.tif \
  data/30m/strategic.tif \
  results/output_30m.tif \
  --config config/config_median.json \
  --local
```

### AWS EMR Processing
```bash
# Create and run EMR cluster
./infrastructure/aws_emr/create_and_run_distributed_cluster.sh \
  s3://bucket/30m/social.tif \
  s3://bucket/30m/environmental.tif \
  s3://bucket/30m/strategic.tif \
  s3://bucket/results/output_30m.tif
```

### AWS Glue Processing
```bash
# Deploy and run Glue job
./infrastructure/aws_glue/deploy_glue_tifffile.sh \
  s3://bucket/30m/social.tif \
  s3://bucket/30m/environmental.tif \
  s3://bucket/30m/strategic.tif \
  s3://bucket/results/output_30m.tif
```

## 📚 Research Contributions

1. **Distributed FIS Processing**: First implementation of distributed fuzzy inference system
2. **Cloud Platform Evaluation**: Comprehensive comparison of AWS EMR vs Glue
3. **Scalability Framework**: Framework for scaling geospatial processing to cloud
4. **Cost-Benefit Analysis**: Economic analysis of cloud computing for scientific processing
5. **Performance Optimization**: Advanced Spark optimizations for geospatial data

## 🔄 Research Evolution Summary

### Phase 1 → Phase 2 → Phase 3
- **1000m**: Single-threaded → **300m**: Multiprocessing → **30m**: Distributed Spark
- **20M pixels**: 9.7 min → **220M pixels**: 17.5 min → **22B pixels**: 45 min
- **Local processing**: 4GB RAM → **Local multiprocessing**: 12GB RAM → **Cloud distributed**: 40GB+ RAM

## ⚠️ Challenges and Solutions

### Challenge 1: Data Transfer Overhead
- **Problem**: Large data transfer to/from cloud
- **Solution**: S3 integration and data locality optimization

### Challenge 2: Cost Management
- **Problem**: Cloud computing costs
- **Solution**: Spot instances and auto-scaling

### Challenge 3: Complexity Management
- **Problem**: Distributed system complexity
- **Solution**: Automated deployment and monitoring

## 📖 Related Documentation

- [Cloud Platform Comparison](../experiments/cloud_platforms/emr_vs_glue.md)
- [Performance Analysis](../experiments/performance_benchmarks/phase3_analysis.md)
- [Cost Analysis](../results/analysis/cloud_cost_analysis.md)
- [Deployment Guide](../infrastructure/deployment_guide.md) 