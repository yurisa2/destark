# Running on Existing EMR Cluster

Since you already have an EMR cluster running and the code is on the master node, here's how to run your raster fuzzy inference system.

## Quick Start

### 1. SSH to EMR Master Node

```bash
# SSH to your EMR master node
ssh -i your-key.pem hadoop@<master-node-public-dns>
```

### 2. Navigate to Your Code

```bash
# Navigate to where your code is located
cd /opt/raster-fuzzy  # or wherever you uploaded your code
```

### 3. Run the Processing

#### **Option A: Using the Helper Script**

```bash
# Basic usage with S3 files
./scripts/run_on_emr_master.sh \
    s3://your-bucket/input/social.tif \
    s3://your-bucket/input/environmental.tif \
    s3://your-bucket/input/strategic.tif \
    s3://your-bucket/output/result.tif

# With custom parameters
./scripts/run_on_emr_master.sh \
    s3://your-bucket/input/social.tif \
    s3://your-bucket/input/environmental.tif \
    s3://your-bucket/input/strategic.tif \
    s3://your-bucket/output/result.tif \
    raster_fis_config.json \
    2000 \
    16
```

#### **Option B: Direct Python Command**

```bash
# Run directly with Python
python3 app/raster_fuzzy_spark_ultra_optimized.py \
    s3://your-bucket/input/social.tif \
    s3://your-bucket/input/environmental.tif \
    s3://your-bucket/input/strategic.tif \
    s3://your-bucket/output/result.tif \
    --config raster_fis_config.json \
    --block-size 1000 \
    --partitions 8 \
    --verbose
```

## Command Line Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `social_tiff` | Path to social factor TIFF file | Required |
| `environmental_tiff` | Path to environmental factor TIFF file | Required |
| `strategic_tiff` | Path to strategic factor TIFF file | Required |
| `output_tiff` | Path for output TIFF file | Required |
| `--config` | Configuration JSON file | `raster_fis_config.json` |
| `--block-size` | Number of rows per block | `1000` |
| `--partitions` | Number of Spark partitions | `8` |
| `--local` | Run in local mode for testing | `false` |
| `--verbose` | Enable verbose output | `false` |

## Examples

### **Small Dataset (Testing)**

```bash
# Local mode with smaller chunks
python3 app/raster_fuzzy_spark_ultra_optimized.py \
    small_social.tif \
    small_env.tif \
    small_strat.tif \
    small_output.tif \
    --local \
    --block-size 500 \
    --partitions 4
```

### **Large Dataset (Production)**

```bash
# Cluster mode with optimized parameters
python3 app/raster_fuzzy_spark_ultra_optimized.py \
    s3://bucket/large_social.tif \
    s3://bucket/large_env.tif \
    s3://bucket/large_strat.tif \
    s3://bucket/large_output.tif \
    --block-size 2000 \
    --partitions 16 \
    --verbose
```

### **Custom Configuration**

```bash
# Using custom fuzzy logic configuration
python3 app/raster_fuzzy_spark_ultra_optimized.py \
    s3://bucket/social.tif \
    s3://bucket/env.tif \
    s3://bucket/strat.tif \
    s3://bucket/output.tif \
    --config my_custom_config.json \
    --block-size 1500 \
    --partitions 12
```

## Monitoring

### **Spark Web UI**
- Access: `http://<master-node>:8080`
- Monitor job progress, executors, and performance

### **YARN Web UI**
- Access: `http://<master-node>:8088`
- View resource usage and application status

### **Logs**
```bash
# View application logs
tail -f /var/log/hadoop-yarn/yarn-yarn-resourcemanager-*.log

# View Spark logs
tail -f /var/log/spark/spark-*.log
```

## Troubleshooting

### **Common Issues**

1. **Permission Denied**
   ```bash
   # Make scripts executable
   chmod +x scripts/*.sh
   ```

2. **Missing Dependencies**
   ```bash
   # Install Python dependencies
   pip3 install -r requirements-spark.txt
   ```

3. **Memory Issues**
   ```bash
   # Reduce block size or partitions
   --block-size 500 --partitions 4
   ```

4. **S3 Access Issues**
   ```bash
   # Check AWS credentials
   aws sts get-caller-identity
   ```

### **Debug Mode**

```bash
# Run with debug output
python3 app/raster_fuzzy_spark_ultra_optimized.py \
    input1.tif input2.tif input3.tif output.tif \
    --verbose --local
```

## Performance Tips

### **For Large Datasets**
- Increase `--block-size` to 2000-5000
- Increase `--partitions` to match available cores
- Use S3 for input/output to avoid disk space issues

### **For Small Datasets**
- Use `--local` mode for faster startup
- Reduce `--block-size` to 500-1000
- Use fewer partitions (4-8)

### **Memory Optimization**
- Monitor memory usage in Spark Web UI
- Adjust `--block-size` based on available memory
- Consider using `--local` mode for memory-constrained environments

## Cleanup

```bash
# Clean up temporary files
rm -rf /tmp/raster_processing/*

# Stop Spark session (if needed)
pkill -f spark
``` 