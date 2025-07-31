# Official EMR Distributed Spark Testing

This approach uses the **official EMR image** directly: `public.ecr.aws/emr-serverless/spark/emr-7.9.0:20250425`

## Why This Approach is Better

✅ **No custom installations** - Uses official EMR image as-is  
✅ **Exact EMR environment** - Same Java, Spark, Python versions  
✅ **All packages pre-installed** - rasterio, scikit-fuzzy, boto3, pyspark  
✅ **Java 8 compatible** - No version conflicts  
✅ **Simple and clean** - Minimal Dockerfile  

## Quick Start

### 1. Build and Run

```bash
chmod +x scripts/build_and_run_official.sh
./scripts/build_and_run_official.sh
```

### 2. Inside Container - Start Distributed Spark

```bash
# Start Spark Master
./start-spark-master.sh

# In another terminal, start Spark Worker
./start-spark-worker.sh

# Check Spark UI at http://localhost:8080
```

### 3. Run Distributed FIS Processing

```bash
python3 emr_fis_distributed_official.py
```

## What This Does

1. **Downloads from S3:**
   - Your 5 config files (config_median.json, etc.)
   - Input raster files (so300m.in, e300m.in, s300m.in)

2. **Processes with distributed Spark:**
   - Uses `spark://localhost:7077` master
   - Distributes blocks across worker nodes
   - Runs your existing FIS logic

3. **Uploads results to S3:**
   - Saves with `result_official_` prefix

## Files Created

- **`Dockerfile.emr-official`** - Uses official EMR image
- **`scripts/emr_fis_distributed_official.py`** - Distributed FIS processing
- **`scripts/build_and_run_official.sh`** - Build and run script

## Expected Results

```
=== OFFICIAL EMR DISTRIBUTED SPARK FIS PROCESSING STARTED ===
Processing config_median.json with distributed Spark...
Processing 14479x15187 raster with distributed Spark...
Created 22040 blocks for distributed processing
Distributing 50 partitions across cluster...
✅ SUCCESS: config_median.json
```

## Monitoring

- **Spark Master UI:** http://localhost:8080
- **Application UI:** http://localhost:4040 (when running)

## Next Steps

Once this works in the container, deploy to real EMR using:

```bash
./scripts/deploy_emr_fixed.sh <cluster-id>
```

The official EMR image approach eliminates all the Java version and package installation issues! 