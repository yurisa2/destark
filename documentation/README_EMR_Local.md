# EMR 7.9.0 Local Environment

This setup creates a local Docker environment that mimics AWS EMR 7.9.0 for testing Spark FIS processing locally.

## Prerequisites

- Docker installed and running
- AWS credentials configured (for S3 access)
- At least 8GB RAM available for Docker

## Quick Start

1. **Set your AWS credentials:**
   ```bash
   export AWS_ACCESS_KEY_ID="your-access-key"
   export AWS_SECRET_ACCESS_KEY="your-secret-key"
   export AWS_DEFAULT_REGION="us-east-2"
   ```

2. **Build and run the environment:**
   ```bash
   ./scripts/build_and_run_emr_local.sh
   ```

3. **Inside the container, run the Spark FIS processing:**
   ```bash
   python3 scripts/spark_fis_local.py
   ```

## What's Included

### Docker Environment
- **Base OS:** Amazon Linux 2 (same as EMR)
- **Java:** Amazon Corretto 8 (same as EMR)
- **Spark:** 3.4.1 (same as EMR 7.9.0)
- **Python:** 3.9 with all required packages

### Python Packages
- `numpy` - Numerical computing
- `scipy` - Scientific computing
- `rasterio` - Geospatial raster I/O
- `scikit-fuzzy` - Fuzzy logic
- `boto3` - AWS SDK
- `pyspark==3.4.1` - Spark Python API

### Available Commands

Inside the container, you can use:

```bash
# Check Spark version
spark-submit --version

# Start PySpark shell
pyspark

# Run Spark FIS processing
python3 scripts/spark_fis_local.py

# Check environment
echo "Spark Home: $SPARK_HOME"
echo "Java Home: $JAVA_HOME"
```

## Ports

The container exposes these ports:
- **4040** - Spark Application UI
- **8080** - Spark Master UI  
- **7077** - Spark Master

## Volumes

The following directories are mounted:
- `./scripts` → `/opt/destark/scripts`
- `./data` → `/opt/destark/data`
- `./logs` → `/opt/destark/logs`

## Troubleshooting

### Java Version Issues
If you encounter Java version conflicts, this environment uses:
- Java 8 (Amazon Corretto) - compatible with EMR
- Spark 3.4.1 - matches EMR 7.9.0

### Memory Issues
If you get out of memory errors:
1. Increase Docker memory limit (8GB+ recommended)
2. Reduce block size in the script (currently 100x100 pixels)
3. Process fewer configurations at once

### S3 Access Issues
Make sure your AWS credentials are properly set:
```bash
export AWS_ACCESS_KEY_ID="your-key"
export AWS_SECRET_ACCESS_KEY="your-secret"
export AWS_DEFAULT_REGION="us-east-2"
```

## Differences from EMR

This local environment:
- Uses `local[*]` Spark master (single node)
- Processes data on local CPU cores
- Downloads data from S3 (same as EMR)
- Uploads results to S3 (same as EMR)

For true distributed processing, you'll still need to run on EMR cluster.

## Next Steps

Once the local environment works:
1. Test with smaller datasets first
2. Verify all packages work correctly
3. Run the full processing pipeline
4. Deploy the working script to EMR

## Files

- `Dockerfile.emr-local` - Docker image definition
- `docker-compose.emr-local.yml` - Docker Compose configuration
- `scripts/spark_fis_local.py` - Local Spark FIS processing script
- `scripts/build_and_run_emr_local.sh` - Build and run script 