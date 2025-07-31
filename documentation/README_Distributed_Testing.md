# Distributed Spark FIS Processing Testing

This guide provides step-by-step instructions for testing distributed Spark FIS processing in a Docker container that mimics the EMR environment.

## Overview

We're building a Docker container that:
1. ✅ **Mimics EMR 7.9.0** environment (Java 8, Spark 3.4.1)
2. ✅ **Runs distributed Spark** across multiple nodes
3. ✅ **Integrates with S3** for data access
4. ✅ **Uses your existing FIS logic** and configs
5. ✅ **Tests Java 8 compatibility** before EMR deployment

## Prerequisites

- Docker Desktop running
- AWS credentials (already configured in the script)
- Internet connection for building the container

## Quick Start

### 1. Build and Run the Container

```bash
chmod +x scripts/build_and_run_distributed.sh
./scripts/build_and_run_distributed.sh
```

This will:
- Build the EMR-like Docker image
- Set AWS credentials
- Start the container with distributed Spark

### 2. Inside the Container - Start Distributed Spark

Once inside the container, you'll see the startup menu. Run these commands in sequence:

```bash
# Start Spark Master
./start-spark-master.sh

# In a new terminal, start Spark Worker
./start-spark-worker.sh

# Check Spark UI at http://localhost:8080
```

### 3. Run Distributed FIS Processing

```bash
# Run the distributed FIS processing
python3 emr_fis_distributed.py
```

## Step-by-Step Process

### Step 1: Build the Container

The build script will:
- Download Amazon Linux 2 base image
- Install Java 8 (Amazon Corretto)
- Install Spark 3.4.1 (same as EMR 7.9.0)
- Install Python packages (rasterio, scikit-fuzzy, boto3)
- Set up environment variables

### Step 2: Start Distributed Spark

1. **Start Spark Master:**
   ```bash
   ./start-spark-master.sh
   ```
   - Starts Spark master on port 7077
   - Web UI available at http://localhost:8080

2. **Start Spark Worker:**
   ```bash
   ./start-spark-worker.sh
   ```
   - Connects worker to master
   - Provides distributed processing capability

### Step 3: Run FIS Processing

The distributed script will:
1. **Download data from S3:**
   - Config files (config_median.json, etc.)
   - Input raster files (so300m.in, e300m.in, s300m.in)

2. **Process with distributed Spark:**
   - Uses `spark://localhost:7077` master
   - Distributes blocks across worker nodes
   - Processes fuzzy logic on each block

3. **Upload results to S3:**
   - Saves processed rasters with `result_distributed_` prefix

## Files Created

### Core Scripts
- **`scripts/emr_fis_distributed.py`** - Distributed Spark FIS processing
- **`scripts/build_and_run_distributed.sh`** - Build and run script
- **`Dockerfile.emr-local`** - EMR-like Docker environment

### Spark Scripts (inside container)
- **`start-spark-master.sh`** - Start Spark master
- **`start-spark-worker.sh`** - Start Spark worker

## Expected Results

### Successful Build
```
=== BUILDING AND RUNNING DISTRIBUTED EMR ENVIRONMENT ===
AWS credentials set for region: us-east-2
Building Docker image...
✅ Docker image built successfully
Starting EMR local environment with distributed Spark...
```

### Successful Spark Startup
```
=== EMR 7.9.0 Local Environment ===
Spark Home: /usr/lib/spark
Java Home: /usr/lib/jvm/java-8-amazon-corretto
Python: /usr/bin/python3
Spark Version: version 3.4.1
```

### Successful Distributed Processing
```
=== DISTRIBUTED SPARK FIS PROCESSING STARTED ===
Processing config_median.json with distributed Spark...
Processing 14479x15187 raster with distributed Spark...
Created 22040 blocks for distributed processing
Distributing 50 partitions across cluster...
✅ SUCCESS: config_median.json
```

## Monitoring

### Spark Web UI
- **Master UI:** http://localhost:8080
- **Application UI:** http://localhost:4040 (when running)

### Container Logs
```bash
# View container logs
docker logs emr-local-spark

# Follow logs in real-time
docker logs -f emr-local-spark
```

## Troubleshooting

### Common Issues

1. **Docker not running:**
   ```bash
   # Start Docker Desktop
   # Or on Linux:
   sudo systemctl start docker
   ```

2. **Port conflicts:**
   ```bash
   # Check if ports are in use
   lsof -i :8080
   lsof -i :7077
   ```

3. **AWS credentials:**
   - Credentials are hardcoded in the build script
   - Ensure they have S3 read/write permissions

4. **Spark connection issues:**
   ```bash
   # Check if Spark master is running
   curl http://localhost:8080
   
   # Check worker connection
   curl http://localhost:8081
   ```

### Debugging Steps

1. **Check container status:**
   ```bash
   docker ps
   docker exec -it emr-local-spark bash
   ```

2. **Check Spark processes:**
   ```bash
   # Inside container
   ps aux | grep spark
   jps
   ```

3. **Check S3 connectivity:**
   ```bash
   # Inside container
   aws s3 ls s3://<AWS-BUCKET>-unifile/unifile_test/
   ```

## Next Steps

Once distributed processing works in the container:

1. **Verify Java 8 compatibility** - No `UnsupportedClassVersionError`
2. **Test S3 integration** - Data download/upload works
3. **Confirm distributed processing** - Multiple nodes working
4. **Deploy to EMR** - Use the working configuration

## EMR Deployment

After successful container testing, deploy to EMR using:

```bash
./scripts/deploy_emr_fixed.sh <cluster-id>
```

This will use the same working configuration that was tested in the container.

## Support

If you encounter issues:

1. Check the troubleshooting section above
2. Verify Docker is running and has sufficient resources
3. Ensure AWS credentials have proper S3 permissions
4. Check Spark UI for detailed error messages
5. Review container logs for specific error details 