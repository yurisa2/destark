# Local Testing for Spark FIS Processing

This guide provides instructions for testing the Spark FIS (Fuzzy Inference System) processing locally without Docker, which can help debug the Java version issues encountered on EMR.

## Overview

Since we encountered persistent Java version compatibility issues on EMR (`UnsupportedClassVersionError`), we've created a local testing environment to:

1. Verify the fuzzy logic works correctly
2. Test the data processing pipeline
3. Debug any issues before deploying to EMR
4. Ensure all dependencies are properly installed

## Prerequisites

- Python 3.7 or higher
- pip3 package manager
- AWS credentials (for S3 access)

## Quick Start

### 1. Set up AWS Credentials

```bash
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_DEFAULT_REGION="us-east-2"
```

### 2. Run Environment Setup

```bash
python3 scripts/setup_local_environment.py
```

This script will:
- Check Python version
- Install missing dependencies
- Test AWS credentials
- Verify the fuzzy system works

### 3. Test the Fuzzy System

```bash
python3 scripts/test_spark_fis_local.py
```

This will test the core fuzzy logic with sample data.

### 4. Run Full Local Processing (Optional)

```bash
python3 scripts/spark_fis_local.py
```

This will process the actual data from S3 using local Spark.

## Files Created

### Core Scripts

- **`scripts/setup_local_environment.py`** - Environment setup and dependency checker
- **`scripts/test_spark_fis_local.py`** - Fuzzy system test with sample data
- **`scripts/spark_fis_local.py`** - Full Spark FIS processing script

### Docker Files (Alternative)

- **`Dockerfile.emr-local`** - Docker image mimicking EMR 7.9.0
- **`docker-compose.emr-local.yml`** - Docker Compose configuration
- **`scripts/build_and_run_emr_local.sh`** - Docker build and run script

## Dependencies

The following Python packages are required:

- `numpy` - Numerical computing
- `scipy` - Scientific computing
- `rasterio` - Geospatial raster I/O
- `scikit-fuzzy` - Fuzzy logic toolkit
- `boto3` - AWS SDK for Python
- `pyspark` - Apache Spark for Python

## Troubleshooting

### Common Issues

1. **Missing Dependencies**
   ```bash
   pip3 install numpy scipy rasterio scikit-fuzzy boto3 pyspark
   ```

2. **AWS Credentials Not Set**
   ```bash
   export AWS_ACCESS_KEY_ID="your-key"
   export AWS_SECRET_ACCESS_KEY="your-secret"
   ```

3. **Permission Issues**
   ```bash
   chmod +x scripts/*.py
   ```

4. **Python Version Issues**
   - Ensure Python 3.7+ is installed
   - Use `python3` instead of `python`

### Docker Issues

If Docker is not working:

1. **Docker not running**
   ```bash
   # Start Docker Desktop or Docker daemon
   sudo systemctl start docker  # Linux
   # Or start Docker Desktop on macOS/Windows
   ```

2. **Permission issues**
   ```bash
   sudo usermod -aG docker $USER
   # Log out and back in
   ```

3. **Build failures**
   - Check internet connection
   - Ensure sufficient disk space
   - Try building without cache: `docker build --no-cache -f Dockerfile.emr-local .`

## Testing Strategy

### 1. Environment Validation
Run the setup script to ensure all dependencies are available:
```bash
python3 scripts/setup_local_environment.py
```

### 2. Fuzzy Logic Testing
Test the core fuzzy inference system:
```bash
python3 scripts/test_spark_fis_local.py
```

### 3. Data Processing Test
Test with actual S3 data:
```bash
python3 scripts/spark_fis_local.py
```

### 4. EMR Comparison
Once local testing works, compare with EMR behavior to identify Java version issues.

## Expected Output

### Successful Setup
```
=== LOCAL ENVIRONMENT SETUP AND TEST ===
=== CHECKING PYTHON VERSION ===
Python 3.9.7
✅ Python version is compatible

=== CHECKING REQUIRED PACKAGES ===
✅ numpy is available
✅ scipy is available
✅ rasterio is available
✅ scikit-fuzzy is available
✅ boto3 is available
✅ pyspark is available

=== CHECKING AWS CREDENTIALS ===
✅ AWS credentials are set
Region: us-east-2

=== TESTING FUZZY SYSTEM ===
✅ Fuzzy system test passed
Output: Fuzzy test result: 3.0

=== SUMMARY ===
✅ Environment is ready for Spark FIS processing!
```

### Successful Fuzzy Test
```
=== LOCAL FUZZY SYSTEM TEST ===
✅ rasterio available
✅ scikit-fuzzy available
✅ PySpark available
=== TESTING FUZZY SYSTEM ===
✅ Fuzzy system test successful
Input social: [[1. 2.]
 [3. 4.]]
Input environmental: [[2. 3.]
 [4. 5.]]
Input strategic: [[1.5 2.5]
 [3.5 4.5]]
Output: [[5. 5.]
 [5. 5.]]

=== ALL TESTS PASSED ===
The fuzzy system is working correctly!
```

## Next Steps

1. **If local testing works**: The issue is likely Java version compatibility on EMR
2. **If local testing fails**: Fix the local issues first, then address EMR deployment
3. **For EMR deployment**: Use the working local configuration to create EMR-compatible scripts

## EMR Deployment After Local Testing

Once local testing is successful, you can:

1. Use the working configuration to create EMR-compatible scripts
2. Ensure Java 8 compatibility on EMR
3. Use the same Python package versions locally and on EMR
4. Test with smaller datasets first

## Support

If you encounter issues:

1. Check the troubleshooting section above
2. Verify all dependencies are installed
3. Ensure AWS credentials are properly set
4. Check Python version compatibility
5. Review error messages for specific package issues 