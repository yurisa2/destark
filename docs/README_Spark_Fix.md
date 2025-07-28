# Spark Processing Fix: 'JavaPackage' object is not callable

## Problem Description

The Spark processing was failing with the error:
```
TypeError: 'JavaPackage' object is not callable
```

This error occurred during Spark session creation and was caused by Java/Python compatibility issues in the PySpark environment.

## Root Causes

1. **Java Version Mismatch**: PySpark 4.0.0 requires Java 17, but the system had Java 8
2. **Missing Environment Variables**: `JAVA_HOME` and `SPARK_HOME` were not properly set
3. **Incorrect Python Path**: Spark was looking for `/opt/conda/bin/python` which doesn't exist on local systems
4. **Log4j Configuration Issues**: Invalid log4j configuration paths were causing startup problems
5. **Spark-submit Path Issues**: Spark was trying to use `spark-submit` from non-existent `/opt/bitnami/spark` path

## Solutions Implemented

### 1. Downgraded PySpark Version
- Changed from PySpark 4.0.0 to PySpark 3.5.0 for Java 8 compatibility
- Updated py4j to version 0.10.9.7

### 2. Enhanced Environment Variable Management
```python
# Set critical environment variables for Java/Python compatibility
python_path = '/opt/conda/bin/python' if os.path.exists('/opt/conda/bin/python') else sys.executable
os.environ['PYSPARK_PYTHON'] = python_path
os.environ['PYSPARK_DRIVER_PYTHON'] = python_path

# Set Java environment variables if not already set
if 'JAVA_HOME' not in os.environ:
    java_paths = [
        '/usr/lib/jvm/java-11-openjdk-amd64',
        '/usr/lib/jvm/java-8-openjdk-amd64',
        '/opt/bitnami/java',
        '/usr/local/openjdk-11',
        '/usr/local/openjdk-8'
    ]
    for java_path in java_paths:
        if os.path.exists(java_path):
            os.environ['JAVA_HOME'] = java_path
            break
```

### 3. Fixed Python Path Detection
- Added automatic detection of Python executable path
- Uses system Python for local development, Docker Python for containerized environments

### 4. Improved Spark Configuration
```python
spark = SparkSession.builder \
    .appName(app_name) \
    .master("local[2]") \
    .config("spark.driver.memory", "2g") \
    .config("spark.executor.memory", "2g") \
    .config("spark.sql.shuffle.partitions", "5") \
    .config("spark.python.worker.python", python_path) \
    .config("spark.driver.extraJavaOptions", "-Dlog4j.configuration=file:///dev/null") \
    .config("spark.executor.extraJavaOptions", "-Dlog4j.configuration=file:///dev/null") \
    .config("spark.sql.adaptive.enabled", "false") \
    .config("spark.sql.adaptive.coalescePartitions.enabled", "false") \
    .config("spark.submit.deployMode", "client") \
    .config("spark.driver.allowMultipleContexts", "true") \
    .getOrCreate()
```

### 5. Fixed Spark-submit Issues
- Added conditional `SPARK_HOME` setting only when the path actually exists
- Added `spark.submit.deployMode` configuration to use client mode
- Added `spark.driver.allowMultipleContexts` to prevent context conflicts
- **For local mode**: Explicitly unset `SPARK_HOME` to prevent spark-submit usage
- Added additional Spark configurations to disable adaptive features that might trigger spark-submit

### 6. Updated Docker Configuration
- Added proper environment variables in `Dockerfile.jupyter`
- Set `JAVA_HOME`, `SPARK_HOME`, and Python paths

## Files Modified

1. **`app/raster_fuzzy_spark_simple.py`**
   - Updated `create_spark_session()` function
   - Added environment variable management
   - Fixed Python path detection

2. **`app/raster_fuzzy_spark.py`**
   - Updated `create_spark_session()` function
   - Applied same fixes as simple version

3. **`Dockerfile.jupyter`**
   - Added Spark and Java environment variables
   - Set proper Python paths

4. **`requirements-spark.txt`**
   - Downgraded PySpark to 3.5.0
   - Updated py4j to 0.10.9.7

## Testing

Created `app/test_spark_fix.py` to verify the fix:
- Tests local Spark session creation
- Tests basic DataFrame operations
- Tests RDD operations
- Tests cluster connection (when available)

## Results

✅ **Local mode**: PASSED - Spark session creates successfully
✅ **DataFrame operations**: WORKING - Can create and manipulate DataFrames
✅ **RDD operations**: WORKING - Can perform distributed operations
✅ **Error resolved**: No more `'JavaPackage' object is not callable` errors
✅ **Spark-submit error resolved**: No more `spark-submit` path errors
✅ **Local mode optimized**: Explicitly unsets SPARK_HOME to prevent external dependencies

## Current Status

✅ **Fixed Issues:**
- `'JavaPackage' object is not callable` error - RESOLVED
- `spark-submit` path errors - RESOLVED  
- Java version compatibility - RESOLVED
- Environment variable configuration - RESOLVED

⚠️ **Known Issues:**
- Cluster mode has SparkContext conflicts (investigation ongoing)
- Local mode works perfectly for testing and development

## Usage

### Recommended: Local Mode
For reliable processing, use local mode:

```bash
# Local mode (recommended)
scripts/run_spark_job.sh \
  app/files/input/base/socioeconomico_1000m.tif \
  app/files/input/base/ambiental_1000m.tif \
  app/files/input/base/estratégico_1000m.tif \
  app/files/output/base/output_1000m_round_down.tif \
  --config config_round_down.json \
  --partitions 8 \
  --local
```

### Cluster Mode (Experimental)
Cluster mode is available but may have SparkContext conflicts:

```bash
# Cluster mode (may have issues)
scripts/run_spark_job.sh \
  app/files/input/base/socioeconomico_1000m.tif \
  app/files/input/base/ambiental_1000m.tif \
  app/files/input/base/estratégico_1000m.tif \
  app/files/output/base/output_1000m_round_down.tif \
  --config config_round_down.json \
  --partitions 8
```

## Error Handling

The Spark system now fails fast if Spark processing encounters issues:
```python
except Exception as e:
    print(f"Spark processing failed: {e}")
    print("Spark processing failed. Please check your Spark configuration or use the dedicated local processing script.")
    sys.exit(1)
```

This ensures clear separation between Spark processing and local processing, with dedicated scripts for each approach. 