# AWS Glue Deployment Guide

This guide explains how to test your raster fuzzy processing script locally and then deploy it to AWS Glue.

## Overview

The `raster_fuzzy_glue.py` script is designed to work both locally and on AWS Glue with minimal changes. It supports:

- S3 input/output paths
- AWS Glue's built-in Spark session
- Environment variable configuration
- Local testing with S3 path simulation

## Local Testing

### 1. Install Dependencies

```bash
pip install -r app/requirements-glue.txt
```

### 2. Run Local Tests

```bash
python scripts/test_glue_local.py
```

This will:
- Set up a test environment with sample data
- Test the script with local files
- Test the script with environment variables (AWS Glue mode)
- Create a deployment template

### 3. Manual Testing

Test with local files:
```bash
python app/raster_fuzzy_glue.py \
  app/files/input/base/socioeconomico_1000m.tif \
  app/files/input/base/ambiental_1000m.tif \
  app/files/input/base/estratégico_1000m.tif \
  output_test.tif \
  --config app/config/config_round_down.json
```

Test with S3 paths (requires AWS credentials):
```bash
python app/raster_fuzzy_glue.py \
  s3://your-bucket/input/social.tif \
  s3://your-bucket/input/env.tif \
  s3://your-bucket/input/strat.tif \
  s3://your-bucket/output/result.tif \
  --config s3://your-bucket/config/config.json
```

## AWS Glue Deployment

### 1. Prepare S3 Data

Upload your files to S3:
```bash
# Upload input files
aws s3 cp app/files/input/base/socioeconomico_1000m.tif s3://your-bucket/input/
aws s3 cp app/files/input/base/ambiental_1000m.tif s3://your-bucket/input/
aws s3 cp app/files/input/base/estratégico_1000m.tif s3://your-bucket/input/

# Upload config file
aws s3 cp app/config/config_round_down.json s3://your-bucket/config/
```

### 2. Create AWS Glue Job

1. Go to AWS Glue Console
2. Click "Jobs" → "Add job"
3. Choose "Spark" as job type
4. Set job name (e.g., "RasterFuzzyProcessing")

### 3. Configure Job Parameters

Add these parameters in the job configuration:

| Parameter | Value |
|-----------|-------|
| `SOCIAL_TIFF` | `s3://your-bucket/input/socioeconomico_1000m.tif` |
| `ENVIRONMENTAL_TIFF` | `s3://your-bucket/input/ambiental_1000m.tif` |
| `STRATEGIC_TIFF` | `s3://your-bucket/input/estratégico_1000m.tif` |
| `OUTPUT_TIFF` | `s3://your-bucket/output/result.tif` |
| `CONFIG_FILE` | `s3://your-bucket/config/config_round_down.json` |

### 4. Paste the Script

Copy the entire content of `app/raster_fuzzy_glue.py` and paste it into the Glue job script editor.

### 5. Configure Job Settings

- **Worker type**: Choose based on your data size:
  - Small datasets (< 1GB): G.1X
  - Medium datasets (1-10GB): G.2X
  - Large datasets (> 10GB): G.4X
- **Number of workers**: Start with 2-4, adjust based on performance
- **Job timeout**: Set to 30-60 minutes for large datasets
- **Max concurrency**: 1 (for this type of job)

### 6. Add Python Libraries

In the job configuration, add these Python libraries:
- `numpy>=1.21.0`
- `scipy>=1.7.0`
- `rasterio>=1.3.0`
- `scikit-fuzzy>=0.4.2`
- `boto3>=1.26.0`

### 7. Run the Job

1. Save the job
2. Click "Run job"
3. Monitor the job in the AWS Glue console

## Script Features

### Dual Mode Operation

The script automatically detects whether it's running in AWS Glue or locally:

```python
# AWS Glue mode (uses environment variables)
if os.environ.get('AWS_EXECUTION_ENV'):
    social_tiff = os.environ.get('SOCIAL_TIFF')
    # ... other variables

# Local mode (uses command line arguments)
else:
    social_tiff = args.social_tiff
    # ... other variables
```

### S3 Integration

The script automatically handles S3 paths:

```python
# Downloads S3 files locally for processing
if social_tiff.startswith('s3://'):
    download_from_s3(social_tiff, local_social)

# Uploads results back to S3
if output_tiff.startswith('s3://'):
    upload_to_s3(local_output, output_tiff)
```

### Spark Session Management

```python
# Uses AWS Glue's built-in Spark session
if os.environ.get('AWS_EXECUTION_ENV'):
    spark = SparkSession.builder.getOrCreate()
else:
    # Creates local Spark session for testing
    spark = SparkSession.builder.master("local[*]").getOrCreate()
```

## Performance Optimization

### Block Processing

The script processes rasters in blocks to manage memory:

```python
# Configurable block size
block_size = 1000  # rows per block

# Creates blocks for parallel processing
for start_row in range(0, height, block_size):
    end_row = min(start_row + block_size, height)
    blocks.append((start_row, end_row))
```

### Spark Partitioning

```python
# Configurable number of partitions
num_partitions = 8  # or auto-detect

# Creates RDD with optimal partitioning
blocks_rdd = spark.sparkContext.parallelize(blocks, numSlices=num_partitions)
```

## Troubleshooting

### Common Issues

1. **Memory Issues**: Reduce block size or increase worker memory
2. **Timeout**: Increase job timeout for large datasets
3. **S3 Access**: Ensure IAM roles have S3 read/write permissions
4. **Dependencies**: Check that all Python libraries are installed

### Debug Mode

Add verbose logging:
```python
python app/raster_fuzzy_glue.py --verbose
```

### Local Debugging

Test with smaller datasets first:
```python
python app/raster_fuzzy_glue.py --block-size 100 --partitions 2
```

## Cost Optimization

### AWS Glue Pricing

- **G.1X**: $0.44 per DPU-hour
- **G.2X**: $0.88 per DPU-hour  
- **G.4X**: $1.76 per DPU-hour

### Recommendations

1. Start with G.1X workers for testing
2. Use G.2X or G.4X for production with large datasets
3. Monitor job duration and adjust worker count
4. Consider using Spot instances for cost savings

## Example Job Configuration

```json
{
  "JobName": "RasterFuzzyProcessing",
  "Role": "AWSGlueServiceRole",
  "Command": {
    "Name": "glueetl",
    "ScriptLocation": "s3://your-bucket/scripts/raster_fuzzy_glue.py",
    "PythonVersion": "3"
  },
  "DefaultArguments": {
    "--job-language": "python",
    "--job-bookmark-option": "job-bookmark-disable"
  },
  "ExecutionProperty": {
    "MaxConcurrentRuns": 1
  },
  "MaxRetries": 0,
  "Timeout": 1800,
  "WorkerType": "G.2X",
  "NumberOfWorkers": 4,
  "GlueVersion": "4.0"
}
```

## Next Steps

1. Test locally with your data
2. Upload files to S3
3. Create AWS Glue job
4. Deploy and monitor
5. Optimize based on performance metrics 