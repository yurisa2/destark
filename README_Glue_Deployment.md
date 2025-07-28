# AWS Glue Deployment for Raster Fuzzy Processing

## Overview

This project provides a stable local testing environment and AWS Glue deployment solution for raster fuzzy inference processing. The system processes three input raster files (social, environmental, strategic) and produces a priority output raster using fuzzy logic.

## Files Created

### Core Scripts
- `app/raster_fuzzy_glue.py` - Full AWS Glue version with Spark
- `app/raster_fuzzy_glue_simple.py` - Simplified local version (recommended for testing)
- `app/requirements-glue.txt` - Dependencies for AWS Glue

### Testing & Documentation
- `scripts/test_glue_local.py` - Local testing suite
- `docs/README_Glue_Deployment.md` - Detailed deployment guide
- `glue_job_template.txt` - AWS Glue job template

## Quick Start

### 1. Local Testing (Recommended)

Test the simplified version locally:

```bash
# Install dependencies
pip install -r app/requirements-glue.txt

# Test with local files
python app/raster_fuzzy_glue_simple.py \
  app/files/input/base/socioeconomico_1000m.tif \
  app/files/input/base/ambiental_1000m.tif \
  app/files/input/base/estratégico_1000m.tif \
  output_test.tif \
  --config app/config/config_round_down.json \
  --block-size 500 \
  --workers 2
```

### 2. AWS Glue Deployment

#### Step 1: Upload Files to S3
```bash
# Upload input files
aws s3 cp app/files/input/base/socioeconomico_1000m.tif s3://your-bucket/input/
aws s3 cp app/files/input/base/ambiental_1000m.tif s3://your-bucket/input/
aws s3 cp app/files/input/base/estratégico_1000m.tif s3://your-bucket/input/

# Upload config file
aws s3 cp app/config/config_round_down.json s3://your-bucket/config/
```

#### Step 2: Create AWS Glue Job

1. Go to AWS Glue Console
2. Click "Jobs" → "Add job"
3. Choose "Spark" as job type
4. Set job name (e.g., "RasterFuzzyProcessing")

#### Step 3: Configure Job Parameters

Add these parameters:

| Parameter | Value |
|-----------|-------|
| `SOCIAL_TIFF` | `s3://your-bucket/input/socioeconomico_1000m.tif` |
| `ENVIRONMENTAL_TIFF` | `s3://your-bucket/input/ambiental_1000m.tif` |
| `STRATEGIC_TIFF` | `s3://your-bucket/input/estratégico_1000m.tif` |
| `OUTPUT_TIFF` | `s3://your-bucket/output/result.tif` |
| `CONFIG_FILE` | `s3://your-bucket/config/config_round_down.json` |

#### Step 4: Paste the Script

Copy the entire content of `app/raster_fuzzy_glue.py` and paste it into the Glue job script editor.

#### Step 5: Configure Job Settings

- **Worker type**: G.2X (for medium datasets)
- **Number of workers**: 4-8 (adjust based on data size)
- **Job timeout**: 60 minutes
- **Max concurrency**: 1

#### Step 6: Add Python Libraries

Add these libraries to the job:
- `numpy>=1.21.0`
- `scipy>=1.7.0`
- `rasterio>=1.3.0`
- `scikit-fuzzy>=0.4.2`
- `boto3>=1.26.0`

#### Step 7: Run the Job

1. Save the job
2. Click "Run job"
3. Monitor in AWS Glue console

## Script Features

### Dual Mode Operation
- **Local Mode**: Uses command line arguments
- **AWS Glue Mode**: Uses environment variables

### S3 Integration
- Automatic download of S3 input files
- Automatic upload of results to S3
- Local file support for testing

### Performance Optimization
- Block-based processing for memory efficiency
- Multiprocessing for local testing
- Spark RDD for AWS Glue deployment
- Configurable block sizes and worker counts

## Key Differences Between Scripts

| Feature | Simple Version | Full Glue Version |
|---------|----------------|-------------------|
| **Local Testing** | ✅ Easy | ⚠️ Spark dependencies |
| **AWS Glue** | ✅ Works | ✅ Optimized |
| **Dependencies** | Minimal | Full Spark stack |
| **Performance** | Good | Excellent |
| **Complexity** | Low | Medium |

## Troubleshooting

### Common Issues

1. **Python Version Mismatch** (Local Spark)
   - Use the simple version for local testing
   - Or set `PYSPARK_PYTHON` environment variable

2. **Memory Issues**
   - Reduce block size: `--block-size 500`
   - Increase worker memory in AWS Glue

3. **Timeout Issues**
   - Increase job timeout for large datasets
   - Use smaller block sizes

4. **S3 Access Issues**
   - Ensure IAM roles have S3 read/write permissions
   - Check bucket and file paths

### Debug Mode

```bash
# Add verbose logging
python app/raster_fuzzy_glue_simple.py --verbose

# Test with smaller data
python app/raster_fuzzy_glue_simple.py --block-size 100 --workers 1
```

## Performance Tips

### Local Testing
- Use `--workers 2-4` for multiprocessing
- Use `--block-size 500-1000` for memory efficiency
- Test with smaller datasets first

### AWS Glue
- Start with G.2X workers for testing
- Use G.4X for large datasets
- Monitor job duration and adjust worker count
- Consider Spot instances for cost savings

## Cost Optimization

### AWS Glue Pricing
- **G.1X**: $0.44 per DPU-hour
- **G.2X**: $0.88 per DPU-hour  
- **G.4X**: $1.76 per DPU-hour

### Recommendations
1. Start with G.1X for testing
2. Use G.2X or G.4X for production
3. Monitor job duration and optimize
4. Consider Spot instances for cost savings

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
  "Timeout": 3600,
  "WorkerType": "G.2X",
  "NumberOfWorkers": 4,
  "GlueVersion": "4.0"
}
```

## Next Steps

1. **Test Locally**: Use the simple version to validate your data
2. **Upload to S3**: Prepare your input files and config
3. **Create Glue Job**: Use the template and script
4. **Deploy**: Run and monitor the job
5. **Optimize**: Adjust parameters based on performance

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the detailed deployment guide in `docs/README_Glue_Deployment.md`
3. Test with the local version first
4. Check AWS Glue logs for detailed error messages

## Success Metrics

- ✅ Local testing works without errors
- ✅ Output file is created with correct size
- ✅ AWS Glue job completes successfully
- ✅ S3 upload/download works correctly
- ✅ Processing time is reasonable for your data size 