# Running All FIS Models

This guide explains how to run all fuzzy inference system (FIS) configurations sequentially to generate multiple output files for comparison.

## Available Scripts

### 1. **`scripts/run_all_fis_models.sh`** - General Purpose
- Works with local files and S3 files
- Uses local config files
- Suitable for any environment

### 2. **`scripts/run_all_fis_models_emr.sh`** - EMR Optimized
- Designed for EMR clusters
- Downloads config files from S3
- Handles EMR environment automatically

## FIS Models Included

The scripts will run these 7 different fuzzy inference configurations:

| Model | Config File | Description |
|-------|-------------|-------------|
| **max** | `config_max.json` | Maximum aggregation (conservative) |
| **minimum** | `config_minimum.json` | Minimum aggregation (optimistic) |
| **median** | `config_median.json` | Median aggregation (balanced) |
| **mode** | `config_mode.json` | Mode aggregation (most common) |
| **round_up** | `config_round_up.json` | Round up results |
| **round_down** | `config_round_down.json` | Round down results |
| **default** | `raster_fis_config.json` | Standard configuration |

## Usage Examples

### **On EMR Cluster:**

```bash
# Navigate to your code directory
cd /home/ssm-user/destark

# Run all models with S3 files
./scripts/run_all_fis_models_emr.sh \
    s3://<AWS-BUCKET>/unifile_test/so300m.in \
    s3://<AWS-BUCKET>/unifile_test/e300m.in \
    s3://<AWS-BUCKET>/unifile_test/s300m.in \
    s3://<AWS-BUCKET>/unifile_test/result_300m

# Run with custom S3 config prefix
./scripts/run_all_fis_models_emr.sh \
    s3://bucket/input/social.tif \
    s3://bucket/input/env.tif \
    s3://bucket/input/strat.tif \
    s3://bucket/output/result \
    s3://bucket/configs
```

### **Local Environment:**

```bash
# Run all models with local files
./scripts/run_all_fis_models.sh \
    ./input/social.tif \
    ./input/env.tif \
    ./input/strat.tif \
    ./output/result

# Run with custom config directory
./scripts/run_all_fis_models.sh \
    ./input/social.tif \
    ./input/env.tif \
    ./input/strat.tif \
    ./output/result \
    /path/to/custom/configs
```

## Output Files

The scripts will generate one output file for each FIS model:

```
result_max.tif          # Maximum aggregation results
result_minimum.tif      # Minimum aggregation results  
result_median.tif       # Median aggregation results
result_mode.tif         # Mode aggregation results
result_round_up.tif     # Round up results
result_round_down.tif   # Round down results
result_default.tif      # Standard configuration results
```

## Logging

Both scripts create detailed logs:

- **Main log**: `logs/all_fis_models_YYYYMMDD_HHMMSS.log`
- **Error log**: `logs/all_fis_models_YYYYMMDD_HHMMSS_error.log`

## Processing Order

The models are processed in this order:
1. **max** - Conservative approach
2. **minimum** - Optimistic approach  
3. **median** - Balanced approach
4. **mode** - Most common value
5. **round_up** - Conservative rounding
6. **round_down** - Conservative rounding
7. **default** - Standard configuration

## Error Handling

- If one model fails, the script continues with the next model
- Failed models are logged in the error log
- The script provides a summary of all generated files at the end

## Performance Considerations

### **Sequential Processing**
- Models run one after another (not parallel)
- Each model uses the same Spark session
- Total time = sum of all individual model times

### **Memory Usage**
- Each model processes the same input data
- Memory is freed between models
- Recommended: 8GB+ RAM for large datasets

### **Storage Requirements**
- Each output file is the same size as input
- Total storage = 7 × input file size
- Consider cleanup of intermediate files

## Comparison Analysis

After running all models, you can compare the results:

```bash
# Check file sizes
ls -lh result_*.tif

# Compare value ranges (if you have GDAL tools)
gdalinfo result_max.tif | grep "Min/Max"
gdalinfo result_minimum.tif | grep "Min/Max"
# ... etc for all models
```

## Troubleshooting

### **Common Issues:**

1. **Config file not found**
   - Check if config files exist in the specified directory
   - For EMR: Verify S3 paths are correct

2. **Memory issues**
   - Reduce `--block-size` parameter
   - Use fewer `--partitions`

3. **S3 access issues**
   - Verify AWS credentials
   - Check S3 bucket permissions

4. **Spark session issues**
   - Check EMR cluster status
   - Verify Spark configuration

### **Debug Commands:**

```bash
# Check if config files exist
ls -la app/config/*.json

# Test single model first
python3 app/raster_fuzzy_spark_ultra_optimized.py \
    input1.tif input2.tif input3.tif test_output.tif \
    --config app/config/config_max.json \
    --verbose

# Check logs
tail -f logs/all_fis_models_*.log
```

## Recommendations

1. **Start with a small dataset** to test all models
2. **Monitor resource usage** during processing
3. **Compare results** to understand model differences
4. **Use the most appropriate model** for your specific use case
5. **Keep logs** for troubleshooting and analysis 