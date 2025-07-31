# Spark Local Distributed Processing Guide

This guide shows how to run Spark locally with distributed processing across multiple cores and proper memory management.

## Quick Start

### Basic Command Structure
```bash
python app/raster_fuzzy_spark_simple.py \
  <social_tiff> \
  <environmental_tiff> \
  <strategic_tiff> \
  <output_tiff> \
  --config <config_file> \
  --local \
  --block-size <rows_per_block> \
  --partitions <num_partitions> \
  --verbose
```

### Example: Run with Median Configuration
```bash
python app/raster_fuzzy_spark_simple.py \
  app/files/input/base/socioeconomico_1000m.tif \
  app/files/input/base/ambiental_1000m.tif \
  app/files/input/base/estratégico_1000m.tif \
  app/files/output/base/result_median_1000m.tif \
  --config app/config/config_median.json \
  --local \
  --block-size 200 \
  --partitions 4 \
  --verbose
```

## Configuration Options

### Memory Management Parameters

| Parameter | Description | Recommended Values |
|-----------|-------------|-------------------|
| `--block-size` | Number of rows per block | 100-500 (smaller = less memory) |
| `--partitions` | Number of Spark partitions | 2-8 (more = more cores used) |
| `--nodata` | NoData value for output | 5.0 (default) |

### Performance Configurations

#### 1. Conservative (Testing/Development)
```bash
--block-size 100 --partitions 2
```
- **Best for**: Testing, development, limited resources
- **Memory usage**: ~2GB
- **CPU cores**: 2

#### 2. Balanced (Standard Processing)
```bash
--block-size 200 --partitions 4
```
- **Best for**: Standard processing, good performance/memory balance
- **Memory usage**: ~4GB
- **CPU cores**: 4
- **Processing time**: ~10 minutes for 4424x4593 raster

#### 3. Performance (High-Performance)
```bash
--block-size 300 --partitions 8
```
- **Best for**: High-performance processing, large datasets
- **Memory usage**: ~8GB
- **CPU cores**: 8

#### 4. Maximum (All Resources)
```bash
--block-size 500 --partitions $(nproc)
```
- **Best for**: Maximum performance, dedicated processing
- **Memory usage**: Maximum available
- **CPU cores**: All available cores

## Available FIS Configurations

| Configuration | Description | File |
|---------------|-------------|------|
| Median | Environmental assessment with median aggregation | `app/config/config_median.json` |
| Minimum | Environmental assessment with minimum aggregation | `app/config/config_minimum.json` |
| Mode | Environmental assessment with mode aggregation | `app/config/config_mode.json` |
| Round Down | Environmental assessment with round down aggregation | `app/config/config_round_down.json` |
| Round Up | Environmental assessment with round up aggregation | `app/config/config_round_up.json` |
| Maximum | Environmental assessment with maximum aggregation | `app/config/config_max.json` |

## Input Files

The system expects three input TIFF files:
- **Social factor**: `socioeconomico_1000m.tif`
- **Environmental factor**: `ambiental_1000m.tif`
- **Strategic factor**: `estratégico_1000m.tif`

These files should be placed in `app/files/input/base/`.

## Output

- **Format**: GeoTIFF
- **Location**: `app/files/output/base/`
- **Naming**: `result_<config_name>_1000m.tif`
- **Value range**: Typically 1.75 to 9.16 (depends on configuration)

## Monitoring and Troubleshooting

### Monitor System Resources
```bash
# Monitor CPU and memory usage
top

# Monitor disk space
df -h

# Check Spark UI (if available)
open http://localhost:4040
```

### Common Issues

1. **Out of Memory**: Reduce `--block-size` or `--partitions`
2. **Slow Processing**: Increase `--partitions` (if memory allows)
3. **File Not Found**: Check input file paths and permissions
4. **Spark Configuration**: Ensure Spark is properly installed and configured

### Performance Tips

1. **Block Size**: Smaller blocks use less memory but may be slower
2. **Partitions**: More partitions use more cores but require more memory
3. **Memory**: Monitor system memory usage during processing
4. **Disk Space**: Ensure sufficient disk space for output files

## Example Runs

### Run All Configurations
```bash
# Conservative approach
for config in median minimum mode round_down round_up max; do
  python app/raster_fuzzy_spark_simple.py \
    app/files/input/base/socioeconomico_1000m.tif \
    app/files/input/base/ambiental_1000m.tif \
    app/files/input/base/estratégico_1000m.tif \
    app/files/output/base/result_${config}_1000m.tif \
    --config app/config/config_${config}.json \
    --local \
    --block-size 100 \
    --partitions 2 \
    --verbose
done
```

### Performance Testing
```bash
# Test different configurations
python app/raster_fuzzy_spark_simple.py \
  app/files/input/base/socioeconomico_1000m.tif \
  app/files/input/base/ambiental_1000m.tif \
  app/files/input/base/estratégico_1000m.tif \
  app/files/output/base/result_performance_test.tif \
  --config app/config/config_median.json \
  --local \
  --block-size 300 \
  --partitions 8 \
  --verbose
```

## System Requirements

- **Python**: 3.7+
- **Spark**: 3.0+
- **Memory**: 4GB+ recommended
- **CPU**: 4+ cores recommended
- **Disk**: Sufficient space for input/output files

## Success Indicators

✅ **Successful run shows**:
- "Spark processing completed successfully!"
- Output file created in specified location
- Processing time displayed
- Output value range displayed

❌ **Failed run shows**:
- Error messages
- Missing output file
- Memory or configuration errors 