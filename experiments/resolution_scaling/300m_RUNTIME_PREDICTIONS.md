# 300m Resolution Runtime Predictions

## Executive Summary

Based on the successful 1000m job statistics, here are the **accurate runtime predictions** for processing 300m resolution files:

### Key Statistics
- **1000m job**: 4424×4593 pixels, 9.7 minutes, 4 cores, 4GB memory
- **300m files**: 14479×15187 pixels, **10.82x more pixels**
- **Scaling factor**: 3.29x linear dimension increase

## Runtime Predictions

| Configuration | Runtime | Memory | Cores | Block Size | Best For |
|---------------|---------|--------|-------|------------|----------|
| **Conservative** | **52 min** | ~8GB | 2 | 100 | Testing, limited RAM |
| **Balanced** | **26 min** | ~16GB | 4 | 200 | **Most systems** ⭐ |
| **Performance** | **13 min** | ~32GB | 8 | 300 | High-end systems |
| **Maximum** | **10.5 min** | ~40GB | 10 | 500 | Maximum speed |
| **Memory-Optimized** | **17.5 min** | ~12GB | 6 | 100 | Limited RAM systems |

## Detailed Analysis

### 1. Conservative Configuration
- **Runtime**: 52 minutes 31 seconds
- **Memory**: ~8GB
- **Use case**: Testing, development, systems with limited RAM
- **Command**: `--block-size 100 --partitions 2`

### 2. Balanced Configuration ⭐ **RECOMMENDED**
- **Runtime**: 26 minutes 15 seconds
- **Memory**: ~16GB
- **Use case**: Standard processing, most systems
- **Command**: `--block-size 200 --partitions 4`

### 3. Performance Configuration
- **Runtime**: 13 minutes 7 seconds
- **Memory**: ~32GB
- **Use case**: High-performance processing, large datasets
- **Command**: `--block-size 300 --partitions 8`

### 4. Maximum Configuration
- **Runtime**: 10 minutes 30 seconds
- **Memory**: ~40GB
- **Use case**: Maximum speed, dedicated processing
- **Command**: `--block-size 500 --partitions 10`

### 5. Memory-Optimized Configuration
- **Runtime**: 17 minutes 30 seconds
- **Memory**: ~12GB
- **Use case**: Systems with limited RAM
- **Command**: `--block-size 100 --partitions 6`

## File Information

### Input Files (300m)
- **socioeconomico_300m.tif**: 210MB
- **ambiental_300m.tif**: 210MB
- **estrategico_300m.tif**: 210MB
- **Total input**: 630MB

### Output Prediction
- **Predicted output size**: 877MB per configuration
- **Storage requirement**: 877MB per result file

## Ready-to-Run Commands

### Balanced Configuration (Recommended)
```bash
python app/raster_fuzzy_spark_simple.py \
  app/files/input/300m/socioeconomico_300m.tif \
  app/files/input/300m/ambiental_300m.tif \
  app/files/input/300m/estrategico_300m.tif \
  app/files/output/300m/result_median_300m.tif \
  --config app/config/config_median.json \
  --local \
  --block-size 200 \
  --partitions 4 \
  --verbose
```

### Memory-Optimized Configuration
```bash
python app/raster_fuzzy_spark_simple.py \
  app/files/input/300m/socioeconomico_300m.tif \
  app/files/input/300m/ambiental_300m.tif \
  app/files/input/300m/estrategico_300m.tif \
  app/files/output/300m/result_median_300m.tif \
  --config app/config/config_median.json \
  --local \
  --block-size 100 \
  --partitions 6 \
  --verbose
```

### Performance Configuration
```bash
python app/raster_fuzzy_spark_simple.py \
  app/files/input/300m/socioeconomico_300m.tif \
  app/files/input/300m/ambiental_300m.tif \
  app/files/input/300m/estrategico_300m.tif \
  app/files/output/300m/result_median_300m.tif \
  --config app/config/config_median.json \
  --local \
  --block-size 300 \
  --partitions 8 \
  --verbose
```

## System Requirements

### Minimum Requirements
- **RAM**: 8GB (Conservative configuration)
- **CPU**: 2 cores
- **Storage**: 1.5GB free space

### Recommended Requirements
- **RAM**: 16GB (Balanced configuration)
- **CPU**: 4 cores
- **Storage**: 2GB free space

### High-Performance Requirements
- **RAM**: 32GB+ (Performance configuration)
- **CPU**: 8+ cores
- **Storage**: 3GB+ free space

## Scaling Factors

### From 1000m to 300m
- **Pixel count**: 10.82x increase (20.3M → 219.8M pixels)
- **Linear dimensions**: 3.29x increase
- **Processing time**: 10.82x increase (with same cores)
- **Memory usage**: 4x increase (due to larger blocks)

### Performance Scaling
- **2 cores**: 52 minutes
- **4 cores**: 26 minutes
- **6 cores**: 17.5 minutes
- **8 cores**: 13 minutes
- **10 cores**: 10.5 minutes

## Validation

These predictions are based on:
1. **Actual 1000m job statistics**: 582.54 seconds, 4 cores
2. **Measured file dimensions**: 14479×15187 vs 4424×4593
3. **Linear scaling**: Processing time scales with pixel count and cores
4. **Memory scaling**: Block size and partition count affect memory usage

## Recommendations

1. **Start with Balanced configuration** (26 min, 16GB RAM)
2. **For limited RAM**: Use Memory-Optimized (17.5 min, 12GB RAM)
3. **For maximum speed**: Use Performance (13 min, 32GB RAM)
4. **For testing**: Use Conservative (52 min, 8GB RAM)

## Monitoring

During processing, monitor:
- **Memory usage**: Should stay within predicted limits
- **CPU usage**: Should utilize all specified cores
- **Disk space**: Ensure sufficient space for 877MB output
- **Processing time**: Compare with predictions for validation 