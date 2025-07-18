# Unified Raster Fuzzy Inference System

This document describes the unified raster fuzzy inference system that combines both sequential and parallel processing capabilities with automatic fallback mechanisms.

## Overview

The unified system (`raster_fuzzy_lib.py`) merges the functionality of both the original sequential and parallel implementations into a single, robust system that:

- **Defaults to sequential processing** for reliability
- **Supports optional parallel processing** for performance
- **Automatically falls back to sequential** if parallel processing fails
- **Provides clear error messages** and progress tracking
- **Maintains the same configuration format** as the original systems

## Key Features

### 1. Unified Interface
- Single class `UnifiedRasterFuzzyInferenceSystem` handles both processing modes
- Same configuration format as original systems
- Consistent input/output handling

### 2. Robust Error Handling
- Automatic fallback from parallel to sequential processing
- Clear error messages and progress reporting
- Graceful handling of multiprocessing issues

### 3. Flexible Processing Options
- Sequential processing (default, most reliable)
- Parallel processing (optional, for performance)
- Configurable number of CPU cores and chunk sizes

## Usage

### Command Line Interface

The unified system can be used via the `raster_fuzzy_cli.py` script:

```bash
# Sequential processing (default)
python app/raster_fuzzy_cli.py social.tif environmental.tif strategic.tif output.tif

# Parallel processing with all available cores
python app/raster_fuzzy_cli.py social.tif environmental.tif strategic.tif output.tif --parallel

# Parallel processing with custom settings
python app/raster_fuzzy_cli.py social.tif environmental.tif strategic.tif output.tif \
    --parallel --cores 8 --chunk-size 1000

# Sequential processing with custom configuration
python app/raster_fuzzy_cli.py social.tif environmental.tif strategic.tif output.tif \
    --config my_config.json --nodata 5
```

### Programmatic Usage

```python
from raster_fuzzy_lib import UnifiedRasterFuzzyInferenceSystem

# Initialize the system
fis = UnifiedRasterFuzzyInferenceSystem('config.json')

# Sequential processing (default)
fis.process_rasters(
    social_tiff='social.tif',
    environmental_tiff='environmental.tif',
    strategic_tiff='strategic.tif',
    output_tiff='output.tif',
    nodata_value=5.0
)

# Parallel processing with fallback
fis.process_rasters(
    social_tiff='social.tif',
    environmental_tiff='environmental.tif',
    strategic_tiff='strategic.tif',
    output_tiff='output.tif',
    nodata_value=5.0,
    parallel=True,
    num_cores=8,
    chunk_size=1000
)
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--config, -c` | Configuration JSON file path | `raster_fis_config.json` |
| `--nodata` | NoData value for output raster | `5.0` |
| `--parallel, -p` | Enable parallel processing | `False` (sequential) |
| `--cores` | Number of CPU cores for parallel processing | All available |
| `--chunk-size` | Rows per chunk for parallel processing | `100` |
| `--verbose, -v` | Enable verbose output | `False` |
| `--create-config` | Create a template configuration file | N/A |

## Batch Processing

The `run_all_models.sh` script has been updated to use the unified system:

```bash
# Run all models with parallel processing (default)
sh scripts/run_all_models.sh

# Run all models with sequential processing
sh scripts/run_all_models.sh --sequential

# Run all models with custom parameters
sh scripts/run_all_models.sh --cores 4 --chunk-size 5000 --nodata 5
```

## Error Handling

### Parallel Processing Failures

If parallel processing fails, the system automatically falls back to sequential processing:

```
Attempting parallel processing...
Parallel processing failed: [error details]
Falling back to sequential processing...
Sequential processing completed successfully!
```

### Common Issues and Solutions

1. **Multiprocessing Errors**: The system automatically falls back to sequential processing
2. **Memory Issues**: Reduce chunk size or use sequential processing
3. **Configuration Errors**: Check JSON syntax and variable names
4. **File Not Found**: Verify input file paths and permissions

## Performance Considerations

### Sequential Processing
- **Pros**: Most reliable, no multiprocessing issues, predictable memory usage
- **Cons**: Slower for large datasets
- **Best for**: Small to medium datasets, debugging, reliable processing

### Parallel Processing
- **Pros**: Faster for large datasets, utilizes multiple CPU cores
- **Cons**: May fail due to multiprocessing issues, higher memory usage
- **Best for**: Large datasets, when speed is critical

### Recommended Settings

| Dataset Size | Recommended Mode | Chunk Size | Cores |
|--------------|------------------|------------|-------|
| < 1000x1000 | Sequential | N/A | N/A |
| 1000x1000 - 5000x5000 | Parallel | 100-500 | 4-8 |
| > 5000x5000 | Parallel | 1000-10000 | 8+ |

## Migration from Previous Versions

### From Sequential Version
- Replace `RasterFuzzyInferenceSystem` with `UnifiedRasterFuzzyInferenceSystem`
- Add `parallel=False` parameter to `process_rasters()` calls
- No other changes needed

### From Parallel Version
- Replace `ParallelRasterFuzzyInferenceSystem` with `UnifiedRasterFuzzyInferenceSystem`
- Add `parallel=True` parameter to `process_rasters()` calls
- Update import statements

### Configuration Files
- No changes needed - same JSON format
- All existing configuration files work with the unified system

## Troubleshooting

### Parallel Processing Issues

1. **Memory Errors**: Reduce chunk size or use sequential processing
2. **Hanging Processes**: Use `--sequential` flag or reduce number of cores
3. **Inconsistent Results**: Check for race conditions in fuzzy system creation

### Performance Issues

1. **Slow Processing**: Enable parallel processing or increase chunk size
2. **High Memory Usage**: Reduce chunk size or use sequential processing
3. **CPU Underutilization**: Increase number of cores or chunk size

## File Structure

```
app/
├── raster_fuzzy_lib.py                 # Main unified system (library)
├── raster_fuzzy_cli.py                 # Command-line interface
├── config/
│   ├── config_round_up.json
│   ├── config_round_down.json
│   └── ...
└── files/
    ├── input/base/
    └── output/

scripts/
└── run_all_models.sh                  # Updated batch script

docs/
└── README_Unified_System.md           # This file
```

## Future Improvements

1. **Better Multiprocessing**: Investigate alternative parallel processing approaches
2. **Memory Optimization**: Implement streaming for very large datasets
3. **GPU Support**: Add CUDA/OpenCL acceleration for fuzzy logic operations
4. **Progress Persistence**: Save and resume interrupted processing
5. **Distributed Processing**: Support for cluster computing

## Support

For issues or questions:
1. Check the error messages for specific guidance
2. Try sequential processing if parallel fails
3. Review the configuration file syntax
4. Check input file formats and paths 