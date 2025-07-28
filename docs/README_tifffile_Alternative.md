# TiffFile Alternative for AWS Glue

This document describes the tifffile-based alternative to GDAL/rasterio for AWS Glue environments where GDAL installation or compatibility issues occur.

## Overview

The tifffile-based implementation provides a GDAL-free alternative for processing raster TIFF files in AWS Glue environments. It uses the `tifffile` Python library instead of `rasterio`/GDAL for reading and writing TIFF files.

## Key Differences

### Advantages of tifffile
- **No GDAL dependency**: Eliminates GDAL installation issues in AWS Glue
- **Lighter weight**: Smaller package size and fewer system dependencies
- **Pure Python**: More portable across different environments
- **Faster installation**: No compilation required

### Limitations of tifffile
- **No geospatial metadata**: CRS, transform, and other geospatial information not preserved
- **Limited format support**: Only supports TIFF format (no GeoTIFF metadata)
- **No spatial operations**: No built-in spatial analysis capabilities

## Files

### Core Implementation
- `app/raster_fuzzy_lib_tifffile.py` - Main library using tifffile
- `app/raster_fuzzy_cli_tifffile.py` - Command-line interface
- `app/test_tifffile_implementation.py` - Test script

### Configuration
- `requirements-glue-tifffile.txt` - Dependencies for tifffile version
- `Dockerfile.glue5-tifffile` - Docker image without GDAL

## Installation

### Local Development
```bash
# Install tifffile-based dependencies
pip install -r requirements-glue-tifffile.txt
```

### AWS Glue Environment
```bash
# Use the tifffile Docker image
docker build -f Dockerfile.glue5-tifffile -t glue-tifffile .
```

## Usage

### Command Line Interface
```bash
# Create a template configuration
python app/raster_fuzzy_cli_tifffile.py --create-config config.json

# Process rasters
python app/raster_fuzzy_cli_tifffile.py \
  --social input/social.tif \
  --environmental input/environmental.tif \
  --strategic input/strategic.tif \
  --output output/result.tif \
  --config config.json

# Parallel processing
python app/raster_fuzzy_cli_tifffile.py \
  --social input/social.tif \
  --environmental input/environmental.tif \
  --strategic input/strategic.tif \
  --output output/result.tif \
  --config config.json \
  --parallel \
  --cores 4
```

### Programmatic Usage
```python
from app.raster_fuzzy_lib_tifffile import UnifiedRasterFuzzyInferenceSystem

# Create FIS instance
fis = UnifiedRasterFuzzyInferenceSystem('config.json')

# Process rasters
fis.process_rasters(
    social_tiff='input/social.tif',
    environmental_tiff='input/environmental.tif',
    strategic_tiff='input/strategic.tif',
    output_tiff='output/result.tif',
    parallel=True,
    num_cores=4
)
```

## Testing

Run the test script to verify the implementation:
```bash
python app/test_tifffile_implementation.py
```

This will:
1. Create test raster files
2. Test the tifffile implementation
3. Compare with rasterio (if available)
4. Verify output quality

## Migration from rasterio

### Code Changes
Replace rasterio imports with tifffile:

```python
# Old (rasterio)
import rasterio
with rasterio.open('file.tif') as src:
    data = src.read(1)
    profile = src.profile

# New (tifffile)
import tifffile
data = tifffile.imread('file.tif')
# Note: No profile/metadata available
```

### Configuration
The configuration format remains the same - only the underlying TIFF reading/writing changes.

## Performance Considerations

### Memory Usage
- tifffile loads entire images into memory
- For large rasters, consider chunked processing
- Use parallel processing for better performance

### Processing Speed
- tifffile is generally faster for simple read/write operations
- No geospatial overhead
- Parallel processing scales well

## AWS Glue Integration

### Glue Job Script
```python
import sys
import os
sys.path.append('/opt/amazon/glue/lib/installation')

from raster_fuzzy_lib_tifffile import UnifiedRasterFuzzyInferenceSystem

def main():
    # Initialize FIS
    fis = UnifiedRasterFuzzyInferenceSystem('config.json')
    
    # Process rasters
    fis.process_rasters(
        social_tiff='s3://bucket/input/social.tif',
        environmental_tiff='s3://bucket/input/environmental.tif',
        strategic_tiff='s3://bucket/input/strategic.tif',
        output_tiff='s3://bucket/output/result.tif',
        parallel=True
    )

if __name__ == "__main__":
    main()
```

### Glue Job Parameters
```json
{
  "job_name": "raster-fuzzy-tifffile",
  "script_location": "s3://bucket/scripts/raster_fuzzy_glue_tifffile.py",
  "python_version": "3",
  "worker_type": "G.1X",
  "number_of_workers": 2,
  "timeout": 2880
}
```

## Troubleshooting

### Common Issues

1. **Import Error**: Ensure tifffile is installed
   ```bash
   pip install tifffile>=2023.0.0
   ```

2. **Memory Issues**: Use chunked processing for large files
   ```python
   fis.process_rasters(..., chunk_size=50)
   ```

3. **S3 Access**: Ensure proper IAM permissions for S3 read/write

4. **Performance**: Use parallel processing and appropriate worker types

### Debug Mode
Enable verbose output for troubleshooting:
```bash
python raster_fuzzy_cli_tifffile.py --verbose [other options]
```

## Comparison with rasterio

| Feature | rasterio | tifffile |
|---------|----------|----------|
| GDAL dependency | Yes | No |
| Geospatial metadata | Full support | Limited |
| Format support | Many formats | TIFF only |
| Installation complexity | High | Low |
| AWS Glue compatibility | Issues common | Good |
| Performance | Good | Better |
| Memory usage | Moderate | Lower |

## Recommendations

### When to use tifffile
- AWS Glue environments with GDAL issues
- Simple TIFF processing without geospatial metadata
- Fast prototyping and development
- Memory-constrained environments

### When to use rasterio
- Geospatial analysis requiring CRS/transform
- Multi-format raster support
- Complex spatial operations
- When geospatial metadata is critical

## Future Enhancements

1. **Geospatial metadata support**: Add optional geospatial metadata handling
2. **Format conversion**: Add support for converting between formats
3. **Spatial operations**: Implement basic spatial analysis functions
4. **Cloud optimization**: Optimize for cloud storage and processing

## Support

For issues with the tifffile implementation:
1. Check the test script output
2. Verify tifffile installation
3. Review memory usage for large files
4. Ensure proper file permissions and paths 