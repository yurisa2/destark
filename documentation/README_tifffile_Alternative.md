# TiffFile Alternative for AWS Glue GDAL Issues

This document provides a complete alternative solution using `tifffile` instead of `rasterio`/GDAL for AWS Glue environments where GDAL installation or compatibility issues occur.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements-glue-tifffile.txt
```

### 2. Test the Implementation
```bash
python app/test_tifffile_implementation.py
```

### 3. Process Your Rasters
```bash
# Create a configuration template
python app/raster_fuzzy_cli_tifffile.py --create-config my_config.json

# Process rasters
python app/raster_fuzzy_cli_tifffile.py \
  --social input/social.tif \
  --environmental input/environmental.tif \
  --strategic input/strategic.tif \
  --output output/result.tif \
  --config my_config.json
```

## 📁 Files Overview

### Core Implementation
- **`app/raster_fuzzy_lib_tifffile.py`** - Main library using tifffile instead of rasterio
- **`app/raster_fuzzy_cli_tifffile.py`** - Command-line interface
- **`app/test_tifffile_implementation.py`** - Test script to verify functionality

### Configuration
- **`requirements-glue-tifffile.txt`** - Dependencies without GDAL
- **`Dockerfile.glue5-tifffile`** - Docker image without GDAL dependencies

### Documentation
- **`docs/README_tifffile_Alternative.md`** - Detailed documentation

## 🔧 Key Differences from rasterio

| Feature | rasterio | tifffile |
|---------|----------|----------|
| **GDAL dependency** | Required | None |
| **Installation complexity** | High (compilation) | Low (pure Python) |
| **AWS Glue compatibility** | Issues common | Excellent |
| **Geospatial metadata** | Full support | Limited |
| **Format support** | Many formats | TIFF only |
| **Performance** | Good | Better |
| **Memory usage** | Moderate | Lower |

## ✅ Advantages of tifffile

1. **No GDAL dependency** - Eliminates installation issues in AWS Glue
2. **Pure Python** - More portable across environments
3. **Faster installation** - No compilation required
4. **Better AWS Glue compatibility** - Works reliably in cloud environments
5. **Lighter weight** - Smaller package size and fewer dependencies

## ⚠️ Limitations

1. **No geospatial metadata** - CRS, transform, and other geospatial information not preserved
2. **Limited format support** - Only supports TIFF format (no GeoTIFF metadata)
3. **No spatial operations** - No built-in spatial analysis capabilities

## 🐳 Docker Usage

### Build the tifffile Docker image
```bash
docker build -f Dockerfile.glue5-tifffile -t glue-tifffile .
```

### Run with Docker
```bash
docker run -it --rm -v $(pwd):/workspace glue-tifffile bash
cd /workspace
python app/test_tifffile_implementation.py
```

## 🔄 Migration from rasterio

### Code Changes
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
The configuration format remains exactly the same - only the underlying TIFF reading/writing changes.

## 🚀 AWS Glue Integration

### Glue Job Script Example
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

## 🧪 Testing

### Run the test suite
```bash
python app/test_tifffile_implementation.py
```

This will:
1. ✅ Create test raster files
2. ✅ Test the tifffile implementation
3. ✅ Compare with rasterio (if available)
4. ✅ Verify output quality

### Expected Output
```
============================================================
Testing tifffile-based raster fuzzy inference system
============================================================
✓ All tests passed! tifffile implementation is working correctly.
============================================================
```

## 📊 Performance Considerations

### Memory Usage
- tifffile loads entire images into memory
- For large rasters, consider chunked processing
- Use parallel processing for better performance

### Processing Speed
- tifffile is generally faster for simple read/write operations
- No geospatial overhead
- Parallel processing scales well

## 🔧 Troubleshooting

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

## 📋 Usage Examples

### Basic Processing
```bash
python app/raster_fuzzy_cli_tifffile.py \
  --social input/social.tif \
  --environmental input/environmental.tif \
  --strategic input/strategic.tif \
  --output output/result.tif \
  --config config.json
```

### Parallel Processing
```bash
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

## 🎯 Recommendations

### When to use tifffile
- ✅ AWS Glue environments with GDAL issues
- ✅ Simple TIFF processing without geospatial metadata
- ✅ Fast prototyping and development
- ✅ Memory-constrained environments

### When to use rasterio
- ✅ Geospatial analysis requiring CRS/transform
- ✅ Multi-format raster support
- ✅ Complex spatial operations
- ✅ When geospatial metadata is critical

## 🔮 Future Enhancements

1. **Geospatial metadata support** - Add optional geospatial metadata handling
2. **Format conversion** - Add support for converting between formats
3. **Spatial operations** - Implement basic spatial analysis functions
4. **Cloud optimization** - Optimize for cloud storage and processing

## 📞 Support

For issues with the tifffile implementation:
1. Check the test script output
2. Verify tifffile installation
3. Review memory usage for large files
4. Ensure proper file permissions and paths

## 📚 Additional Resources

- [tifffile Documentation](https://github.com/cgohlke/tifffile)
- [AWS Glue Documentation](https://docs.aws.amazon.com/glue/)
- [SciKit-Fuzzy Documentation](https://pythonhosted.org/scikit-fuzzy/)

---

**Note**: This tifffile-based implementation provides a robust alternative to GDAL/rasterio for AWS Glue environments while maintaining the same fuzzy logic processing capabilities. 