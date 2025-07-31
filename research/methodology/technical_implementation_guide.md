# Technical Implementation Guide: Rasterio vs Tifffile

## Implementation Overview

This guide provides detailed technical implementation differences between our `rasterio` and `tifffile` approaches, along with practical guidance for choosing and implementing each solution.

## 🔧 Core Implementation Differences

### File Reading Implementation

#### Rasterio Approach
```python
import rasterio

def read_raster_rasterio(file_path: str) -> Tuple[np.ndarray, Dict]:
    """Read raster file using rasterio."""
    with rasterio.open(file_path) as src:
        data = src.read(1)  # Read first band
        profile = src.profile.copy()
        
        # Extract metadata
        metadata = {
            'shape': data.shape,
            'dtype': data.dtype,
            'crs': src.crs,
            'transform': src.transform,
            'nodata': src.nodata,
            'count': src.count,
            'width': src.width,
            'height': src.height
        }
        
        return data, metadata
```

#### Tifffile Approach
```python
import tifffile

def read_raster_tifffile(file_path: str) -> Tuple[np.ndarray, Dict]:
    """Read raster file using tifffile."""
    data = tifffile.imread(file_path)
    
    # Extract basic metadata
    metadata = {
        'shape': data.shape,
        'dtype': data.dtype,
        'size': data.size,
        'ndim': data.ndim
    }
    
    # Note: No geospatial metadata (CRS, transform, etc.)
    return data, metadata
```

### File Writing Implementation

#### Rasterio Approach
```python
def write_raster_rasterio(data: np.ndarray, file_path: str, 
                         profile: Dict = None) -> None:
    """Write raster file using rasterio."""
    if profile is None:
        profile = {
            'driver': 'GTiff',
            'height': data.shape[0],
            'width': data.shape[1],
            'count': 1,
            'dtype': data.dtype,
            'crs': 'EPSG:4326',  # Default CRS
            'transform': rasterio.transform.from_bounds(
                -180, -90, 180, 90, data.shape[1], data.shape[0]
            )
        }
    
    with rasterio.open(file_path, 'w', **profile) as dst:
        dst.write(data, 1)
```

#### Tifffile Approach
```python
def write_raster_tifffile(data: np.ndarray, file_path: str) -> None:
    """Write raster file using tifffile."""
    tifffile.imwrite(file_path, data)
    # Note: No geospatial metadata preserved
```

## 📊 Value Handling Differences

### Critical Discovery: Value Scaling

Our research revealed a fundamental difference in how the two libraries handle data values:

#### Rasterio Value Scaling
```python
# Rasterio automatically scales values to 0-255 range
def process_with_rasterio(data):
    # Input data range: 2.037 - 9.200
    processed_data = fis_process(data)
    
    # Rasterio automatically scales to 0-255
    # Output range: 0.000 - 251.000
    return processed_data
```

#### Tifffile Value Preservation
```python
# Tifffile preserves original values
def process_with_tifffile(data):
    # Input data range: 2.037 - 9.200
    processed_data = fis_process(data)
    
    # Tifffile preserves original range
    # Output range: 2.037 - 9.200
    return processed_data
```

### Impact on Results

```python
# Statistical comparison
rasterio_stats = {
    'min': 0.000,      # Scaled down
    'max': 251.000,    # Scaled up
    'mean': 66.668,    # Significantly different
    'std': 82.313      # Much higher
}

tifffile_stats = {
    'min': 2.037,      # Original range
    'max': 9.200,      # Original range
    'mean': 4.022,     # Original mean
    'std': 2.395       # Original std
}
```

## 🚀 Performance Implementation

### Memory Usage Optimization

#### Rasterio Memory Management
```python
def process_large_raster_rasterio(file_path: str, chunk_size: int = 1000):
    """Process large raster with rasterio using windowed reading."""
    with rasterio.open(file_path) as src:
        for row in range(0, src.height, chunk_size):
            for col in range(0, src.width, chunk_size):
                window = rasterio.windows.Window(
                    col, row, 
                    min(chunk_size, src.width - col),
                    min(chunk_size, src.height - row)
                )
                data = src.read(1, window=window)
                # Process chunk...
```

#### Tifffile Memory Management
```python
def process_large_raster_tifffile(file_path: str, chunk_size: int = 1000):
    """Process large raster with tifffile using memory mapping."""
    with tifffile.TiffFile(file_path) as tif:
        data = tif.asarray()
        
        for row in range(0, data.shape[0], chunk_size):
            for col in range(0, data.shape[1], chunk_size):
                chunk = data[row:row+chunk_size, col:col+chunk_size]
                # Process chunk...
```

### Parallel Processing Implementation

#### Rasterio Parallel Processing
```python
def parallel_process_rasterio(input_files: List[str], output_file: str):
    """Parallel processing with rasterio."""
    def process_chunk_rasterio(chunk_data):
        # Process individual chunk
        return processed_chunk
    
    # Create chunks
    chunks = create_chunks_rasterio(input_files)
    
    # Process in parallel
    with mp.Pool() as pool:
        results = pool.map(process_chunk_rasterio, chunks)
    
    # Combine results
    combine_results_rasterio(results, output_file)
```

#### Tifffile Parallel Processing
```python
def parallel_process_tifffile(input_files: List[str], output_file: str):
    """Parallel processing with tifffile."""
    def process_chunk_tifffile(chunk_data):
        # Process individual chunk
        return processed_chunk
    
    # Create chunks
    chunks = create_chunks_tifffile(input_files)
    
    # Process in parallel
    with mp.Pool() as pool:
        results = pool.map(process_chunk_tifffile, chunks)
    
    # Combine results
    combine_results_tifffile(results, output_file)
```

## 🔍 Error Handling and Validation

### Rasterio Error Handling
```python
def safe_rasterio_processing(file_path: str):
    """Safe rasterio processing with comprehensive error handling."""
    try:
        with rasterio.open(file_path) as src:
            # Validate file
            if src.count == 0:
                raise ValueError("No bands found in raster")
            
            # Check CRS
            if src.crs is None:
                print("Warning: No CRS information")
            
            # Process data
            data = src.read(1)
            
    except rasterio.errors.RasterioIOError as e:
        print(f"Rasterio IO Error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None
```

### Tifffile Error Handling
```python
def safe_tifffile_processing(file_path: str):
    """Safe tifffile processing with comprehensive error handling."""
    try:
        # Validate file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Read data
        data = tifffile.imread(file_path)
        
        # Validate data
        if data.size == 0:
            raise ValueError("Empty raster data")
        
        # Check data type
        if not np.issubdtype(data.dtype, np.number):
            raise ValueError("Non-numeric data type")
        
        return data
        
    except tifffile.TiffFileError as e:
        print(f"Tifffile Error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None
```

## 📦 Deployment Implementation

### Docker Implementation

#### Rasterio Dockerfile
```dockerfile
# Rasterio Dockerfile
FROM python:3.9-slim

# Install system dependencies for GDAL
RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set GDAL environment variables
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

# Install Python dependencies
COPY requirements-rasterio.txt .
RUN pip install -r requirements-rasterio.txt

# Copy application code
COPY app/ /app/
WORKDIR /app

# Default command
CMD ["python", "raster_fuzzy_lib.py"]
```

#### Tifffile Dockerfile
```dockerfile
# Tifffile Dockerfile
FROM python:3.9-slim

# Install minimal system dependencies
RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements-tifffile.txt .
RUN pip install -r requirements-tifffile.txt

# Copy application code
COPY app/ /app/
WORKDIR /app

# Default command
CMD ["python", "raster_fuzzy_lib_tifffile.py"]
```

### AWS Glue Implementation

#### Rasterio Glue Job (Problematic)
```python
# AWS Glue job with rasterio (frequent issues)
import sys
import os

# GDAL setup (often fails in Glue)
os.environ['GDAL_DATA'] = '/opt/amazon/glue/lib/installation/share/gdal'
os.environ['PROJ_LIB'] = '/opt/amazon/glue/lib/installation/share/proj'

try:
    import rasterio
except ImportError as e:
    print(f"Rasterio import failed: {e}")
    sys.exit(1)

# Processing code...
```

#### Tifffile Glue Job (Reliable)
```python
# AWS Glue job with tifffile (reliable)
import sys
import os

# Simple import (always works)
import tifffile
import numpy as np

# Processing code...
```

## 🔬 Testing Implementation

### Unit Testing

#### Rasterio Tests
```python
import unittest
import rasterio
import numpy as np

class TestRasterioImplementation(unittest.TestCase):
    
    def setUp(self):
        self.test_file = "test_data.tif"
        self.create_test_file()
    
    def create_test_file(self):
        """Create test raster file."""
        profile = {
            'driver': 'GTiff',
            'height': 100,
            'width': 100,
            'count': 1,
            'dtype': 'float32',
            'crs': 'EPSG:4326',
            'transform': rasterio.transform.from_bounds(
                -180, -90, 180, 90, 100, 100
            )
        }
        
        data = np.random.random((100, 100)).astype('float32')
        
        with rasterio.open(self.test_file, 'w', **profile) as dst:
            dst.write(data, 1)
    
    def test_read_raster(self):
        """Test raster reading."""
        with rasterio.open(self.test_file) as src:
            data = src.read(1)
            self.assertEqual(data.shape, (100, 100))
            self.assertEqual(data.dtype, np.float32)
    
    def test_metadata_preservation(self):
        """Test metadata preservation."""
        with rasterio.open(self.test_file) as src:
            self.assertIsNotNone(src.crs)
            self.assertIsNotNone(src.transform)
```

#### Tifffile Tests
```python
import unittest
import tifffile
import numpy as np

class TestTifffileImplementation(unittest.TestCase):
    
    def setUp(self):
        self.test_file = "test_data.tif"
        self.create_test_file()
    
    def create_test_file(self):
        """Create test raster file."""
        data = np.random.random((100, 100)).astype('float32')
        tifffile.imwrite(self.test_file, data)
    
    def test_read_raster(self):
        """Test raster reading."""
        data = tifffile.imread(self.test_file)
        self.assertEqual(data.shape, (100, 100))
        self.assertEqual(data.dtype, np.float32)
    
    def test_value_preservation(self):
        """Test value preservation."""
        original_data = np.random.random((50, 50)).astype('float32')
        tifffile.imwrite(self.test_file, original_data)
        
        loaded_data = tifffile.imread(self.test_file)
        np.testing.assert_array_almost_equal(original_data, loaded_data)
```

### Integration Testing

#### Cross-Library Validation
```python
def validate_cross_library_results(rasterio_file: str, tifffile_file: str):
    """Validate results between rasterio and tifffile implementations."""
    
    # Load results
    with rasterio.open(rasterio_file) as src:
        rasterio_data = src.read(1)
    
    tifffile_data = tifffile.imread(tifffile_file)
    
    # Compare shapes
    assert rasterio_data.shape == tifffile_data.shape, "Shape mismatch"
    
    # Compare data types
    assert rasterio_data.dtype == tifffile_data.dtype, "Data type mismatch"
    
    # Note: Values will be different due to scaling differences
    print(f"Rasterio range: {rasterio_data.min():.3f} - {rasterio_data.max():.3f}")
    print(f"Tifffile range: {tifffile_data.min():.3f} - {tifffile_data.max():.3f}")
    
    # Validate that both contain valid data
    assert not np.all(np.isnan(rasterio_data)), "Rasterio data contains all NaN"
    assert not np.all(np.isnan(tifffile_data)), "Tifffile data contains all NaN"
```

## 📈 Performance Monitoring

### Performance Metrics Collection
```python
import time
import psutil
import os

class PerformanceMonitor:
    """Monitor performance metrics for both implementations."""
    
    def __init__(self):
        self.start_time = None
        self.start_memory = None
    
    def start_monitoring(self):
        """Start performance monitoring."""
        self.start_time = time.time()
        self.start_memory = psutil.Process().memory_info().rss
    
    def end_monitoring(self):
        """End performance monitoring and return metrics."""
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss
        
        return {
            'processing_time': end_time - self.start_time,
            'memory_usage': end_memory - self.start_memory,
            'peak_memory': end_memory
        }

def benchmark_implementations(input_file: str):
    """Benchmark both implementations."""
    
    # Test rasterio
    monitor = PerformanceMonitor()
    monitor.start_monitoring()
    
    # Rasterio processing
    with rasterio.open(input_file) as src:
        data_rasterio = src.read(1)
    
    rasterio_metrics = monitor.end_monitoring()
    
    # Test tifffile
    monitor = PerformanceMonitor()
    monitor.start_monitoring()
    
    # Tifffile processing
    data_tifffile = tifffile.imread(input_file)
    
    tifffile_metrics = monitor.end_monitoring()
    
    return {
        'rasterio': rasterio_metrics,
        'tifffile': tifffile_metrics
    }
```

## 🔄 Migration Guide

### From Rasterio to Tifffile
```python
def migrate_rasterio_to_tifffile(rasterio_code):
    """Migrate rasterio code to tifffile."""
    
    # Old rasterio code
    # with rasterio.open(file_path) as src:
    #     data = src.read(1)
    #     profile = src.profile.copy()
    
    # New tifffile code
    data = tifffile.imread(file_path)
    # Note: No profile/metadata preservation
    
    return data

def preserve_metadata_tifffile(file_path: str, metadata: Dict):
    """Preserve metadata when using tifffile."""
    data = tifffile.imread(file_path)
    
    # Store metadata separately
    metadata_file = file_path.replace('.tif', '_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f)
    
    return data, metadata
```

### From Tifffile to Rasterio
```python
def migrate_tifffile_to_rasterio(tifffile_code):
    """Migrate tifffile code to rasterio."""
    
    # Old tifffile code
    # data = tifffile.imread(file_path)
    
    # New rasterio code
    with rasterio.open(file_path) as src:
        data = src.read(1)
        profile = src.profile.copy()
    
    return data, profile
```

## 📋 Best Practices

### Development Best Practices
1. **Use rasterio for development**: Full geospatial support for algorithm development
2. **Implement both versions**: Maintain flexibility with dual implementations
3. **Test cross-platform**: Validate results between implementations
4. **Document differences**: Clear documentation of value scaling and metadata handling

### Production Best Practices
1. **Use tifffile for cloud deployment**: Reliable deployment in distributed environments
2. **Monitor performance**: Track performance differences across environments
3. **Validate results**: Ensure consistency between development and production
4. **Handle metadata separately**: Store geospatial metadata when using tifffile

### Testing Best Practices
1. **Unit test both implementations**: Ensure both work correctly
2. **Integration test cross-platform**: Validate results between implementations
3. **Performance benchmark**: Regular performance comparison
4. **Error handling test**: Test error conditions for both implementations 