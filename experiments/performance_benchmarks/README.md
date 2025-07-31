# Performance Benchmarks

This directory contains performance benchmarking experiments and results for our FIS processing system.

## Contents

### Scripts
- **`quick_performance_test.py`** - Quick performance testing script
- **`simple_performance_test.py`** - Comprehensive performance testing
- **`performance_comparison_test.py`** - Compare different implementations

### Results
- **`performance_test_summary.md`** - Summary of performance test results
- **`requirements-performance-test.txt`** - Dependencies for performance testing

## Performance Metrics

### Tested Configurations
- Single-threaded vs multiprocessing
- Rasterio vs tifffile implementations
- Different chunk sizes and core counts
- Memory usage and processing time

### Key Results
- **Baseline Performance**: 9.7 minutes for 1000m data
- **Multiprocessing Speedup**: 6-10x improvement
- **Memory Optimization**: Configurable chunk sizes
- **Library Comparison**: Rasterio vs tifffile performance

## Running Benchmarks

### Quick Test
```bash
python quick_performance_test.py
```

### Comprehensive Test
```bash
python simple_performance_test.py
```

### Comparison Test
```bash
python performance_comparison_test.py
```

## Dependencies
```bash
pip install -r requirements-performance-test.txt
```

## Output Files
- Performance metrics and timing data
- Memory usage statistics
- Processing efficiency analysis
- Comparative results between implementations 