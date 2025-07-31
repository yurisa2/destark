#!/usr/bin/env python3
"""
Runtime Prediction Analysis for 300m Resolution Processing
Based on 1000m job statistics and scaling factors
"""

import math
import time
from datetime import timedelta

def format_time(seconds):
    """Format seconds into human readable time"""
    return str(timedelta(seconds=int(seconds)))

def calculate_runtime_predictions():
    """Calculate runtime predictions for 300m resolution"""
    
    print("=== RUNTIME PREDICTION ANALYSIS ===")
    print("Based on 1000m job statistics and scaling factors")
    print()
    
    # 1000m job statistics (from completed run)
    print("=== 1000m JOB STATISTICS ===")
    print("Raster dimensions: 4424 x 4593 pixels")
    print("Total pixels: 20,331,432")
    print("Processing time: 582.54 seconds (~9.7 minutes)")
    print("Block size: 200 rows")
    print("Partitions: 4")
    print("Memory usage: ~4GB")
    print("Output file size: 81MB")
    print()
    
    # 300m file dimensions
    print("=== 300m FILE DIMENSIONS ===")
    print("Raster dimensions: 14479 x 15187 pixels")
    print("Total pixels: 219,834,973")
    print("File sizes: 210MB each (3 files = 630MB total)")
    print()
    
    # Calculate scaling factors
    print("=== SCALING FACTORS ===")
    pixel_ratio = (14479 * 15187) / (4424 * 4593)
    print(f"Pixel ratio (300m/1000m): {pixel_ratio:.2f}x")
    print(f"Linear dimension ratio: {math.sqrt(pixel_ratio):.2f}x")
    print()
    
    # Base processing time per pixel
    base_time_per_pixel = 582.54 / (4424 * 4593)
    print(f"Base processing time per pixel: {base_time_per_pixel:.8f} seconds")
    print()
    
    # Runtime predictions for different configurations
    print("=== RUNTIME PREDICTIONS FOR 300m RESOLUTION ===")
    print()
    
    # Conservative configuration (2 cores, 2GB memory)
    print("1. CONSERVATIVE CONFIGURATION")
    print("   Block size: 100 rows")
    print("   Partitions: 2")
    print("   Memory: ~2GB")
    conservative_time = (219834973 * base_time_per_pixel) / 2  # 2 cores
    print(f"   Predicted runtime: {format_time(conservative_time)} ({conservative_time:.0f} seconds)")
    print(f"   Estimated memory: ~8GB (4x more pixels)")
    print()
    
    # Balanced configuration (4 cores, 4GB memory) - same as 1000m
    print("2. BALANCED CONFIGURATION (Same as 1000m)")
    print("   Block size: 200 rows")
    print("   Partitions: 4")
    print("   Memory: ~4GB")
    balanced_time = (219834973 * base_time_per_pixel) / 4  # 4 cores
    print(f"   Predicted runtime: {format_time(balanced_time)} ({balanced_time:.0f} seconds)")
    print(f"   Estimated memory: ~16GB (4x more pixels)")
    print()
    
    # Performance configuration (8 cores, 8GB memory)
    print("3. PERFORMANCE CONFIGURATION")
    print("   Block size: 300 rows")
    print("   Partitions: 8")
    print("   Memory: ~8GB")
    performance_time = (219834973 * base_time_per_pixel) / 8  # 8 cores
    print(f"   Predicted runtime: {format_time(performance_time)} ({performance_time:.0f} seconds)")
    print(f"   Estimated memory: ~32GB (4x more pixels)")
    print()
    
    # Maximum configuration (10 cores, all memory)
    print("4. MAXIMUM CONFIGURATION")
    print("   Block size: 500 rows")
    print("   Partitions: 10")
    print("   Memory: Maximum available")
    maximum_time = (219834973 * base_time_per_pixel) / 10  # 10 cores
    print(f"   Predicted runtime: {format_time(maximum_time)} ({maximum_time:.0f} seconds)")
    print(f"   Estimated memory: ~40GB (4x more pixels)")
    print()
    
    # Memory-optimized configuration
    print("5. MEMORY-OPTIMIZED CONFIGURATION")
    print("   Block size: 100 rows (smaller blocks)")
    print("   Partitions: 6")
    print("   Memory: ~12GB")
    memory_optimized_time = (219834973 * base_time_per_pixel) / 6  # 6 cores
    print(f"   Predicted runtime: {format_time(memory_optimized_time)} ({memory_optimized_time:.0f} seconds)")
    print(f"   Estimated memory: ~12GB (smaller blocks)")
    print()
    
    # Summary table
    print("=== SUMMARY TABLE ===")
    print("Configuration          | Runtime    | Memory | Cores | Block Size")
    print("----------------------|------------|--------|-------|-----------")
    print(f"Conservative          | {format_time(conservative_time):<10} | ~8GB   | 2     | 100")
    print(f"Balanced              | {format_time(balanced_time):<10} | ~16GB  | 4     | 200")
    print(f"Performance           | {format_time(performance_time):<10} | ~32GB  | 8     | 300")
    print(f"Maximum               | {format_time(maximum_time):<10} | ~40GB  | 10    | 500")
    print(f"Memory-Optimized      | {format_time(memory_optimized_time):<10} | ~12GB  | 6     | 100")
    print()
    
    # Recommendations
    print("=== RECOMMENDATIONS ===")
    print("1. For systems with 16GB+ RAM: Use Balanced configuration")
    print("2. For systems with 32GB+ RAM: Use Performance configuration")
    print("3. For systems with limited RAM: Use Memory-Optimized configuration")
    print("4. For maximum speed: Use Maximum configuration (if sufficient RAM)")
    print()
    
    # Output file size prediction
    print("=== OUTPUT FILE SIZE PREDICTION ===")
    output_size_ratio = pixel_ratio
    predicted_output_size = 81 * output_size_ratio
    print(f"Predicted output file size: {predicted_output_size:.0f}MB")
    print(f"Storage requirement: {predicted_output_size:.0f}MB per configuration")
    print()
    
    # Command examples
    print("=== COMMAND EXAMPLES ===")
    print("Balanced configuration (recommended for most systems):")
    print("python app/raster_fuzzy_spark_simple.py \\")
    print("  app/files/input/300m/socioeconomico_300m.tif \\")
    print("  app/files/input/300m/ambiental_300m.tif \\")
    print("  app/files/input/300m/estrategico_300m.tif \\")
    print("  app/files/output/300m/result_median_300m.tif \\")
    print("  --config app/config/config_median.json \\")
    print("  --local \\")
    print("  --block-size 200 \\")
    print("  --partitions 4 \\")
    print("  --verbose")
    print()
    
    print("Memory-optimized configuration (for limited RAM):")
    print("python app/raster_fuzzy_spark_simple.py \\")
    print("  app/files/input/300m/socioeconomico_300m.tif \\")
    print("  app/files/input/300m/ambiental_300m.tif \\")
    print("  app/files/input/300m/estrategico_300m.tif \\")
    print("  app/files/output/300m/result_median_300m.tif \\")
    print("  --config app/config/config_median.json \\")
    print("  --local \\")
    print("  --block-size 100 \\")
    print("  --partitions 6 \\")
    print("  --verbose")
    print()
    
    return {
        'conservative': conservative_time,
        'balanced': balanced_time,
        'performance': performance_time,
        'maximum': maximum_time,
        'memory_optimized': memory_optimized_time,
        'pixel_ratio': pixel_ratio,
        'predicted_output_size': predicted_output_size
    }

if __name__ == "__main__":
    results = calculate_runtime_predictions() 