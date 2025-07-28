#!/usr/bin/env python3
"""
Quick Benchmark Script - Compare Original vs Ultra-Optimized
Tests both versions with same parameters for accurate comparison.
"""

import os
import sys
import time
import subprocess
import numpy as np
import rasterio

def run_benchmark_test(script_name, input_files, output_file, config_file, extra_args=None):
    """Run a benchmark test and return timing."""
    cmd = [
        'python', script_name,
        input_files['social'],
        input_files['environmental'], 
        input_files['strategic'],
        output_file,
        '--config', config_file,
        '--local',
        '--block-size', '500'
    ]
    
    if extra_args:
        cmd.extend(extra_args)
    
    print(f"Running: {' '.join(cmd)}")
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        end_time = time.time()
        
        if result.returncode == 0:
            print(f"✅ Success: {end_time - start_time:.2f}s")
            return end_time - start_time, True
        else:
            print(f"❌ Failed: {result.stderr}")
            return 0, False
            
    except subprocess.TimeoutExpired:
        print("⏰ Timeout after 10 minutes")
        return 600, False
    except Exception as e:
        print(f"💥 Error: {e}")
        return 0, False

def analyze_output(output_file):
    """Analyze output file for quality metrics."""
    try:
        with rasterio.open(output_file) as src:
            data = src.read(1)
            valid_data = data[data != src.nodata]
            
            if len(valid_data) == 0:
                return None
            
            return {
                'min': float(np.min(valid_data)),
                'max': float(np.max(valid_data)),
                'mean': float(np.mean(valid_data)),
                'std': float(np.std(valid_data)),
                'valid_pixels': len(valid_data)
            }
    except Exception as e:
        print(f"Error analyzing {output_file}: {e}")
        return None

def main():
    """Run quick benchmark comparison."""
    print("🚀 Quick Benchmark: Original vs Ultra-Optimized")
    print("=" * 60)
    
    # Test configuration
    input_files = {
        'social': 'app/files/input/base/socioeconomico_1000m.tif',
        'environmental': 'app/files/input/base/ambiental_1000m.tif',
        'strategic': 'app/files/input/base/estrategico_1000m.tif'
    }
    
    config_file = 'app/config/raster_fis_config.json'
    
    # Test scenarios
    tests = [
        {
            'name': 'Original_Simple',
            'script': 'app/raster_fuzzy_spark_simple.py',
            'output': 'app/files/output/benchmark_original.tif'
        },
        {
            'name': 'Ultra_Optimized',
            'script': 'app/raster_fuzzy_spark_ultra_optimized.py',
            'output': 'app/files/output/benchmark_ultra_optimized.tif'
        }
    ]
    
    results = {}
    
    for test in tests:
        print(f"\n🔬 Testing: {test['name']}")
        print("-" * 40)
        
        # Run benchmark
        execution_time, success = run_benchmark_test(
            test['script'],
            input_files,
            test['output'],
            config_file
        )
        
        # Analyze output quality
        quality_metrics = analyze_output(test['output'])
        
        results[test['name']] = {
            'execution_time': execution_time,
            'success': success,
            'quality_metrics': quality_metrics
        }
        
        if quality_metrics:
            print(f"📊 Quality: min={quality_metrics['min']:.3f}, "
                  f"max={quality_metrics['max']:.3f}, "
                  f"mean={quality_metrics['mean']:.3f}")
    
    # Generate comparison report
    print("\n" + "=" * 60)
    print("📊 BENCHMARK COMPARISON RESULTS")
    print("=" * 60)
    
    baseline_time = None
    for name, result in results.items():
        if result['success']:
            time = result['execution_time']
            if baseline_time is None:
                baseline_time = time
                speedup = 1.0
            else:
                speedup = baseline_time / time
            
            print(f"{name:20} | {time:8.2f}s | {speedup:6.2f}x")
            
            if result['quality_metrics']:
                qm = result['quality_metrics']
                print(f"{'':20} | min={qm['min']:.3f}, max={qm['max']:.3f}, mean={qm['mean']:.3f}")
    
    # Find winner
    fastest = None
    for name, result in results.items():
        if result['success']:
            if fastest is None or result['execution_time'] < fastest['execution_time']:
                fastest = (name, result)
    
    if fastest:
        print(f"\n🏆 Winner: {fastest[0]} ({fastest[1]['execution_time']:.2f}s)")
    
    # Quality comparison
    print("\n📈 Quality Comparison:")
    print("-" * 40)
    
    original_quality = results.get('Original_Simple', {}).get('quality_metrics')
    optimized_quality = results.get('Ultra_Optimized', {}).get('quality_metrics')
    
    if original_quality and optimized_quality:
        mean_diff = abs(original_quality['mean'] - optimized_quality['mean'])
        print(f"Mean difference: {mean_diff:.6f}")
        
        if mean_diff < 0.001:
            print("✅ Quality maintained (difference < 0.001)")
        else:
            print("⚠️  Quality difference detected")
    
    print("\n🎉 Benchmark completed!")

if __name__ == "__main__":
    main() 