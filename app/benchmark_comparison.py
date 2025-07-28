#!/usr/bin/env python3
"""
Comprehensive Benchmark Comparison Script
Compares original vs optimized Spark raster processing with detailed metrics.
"""

import os
import sys
import time
import json
import subprocess
import logging
from datetime import datetime
from pathlib import Path
import psutil
import numpy as np

def setup_benchmark_logging():
    """Setup logging for benchmark comparison."""
    log_file = f"logs/benchmark_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__), log_file

def get_system_info():
    """Get comprehensive system information."""
    return {
        'cpu_count': psutil.cpu_count(),
        'memory_total_gb': psutil.virtual_memory().total / (1024**3),
        'memory_available_gb': psutil.virtual_memory().available / (1024**3),
        'disk_free_gb': psutil.disk_usage('/').free / (1024**3),
        'python_version': sys.version,
        'platform': sys.platform
    }

def run_benchmark_test(script_path, input_files, output_file, config_file, 
                      sample_rate=0.1, local_mode=True, extra_args=None):
    """Run a benchmark test and capture metrics."""
    logger = logging.getLogger(__name__)
    
    # Prepare command
    cmd = [
        'python', script_path,
        input_files['social'],
        input_files['environmental'], 
        input_files['strategic'],
        output_file,
        '--config', config_file,
        '--sample-rate', str(sample_rate)
    ]
    
    if local_mode:
        cmd.append('--local')
    
    if extra_args:
        cmd.extend(extra_args)
    
    logger.info(f"🚀 Running command: {' '.join(cmd)}")
    
    # Capture start metrics
    start_time = time.time()
    start_memory = psutil.virtual_memory()
    start_cpu = psutil.cpu_percent(interval=1)
    
    # Run the command
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        end_time = time.time()
        end_memory = psutil.virtual_memory()
        end_cpu = psutil.cpu_percent(interval=1)
        
        # Calculate metrics
        execution_time = end_time - start_time
        memory_used = (end_memory.used - start_memory.used) / (1024**3)  # GB
        
        metrics = {
            'execution_time': execution_time,
            'memory_used_gb': memory_used,
            'cpu_usage_start': start_cpu,
            'cpu_usage_end': end_cpu,
            'return_code': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'success': result.returncode == 0
        }
        
        if result.returncode == 0:
            logger.info(f"✅ Test completed successfully in {execution_time:.2f}s")
        else:
            logger.error(f"❌ Test failed with return code {result.returncode}")
            logger.error(f"STDERR: {result.stderr}")
        
        return metrics
        
    except subprocess.TimeoutExpired:
        logger.error("⏰ Test timed out after 1 hour")
        return {
            'execution_time': 3600,
            'memory_used_gb': 0,
            'cpu_usage_start': 0,
            'cpu_usage_end': 0,
            'return_code': -1,
            'stdout': '',
            'stderr': 'Timeout after 1 hour',
            'success': False
        }
    except Exception as e:
        logger.error(f"💥 Test failed with exception: {e}")
        return {
            'execution_time': 0,
            'memory_used_gb': 0,
            'cpu_usage_start': 0,
            'cpu_usage_end': 0,
            'return_code': -1,
            'stdout': '',
            'stderr': str(e),
            'success': False
        }

def analyze_output_file(output_file):
    """Analyze the output file for quality metrics."""
    try:
        import rasterio
        with rasterio.open(output_file) as src:
            data = src.read(1)
            
            # Calculate statistics
            valid_data = data[data != src.nodata]
            if len(valid_data) == 0:
                return {
                    'valid_pixels': 0,
                    'total_pixels': data.size,
                    'min_value': None,
                    'max_value': None,
                    'mean_value': None,
                    'std_value': None
                }
            
            return {
                'valid_pixels': len(valid_data),
                'total_pixels': data.size,
                'min_value': float(np.min(valid_data)),
                'max_value': float(np.max(valid_data)),
                'mean_value': float(np.mean(valid_data)),
                'std_value': float(np.std(valid_data))
            }
    except Exception as e:
        logging.getLogger(__name__).error(f"Error analyzing output file: {e}")
        return None

def run_comprehensive_benchmark():
    """Run comprehensive benchmark comparison."""
    logger, log_file = setup_benchmark_logging()
    
    logger.info("🎯 Starting Comprehensive Benchmark Comparison")
    logger.info(f"📝 Log file: {log_file}")
    
    # System information
    system_info = get_system_info()
    logger.info(f"💻 System Info: {system_info}")
    
    # Test configuration
    input_files = {
        'social': 'app/files/input/base/socioeconomico_1000m.tif',
        'environmental': 'app/files/input/base/ambiental_1000m.tif',
        'strategic': 'app/files/input/base/estratégico_1000m.tif'
    }
    
    config_file = 'app/config/raster_fis_config.json'
    sample_rate = 0.1  # 10% sampling
    
    # Test scenarios
    test_scenarios = [
        {
            'name': 'Original_Simple',
            'script': 'app/raster_fuzzy_spark_simple.py',
            'extra_args': ['--block-size', '500']
        },
        {
            'name': 'Optimized',
            'script': 'app/raster_fuzzy_spark_optimized.py',
            'extra_args': ['--block-size', '1000']
        },
        {
            'name': 'Optimized_Large_Blocks',
            'script': 'app/raster_fuzzy_spark_optimized.py',
            'extra_args': ['--block-size', '2000']
        },
        {
            'name': 'Optimized_Small_Blocks',
            'script': 'app/raster_fuzzy_spark_optimized.py',
            'extra_args': ['--block-size', '500']
        }
    ]
    
    results = {}
    
    for scenario in test_scenarios:
        logger.info(f"\n🔬 Testing Scenario: {scenario['name']}")
        
        # Create unique output file
        output_file = f"app/files/output/benchmark_{scenario['name']}_{datetime.now().strftime('%H%M%S')}.tif"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Run benchmark
        metrics = run_benchmark_test(
            script_path=scenario['script'],
            input_files=input_files,
            output_file=output_file,
            config_file=config_file,
            sample_rate=sample_rate,
            local_mode=True,
            extra_args=scenario['extra_args']
        )
        
        # Analyze output quality
        quality_metrics = analyze_output_file(output_file)
        
        # Store results
        results[scenario['name']] = {
            'performance_metrics': metrics,
            'quality_metrics': quality_metrics,
            'output_file': output_file
        }
        
        logger.info(f"📊 Performance: {metrics['execution_time']:.2f}s, "
                   f"Memory: {metrics['memory_used_gb']:.2f}GB, "
                   f"Success: {metrics['success']}")
        
        if quality_metrics:
            logger.info(f"📈 Quality: {quality_metrics['valid_pixels']} valid pixels, "
                       f"Mean: {quality_metrics['mean_value']:.3f}")
    
    # Generate comparison report
    generate_comparison_report(results, log_file)
    
    return results

def generate_comparison_report(results, log_file):
    """Generate a comprehensive comparison report."""
    logger = logging.getLogger(__name__)
    
    logger.info("\n" + "="*80)
    logger.info("📊 COMPREHENSIVE BENCHMARK COMPARISON REPORT")
    logger.info("="*80)
    
    # Performance comparison
    logger.info("\n🏃 PERFORMANCE COMPARISON:")
    logger.info("-" * 50)
    
    baseline_time = None
    for name, result in results.items():
        if result['performance_metrics']['success']:
            time = result['performance_metrics']['execution_time']
            memory = result['performance_metrics']['memory_used_gb']
            
            if baseline_time is None:
                baseline_time = time
                speedup = 1.0
            else:
                speedup = baseline_time / time
            
            logger.info(f"{name:25} | {time:8.2f}s | {memory:6.2f}GB | {speedup:6.2f}x")
    
    # Quality comparison
    logger.info("\n📈 QUALITY COMPARISON:")
    logger.info("-" * 50)
    
    for name, result in results.items():
        if result['quality_metrics']:
            qm = result['quality_metrics']
            logger.info(f"{name:25} | {qm['valid_pixels']:8d} | "
                       f"{qm['mean_value']:8.3f} | {qm['std_value']:8.3f}")
    
    # Recommendations
    logger.info("\n💡 RECOMMENDATIONS:")
    logger.info("-" * 50)
    
    # Find fastest successful run
    fastest = None
    for name, result in results.items():
        if result['performance_metrics']['success']:
            if fastest is None or result['performance_metrics']['execution_time'] < fastest['performance_metrics']['execution_time']:
                fastest = (name, result)
    
    if fastest:
        logger.info(f"🏆 Fastest: {fastest[0]} ({fastest[1]['performance_metrics']['execution_time']:.2f}s)")
    
    # Find most memory efficient
    most_efficient = None
    for name, result in results.items():
        if result['performance_metrics']['success']:
            if most_efficient is None or result['performance_metrics']['memory_used_gb'] < most_efficient['performance_metrics']['memory_used_gb']:
                most_efficient = (name, result)
    
    if most_efficient:
        logger.info(f"💾 Most Memory Efficient: {most_efficient[0]} ({most_efficient[1]['performance_metrics']['memory_used_gb']:.2f}GB)")
    
    # Save detailed results to JSON
    results_file = log_file.replace('.log', '_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"\n📄 Detailed results saved to: {results_file}")
    logger.info(f"📝 Full log saved to: {log_file}")

def main():
    """Main function for benchmark comparison."""
    try:
        results = run_comprehensive_benchmark()
        print("\n🎉 Benchmark comparison completed successfully!")
        print("Check the log files for detailed results.")
        return results
    except Exception as e:
        print(f"❌ Benchmark comparison failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main() 