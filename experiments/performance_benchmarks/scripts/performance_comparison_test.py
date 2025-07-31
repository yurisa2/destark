#!/usr/bin/env python3
"""
Performance Comparison Test: Rasterio vs Tifffile FIS Implementations
This script measures and compares performance metrics between the two libraries.
"""

import os
import sys
import time
import psutil
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
import gc

# Add the app directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from raster_fuzzy_lib import UnifiedRasterFuzzyInferenceSystem
from raster_fuzzy_lib_tifffile import UnifiedRasterFuzzyInferenceSystem as UnifiedRasterFuzzyInferenceSystemTifffile


class PerformanceMonitor:
    """Monitor system performance during processing."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.start_time = None
        self.start_memory = None
        self.peak_memory = 0
        self.memory_samples = []
        self.cpu_samples = []
        self.sample_times = []
    
    def start_monitoring(self):
        """Start performance monitoring."""
        self.start_time = time.time()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.peak_memory = self.start_memory
        self.memory_samples = [self.start_memory]
        self.cpu_samples = [self.process.cpu_percent()]
        self.sample_times = [0]
        
        print(f"📊 Starting performance monitoring...")
        print(f"   Initial memory: {self.start_memory:.1f} MB")
    
    def sample_performance(self):
        """Take a performance sample."""
        current_time = time.time() - self.start_time
        current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        current_cpu = self.process.cpu_percent()
        
        self.memory_samples.append(current_memory)
        self.cpu_samples.append(current_cpu)
        self.sample_times.append(current_time)
        
        if current_memory > self.peak_memory:
            self.peak_memory = current_memory
    
    def stop_monitoring(self):
        """Stop performance monitoring and return results."""
        end_time = time.time()
        end_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        total_time = end_time - self.start_time
        memory_increase = end_memory - self.start_memory
        
        results = {
            'total_time': total_time,
            'start_memory': self.start_memory,
            'end_memory': end_memory,
            'peak_memory': self.peak_memory,
            'memory_increase': memory_increase,
            'memory_samples': self.memory_samples,
            'cpu_samples': self.cpu_samples,
            'sample_times': self.sample_times
        }
        
        print(f"📊 Performance monitoring completed:")
        print(f"   Total time: {total_time:.2f} seconds")
        print(f"   Memory increase: {memory_increase:.1f} MB")
        print(f"   Peak memory: {self.peak_memory:.1f} MB")
        
        return results


class FISPerformanceTester:
    """Test FIS performance with different libraries."""
    
    def __init__(self, config_name="config_max", resolution="1000m"):
        self.config_name = config_name
        self.resolution = resolution
        self.base_dir = Path(__file__).parent
        self.config_path = self.base_dir / "app" / "config" / f"{config_name}.json"
        self.input_dir = self.base_dir / "app" / "files" / "input"
        self.output_dir = self.base_dir / "app" / "files" / "output"
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define input files based on resolution
        if resolution == "1000m":
            self.input_subdir = self.input_dir / "base"
            self.social_file = self.input_subdir / "socioeconomico_1000m.tif"
            self.environmental_file = self.input_subdir / "ambiental_1000m.tif"
            self.strategic_file = self.input_subdir / "estratégico_1000m.tif"
        elif resolution == "300m":
            self.input_subdir = self.input_dir / "300m"
            self.social_file = self.input_subdir / "socioeconomico_300m.tif"
            self.environmental_file = self.input_subdir / "ambiental_300m.tif"
            self.strategic_file = self.input_subdir / "estrategico_300m.tif"
        else:
            raise ValueError(f"Unsupported resolution: {resolution}")
        
        # Verify files exist
        self.verify_input_files()
        
        # Get optimal core count
        self.optimal_cores, self.total_cores = self.get_optimal_core_count()
        
        print(f"🔧 FIS Performance Tester initialized:")
        print(f"   Config: {config_name}")
        print(f"   Resolution: {resolution}")
        print(f"   Cores: {self.optimal_cores}/{self.total_cores}")
    
    def verify_input_files(self):
        """Verify that all input files exist."""
        input_files = [self.social_file, self.environmental_file, self.strategic_file]
        for file_path in input_files:
            if not file_path.exists():
                raise FileNotFoundError(f"Input file not found: {file_path}")
        
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
    
    def get_optimal_core_count(self):
        """Get optimal number of cores for processing."""
        total_cores = mp.cpu_count()
        optimal_cores = max(1, int(total_cores * 0.75))
        return optimal_cores, total_cores
    
    def test_rasterio_performance(self, monitor_interval=1.0):
        """Test FIS performance using rasterio implementation."""
        print(f"\n{'='*80}")
        print(f"TESTING RASTERIO IMPLEMENTATION")
        print(f"{'='*80}")
        
        # Initialize performance monitor
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        # Output file
        output_file = self.output_dir / f"output_{self.resolution}_{self.config_name}_rasterio_perf.tif"
        
        try:
            # Initialize FIS system
            print("🔄 Initializing FIS system (rasterio)...")
            init_start = time.time()
            
            fis = UnifiedRasterFuzzyInferenceSystem(str(self.config_path))
            
            init_time = time.time() - init_start
            print(f"✅ FIS system initialized in {init_time:.2f} seconds")
            
            # Process rasters
            print("🔄 Processing rasters with rasterio...")
            processing_start = time.time()
            
            # Start monitoring thread for periodic sampling
            import threading
            stop_monitoring = threading.Event()
            
            def monitor_thread():
                while not stop_monitoring.is_set():
                    monitor.sample_performance()
                    time.sleep(monitor_interval)
            
            monitor_thread = threading.Thread(target=monitor_thread)
            monitor_thread.start()
            
            # Process the rasters
            fis.process_rasters(
                social_tiff=str(self.social_file),
                environmental_tiff=str(self.environmental_file),
                strategic_tiff=str(self.strategic_file),
                output_tiff=str(output_file),
                nodata_value=5.0,
                parallel=True,
                num_cores=self.optimal_cores,
                chunk_size=100
            )
            
            # Stop monitoring
            stop_monitoring.set()
            monitor_thread.join()
            
            processing_time = time.time() - processing_start
            total_time = time.time() - monitor.start_time
            
            # Final performance sample
            monitor.sample_performance()
            performance_results = monitor.stop_monitoring()
            
            # Check if output was created
            if output_file.exists():
                output_size_mb = output_file.stat().st_size / (1024 * 1024)
                print(f"✅ Rasterio processing completed successfully!")
                print(f"   Processing time: {processing_time:.2f} seconds")
                print(f"   Total time: {total_time:.2f} seconds")
                print(f"   Output file size: {output_size_mb:.1f} MB")
                
                performance_results.update({
                    'library': 'rasterio',
                    'init_time': init_time,
                    'processing_time': processing_time,
                    'output_file': str(output_file),
                    'output_size_mb': output_size_mb,
                    'success': True
                })
                
                return performance_results
            else:
                print(f"❌ Rasterio processing failed - output file not created")
                performance_results.update({
                    'library': 'rasterio',
                    'success': False
                })
                return performance_results
                
        except Exception as e:
            print(f"❌ Error during rasterio processing: {e}")
            import traceback
            traceback.print_exc()
            
            performance_results = monitor.stop_monitoring()
            performance_results.update({
                'library': 'rasterio',
                'success': False,
                'error': str(e)
            })
            return performance_results
    
    def test_tifffile_performance(self, monitor_interval=1.0):
        """Test FIS performance using tifffile implementation."""
        print(f"\n{'='*80}")
        print(f"TESTING TIFFFILE IMPLEMENTATION")
        print(f"{'='*80}")
        
        # Initialize performance monitor
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        # Output file
        output_file = self.output_dir / f"output_{self.resolution}_{self.config_name}_tifffile_perf.tif"
        
        try:
            # Initialize FIS system
            print("🔄 Initializing FIS system (tifffile)...")
            init_start = time.time()
            
            fis = UnifiedRasterFuzzyInferenceSystemTifffile(str(self.config_path))
            
            init_time = time.time() - init_start
            print(f"✅ FIS system initialized in {init_time:.2f} seconds")
            
            # Process rasters
            print("🔄 Processing rasters with tifffile...")
            processing_start = time.time()
            
            # Start monitoring thread for periodic sampling
            import threading
            stop_monitoring = threading.Event()
            
            def monitor_thread():
                while not stop_monitoring.is_set():
                    monitor.sample_performance()
                    time.sleep(monitor_interval)
            
            monitor_thread = threading.Thread(target=monitor_thread)
            monitor_thread.start()
            
            # Process the rasters
            fis.process_rasters(
                social_tiff=str(self.social_file),
                environmental_tiff=str(self.environmental_file),
                strategic_tiff=str(self.strategic_file),
                output_tiff=str(output_file),
                nodata_value=5.0,
                parallel=True,
                num_cores=self.optimal_cores,
                chunk_size=100
            )
            
            # Stop monitoring
            stop_monitoring.set()
            monitor_thread.join()
            
            processing_time = time.time() - processing_start
            total_time = time.time() - monitor.start_time
            
            # Final performance sample
            monitor.sample_performance()
            performance_results = monitor.stop_monitoring()
            
            # Check if output was created
            if output_file.exists():
                output_size_mb = output_file.stat().st_size / (1024 * 1024)
                print(f"✅ Tifffile processing completed successfully!")
                print(f"   Processing time: {processing_time:.2f} seconds")
                print(f"   Total time: {total_time:.2f} seconds")
                print(f"   Output file size: {output_size_mb:.1f} MB")
                
                performance_results.update({
                    'library': 'tifffile',
                    'init_time': init_time,
                    'processing_time': processing_time,
                    'output_file': str(output_file),
                    'output_size_mb': output_size_mb,
                    'success': True
                })
                
                return performance_results
            else:
                print(f"❌ Tifffile processing failed - output file not created")
                performance_results.update({
                    'library': 'tifffile',
                    'success': False
                })
                return performance_results
                
        except Exception as e:
            print(f"❌ Error during tifffile processing: {e}")
            import traceback
            traceback.print_exc()
            
            performance_results = monitor.stop_monitoring()
            performance_results.update({
                'library': 'tifffile',
                'success': False,
                'error': str(e)
            })
            return performance_results
    
    def compare_outputs(self, rasterio_output, tifffile_output):
        """Compare the outputs from both libraries."""
        print(f"\n{'='*80}")
        print(f"COMPARING OUTPUTS")
        print(f"{'='*80}")
        
        try:
            import tifffile
            
            # Load both outputs
            print("Loading rasterio output...")
            rasterio_data = tifffile.imread(rasterio_output)
            
            print("Loading tifffile output...")
            tifffile_data = tifffile.imread(tifffile_output)
            
            # Basic comparison
            print(f"Rasterio output shape: {rasterio_data.shape}")
            print(f"Tifffile output shape: {tifffile_data.shape}")
            print(f"Rasterio output dtype: {rasterio_data.dtype}")
            print(f"Tifffile output dtype: {tifffile_data.dtype}")
            
            # Statistical comparison
            rasterio_stats = {
                'min': float(rasterio_data.min()),
                'max': float(rasterio_data.max()),
                'mean': float(rasterio_data.mean()),
                'std': float(rasterio_data.std()),
                'median': float(np.median(rasterio_data))
            }
            
            tifffile_stats = {
                'min': float(tifffile_data.min()),
                'max': float(tifffile_data.max()),
                'mean': float(tifffile_data.mean()),
                'std': float(tifffile_data.std()),
                'median': float(np.median(tifffile_data))
            }
            
            print(f"\nRasterio statistics:")
            for key, value in rasterio_stats.items():
                print(f"  {key}: {value:.3f}")
            
            print(f"\nTifffile statistics:")
            for key, value in tifffile_stats.items():
                print(f"  {key}: {value:.3f}")
            
            # Calculate differences
            if rasterio_data.shape == tifffile_data.shape:
                difference = rasterio_data - tifffile_data
                diff_stats = {
                    'max_abs_diff': float(np.abs(difference).max()),
                    'mean_abs_diff': float(np.abs(difference).mean()),
                    'std_diff': float(difference.std()),
                    'non_zero_pixels': int(np.count_nonzero(difference)),
                    'total_pixels': int(difference.size)
                }
                
                print(f"\nDifference statistics:")
                for key, value in diff_stats.items():
                    print(f"  {key}: {value}")
                
                return {
                    'rasterio_stats': rasterio_stats,
                    'tifffile_stats': tifffile_stats,
                    'difference_stats': diff_stats,
                    'shapes_match': True
                }
            else:
                print(f"❌ Output shapes don't match!")
                return {
                    'rasterio_stats': rasterio_stats,
                    'tifffile_stats': tifffile_stats,
                    'shapes_match': False
                }
                
        except Exception as e:
            print(f"❌ Error comparing outputs: {e}")
            return {'error': str(e)}
    
    def create_performance_report(self, rasterio_results, tifffile_results, comparison_results=None):
        """Create a comprehensive performance report."""
        print(f"\n{'='*80}")
        print(f"PERFORMANCE REPORT")
        print(f"{'='*80}")
        
        # Create results directory
        results_dir = Path("performance_results")
        results_dir.mkdir(exist_ok=True)
        
        # Prepare report data
        report_data = {
            'test_info': {
                'timestamp': datetime.now().isoformat(),
                'config': self.config_name,
                'resolution': self.resolution,
                'cores_used': self.optimal_cores,
                'total_cores': self.total_cores
            },
            'rasterio_results': rasterio_results,
            'tifffile_results': tifffile_results,
            'comparison_results': comparison_results
        }
        
        # Save detailed results
        results_file = results_dir / f"performance_test_{self.resolution}_{self.config_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        # Create summary table
        summary_data = []
        
        if rasterio_results.get('success'):
            summary_data.append({
                'Library': 'Rasterio',
                'Total Time (s)': f"{rasterio_results['total_time']:.2f}",
                'Processing Time (s)': f"{rasterio_results['processing_time']:.2f}",
                'Init Time (s)': f"{rasterio_results['init_time']:.2f}",
                'Peak Memory (MB)': f"{rasterio_results['peak_memory']:.1f}",
                'Memory Increase (MB)': f"{rasterio_results['memory_increase']:.1f}",
                'Output Size (MB)': f"{rasterio_results['output_size_mb']:.1f}",
                'Status': '✅ Success'
            })
        
        if tifffile_results.get('success'):
            summary_data.append({
                'Library': 'Tifffile',
                'Total Time (s)': f"{tifffile_results['total_time']:.2f}",
                'Processing Time (s)': f"{tifffile_results['processing_time']:.2f}",
                'Init Time (s)': f"{tifffile_results['init_time']:.2f}",
                'Peak Memory (MB)': f"{tifffile_results['peak_memory']:.1f}",
                'Memory Increase (MB)': f"{tifffile_results['memory_increase']:.1f}",
                'Output Size (MB)': f"{tifffile_results['output_size_mb']:.1f}",
                'Status': '✅ Success'
            })
        
        # Create DataFrame and display
        df = pd.DataFrame(summary_data)
        print("\n📊 Performance Summary:")
        print(df.to_string(index=False))
        
        # Calculate performance ratios
        if len(summary_data) == 2:
            rasterio_time = float(summary_data[0]['Total Time (s)'])
            tifffile_time = float(summary_data[1]['Total Time (s)'])
            
            if rasterio_time > 0 and tifffile_time > 0:
                time_ratio = tifffile_time / rasterio_time
                print(f"\n⏱️  Performance Ratios:")
                print(f"   Tifffile/Rasterio Time: {time_ratio:.2f}x")
                if time_ratio > 1:
                    print(f"   Rasterio is {time_ratio:.2f}x faster")
                else:
                    print(f"   Tifffile is {1/time_ratio:.2f}x faster")
        
        # Create performance plots
        self.create_performance_plots(rasterio_results, tifffile_results, results_dir)
        
        print(f"\n📁 Detailed results saved to: {results_file}")
        print(f"📊 Performance plots saved to: {results_dir}")
        
        return report_data
    
    def create_performance_plots(self, rasterio_results, tifffile_results, results_dir):
        """Create performance visualization plots."""
        try:
            # Memory usage over time
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # Memory usage comparison
            if rasterio_results.get('success') and tifffile_results.get('success'):
                ax1.plot(rasterio_results['sample_times'], rasterio_results['memory_samples'], 
                        label='Rasterio', linewidth=2, color='blue')
                ax1.plot(tifffile_results['sample_times'], tifffile_results['memory_samples'], 
                        label='Tifffile', linewidth=2, color='red')
                ax1.set_title('Memory Usage Over Time')
                ax1.set_xlabel('Time (seconds)')
                ax1.set_ylabel('Memory (MB)')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
            
            # CPU usage comparison
            if rasterio_results.get('success') and tifffile_results.get('success'):
                ax2.plot(rasterio_results['sample_times'], rasterio_results['cpu_samples'], 
                        label='Rasterio', linewidth=2, color='blue')
                ax2.plot(tifffile_results['sample_times'], tifffile_results['cpu_samples'], 
                        label='Tifffile', linewidth=2, color='red')
                ax2.set_title('CPU Usage Over Time')
                ax2.set_xlabel('Time (seconds)')
                ax2.set_ylabel('CPU (%)')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            
            # Performance metrics bar chart
            if rasterio_results.get('success') and tifffile_results.get('success'):
                metrics = ['Total Time (s)', 'Processing Time (s)', 'Peak Memory (MB)', 'Output Size (MB)']
                rasterio_values = [
                    rasterio_results['total_time'],
                    rasterio_results['processing_time'],
                    rasterio_results['peak_memory'],
                    rasterio_results['output_size_mb']
                ]
                tifffile_values = [
                    tifffile_results['total_time'],
                    tifffile_results['processing_time'],
                    tifffile_results['peak_memory'],
                    tifffile_results['output_size_mb']
                ]
                
                x = np.arange(len(metrics))
                width = 0.35
                
                ax3.bar(x - width/2, rasterio_values, width, label='Rasterio', color='blue', alpha=0.7)
                ax3.bar(x + width/2, tifffile_values, width, label='Tifffile', color='red', alpha=0.7)
                
                ax3.set_title('Performance Metrics Comparison')
                ax3.set_xticks(x)
                ax3.set_xticklabels(metrics, rotation=45)
                ax3.legend()
                ax3.grid(True, alpha=0.3)
            
            # Speed comparison
            if rasterio_results.get('success') and tifffile_results.get('success'):
                libraries = ['Rasterio', 'Tifffile']
                times = [rasterio_results['total_time'], tifffile_results['total_time']]
                colors = ['blue', 'red']
                
                bars = ax4.bar(libraries, times, color=colors, alpha=0.7)
                ax4.set_title('Total Processing Time')
                ax4.set_ylabel('Time (seconds)')
                ax4.grid(True, alpha=0.3)
                
                # Add value labels on bars
                for bar, time_val in zip(bars, times):
                    height = bar.get_height()
                    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                            f'{time_val:.1f}s', ha='center', va='bottom')
            
            plt.tight_layout()
            plot_file = results_dir / f"performance_comparison_{self.resolution}_{self.config_name}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"📊 Performance plots created: {plot_file}")
            
        except Exception as e:
            print(f"⚠️  Could not create performance plots: {e}")
    
    def run_complete_test(self):
        """Run the complete performance test."""
        print(f"🚀 Starting FIS Performance Comparison Test")
        print(f"   Config: {self.config_name}")
        print(f"   Resolution: {self.resolution}")
        print(f"   Cores: {self.optimal_cores}/{self.total_cores}")
        
        # Force garbage collection before tests
        gc.collect()
        
        # Test rasterio implementation
        rasterio_results = self.test_rasterio_performance()
        
        # Force garbage collection between tests
        gc.collect()
        time.sleep(2)  # Brief pause between tests
        
        # Test tifffile implementation
        tifffile_results = self.test_tifffile_performance()
        
        # Compare outputs if both succeeded
        comparison_results = None
        if rasterio_results.get('success') and tifffile_results.get('success'):
            rasterio_output = rasterio_results['output_file']
            tifffile_output = tifffile_results['output_file']
            comparison_results = self.compare_outputs(rasterio_output, tifffile_output)
        
        # Create performance report
        report_data = self.create_performance_report(rasterio_results, tifffile_results, comparison_results)
        
        return report_data


def main():
    """Main function to run performance tests."""
    
    # Test configurations
    test_configs = [
        {"config": "config_max", "resolution": "1000m"},
        # {"config": "config_median", "resolution": "1000m"},
        # {"config": "config_minimum", "resolution": "1000m"},
        # {"config": "config_max", "resolution": "300m"},
    ]
    
    all_results = []
    
    for test_config in test_configs:
        print(f"\n{'='*100}")
        print(f"PERFORMANCE TEST: {test_config['config']} - {test_config['resolution']}")
        print(f"{'='*100}")
        
        try:
            tester = FISPerformanceTester(
                config_name=test_config['config'],
                resolution=test_config['resolution']
            )
            
            results = tester.run_complete_test()
            all_results.append(results)
            
        except Exception as e:
            print(f"❌ Test failed for {test_config}: {e}")
            import traceback
            traceback.print_exc()
    
    # Create overall summary
    print(f"\n{'='*100}")
    print(f"OVERALL TEST SUMMARY")
    print(f"{'='*100}")
    
    successful_tests = [r for r in all_results if r is not None]
    print(f"✅ Completed tests: {len(successful_tests)}/{len(test_configs)}")
    
    if successful_tests:
        print(f"📁 Results saved to: performance_results/")
        print(f"📊 Check the generated plots and JSON files for detailed analysis")


if __name__ == "__main__":
    main() 