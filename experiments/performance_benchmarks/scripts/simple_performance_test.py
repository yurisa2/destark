#!/usr/bin/env python3
"""
Simple Performance Comparison Test: Rasterio vs Tifffile FIS Implementations
This script measures basic performance metrics between the two libraries.
"""

import os
import sys
import time
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


class SimplePerformanceTester:
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
        
        print(f"🔧 Simple Performance Tester initialized:")
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
    
    def test_rasterio_performance(self):
        """Test FIS performance using rasterio implementation."""
        print(f"\n{'='*80}")
        print(f"TESTING RASTERIO IMPLEMENTATION")
        print(f"{'='*80}")
        
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
            
            processing_time = time.time() - processing_start
            total_time = init_time + processing_time
            
            # Check if output was created
            if output_file.exists():
                output_size_mb = output_file.stat().st_size / (1024 * 1024)
                print(f"✅ Rasterio processing completed successfully!")
                print(f"   Init time: {init_time:.2f} seconds")
                print(f"   Processing time: {processing_time:.2f} seconds")
                print(f"   Total time: {total_time:.2f} seconds")
                print(f"   Output file size: {output_size_mb:.1f} MB")
                
                return {
                    'library': 'rasterio',
                    'init_time': init_time,
                    'processing_time': processing_time,
                    'total_time': total_time,
                    'output_file': str(output_file),
                    'output_size_mb': output_size_mb,
                    'success': True
                }
            else:
                print(f"❌ Rasterio processing failed - output file not created")
                return {
                    'library': 'rasterio',
                    'success': False
                }
                
        except Exception as e:
            print(f"❌ Error during rasterio processing: {e}")
            import traceback
            traceback.print_exc()
            return {
                'library': 'rasterio',
                'success': False,
                'error': str(e)
            }
    
    def test_tifffile_performance(self):
        """Test FIS performance using tifffile implementation."""
        print(f"\n{'='*80}")
        print(f"TESTING TIFFFILE IMPLEMENTATION")
        print(f"{'='*80}")
        
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
            
            processing_time = time.time() - processing_start
            total_time = init_time + processing_time
            
            # Check if output was created
            if output_file.exists():
                output_size_mb = output_file.stat().st_size / (1024 * 1024)
                print(f"✅ Tifffile processing completed successfully!")
                print(f"   Init time: {init_time:.2f} seconds")
                print(f"   Processing time: {processing_time:.2f} seconds")
                print(f"   Total time: {total_time:.2f} seconds")
                print(f"   Output file size: {output_size_mb:.1f} MB")
                
                return {
                    'library': 'tifffile',
                    'init_time': init_time,
                    'processing_time': processing_time,
                    'total_time': total_time,
                    'output_file': str(output_file),
                    'output_size_mb': output_size_mb,
                    'success': True
                }
            else:
                print(f"❌ Tifffile processing failed - output file not created")
                return {
                    'library': 'tifffile',
                    'success': False
                }
                
        except Exception as e:
            print(f"❌ Error during tifffile processing: {e}")
            import traceback
            traceback.print_exc()
            return {
                'library': 'tifffile',
                'success': False,
                'error': str(e)
            }
    
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
        results_file = results_dir / f"simple_performance_test_{self.resolution}_{self.config_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        # Create summary table
        summary_data = []
        
        if rasterio_results.get('success'):
            summary_data.append({
                'Library': 'Rasterio',
                'Init Time (s)': f"{rasterio_results['init_time']:.2f}",
                'Processing Time (s)': f"{rasterio_results['processing_time']:.2f}",
                'Total Time (s)': f"{rasterio_results['total_time']:.2f}",
                'Output Size (MB)': f"{rasterio_results['output_size_mb']:.1f}",
                'Status': '✅ Success'
            })
        
        if tifffile_results.get('success'):
            summary_data.append({
                'Library': 'Tifffile',
                'Init Time (s)': f"{tifffile_results['init_time']:.2f}",
                'Processing Time (s)': f"{tifffile_results['processing_time']:.2f}",
                'Total Time (s)': f"{tifffile_results['total_time']:.2f}",
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
        
        # Create simple performance plot
        self.create_performance_plot(rasterio_results, tifffile_results, results_dir)
        
        print(f"\n📁 Detailed results saved to: {results_file}")
        print(f"📊 Performance plot saved to: {results_dir}")
        
        return report_data
    
    def create_performance_plot(self, rasterio_results, tifffile_results, results_dir):
        """Create performance visualization plot."""
        try:
            if rasterio_results.get('success') and tifffile_results.get('success'):
                # Create comparison plot
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
                
                # Time comparison
                libraries = ['Rasterio', 'Tifffile']
                init_times = [rasterio_results['init_time'], tifffile_results['init_time']]
                processing_times = [rasterio_results['processing_time'], tifffile_results['processing_time']]
                total_times = [rasterio_results['total_time'], tifffile_results['total_time']]
                
                x = np.arange(len(libraries))
                width = 0.25
                
                ax1.bar(x - width, init_times, width, label='Init Time', color='lightblue')
                ax1.bar(x, processing_times, width, label='Processing Time', color='orange')
                ax1.bar(x + width, total_times, width, label='Total Time', color='green')
                
                ax1.set_title('Time Comparison')
                ax1.set_ylabel('Time (seconds)')
                ax1.set_xticks(x)
                ax1.set_xticklabels(libraries)
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Output size comparison
                output_sizes = [rasterio_results['output_size_mb'], tifffile_results['output_size_mb']]
                bars = ax2.bar(libraries, output_sizes, color=['blue', 'red'], alpha=0.7)
                ax2.set_title('Output File Size')
                ax2.set_ylabel('Size (MB)')
                ax2.grid(True, alpha=0.3)
                
                # Add value labels on bars
                for bar, size in zip(bars, output_sizes):
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                            f'{size:.1f}MB', ha='center', va='bottom')
                
                # Speed comparison (inverse of time)
                speeds = [1/rasterio_results['total_time'], 1/tifffile_results['total_time']]
                bars = ax3.bar(libraries, speeds, color=['blue', 'red'], alpha=0.7)
                ax3.set_title('Processing Speed (1/time)')
                ax3.set_ylabel('Speed (1/seconds)')
                ax3.grid(True, alpha=0.3)
                
                # Add value labels on bars
                for bar, speed in zip(bars, speeds):
                    height = bar.get_height()
                    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                            f'{speed:.3f}', ha='center', va='bottom')
                
                # Performance summary table
                ax4.axis('tight')
                ax4.axis('off')
                
                # Create summary table data
                table_data = [
                    ['Metric', 'Rasterio', 'Tifffile', 'Ratio (T/R)'],
                    ['Init Time (s)', f"{rasterio_results['init_time']:.2f}", f"{tifffile_results['init_time']:.2f}", f"{tifffile_results['init_time']/rasterio_results['init_time']:.2f}"],
                    ['Processing Time (s)', f"{rasterio_results['processing_time']:.2f}", f"{tifffile_results['processing_time']:.2f}", f"{tifffile_results['processing_time']/rasterio_results['processing_time']:.2f}"],
                    ['Total Time (s)', f"{rasterio_results['total_time']:.2f}", f"{tifffile_results['total_time']:.2f}", f"{tifffile_results['total_time']/rasterio_results['total_time']:.2f}"],
                    ['Output Size (MB)', f"{rasterio_results['output_size_mb']:.1f}", f"{tifffile_results['output_size_mb']:.1f}", f"{tifffile_results['output_size_mb']/rasterio_results['output_size_mb']:.2f}"]
                ]
                
                table = ax4.table(cellText=table_data, cellLoc='center', loc='center')
                table.auto_set_font_size(False)
                table.set_fontsize(10)
                table.scale(1.2, 1.5)
                
                # Color header row
                for i in range(len(table_data[0])):
                    table[(0, i)].set_facecolor('#4CAF50')
                    table[(0, i)].set_text_props(weight='bold', color='white')
                
                plt.tight_layout()
                plot_file = results_dir / f"simple_performance_comparison_{self.resolution}_{self.config_name}.png"
                plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                plt.show()
                
                print(f"📊 Performance plot created: {plot_file}")
                
        except Exception as e:
            print(f"⚠️  Could not create performance plot: {e}")
    
    def run_complete_test(self):
        """Run the complete performance test."""
        print(f"🚀 Starting Simple FIS Performance Comparison Test")
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
    ]
    
    all_results = []
    
    for test_config in test_configs:
        print(f"\n{'='*100}")
        print(f"PERFORMANCE TEST: {test_config['config']} - {test_config['resolution']}")
        print(f"{'='*100}")
        
        try:
            tester = SimplePerformanceTester(
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