#!/usr/bin/env python3
"""
Quick Performance Test: Rasterio vs Tifffile FIS
Simple performance comparison without complex dependencies
"""

import os
import sys
import time
import multiprocessing as mp
from pathlib import Path

# Add the app directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

def get_optimal_core_count():
    """Get optimal number of cores."""
    total_cores = mp.cpu_count()
    optimal_cores = max(1, int(total_cores * 0.75))
    return optimal_cores, total_cores

def test_rasterio_performance():
    """Test FIS performance using rasterio implementation."""
    print(f"\n{'='*60}")
    print(f"TESTING RASTERIO IMPLEMENTATION")
    print(f"{'='*60}")
    
    try:
        from raster_fuzzy_lib import UnifiedRasterFuzzyInferenceSystem
        
        # Setup paths
        base_dir = Path(__file__).parent
        config_path = base_dir / "app" / "config" / "config_max.json"
        input_dir = base_dir / "app" / "files" / "input" / "base"
        output_dir = base_dir / "app" / "files" / "output"
        
        # Input files
        social_file = input_dir / "socioeconomico_1000m.tif"
        environmental_file = input_dir / "ambiental_1000m.tif"
        strategic_file = input_dir / "estratégico_1000m.tif"
        output_file = output_dir / "output_1000m_config_max_rasterio_quick.tif"
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get core count
        optimal_cores, total_cores = get_optimal_core_count()
        print(f"Using {optimal_cores}/{total_cores} CPU cores")
        
        # Initialize FIS system
        print("🔄 Initializing FIS system (rasterio)...")
        init_start = time.time()
        
        fis = UnifiedRasterFuzzyInferenceSystem(str(config_path))
        
        init_time = time.time() - init_start
        print(f"✅ FIS system initialized in {init_time:.2f} seconds")
        
        # Process rasters
        print("🔄 Processing rasters with rasterio...")
        processing_start = time.time()
        
        fis.process_rasters(
            social_tiff=str(social_file),
            environmental_tiff=str(environmental_file),
            strategic_tiff=str(strategic_file),
            output_tiff=str(output_file),
            nodata_value=5.0,
            parallel=True,
            num_cores=optimal_cores,
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
            return {'library': 'rasterio', 'success': False}
            
    except Exception as e:
        print(f"❌ Error during rasterio processing: {e}")
        import traceback
        traceback.print_exc()
        return {'library': 'rasterio', 'success': False, 'error': str(e)}

def test_tifffile_performance():
    """Test FIS performance using tifffile implementation."""
    print(f"\n{'='*60}")
    print(f"TESTING TIFFFILE IMPLEMENTATION")
    print(f"{'='*60}")
    
    try:
        from raster_fuzzy_lib_tifffile import UnifiedRasterFuzzyInferenceSystem as UnifiedRasterFuzzyInferenceSystemTifffile
        
        # Setup paths
        base_dir = Path(__file__).parent
        config_path = base_dir / "app" / "config" / "config_max.json"
        input_dir = base_dir / "app" / "files" / "input" / "base"
        output_dir = base_dir / "app" / "files" / "output"
        
        # Input files
        social_file = input_dir / "socioeconomico_1000m.tif"
        environmental_file = input_dir / "ambiental_1000m.tif"
        strategic_file = input_dir / "estratégico_1000m.tif"
        output_file = output_dir / "output_1000m_config_max_tifffile_quick.tif"
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get core count
        optimal_cores, total_cores = get_optimal_core_count()
        print(f"Using {optimal_cores}/{total_cores} CPU cores")
        
        # Initialize FIS system
        print("🔄 Initializing FIS system (tifffile)...")
        init_start = time.time()
        
        fis = UnifiedRasterFuzzyInferenceSystemTifffile(str(config_path))
        
        init_time = time.time() - init_start
        print(f"✅ FIS system initialized in {init_time:.2f} seconds")
        
        # Process rasters
        print("🔄 Processing rasters with tifffile...")
        processing_start = time.time()
        
        fis.process_rasters(
            social_tiff=str(social_file),
            environmental_tiff=str(environmental_file),
            strategic_tiff=str(strategic_file),
            output_tiff=str(output_file),
            nodata_value=5.0,
            parallel=True,
            num_cores=optimal_cores,
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
            return {'library': 'tifffile', 'success': False}
            
    except Exception as e:
        print(f"❌ Error during tifffile processing: {e}")
        import traceback
        traceback.print_exc()
        return {'library': 'tifffile', 'success': False, 'error': str(e)}

def compare_outputs(rasterio_output, tifffile_output):
    """Compare the outputs from both libraries."""
    print(f"\n{'='*60}")
    print(f"COMPARING OUTPUTS")
    print(f"{'='*60}")
    
    try:
        import tifffile
        import numpy as np
        
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

def create_performance_summary(rasterio_results, tifffile_results, comparison_results=None):
    """Create a performance summary."""
    print(f"\n{'='*60}")
    print(f"PERFORMANCE SUMMARY")
    print(f"{'='*60}")
    
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
    
    # Display summary
    print("\n📊 Performance Summary:")
    print("-" * 80)
    for row in summary_data:
        print(f"{row['Library']:10} | {row['Init Time (s)']:12} | {row['Processing Time (s)']:18} | {row['Total Time (s)']:12} | {row['Output Size (MB)']:15} | {row['Status']}")
    
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
    
    return summary_data

def main():
    """Main function to run performance tests."""
    print(f"🚀 Starting Quick FIS Performance Comparison Test")
    print(f"   Config: config_max")
    print(f"   Resolution: 1000m")
    
    # Test rasterio implementation
    rasterio_results = test_rasterio_performance()
    
    # Brief pause between tests
    time.sleep(2)
    
    # Test tifffile implementation
    tifffile_results = test_tifffile_performance()
    
    # Compare outputs if both succeeded
    comparison_results = None
    if rasterio_results.get('success') and tifffile_results.get('success'):
        rasterio_output = rasterio_results['output_file']
        tifffile_output = tifffile_results['output_file']
        comparison_results = compare_outputs(rasterio_output, tifffile_output)
    
    # Create performance summary
    summary_data = create_performance_summary(rasterio_results, tifffile_results, comparison_results)
    
    print(f"\n{'='*60}")
    print(f"TEST COMPLETED")
    print(f"{'='*60}")
    
    successful_tests = sum(1 for r in [rasterio_results, tifffile_results] if r.get('success'))
    print(f"✅ Completed tests: {successful_tests}/2")
    
    if successful_tests == 2:
        print(f"📊 Performance comparison completed successfully!")
        print(f"📁 Check the output files for detailed results")

if __name__ == "__main__":
    main() 