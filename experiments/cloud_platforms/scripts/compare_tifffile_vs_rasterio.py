#!/usr/bin/env python3
"""
Compare FIS processing using tifffile vs rasterio libraries
This script processes the same model (config_max) with 1000m inputs using both libraries
and compares the results to check for consistency.
"""

import os
import sys
import multiprocessing as mp
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import rasterio
import pandas as pd

# Add the app directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from raster_fuzzy_lib import UnifiedRasterFuzzyInferenceSystem
from raster_fuzzy_lib_tifffile import UnifiedRasterFuzzyInferenceSystem as UnifiedRasterFuzzyInferenceSystemTifffile


def get_optimal_core_count():
    """Get optimal number of cores (all available minus one)."""
    total_cores = mp.cpu_count()
    optimal_cores = max(1, total_cores - 1)
    return optimal_cores, total_cores


def run_fis_with_rasterio():
    """Run FIS processing using rasterio library."""
    print(f"\n{'='*80}")
    print(f"PROCESSING WITH RASTERIO LIBRARY")
    print(f"{'='*80}")
    
    # Get optimal core count
    optimal_cores, total_cores = get_optimal_core_count()
    
    print(f"Total CPU cores available: {total_cores}")
    print(f"Using cores: {optimal_cores} (all minus one)")
    print(f"Configuration: config_max.json")
    print(f"Resolution: 1000m")
    print(f"Library: rasterio")
    
    # Define file paths
    base_dir = Path(__file__).parent
    config_path = base_dir / "app" / "config" / "config_max.json"
    input_dir = base_dir / "app" / "files" / "input" / "base"
    output_dir = base_dir / "app" / "files" / "output"
    
    # Input files (1000m resolution)
    social_tiff = input_dir / "socioeconomico_1000m.tif"
    environmental_tiff = input_dir / "ambiental_1000m.tif"
    strategic_tiff = input_dir / "estratégico_1000m.tif"
    
    # Output file
    output_tiff = output_dir / "output_1000m_config_max_rasterio.tif"
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Verify input files exist
    input_files = [social_tiff, environmental_tiff, strategic_tiff]
    for file_path in input_files:
        if not file_path.exists():
            print(f"❌ Error: Input file not found: {file_path}")
            return False, None
    
    # Verify config file exists
    if not config_path.exists():
        print(f"❌ Error: Configuration file not found: {config_path}")
        return False, None
    
    print(f"📁 Output: {output_tiff}")
    
    try:
        # Initialize the FIS system (rasterio version)
        print("🔄 Initializing FIS system (rasterio)...")
        start_time = time.time()
        
        fis = UnifiedRasterFuzzyInferenceSystem(str(config_path))
        
        init_time = time.time() - start_time
        print(f"✅ FIS system initialized in {init_time:.2f} seconds")
        
        # Process the rasters
        print("🔄 Processing rasters with rasterio...")
        processing_start = time.time()
        
        fis.process_rasters(
            social_tiff=str(social_tiff),
            environmental_tiff=str(environmental_tiff),
            strategic_tiff=str(strategic_tiff),
            output_tiff=str(output_tiff),
            nodata_value=5.0,
            parallel=True,
            num_cores=optimal_cores,
            chunk_size=100
        )
        
        processing_time = time.time() - processing_start
        total_time = time.time() - start_time
        
        # Check if output file was created
        if output_tiff.exists():
            output_size_mb = output_tiff.stat().st_size / (1024 * 1024)
            print(f"\n✅ RASTERIO PROCESSING COMPLETED SUCCESSFULLY!")
            print(f"   Processing time: {processing_time:.2f} seconds")
            print(f"   Total time: {total_time:.2f} seconds")
            print(f"   Output file size: {output_size_mb:.1f} MB")
            print(f"   Output location: {output_tiff}")
            return True, output_tiff
        else:
            print(f"❌ Error: Output file was not created")
            return False, None
            
    except Exception as e:
        print(f"❌ Error during rasterio processing: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def run_fis_with_tifffile():
    """Run FIS processing using tifffile library."""
    print(f"\n{'='*80}")
    print(f"PROCESSING WITH TIFFFILE LIBRARY")
    print(f"{'='*80}")
    
    # Get optimal core count
    optimal_cores, total_cores = get_optimal_core_count()
    
    print(f"Total CPU cores available: {total_cores}")
    print(f"Using cores: {optimal_cores} (all minus one)")
    print(f"Configuration: config_max.json")
    print(f"Resolution: 1000m")
    print(f"Library: tifffile")
    
    # Define file paths
    base_dir = Path(__file__).parent
    config_path = base_dir / "app" / "config" / "config_max.json"
    input_dir = base_dir / "app" / "files" / "input" / "base"
    output_dir = base_dir / "app" / "files" / "output"
    
    # Input files (1000m resolution)
    social_tiff = input_dir / "socioeconomico_1000m.tif"
    environmental_tiff = input_dir / "ambiental_1000m.tif"
    strategic_tiff = input_dir / "estratégico_1000m.tif"
    
    # Output file
    output_tiff = output_dir / "output_1000m_config_max_tifffile.tif"
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Verify input files exist
    input_files = [social_tiff, environmental_tiff, strategic_tiff]
    for file_path in input_files:
        if not file_path.exists():
            print(f"❌ Error: Input file not found: {file_path}")
            return False, None
    
    # Verify config file exists
    if not config_path.exists():
        print(f"❌ Error: Configuration file not found: {config_path}")
        return False, None
    
    print(f"📁 Output: {output_tiff}")
    
    try:
        # Initialize the FIS system (tifffile version)
        print("🔄 Initializing FIS system (tifffile)...")
        start_time = time.time()
        
        fis = UnifiedRasterFuzzyInferenceSystemTifffile(str(config_path))
        
        init_time = time.time() - start_time
        print(f"✅ FIS system initialized in {init_time:.2f} seconds")
        
        # Process the rasters
        print("🔄 Processing rasters with tifffile...")
        processing_start = time.time()
        
        fis.process_rasters(
            social_tiff=str(social_tiff),
            environmental_tiff=str(environmental_tiff),
            strategic_tiff=str(strategic_tiff),
            output_tiff=str(output_tiff),
            nodata_value=5.0,
            parallel=True,
            num_cores=optimal_cores,
            chunk_size=100
        )
        
        processing_time = time.time() - processing_start
        total_time = time.time() - start_time
        
        # Check if output file was created
        if output_tiff.exists():
            output_size_mb = output_tiff.stat().st_size / (1024 * 1024)
            print(f"\n✅ TIFFFILE PROCESSING COMPLETED SUCCESSFULLY!")
            print(f"   Processing time: {processing_time:.2f} seconds")
            print(f"   Total time: {total_time:.2f} seconds")
            print(f"   Output file size: {output_size_mb:.1f} MB")
            print(f"   Output location: {output_tiff}")
            return True, output_tiff
        else:
            print(f"❌ Error: Output file was not created")
            return False, None
            
    except Exception as e:
        print(f"❌ Error during tifffile processing: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def load_raster_data(file_path):
    """Load raster data and return the array along with metadata."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")
    
    with rasterio.open(file_path) as src:
        data = src.read(1)  # Read first band
        metadata = {
            'transform': src.transform,
            'crs': src.crs,
            'nodata': src.nodata,
            'shape': data.shape,
            'dtype': data.dtype,
            'file_size_mb': os.path.getsize(file_path) / (1024 * 1024)
        }
        
        # Handle NoData values
        if metadata['nodata'] is not None:
            valid_data = data[data != metadata['nodata']]
        else:
            valid_data = data.flatten()
        
        metadata['valid_pixels'] = len(valid_data)
        metadata['min'] = float(valid_data.min()) if len(valid_data) > 0 else 0
        metadata['max'] = float(valid_data.max()) if len(valid_data) > 0 else 0
        metadata['mean'] = float(valid_data.mean()) if len(valid_data) > 0 else 0
        metadata['std'] = float(valid_data.std()) if len(valid_data) > 0 else 0
        metadata['median'] = float(np.median(valid_data)) if len(valid_data) > 0 else 0
        
        return data, metadata


def compare_results(file_rasterio, file_tifffile, output_plot=None):
    """Compare the results from both libraries."""
    print(f"\n{'='*80}")
    print(f"COMPARING RESULTS: RASTERIO vs TIFFFILE")
    print(f"{'='*80}")
    
    # Load both datasets
    print("Loading rasterio result...")
    data_rasterio, meta_rasterio = load_raster_data(file_rasterio)
    
    print("Loading tifffile result...")
    data_tifffile, meta_tifffile = load_raster_data(file_tifffile)
    
    # Determine global min/max for consistent color scaling
    global_min = min(meta_rasterio['min'], meta_tifffile['min'])
    global_max = max(meta_rasterio['max'], meta_tifffile['max'])
    
    print(f"Global value range: {global_min:.3f} to {global_max:.3f}")
    
    # Create comparison plot
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 8))
    
    # Plot rasterio result
    im1 = ax1.imshow(data_rasterio, cmap='viridis', vmin=global_min, vmax=global_max)
    ax1.set_title(f'Rasterio Result\n{meta_rasterio["shape"][0]}x{meta_rasterio["shape"][1]} pixels')
    ax1.set_xlabel('Column')
    ax1.set_ylabel('Row')
    
    # Add statistics for rasterio
    stats_text_rasterio = f"""Statistics:
Min: {meta_rasterio['min']:.3f}
Max: {meta_rasterio['max']:.3f}
Mean: {meta_rasterio['mean']:.3f}
Std: {meta_rasterio['std']:.3f}
Valid pixels: {meta_rasterio['valid_pixels']:,}
File size: {meta_rasterio['file_size_mb']:.1f} MB"""
    
    ax1.text(0.02, 0.98, stats_text_rasterio, transform=ax1.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', 
             facecolor='white', alpha=0.9), fontsize=9)
    
    # Plot tifffile result
    im2 = ax2.imshow(data_tifffile, cmap='viridis', vmin=global_min, vmax=global_max)
    ax2.set_title(f'Tifffile Result\n{meta_tifffile["shape"][0]}x{meta_tifffile["shape"][1]} pixels')
    ax2.set_xlabel('Column')
    ax2.set_ylabel('Row')
    
    # Add statistics for tifffile
    stats_text_tifffile = f"""Statistics:
Min: {meta_tifffile['min']:.3f}
Max: {meta_tifffile['max']:.3f}
Mean: {meta_tifffile['mean']:.3f}
Std: {meta_tifffile['std']:.3f}
Valid pixels: {meta_tifffile['valid_pixels']:,}
File size: {meta_tifffile['file_size_mb']:.1f} MB"""
    
    ax2.text(0.02, 0.98, stats_text_tifffile, transform=ax2.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', 
             facecolor='white', alpha=0.9), fontsize=9)
    
    # Plot difference
    difference = data_rasterio - data_tifffile
    im3 = ax3.imshow(difference, cmap='RdBu_r')
    ax3.set_title(f'Difference (Rasterio - Tifffile)\nMax diff: {np.abs(difference).max():.6f}')
    ax3.set_xlabel('Column')
    ax3.set_ylabel('Row')
    
    # Add difference statistics
    diff_stats = f"""Difference Stats:
Max abs diff: {np.abs(difference).max():.6f}
Mean abs diff: {np.abs(difference).mean():.6f}
Std diff: {difference.std():.6f}
Non-zero pixels: {np.count_nonzero(difference):,}"""
    
    ax3.text(0.02, 0.98, diff_stats, transform=ax3.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', 
             facecolor='white', alpha=0.9), fontsize=9)
    
    # Add colorbars
    plt.colorbar(im1, ax=ax1, label='FIS Output Value')
    plt.colorbar(im2, ax=ax2, label='FIS Output Value')
    plt.colorbar(im3, ax=ax3, label='Difference')
    
    # Add overall title
    fig.suptitle('FIS Results Comparison: Rasterio vs Tifffile Libraries', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    # Save if output file specified
    if output_plot:
        plt.savefig(output_plot, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to: {output_plot}")
    
    plt.show()
    
    return meta_rasterio, meta_tifffile, difference


def create_comparison_table(meta_rasterio, meta_tifffile, difference, output_file=None):
    """Create a comparison table for the results."""
    
    # Calculate difference statistics
    max_abs_diff = np.abs(difference).max()
    mean_abs_diff = np.abs(difference).mean()
    std_diff = difference.std()
    non_zero_pixels = np.count_nonzero(difference)
    total_pixels = difference.size
    
    # Create comparison data
    comparison_data = {
        'Metric': [
            'Resolution',
            'Dimensions (pixels)',
            'Total Pixels',
            'Valid Pixels',
            'File Size (MB)',
            'Min Value',
            'Max Value',
            'Mean Value',
            'Median Value',
            'Standard Deviation',
            'Value Range',
            'Data Density (%)'
        ],
        'Rasterio': [
            '1000m',
            f"{meta_rasterio['shape'][0]} × {meta_rasterio['shape'][1]}",
            f"{meta_rasterio['shape'][0] * meta_rasterio['shape'][1]:,}",
            f"{meta_rasterio['valid_pixels']:,}",
            f"{meta_rasterio['file_size_mb']:.1f}",
            f"{meta_rasterio['min']:.3f}",
            f"{meta_rasterio['max']:.3f}",
            f"{meta_rasterio['mean']:.3f}",
            f"{meta_rasterio['median']:.3f}",
            f"{meta_rasterio['std']:.3f}",
            f"{meta_rasterio['max'] - meta_rasterio['min']:.3f}",
            f"{meta_rasterio['valid_pixels'] / (meta_rasterio['shape'][0] * meta_rasterio['shape'][1]) * 100:.1f}%"
        ],
        'Tifffile': [
            '1000m',
            f"{meta_tifffile['shape'][0]} × {meta_tifffile['shape'][1]}",
            f"{meta_tifffile['shape'][0] * meta_tifffile['shape'][1]:,}",
            f"{meta_tifffile['valid_pixels']:,}",
            f"{meta_tifffile['file_size_mb']:.1f}",
            f"{meta_tifffile['min']:.3f}",
            f"{meta_tifffile['max']:.3f}",
            f"{meta_tifffile['mean']:.3f}",
            f"{meta_tifffile['median']:.3f}",
            f"{meta_tifffile['std']:.3f}",
            f"{meta_tifffile['max'] - meta_tifffile['min']:.3f}",
            f"{meta_tifffile['valid_pixels'] / (meta_tifffile['shape'][0] * meta_tifffile['shape'][1]) * 100:.1f}%"
        ],
        'Difference': [
            'Same',
            'Same',
            'Same',
            'Same',
            f"{meta_tifffile['file_size_mb'] - meta_rasterio['file_size_mb']:+.1f}",
            f"{meta_tifffile['min'] - meta_rasterio['min']:+.3f}",
            f"{meta_tifffile['max'] - meta_rasterio['max']:+.3f}",
            f"{meta_tifffile['mean'] - meta_rasterio['mean']:+.3f}",
            f"{meta_tifffile['median'] - meta_rasterio['median']:+.3f}",
            f"{meta_tifffile['std'] - meta_rasterio['std']:+.3f}",
            f"{(meta_tifffile['max'] - meta_tifffile['min']) - (meta_rasterio['max'] - meta_rasterio['min']):+.3f}",
            f"{meta_tifffile['valid_pixels'] / (meta_tifffile['shape'][0] * meta_tifffile['shape'][1]) * 100 - meta_rasterio['valid_pixels'] / (meta_rasterio['shape'][0] * meta_rasterio['shape'][1]) * 100:+.1f}%"
        ]
    }
    
    # Create DataFrame
    df = pd.DataFrame(comparison_data)
    
    # Print the table
    print("\n" + "="*100)
    print("LIBRARY COMPARISON STATISTICS")
    print("Rasterio vs Tifffile")
    print("="*100)
    print(df.to_string(index=False))
    
    # Print difference analysis
    print(f"\n" + "="*80)
    print("DIFFERENCE ANALYSIS")
    print("="*80)
    print(f"Maximum absolute difference: {max_abs_diff:.6f}")
    print(f"Mean absolute difference: {mean_abs_diff:.6f}")
    print(f"Standard deviation of differences: {std_diff:.6f}")
    print(f"Non-zero difference pixels: {non_zero_pixels:,} ({non_zero_pixels/total_pixels*100:.2f}%)")
    print(f"Total pixels: {total_pixels:,}")
    
    # Quality assessment
    print(f"\nQUALITY ASSESSMENT:")
    if max_abs_diff < 1e-10:
        print(f"  ✅ EXCELLENT: Results are essentially identical (max diff: {max_abs_diff:.2e})")
    elif max_abs_diff < 1e-6:
        print(f"  ✅ VERY GOOD: Results are nearly identical (max diff: {max_abs_diff:.2e})")
    elif max_abs_diff < 1e-3:
        print(f"  ✅ GOOD: Results are very similar (max diff: {max_abs_diff:.2e})")
    elif max_abs_diff < 0.01:
        print(f"  ⚠️  MODERATE: Results show some differences (max diff: {max_abs_diff:.6f})")
    else:
        print(f"  ❌ POOR: Results show significant differences (max diff: {max_abs_diff:.6f})")
    
    # Save to CSV if requested
    if output_file:
        df.to_csv(output_file, index=False)
        print(f"\nComparison table saved to: {output_file}")
    
    return df


def main():
    """Main function to run the library comparison."""
    print("="*80)
    print("FIS LIBRARY COMPARISON: RASTERIO vs TIFFFILE")
    print("Processing config_max with 1000m inputs")
    print("="*80)
    
    # Create plots directory
    os.makedirs("plots", exist_ok=True)
    
    # Run both processing methods
    success_rasterio, file_rasterio = run_fis_with_rasterio()
    success_tifffile, file_tifffile = run_fis_with_tifffile()
    
    if not success_rasterio or not success_tifffile:
        print("\n❌ One or both processing methods failed. Cannot compare results.")
        return
    
    # Compare results
    output_plot = "plots/rasterio_vs_tifffile_comparison.png"
    output_stats = "plots/rasterio_vs_tifffile_statistics.csv"
    
    meta_rasterio, meta_tifffile, difference = compare_results(
        file_rasterio, file_tifffile, output_plot
    )
    
    # Create comparison table
    stats_df = create_comparison_table(meta_rasterio, meta_tifffile, difference, output_stats)
    
    print(f"\n✅ Library comparison completed successfully!")
    print(f"📊 Comparison plot saved to: {output_plot}")
    print(f"📋 Statistics saved to: {output_stats}")
    print(f"📁 Rasterio result: {file_rasterio}")
    print(f"📁 Tifffile result: {file_tifffile}")


if __name__ == "__main__":
    main() 