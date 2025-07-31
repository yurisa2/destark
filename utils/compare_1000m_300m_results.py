#!/usr/bin/env python3
"""
Script to compare 1000m and 300m FIS results side by side with the same color coding.
This helps identify any differences or issues between the two resolutions.
"""

import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import show
import os
from pathlib import Path


def load_raster_data(file_path):
    """
    Load raster data and return the array along with metadata.
    
    Args:
        file_path (str): Path to the TIFF file
        
    Returns:
        tuple: (data_array, metadata_dict)
    """
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
        
        return data, metadata


def plot_comparison_side_by_side(file_1000m, file_300m, output_file=None, 
                                colormap='viridis', dpi=300, figsize=(16, 8)):
    """
    Plot 1000m and 300m results side by side with the same color coding.
    
    Args:
        file_1000m (str): Path to 1000m result file
        file_300m (str): Path to 300m result file
        output_file (str): Path to save the comparison plot (optional)
        colormap (str): Matplotlib colormap name
        dpi (int): DPI for saved image
        figsize (tuple): Figure size (width, height)
    """
    
    print("Loading 1000m data...")
    data_1000m, meta_1000m = load_raster_data(file_1000m)
    
    print("Loading 300m data...")
    data_300m, meta_300m = load_raster_data(file_300m)
    
    # Determine global min/max for consistent color scaling
    global_min = min(meta_1000m['min'], meta_300m['min'])
    global_max = max(meta_1000m['max'], meta_300m['max'])
    
    print(f"Global value range: {global_min:.3f} to {global_max:.3f}")
    
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1000m result
    im1 = ax1.imshow(data_1000m, cmap=colormap, vmin=global_min, vmax=global_max)
    ax1.set_title(f'1000m Resolution Result\n{meta_1000m["shape"][0]}x{meta_1000m["shape"][1]} pixels')
    ax1.set_xlabel('Column')
    ax1.set_ylabel('Row')
    
    # Add statistics for 1000m
    stats_text_1000m = f"""Statistics:
Min: {meta_1000m['min']:.3f}
Max: {meta_1000m['max']:.3f}
Mean: {meta_1000m['mean']:.3f}
Std: {meta_1000m['std']:.3f}
Valid pixels: {meta_1000m['valid_pixels']:,}
File size: {meta_1000m['file_size_mb']:.1f} MB"""
    
    ax1.text(0.02, 0.98, stats_text_1000m, transform=ax1.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', 
             facecolor='white', alpha=0.9), fontsize=9)
    
    # Plot 300m result
    im2 = ax2.imshow(data_300m, cmap=colormap, vmin=global_min, vmax=global_max)
    ax2.set_title(f'300m Resolution Result\n{meta_300m["shape"][0]}x{meta_300m["shape"][1]} pixels')
    ax2.set_xlabel('Column')
    ax2.set_ylabel('Row')
    
    # Add statistics for 300m
    stats_text_300m = f"""Statistics:
Min: {meta_300m['min']:.3f}
Max: {meta_300m['max']:.3f}
Mean: {meta_300m['mean']:.3f}
Std: {meta_300m['std']:.3f}
Valid pixels: {meta_300m['valid_pixels']:,}
File size: {meta_300m['file_size_mb']:.1f} MB"""
    
    ax2.text(0.02, 0.98, stats_text_300m, transform=ax2.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', 
             facecolor='white', alpha=0.9), fontsize=9)
    
    # Add a single colorbar for both plots
    cbar = plt.colorbar(im1, ax=[ax1, ax2], shrink=0.8, aspect=30)
    cbar.set_label('FIS Output Value')
    
    # Add overall title
    fig.suptitle('FIS Results Comparison: 1000m vs 300m Resolution', fontsize=16, fontweight='bold')
    
    # Add comparison statistics
    comparison_text = f"""Comparison:
Value Range Overlap: {global_min:.3f} to {global_max:.3f}
1000m Mean: {meta_1000m['mean']:.3f} ± {meta_1000m['std']:.3f}
300m Mean: {meta_300m['mean']:.3f} ± {meta_300m['std']:.3f}
Mean Difference: {abs(meta_1000m['mean'] - meta_300m['mean']):.3f}
Resolution Ratio: {meta_300m['shape'][0] * meta_300m['shape'][1] / (meta_1000m['shape'][0] * meta_1000m['shape'][1]):.1f}x more pixels"""
    
    fig.text(0.02, 0.02, comparison_text, fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    # Save if output file specified
    if output_file:
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
        print(f"Comparison plot saved to: {output_file}")
    
    plt.show()
    
    return meta_1000m, meta_300m


def analyze_differences(data_1000m, data_300m, meta_1000m, meta_300m):
    """
    Analyze differences between the two datasets.
    
    Args:
        data_1000m: 1000m data array
        data_300m: 300m data array
        meta_1000m: 1000m metadata
        meta_300m: 300m metadata
    """
    
    print("\n" + "="*60)
    print("DETAILED COMPARISON ANALYSIS")
    print("="*60)
    
    # Basic statistics comparison
    print(f"\nValue Range Comparison:")
    print(f"  1000m: {meta_1000m['min']:.3f} to {meta_1000m['max']:.3f}")
    print(f"  300m:  {meta_300m['min']:.3f} to {meta_300m['max']:.3f}")
    
    print(f"\nCentral Tendency:")
    print(f"  1000m Mean: {meta_1000m['mean']:.3f} ± {meta_1000m['std']:.3f}")
    print(f"  300m Mean:  {meta_300m['mean']:.3f} ± {meta_300m['std']:.3f}")
    print(f"  Mean Difference: {abs(meta_1000m['mean'] - meta_300m['mean']):.3f}")
    
    print(f"\nData Distribution:")
    print(f"  1000m Valid Pixels: {meta_1000m['valid_pixels']:,}")
    print(f"  300m Valid Pixels:  {meta_300m['valid_pixels']:,}")
    print(f"  Pixel Ratio: {meta_300m['valid_pixels'] / meta_1000m['valid_pixels']:.1f}x")
    
    print(f"\nFile Sizes:")
    print(f"  1000m: {meta_1000m['file_size_mb']:.1f} MB")
    print(f"  300m:  {meta_300m['file_size_mb']:.1f} MB")
    print(f"  Size Ratio: {meta_300m['file_size_mb'] / meta_1000m['file_size_mb']:.1f}x")
    
    # Check for potential issues
    print(f"\nPotential Issues:")
    
    if abs(meta_1000m['mean'] - meta_300m['mean']) > 0.1:
        print(f"  ⚠️  Large mean difference: {abs(meta_1000m['mean'] - meta_300m['mean']):.3f}")
    
    if abs(meta_1000m['std'] - meta_300m['std']) > 0.1:
        print(f"  ⚠️  Large std difference: {abs(meta_1000m['std'] - meta_300m['std']):.3f}")
    
    if meta_1000m['min'] == meta_1000m['max']:
        print(f"  ⚠️  1000m data has no variation (all values are {meta_1000m['min']})")
    
    if meta_300m['min'] == meta_300m['max']:
        print(f"  ⚠️  300m data has no variation (all values are {meta_300m['min']})")
    
    # Check for reasonable file size ratios
    expected_size_ratio = (meta_300m['valid_pixels'] / meta_1000m['valid_pixels'])
    actual_size_ratio = meta_300m['file_size_mb'] / meta_1000m['file_size_mb']
    
    if abs(actual_size_ratio - expected_size_ratio) > expected_size_ratio * 0.5:
        print(f"  ⚠️  File size ratio ({actual_size_ratio:.1f}x) doesn't match pixel ratio ({expected_size_ratio:.1f}x)")


def main():
    """Main function to run the comparison."""
    
    # Define file paths
    file_1000m = "app/files/output/1000m_output_tifffile.tif"
    file_300m = "app/files/output/output_300m_config_minimum.tif"
    output_plot = "plots/1000m_300m_comparison.png"
    
    print("="*60)
    print("FIS RESULTS COMPARISON: 1000m vs 300m Resolution")
    print("="*60)
    
    try:
        # Create plots directory if it doesn't exist
        os.makedirs("plots", exist_ok=True)
        
        # Load data and create comparison plot
        meta_1000m, meta_300m = plot_comparison_side_by_side(
            file_1000m=file_1000m,
            file_300m=file_300m,
            output_file=output_plot,
            colormap='viridis',
            dpi=300,
            figsize=(16, 8)
        )
        
        # Load data for detailed analysis
        data_1000m, _ = load_raster_data(file_1000m)
        data_300m, _ = load_raster_data(file_300m)
        
        # Perform detailed analysis
        analyze_differences(data_1000m, data_300m, meta_1000m, meta_300m)
        
        print(f"\n✅ Comparison completed successfully!")
        print(f"📊 Plot saved to: {output_plot}")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please ensure both 1000m and 300m output files exist.")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 