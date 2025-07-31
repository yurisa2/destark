#!/usr/bin/env python3
"""
Comprehensive comparison of all FIS results:
- 1000m tifffile output
- 1000m config_max output  
- 300m config_minimum output

This helps identify any differences or issues between different configurations and resolutions.
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


def plot_three_way_comparison(file_1000m_tifffile, file_1000m_config_max, file_300m_config_min, 
                             output_file=None, colormap='viridis', dpi=300, figsize=(20, 12)):
    """
    Plot all three results side by side with the same color coding.
    
    Args:
        file_1000m_tifffile (str): Path to 1000m tifffile result file
        file_1000m_config_max (str): Path to 1000m config_max result file
        file_300m_config_min (str): Path to 300m config_minimum result file
        output_file (str): Path to save the comparison plot (optional)
        colormap (str): Matplotlib colormap name
        dpi (int): DPI for saved image
        figsize (tuple): Figure size (width, height)
    """
    
    print("Loading 1000m tifffile data...")
    data_1000m_tifffile, meta_1000m_tifffile = load_raster_data(file_1000m_tifffile)
    
    print("Loading 1000m config_max data...")
    data_1000m_config_max, meta_1000m_config_max = load_raster_data(file_1000m_config_max)
    
    print("Loading 300m config_minimum data...")
    data_300m_config_min, meta_300m_config_min = load_raster_data(file_300m_config_min)
    
    # Determine global min/max for consistent color scaling
    global_min = min(meta_1000m_tifffile['min'], meta_1000m_config_max['min'], meta_300m_config_min['min'])
    global_max = max(meta_1000m_tifffile['max'], meta_1000m_config_max['max'], meta_300m_config_min['max'])
    
    print(f"Global value range: {global_min:.3f} to {global_max:.3f}")
    
    # Create figure with three subplots side by side
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
    
    # Plot 1000m tifffile result
    im1 = ax1.imshow(data_1000m_tifffile, cmap=colormap, vmin=global_min, vmax=global_max)
    ax1.set_title(f'1000m Tifffile Result\n{meta_1000m_tifffile["shape"][0]}x{meta_1000m_tifffile["shape"][1]} pixels')
    ax1.set_xlabel('Column')
    ax1.set_ylabel('Row')
    
    # Add statistics for 1000m tifffile
    stats_text_1000m_tifffile = f"""Statistics:
Min: {meta_1000m_tifffile['min']:.3f}
Max: {meta_1000m_tifffile['max']:.3f}
Mean: {meta_1000m_tifffile['mean']:.3f}
Std: {meta_1000m_tifffile['std']:.3f}
Valid pixels: {meta_1000m_tifffile['valid_pixels']:,}
File size: {meta_1000m_tifffile['file_size_mb']:.1f} MB"""
    
    ax1.text(0.02, 0.98, stats_text_1000m_tifffile, transform=ax1.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', 
             facecolor='white', alpha=0.9), fontsize=9)
    
    # Plot 1000m config_max result
    im2 = ax2.imshow(data_1000m_config_max, cmap=colormap, vmin=global_min, vmax=global_max)
    ax2.set_title(f'1000m Config Max Result\n{meta_1000m_config_max["shape"][0]}x{meta_1000m_config_max["shape"][1]} pixels')
    ax2.set_xlabel('Column')
    ax2.set_ylabel('Row')
    
    # Add statistics for 1000m config_max
    stats_text_1000m_config_max = f"""Statistics:
Min: {meta_1000m_config_max['min']:.3f}
Max: {meta_1000m_config_max['max']:.3f}
Mean: {meta_1000m_config_max['mean']:.3f}
Std: {meta_1000m_config_max['std']:.3f}
Valid pixels: {meta_1000m_config_max['valid_pixels']:,}
File size: {meta_1000m_config_max['file_size_mb']:.1f} MB"""
    
    ax2.text(0.02, 0.98, stats_text_1000m_config_max, transform=ax2.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', 
             facecolor='white', alpha=0.9), fontsize=9)
    
    # Plot 300m config_minimum result
    im3 = ax3.imshow(data_300m_config_min, cmap=colormap, vmin=global_min, vmax=global_max)
    ax3.set_title(f'300m Config Minimum Result\n{meta_300m_config_min["shape"][0]}x{meta_300m_config_min["shape"][1]} pixels')
    ax3.set_xlabel('Column')
    ax3.set_ylabel('Row')
    
    # Add statistics for 300m config_minimum
    stats_text_300m_config_min = f"""Statistics:
Min: {meta_300m_config_min['min']:.3f}
Max: {meta_300m_config_min['max']:.3f}
Mean: {meta_300m_config_min['mean']:.3f}
Std: {meta_300m_config_min['std']:.3f}
Valid pixels: {meta_300m_config_min['valid_pixels']:,}
File size: {meta_300m_config_min['file_size_mb']:.1f} MB"""
    
    ax3.text(0.02, 0.98, stats_text_300m_config_min, transform=ax3.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', 
             facecolor='white', alpha=0.9), fontsize=9)
    
    # Add a single colorbar for all plots
    cbar = plt.colorbar(im1, ax=[ax1, ax2, ax3], shrink=0.8, aspect=30)
    cbar.set_label('FIS Output Value')
    
    # Add overall title
    fig.suptitle('FIS Results Comparison: All Configurations and Resolutions', fontsize=16, fontweight='bold')
    
    # Add comparison statistics
    comparison_text = f"""Comparison Summary:
Global Value Range: {global_min:.3f} to {global_max:.3f}

1000m Tifffile: {meta_1000m_tifffile['mean']:.3f} ± {meta_1000m_tifffile['std']:.3f}
1000m Config Max: {meta_1000m_config_max['mean']:.3f} ± {meta_1000m_config_max['std']:.3f}
300m Config Min: {meta_300m_config_min['mean']:.3f} ± {meta_300m_config_min['std']:.3f}

1000m Config Diff: {abs(meta_1000m_tifffile['mean'] - meta_1000m_config_max['mean']):.3f}
300m vs 1000m Diff: {abs(meta_300m_config_min['mean'] - meta_1000m_config_max['mean']):.3f}
Resolution Ratio: {meta_300m_config_min['shape'][0] * meta_300m_config_min['shape'][1] / (meta_1000m_config_max['shape'][0] * meta_1000m_config_max['shape'][1]):.1f}x more pixels"""
    
    fig.text(0.02, 0.02, comparison_text, fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    # Save if output file specified
    if output_file:
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
        print(f"Comparison plot saved to: {output_file}")
    
    plt.show()
    
    return meta_1000m_tifffile, meta_1000m_config_max, meta_300m_config_min


def analyze_all_differences(data_1000m_tifffile, data_1000m_config_max, data_300m_config_min,
                           meta_1000m_tifffile, meta_1000m_config_max, meta_300m_config_min):
    """
    Analyze differences between all three datasets.
    """
    
    print("\n" + "="*80)
    print("COMPREHENSIVE COMPARISON ANALYSIS")
    print("="*80)
    
    # Basic statistics comparison
    print(f"\nValue Range Comparison:")
    print(f"  1000m Tifffile: {meta_1000m_tifffile['min']:.3f} to {meta_1000m_tifffile['max']:.3f}")
    print(f"  1000m Config Max: {meta_1000m_config_max['min']:.3f} to {meta_1000m_config_max['max']:.3f}")
    print(f"  300m Config Min:  {meta_300m_config_min['min']:.3f} to {meta_300m_config_min['max']:.3f}")
    
    print(f"\nCentral Tendency:")
    print(f"  1000m Tifffile: {meta_1000m_tifffile['mean']:.3f} ± {meta_1000m_tifffile['std']:.3f}")
    print(f"  1000m Config Max: {meta_1000m_config_max['mean']:.3f} ± {meta_1000m_config_max['std']:.3f}")
    print(f"  300m Config Min:  {meta_300m_config_min['mean']:.3f} ± {meta_300m_config_min['std']:.3f}")
    
    print(f"\nConfiguration Differences (1000m):")
    mean_diff_1000m = abs(meta_1000m_tifffile['mean'] - meta_1000m_config_max['mean'])
    std_diff_1000m = abs(meta_1000m_tifffile['std'] - meta_1000m_config_max['std'])
    print(f"  Mean Difference: {mean_diff_1000m:.3f}")
    print(f"  Std Difference: {std_diff_1000m:.3f}")
    
    print(f"\nResolution Differences (Config Max vs Config Min):")
    mean_diff_res = abs(meta_1000m_config_max['mean'] - meta_300m_config_min['mean'])
    std_diff_res = abs(meta_1000m_config_max['std'] - meta_300m_config_min['std'])
    print(f"  Mean Difference: {mean_diff_res:.3f}")
    print(f"  Std Difference: {std_diff_res:.3f}")
    
    print(f"\nData Distribution:")
    print(f"  1000m Tifffile Valid Pixels: {meta_1000m_tifffile['valid_pixels']:,}")
    print(f"  1000m Config Max Valid Pixels: {meta_1000m_config_max['valid_pixels']:,}")
    print(f"  300m Config Min Valid Pixels: {meta_300m_config_min['valid_pixels']:,}")
    
    print(f"\nFile Sizes:")
    print(f"  1000m Tifffile: {meta_1000m_tifffile['file_size_mb']:.1f} MB")
    print(f"  1000m Config Max: {meta_1000m_config_max['file_size_mb']:.1f} MB")
    print(f"  300m Config Min: {meta_300m_config_min['file_size_mb']:.1f} MB")
    
    # Check for potential issues
    print(f"\nPotential Issues:")
    
    if mean_diff_1000m > 0.1:
        print(f"  ⚠️  Large mean difference between 1000m configs: {mean_diff_1000m:.3f}")
    
    if mean_diff_res > 0.1:
        print(f"  ⚠️  Large mean difference between resolutions: {mean_diff_res:.3f}")
    
    if std_diff_1000m > 0.1:
        print(f"  ⚠️  Large std difference between 1000m configs: {std_diff_1000m:.3f}")
    
    if std_diff_res > 0.1:
        print(f"  ⚠️  Large std difference between resolutions: {std_diff_res:.3f}")
    
    # Check for reasonable file size ratios
    expected_size_ratio = (meta_300m_config_min['valid_pixels'] / meta_1000m_config_max['valid_pixels'])
    actual_size_ratio = meta_300m_config_min['file_size_mb'] / meta_1000m_config_max['file_size_mb']
    
    if abs(actual_size_ratio - expected_size_ratio) > expected_size_ratio * 0.5:
        print(f"  ⚠️  File size ratio ({actual_size_ratio:.1f}x) doesn't match pixel ratio ({expected_size_ratio:.1f}x)")
    
    # Check for consistency between 1000m results
    if mean_diff_1000m < 0.01 and std_diff_1000m < 0.01:
        print(f"  ✅ 1000m results are very consistent (same configuration)")
    elif mean_diff_1000m < 0.1 and std_diff_1000m < 0.1:
        print(f"  ✅ 1000m results are reasonably consistent")
    else:
        print(f"  ⚠️  1000m results show significant differences")


def main():
    """Main function to run the comprehensive comparison."""
    
    # Define file paths
    file_1000m_tifffile = "app/files/output/1000m_output_tifffile.tif"
    file_1000m_config_max = "app/files/output/output_1000m_config_max.tif"
    file_300m_config_min = "app/files/output/output_300m_config_minimum.tif"
    output_plot = "plots/all_results_comparison.png"
    
    print("="*80)
    print("COMPREHENSIVE FIS RESULTS COMPARISON")
    print("1000m Tifffile vs 1000m Config Max vs 300m Config Minimum")
    print("="*80)
    
    try:
        # Create plots directory if it doesn't exist
        os.makedirs("plots", exist_ok=True)
        
        # Load data and create comparison plot
        meta_1000m_tifffile, meta_1000m_config_max, meta_300m_config_min = plot_three_way_comparison(
            file_1000m_tifffile=file_1000m_tifffile,
            file_1000m_config_max=file_1000m_config_max,
            file_300m_config_min=file_300m_config_min,
            output_file=output_plot,
            colormap='viridis',
            dpi=300,
            figsize=(20, 12)
        )
        
        # Load data for detailed analysis
        data_1000m_tifffile, _ = load_raster_data(file_1000m_tifffile)
        data_1000m_config_max, _ = load_raster_data(file_1000m_config_max)
        data_300m_config_min, _ = load_raster_data(file_300m_config_min)
        
        # Perform detailed analysis
        analyze_all_differences(data_1000m_tifffile, data_1000m_config_max, data_300m_config_min,
                               meta_1000m_tifffile, meta_1000m_config_max, meta_300m_config_min)
        
        print(f"\n✅ Comprehensive comparison completed successfully!")
        print(f"📊 Plot saved to: {output_plot}")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please ensure all three output files exist:")
        print(f"  - {file_1000m_tifffile}")
        print(f"  - {file_1000m_config_max}")
        print(f"  - {file_300m_config_min}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 