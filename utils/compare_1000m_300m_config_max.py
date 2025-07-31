#!/usr/bin/env python3
"""
Proper comparison of FIS results: 1000m config_max vs 300m config_max
This compares the same configuration across different resolutions.
"""

import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import show
import os
from pathlib import Path
import pandas as pd


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
        metadata['median'] = float(np.median(valid_data)) if len(valid_data) > 0 else 0
        
        return data, metadata


def plot_side_by_side_comparison(file_1000m, file_300m, output_file=None, 
                                colormap='viridis', dpi=300, figsize=(16, 8)):
    """
    Plot 1000m and 300m config_max results side by side with the same color coding.
    
    Args:
        file_1000m (str): Path to 1000m config_max result file
        file_300m (str): Path to 300m config_max result file
        output_file (str): Path to save the comparison plot (optional)
        colormap (str): Matplotlib colormap name
        dpi (int): DPI for saved image
        figsize (tuple): Figure size (width, height)
    """
    
    print("Loading 1000m config_max data...")
    data_1000m, meta_1000m = load_raster_data(file_1000m)
    
    print("Loading 300m config_max data...")
    data_300m, meta_300m = load_raster_data(file_300m)
    
    # Determine global min/max for consistent color scaling
    global_min = min(meta_1000m['min'], meta_300m['min'])
    global_max = max(meta_1000m['max'], meta_300m['max'])
    
    print(f"Global value range: {global_min:.3f} to {global_max:.3f}")
    
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1000m result
    im1 = ax1.imshow(data_1000m, cmap=colormap, vmin=global_min, vmax=global_max)
    ax1.set_title(f'1000m Resolution (Config Max)\n{meta_1000m["shape"][0]}x{meta_1000m["shape"][1]} pixels')
    ax1.set_xlabel('Column')
    ax1.set_ylabel('Row')
    
    # Plot 300m result
    im2 = ax2.imshow(data_300m, cmap=colormap, vmin=global_min, vmax=global_max)
    ax2.set_title(f'300m Resolution (Config Max)\n{meta_300m["shape"][0]}x{meta_300m["shape"][1]} pixels')
    ax2.set_xlabel('Column')
    ax2.set_ylabel('Row')
    
    # Add a single colorbar for both plots
    cbar = plt.colorbar(im1, ax=[ax1, ax2], shrink=0.8, aspect=30)
    cbar.set_label('FIS Output Value')
    
    # Add overall title
    fig.suptitle('FIS Results Comparison: 1000m vs 300m Resolution (Config Max)', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    # Save if output file specified
    if output_file:
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
        print(f"Comparison plot saved to: {output_file}")
    
    plt.show()
    
    return meta_1000m, meta_300m


def create_statistics_table(meta_1000m, meta_300m, output_file=None):
    """
    Create a nice statistics comparison table.
    
    Args:
        meta_1000m: 1000m metadata
        meta_300m: 300m metadata
        output_file: optional output file for the table
    """
    
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
            'Pixels per km²',
            'Data Density (%)'
        ],
        '1000m Config Max': [
            '1000m',
            f"{meta_1000m['shape'][0]} × {meta_1000m['shape'][1]}",
            f"{meta_1000m['shape'][0] * meta_1000m['shape'][1]:,}",
            f"{meta_1000m['valid_pixels']:,}",
            f"{meta_1000m['file_size_mb']:.1f}",
            f"{meta_1000m['min']:.3f}",
            f"{meta_1000m['max']:.3f}",
            f"{meta_1000m['mean']:.3f}",
            f"{meta_1000m['median']:.3f}",
            f"{meta_1000m['std']:.3f}",
            f"{meta_1000m['max'] - meta_1000m['min']:.3f}",
            f"{meta_1000m['valid_pixels'] / (meta_1000m['shape'][0] * meta_1000m['shape'][1]):.1f}",
            f"{meta_1000m['valid_pixels'] / (meta_1000m['shape'][0] * meta_1000m['shape'][1]) * 100:.1f}%"
        ],
        '300m Config Max': [
            '300m',
            f"{meta_300m['shape'][0]} × {meta_300m['shape'][1]}",
            f"{meta_300m['shape'][0] * meta_300m['shape'][1]:,}",
            f"{meta_300m['valid_pixels']:,}",
            f"{meta_300m['file_size_mb']:.1f}",
            f"{meta_300m['min']:.3f}",
            f"{meta_300m['max']:.3f}",
            f"{meta_300m['mean']:.3f}",
            f"{meta_300m['median']:.3f}",
            f"{meta_300m['std']:.3f}",
            f"{meta_300m['max'] - meta_300m['min']:.3f}",
            f"{meta_300m['valid_pixels'] / (meta_300m['shape'][0] * meta_300m['shape'][1]):.1f}",
            f"{meta_300m['valid_pixels'] / (meta_300m['shape'][0] * meta_300m['shape'][1]) * 100:.1f}%"
        ],
        'Difference': [
            '3.33x finer',
            f"{meta_300m['shape'][0] / meta_1000m['shape'][0]:.1f}x rows, {meta_300m['shape'][1] / meta_1000m['shape'][1]:.1f}x cols",
            f"{meta_300m['shape'][0] * meta_300m['shape'][1] / (meta_1000m['shape'][0] * meta_1000m['shape'][1]):.1f}x",
            f"{meta_300m['valid_pixels'] / meta_1000m['valid_pixels']:.1f}x",
            f"{meta_300m['file_size_mb'] / meta_1000m['file_size_mb']:.1f}x",
            f"{meta_300m['min'] - meta_1000m['min']:+.3f}",
            f"{meta_300m['max'] - meta_1000m['max']:+.3f}",
            f"{meta_300m['mean'] - meta_1000m['mean']:+.3f}",
            f"{meta_300m['median'] - meta_1000m['median']:+.3f}",
            f"{meta_300m['std'] - meta_1000m['std']:+.3f}",
            f"{(meta_300m['max'] - meta_300m['min']) - (meta_1000m['max'] - meta_1000m['min']):+.3f}",
            f"{meta_300m['valid_pixels'] / (meta_300m['shape'][0] * meta_300m['shape'][1]) / (meta_1000m['valid_pixels'] / (meta_1000m['shape'][0] * meta_1000m['shape'][1])):.1f}x",
            f"{meta_300m['valid_pixels'] / (meta_300m['shape'][0] * meta_300m['shape'][1]) * 100 - meta_1000m['valid_pixels'] / (meta_1000m['shape'][0] * meta_1000m['shape'][1]) * 100:+.1f}%"
        ]
    }
    
    # Create DataFrame
    df = pd.DataFrame(comparison_data)
    
    # Print the table
    print("\n" + "="*100)
    print("COMPREHENSIVE STATISTICS COMPARISON")
    print("1000m Config Max vs 300m Config Max")
    print("="*100)
    print(df.to_string(index=False))
    
    # Save to CSV if requested
    if output_file:
        df.to_csv(output_file, index=False)
        print(f"\nStatistics table saved to: {output_file}")
    
    return df


def analyze_differences(meta_1000m, meta_300m):
    """
    Analyze differences between the two datasets.
    """
    
    print("\n" + "="*80)
    print("DETAILED DIFFERENCE ANALYSIS")
    print("="*80)
    
    # Calculate key differences
    mean_diff = abs(meta_1000m['mean'] - meta_300m['mean'])
    std_diff = abs(meta_1000m['std'] - meta_300m['std'])
    pixel_ratio = meta_300m['valid_pixels'] / meta_1000m['valid_pixels']
    size_ratio = meta_300m['file_size_mb'] / meta_1000m['file_size_mb']
    
    print(f"\nKey Differences:")
    print(f"  Mean Value Difference: {mean_diff:.3f}")
    print(f"  Standard Deviation Difference: {std_diff:.3f}")
    print(f"  Pixel Count Ratio: {pixel_ratio:.1f}x")
    print(f"  File Size Ratio: {size_ratio:.1f}x")
    
    # Check for potential issues
    print(f"\nQuality Assessment:")
    
    if mean_diff < 0.01:
        print(f"  ✅ Excellent consistency in mean values")
    elif mean_diff < 0.1:
        print(f"  ✅ Good consistency in mean values")
    elif mean_diff < 0.5:
        print(f"  ⚠️  Moderate difference in mean values")
    else:
        print(f"  ❌ Large difference in mean values - potential issue")
    
    if std_diff < 0.01:
        print(f"  ✅ Excellent consistency in standard deviation")
    elif std_diff < 0.1:
        print(f"  ✅ Good consistency in standard deviation")
    elif std_diff < 0.5:
        print(f"  ⚠️  Moderate difference in standard deviation")
    else:
        print(f"  ❌ Large difference in standard deviation - potential issue")
    
    # Check file size consistency
    expected_size_ratio = pixel_ratio
    if abs(size_ratio - expected_size_ratio) < expected_size_ratio * 0.2:
        print(f"  ✅ File sizes are consistent with pixel counts")
    else:
        print(f"  ⚠️  File size ratio ({size_ratio:.1f}x) doesn't match pixel ratio ({expected_size_ratio:.1f}x)")
    
    # Resolution quality assessment
    if pixel_ratio > 9 and pixel_ratio < 12:  # Expected ~11.1x for 3.33x resolution
        print(f"  ✅ Resolution ratio is as expected (~11x)")
    else:
        print(f"  ⚠️  Unexpected resolution ratio ({pixel_ratio:.1f}x)")


def main():
    """Main function to run the comparison."""
    
    # Define file paths
    file_1000m_config_max = "app/files/output/output_1000m_config_max.tif"
    file_300m_config_max = "app/files/output/output_300m_config_max.tif"
    output_plot = "plots/1000m_vs_300m_config_max_comparison.png"
    output_stats = "plots/1000m_vs_300m_config_max_statistics.csv"
    
    print("="*80)
    print("FIS RESULTS COMPARISON: 1000m vs 300m Resolution")
    print("Both using Config Max configuration")
    print("="*80)
    
    try:
        # Create plots directory if it doesn't exist
        os.makedirs("plots", exist_ok=True)
        
        # Check if 300m config_max exists, if not, we need to run it
        if not os.path.exists(file_300m_config_max):
            print(f"❌ 300m config_max file not found: {file_300m_config_max}")
            print("You need to run the 300m config_max processing first.")
            print("Run: python run_all_remaining_fis_models.py")
            return
        
        # Load data and create comparison plot
        meta_1000m, meta_300m = plot_side_by_side_comparison(
            file_1000m=file_1000m_config_max,
            file_300m=file_300m_config_max,
            output_file=output_plot,
            colormap='viridis',
            dpi=300,
            figsize=(16, 8)
        )
        
        # Create statistics table
        stats_df = create_statistics_table(meta_1000m, meta_300m, output_stats)
        
        # Perform detailed analysis
        analyze_differences(meta_1000m, meta_300m)
        
        print(f"\n✅ Comparison completed successfully!")
        print(f"📊 Plot saved to: {output_plot}")
        print(f"📋 Statistics saved to: {output_stats}")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please ensure both 1000m and 300m config_max files exist:")
        print(f"  - {file_1000m_config_max}")
        print(f"  - {file_300m_config_max}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 