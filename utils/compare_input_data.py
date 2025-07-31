#!/usr/bin/env python3
"""
Compare input data for 300m and 1000m resolutions
This script creates side-by-side comparisons of the input rasters to identify any differences
that might explain the FIS output variations.
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


def plot_input_comparison(input_files_1000m, input_files_300m, output_file=None, 
                         colormap='viridis', dpi=300, figsize=(20, 15)):
    """
    Plot input data comparison for both resolutions.
    
    Args:
        input_files_1000m: dict with keys 'social', 'environmental', 'strategic'
        input_files_300m: dict with keys 'social', 'environmental', 'strategic'
        output_file: Path to save the comparison plot
        colormap: Matplotlib colormap name
        dpi: DPI for saved image
        figsize: Figure size (width, height)
    """
    
    # Load all data
    data_1000m = {}
    data_300m = {}
    meta_1000m = {}
    meta_300m = {}
    
    input_types = ['social', 'environmental', 'strategic']
    
    print("Loading 1000m input data...")
    for input_type in input_types:
        data_1000m[input_type], meta_1000m[input_type] = load_raster_data(input_files_1000m[input_type])
        print(f"  {input_type}: {meta_1000m[input_type]['shape']}, range: {meta_1000m[input_type]['min']:.3f} to {meta_1000m[input_type]['max']:.3f}")
    
    print("Loading 300m input data...")
    for input_type in input_types:
        data_300m[input_type], meta_300m[input_type] = load_raster_data(input_files_300m[input_type])
        print(f"  {input_type}: {meta_300m[input_type]['shape']}, range: {meta_300m[input_type]['min']:.3f} to {meta_300m[input_type]['max']:.3f}")
    
    # Determine global min/max for each input type
    global_ranges = {}
    for input_type in input_types:
        global_min = min(meta_1000m[input_type]['min'], meta_300m[input_type]['min'])
        global_max = max(meta_1000m[input_type]['max'], meta_300m[input_type]['max'])
        global_ranges[input_type] = (global_min, global_max)
        print(f"Global range for {input_type}: {global_min:.3f} to {global_max:.3f}")
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=figsize)
    
    # Plot each input type
    for i, input_type in enumerate(input_types):
        # 1000m plot
        im1 = axes[i, 0].imshow(data_1000m[input_type], cmap=colormap, 
                               vmin=global_ranges[input_type][0], 
                               vmax=global_ranges[input_type][1])
        axes[i, 0].set_title(f'1000m {input_type.title()}\n{meta_1000m[input_type]["shape"][0]}x{meta_1000m[input_type]["shape"][1]} pixels')
        axes[i, 0].set_xlabel('Column')
        axes[i, 0].set_ylabel('Row')
        
        # Add statistics for 1000m
        stats_text_1000m = f"""Statistics:
Min: {meta_1000m[input_type]['min']:.3f}
Max: {meta_1000m[input_type]['max']:.3f}
Mean: {meta_1000m[input_type]['mean']:.3f}
Std: {meta_1000m[input_type]['std']:.3f}
Valid pixels: {meta_1000m[input_type]['valid_pixels']:,}"""
        
        axes[i, 0].text(0.02, 0.98, stats_text_1000m, transform=axes[i, 0].transAxes, 
                       verticalalignment='top', bbox=dict(boxstyle='round', 
                       facecolor='white', alpha=0.9), fontsize=8)
        
        # 300m plot
        im2 = axes[i, 1].imshow(data_300m[input_type], cmap=colormap, 
                               vmin=global_ranges[input_type][0], 
                               vmax=global_ranges[input_type][1])
        axes[i, 1].set_title(f'300m {input_type.title()}\n{meta_300m[input_type]["shape"][0]}x{meta_300m[input_type]["shape"][1]} pixels')
        axes[i, 1].set_xlabel('Column')
        axes[i, 1].set_ylabel('Row')
        
        # Add statistics for 300m
        stats_text_300m = f"""Statistics:
Min: {meta_300m[input_type]['min']:.3f}
Max: {meta_300m[input_type]['max']:.3f}
Mean: {meta_300m[input_type]['mean']:.3f}
Std: {meta_300m[input_type]['std']:.3f}
Valid pixels: {meta_300m[input_type]['valid_pixels']:,}"""
        
        axes[i, 1].text(0.02, 0.98, stats_text_300m, transform=axes[i, 1].transAxes, 
                       verticalalignment='top', bbox=dict(boxstyle='round', 
                       facecolor='white', alpha=0.9), fontsize=8)
        
        # Add colorbar for each row
        cbar = plt.colorbar(im1, ax=[axes[i, 0], axes[i, 1]], shrink=0.8, aspect=30)
        cbar.set_label(f'{input_type.title()} Value')
    
    # Add overall title
    fig.suptitle('Input Data Comparison: 1000m vs 300m Resolution', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    # Save if output file specified
    if output_file:
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
        print(f"Input comparison plot saved to: {output_file}")
    
    plt.show()
    
    return meta_1000m, meta_300m


def create_input_statistics_table(meta_1000m, meta_300m, output_file=None):
    """
    Create a statistics comparison table for input data.
    
    Args:
        meta_1000m: 1000m metadata dict
        meta_300m: 300m metadata dict
        output_file: optional output file for the table
    """
    
    input_types = ['social', 'environmental', 'strategic']
    
    # Create comparison data
    comparison_data = {
        'Input Type': [],
        'Metric': [],
        '1000m': [],
        '300m': [],
        'Difference': []
    }
    
    for input_type in input_types:
        # Add metrics for each input type
        metrics = [
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
        ]
        
        for metric in metrics:
            comparison_data['Input Type'].append(input_type.title())
            comparison_data['Metric'].append(metric)
            
            if metric == 'Resolution':
                comparison_data['1000m'].append('1000m')
                comparison_data['300m'].append('300m')
                comparison_data['Difference'].append('3.33x finer')
            elif metric == 'Dimensions (pixels)':
                comparison_data['1000m'].append(f"{meta_1000m[input_type]['shape'][0]} × {meta_1000m[input_type]['shape'][1]}")
                comparison_data['300m'].append(f"{meta_300m[input_type]['shape'][0]} × {meta_300m[input_type]['shape'][1]}")
                comparison_data['Difference'].append(f"{meta_300m[input_type]['shape'][0] / meta_1000m[input_type]['shape'][0]:.1f}x rows, {meta_300m[input_type]['shape'][1] / meta_1000m[input_type]['shape'][1]:.1f}x cols")
            elif metric == 'Total Pixels':
                total_1000m = meta_1000m[input_type]['shape'][0] * meta_1000m[input_type]['shape'][1]
                total_300m = meta_300m[input_type]['shape'][0] * meta_300m[input_type]['shape'][1]
                comparison_data['1000m'].append(f"{total_1000m:,}")
                comparison_data['300m'].append(f"{total_300m:,}")
                comparison_data['Difference'].append(f"{total_300m / total_1000m:.1f}x")
            elif metric == 'Valid Pixels':
                comparison_data['1000m'].append(f"{meta_1000m[input_type]['valid_pixels']:,}")
                comparison_data['300m'].append(f"{meta_300m[input_type]['valid_pixels']:,}")
                comparison_data['Difference'].append(f"{meta_300m[input_type]['valid_pixels'] / meta_1000m[input_type]['valid_pixels']:.1f}x")
            elif metric == 'File Size (MB)':
                comparison_data['1000m'].append(f"{meta_1000m[input_type]['file_size_mb']:.1f}")
                comparison_data['300m'].append(f"{meta_300m[input_type]['file_size_mb']:.1f}")
                comparison_data['Difference'].append(f"{meta_300m[input_type]['file_size_mb'] / meta_1000m[input_type]['file_size_mb']:.1f}x")
            elif metric == 'Min Value':
                comparison_data['1000m'].append(f"{meta_1000m[input_type]['min']:.3f}")
                comparison_data['300m'].append(f"{meta_300m[input_type]['min']:.3f}")
                comparison_data['Difference'].append(f"{meta_300m[input_type]['min'] - meta_1000m[input_type]['min']:+.3f}")
            elif metric == 'Max Value':
                comparison_data['1000m'].append(f"{meta_1000m[input_type]['max']:.3f}")
                comparison_data['300m'].append(f"{meta_300m[input_type]['max']:.3f}")
                comparison_data['Difference'].append(f"{meta_300m[input_type]['max'] - meta_1000m[input_type]['max']:+.3f}")
            elif metric == 'Mean Value':
                comparison_data['1000m'].append(f"{meta_1000m[input_type]['mean']:.3f}")
                comparison_data['300m'].append(f"{meta_300m[input_type]['mean']:.3f}")
                comparison_data['Difference'].append(f"{meta_300m[input_type]['mean'] - meta_1000m[input_type]['mean']:+.3f}")
            elif metric == 'Median Value':
                comparison_data['1000m'].append(f"{meta_1000m[input_type]['median']:.3f}")
                comparison_data['300m'].append(f"{meta_300m[input_type]['median']:.3f}")
                comparison_data['Difference'].append(f"{meta_300m[input_type]['median'] - meta_1000m[input_type]['median']:+.3f}")
            elif metric == 'Standard Deviation':
                comparison_data['1000m'].append(f"{meta_1000m[input_type]['std']:.3f}")
                comparison_data['300m'].append(f"{meta_300m[input_type]['std']:.3f}")
                comparison_data['Difference'].append(f"{meta_300m[input_type]['std'] - meta_1000m[input_type]['std']:+.3f}")
            elif metric == 'Value Range':
                range_1000m = meta_1000m[input_type]['max'] - meta_1000m[input_type]['min']
                range_300m = meta_300m[input_type]['max'] - meta_300m[input_type]['min']
                comparison_data['1000m'].append(f"{range_1000m:.3f}")
                comparison_data['300m'].append(f"{range_300m:.3f}")
                comparison_data['Difference'].append(f"{range_300m - range_1000m:+.3f}")
            elif metric == 'Data Density (%)':
                density_1000m = meta_1000m[input_type]['valid_pixels'] / (meta_1000m[input_type]['shape'][0] * meta_1000m[input_type]['shape'][1]) * 100
                density_300m = meta_300m[input_type]['valid_pixels'] / (meta_300m[input_type]['shape'][0] * meta_300m[input_type]['shape'][1]) * 100
                comparison_data['1000m'].append(f"{density_1000m:.1f}%")
                comparison_data['300m'].append(f"{density_300m:.1f}%")
                comparison_data['Difference'].append(f"{density_300m - density_1000m:+.1f}%")
    
    # Create DataFrame
    df = pd.DataFrame(comparison_data)
    
    # Print the table
    print("\n" + "="*120)
    print("INPUT DATA STATISTICS COMPARISON")
    print("1000m vs 300m Resolution")
    print("="*120)
    print(df.to_string(index=False))
    
    # Save to CSV if requested
    if output_file:
        df.to_csv(output_file, index=False)
        print(f"\nInput statistics table saved to: {output_file}")
    
    return df


def analyze_input_differences(meta_1000m, meta_300m):
    """
    Analyze differences between input datasets.
    """
    
    print("\n" + "="*80)
    print("INPUT DATA DIFFERENCE ANALYSIS")
    print("="*80)
    
    input_types = ['social', 'environmental', 'strategic']
    
    for input_type in input_types:
        print(f"\n{input_type.upper()} INPUT ANALYSIS:")
        
        # Calculate differences
        mean_diff = abs(meta_1000m[input_type]['mean'] - meta_300m[input_type]['mean'])
        std_diff = abs(meta_1000m[input_type]['std'] - meta_300m[input_type]['std'])
        range_diff = abs((meta_1000m[input_type]['max'] - meta_1000m[input_type]['min']) - 
                        (meta_300m[input_type]['max'] - meta_300m[input_type]['min']))
        
        print(f"  Mean Value Difference: {mean_diff:.3f}")
        print(f"  Standard Deviation Difference: {std_diff:.3f}")
        print(f"  Value Range Difference: {range_diff:.3f}")
        
        # Quality assessment
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


def main():
    """Main function to run the input data comparison."""
    
    # Define file paths
    input_files_1000m = {
        'social': "app/files/input/base/socioeconomico_1000m.tif",
        'environmental': "app/files/input/base/ambiental_1000m.tif",
        'strategic': "app/files/input/base/estratégico_1000m.tif"
    }
    
    input_files_300m = {
        'social': "app/files/input/300m/socioeconomico_300m.tif",
        'environmental': "app/files/input/300m/ambiental_300m.tif",
        'strategic': "app/files/input/300m/estrategico_300m.tif"
    }
    
    output_plot = "plots/input_data_comparison.png"
    output_stats = "plots/input_data_statistics.csv"
    
    print("="*80)
    print("INPUT DATA COMPARISON: 1000m vs 300m Resolution")
    print("="*80)
    
    try:
        # Create plots directory if it doesn't exist
        os.makedirs("plots", exist_ok=True)
        
        # Load data and create comparison plot
        meta_1000m, meta_300m = plot_input_comparison(
            input_files_1000m=input_files_1000m,
            input_files_300m=input_files_300m,
            output_file=output_plot,
            colormap='viridis',
            dpi=300,
            figsize=(20, 15)
        )
        
        # Create statistics table
        stats_df = create_input_statistics_table(meta_1000m, meta_300m, output_stats)
        
        # Perform detailed analysis
        analyze_input_differences(meta_1000m, meta_300m)
        
        print(f"\n✅ Input data comparison completed successfully!")
        print(f"📊 Plot saved to: {output_plot}")
        print(f"📋 Statistics saved to: {output_stats}")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please ensure all input files exist:")
        for res, files in [("1000m", input_files_1000m), ("300m", input_files_300m)]:
            print(f"  {res} files:")
            for input_type, file_path in files.items():
                print(f"    {input_type}: {file_path}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 