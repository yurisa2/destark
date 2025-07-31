#!/usr/bin/env python3
"""
Detailed analysis of FIS results to identify potential issues.
This script examines value distributions, spatial patterns, and compares results with input data.
"""

import numpy as np
import matplotlib.pyplot as plt
import rasterio
import os
from pathlib import Path
import seaborn as sns


def analyze_value_distribution(data, title, output_file=None):
    """
    Analyze the distribution of values in the raster data.
    
    Args:
        data: numpy array of raster data
        title: title for the plot
        output_file: optional output file path
    """
    
    # Flatten data and remove any NaN/inf values
    flat_data = data.flatten()
    flat_data = flat_data[np.isfinite(flat_data)]
    
    print(f"\n{title} - Value Distribution Analysis:")
    print(f"  Total pixels: {len(flat_data):,}")
    print(f"  Min: {flat_data.min():.3f}")
    print(f"  Max: {flat_data.max():.3f}")
    print(f"  Mean: {flat_data.mean():.3f}")
    print(f"  Median: {np.median(flat_data):.3f}")
    print(f"  Std: {flat_data.std():.3f}")
    print(f"  Unique values: {len(np.unique(flat_data))}")
    
    # Check for suspicious patterns
    unique_vals, counts = np.unique(flat_data, return_counts=True)
    print(f"  Most common values:")
    for i in range(min(5, len(unique_vals))):
        print(f"    {unique_vals[i]:.3f}: {counts[i]:,} pixels ({counts[i]/len(flat_data)*100:.1f}%)")
    
    # Create distribution plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Histogram
    ax1.hist(flat_data, bins=50, alpha=0.7, edgecolor='black')
    ax1.set_title(f'{title} - Value Distribution')
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Frequency')
    ax1.axvline(flat_data.mean(), color='red', linestyle='--', label=f'Mean: {flat_data.mean():.3f}')
    ax1.axvline(np.median(flat_data), color='green', linestyle='--', label=f'Median: {np.median(flat_data):.3f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Box plot
    ax2.boxplot(flat_data, vert=False)
    ax2.set_title(f'{title} - Box Plot')
    ax2.set_xlabel('Value')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Distribution plot saved to: {output_file}")
    
    plt.show()
    
    return flat_data


def analyze_spatial_patterns(data, title, output_file=None):
    """
    Analyze spatial patterns in the raster data.
    
    Args:
        data: numpy array of raster data
        title: title for the plot
        output_file: optional output file path
    """
    
    print(f"\n{title} - Spatial Pattern Analysis:")
    
    # Calculate spatial statistics
    row_means = np.mean(data, axis=1)
    col_means = np.mean(data, axis=0)
    
    print(f"  Row means: {row_means.min():.3f} to {row_means.max():.3f}")
    print(f"  Column means: {col_means.min():.3f} to {col_means.max():.3f}")
    
    # Create spatial pattern plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Original data
    im1 = ax1.imshow(data, cmap='viridis')
    ax1.set_title(f'{title} - Full Data')
    plt.colorbar(im1, ax=ax1)
    
    # Row means
    ax2.plot(row_means, range(len(row_means)))
    ax2.set_title('Row Means')
    ax2.set_xlabel('Mean Value')
    ax2.set_ylabel('Row Index')
    ax2.grid(True, alpha=0.3)
    
    # Column means
    ax3.plot(range(len(col_means)), col_means)
    ax3.set_title('Column Means')
    ax3.set_xlabel('Column Index')
    ax3.set_ylabel('Mean Value')
    ax3.grid(True, alpha=0.3)
    
    # Value range by position
    value_range = data.max(axis=0) - data.min(axis=0)
    im4 = ax4.imshow(value_range.reshape(1, -1), cmap='plasma', aspect='auto')
    ax4.set_title('Value Range by Column')
    ax4.set_xlabel('Column Index')
    ax4.set_ylabel('Value Range')
    plt.colorbar(im4, ax=ax4)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Spatial pattern plot saved to: {output_file}")
    
    plt.show()


def compare_with_input_data(output_file, input_files, output_plot=None):
    """
    Compare output results with input data to check for consistency.
    
    Args:
        output_file: path to output file
        input_files: list of input file paths
        output_plot: optional output plot path
    """
    
    print(f"\nComparing output with input data...")
    
    # Load output data
    with rasterio.open(output_file) as src:
        output_data = src.read(1)
    
    # Load input data
    input_data = []
    input_names = ['Environmental', 'Social', 'Strategic']
    
    for i, input_file in enumerate(input_files):
        if os.path.exists(input_file):
            with rasterio.open(input_file) as src:
                data = src.read(1)
                input_data.append(data)
                print(f"  Loaded {input_names[i]}: {data.shape}, range: {data.min():.3f} to {data.max():.3f}")
        else:
            print(f"  Warning: {input_file} not found")
            input_data.append(None)
    
    # Create comparison plots
    if any(data is not None for data in input_data):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Output data
        im1 = axes[0, 0].imshow(output_data, cmap='viridis')
        axes[0, 0].set_title('FIS Output')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # Input data plots
        for i, (data, name) in enumerate(zip(input_data, input_names)):
            if data is not None:
                row = (i + 1) // 2
                col = (i + 1) % 2
                im = axes[row, col].imshow(data, cmap='viridis')
                axes[row, col].set_title(f'{name} Input')
                plt.colorbar(im, ax=axes[row, col])
        
        # Hide unused subplot
        if len(input_data) < 3:
            axes[1, 1].set_visible(False)
        
        plt.tight_layout()
        
        if output_plot:
            plt.savefig(output_plot, dpi=300, bbox_inches='tight')
            print(f"Input-output comparison saved to: {output_plot}")
        
        plt.show()
        
        # Check for correlations
        print(f"\nCorrelation Analysis:")
        for i, (data, name) in enumerate(zip(input_data, input_names)):
            if data is not None and data.shape == output_data.shape:
                # Reshape for correlation calculation
                flat_output = output_data.flatten()
                flat_input = data.flatten()
                
                # Remove any NaN/inf values
                valid_mask = np.isfinite(flat_output) & np.isfinite(flat_input)
                if np.sum(valid_mask) > 0:
                    correlation = np.corrcoef(flat_output[valid_mask], flat_input[valid_mask])[0, 1]
                    print(f"  {name} correlation with output: {correlation:.3f}")
                else:
                    print(f"  {name}: No valid data for correlation")


def check_for_anomalies(data, title):
    """
    Check for anomalies in the data.
    
    Args:
        data: numpy array of raster data
        title: title for the analysis
    """
    
    print(f"\n{title} - Anomaly Detection:")
    
    flat_data = data.flatten()
    flat_data = flat_data[np.isfinite(flat_data)]
    
    # Check for constant values
    if len(np.unique(flat_data)) == 1:
        print(f"  ⚠️  WARNING: All values are identical ({flat_data[0]})")
    
    # Check for extreme outliers
    q1, q3 = np.percentile(flat_data, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    outliers = flat_data[(flat_data < lower_bound) | (flat_data > upper_bound)]
    print(f"  Outliers (IQR method): {len(outliers):,} pixels ({len(outliers)/len(flat_data)*100:.1f}%)")
    
    # Check for suspicious patterns
    unique_vals, counts = np.unique(flat_data, return_counts=True)
    dominant_value = unique_vals[np.argmax(counts)]
    dominant_percentage = np.max(counts) / len(flat_data) * 100
    
    if dominant_percentage > 50:
        print(f"  ⚠️  WARNING: {dominant_percentage:.1f}% of pixels have the same value ({dominant_value})")
    
    # Check for reasonable value ranges
    if flat_data.max() - flat_data.min() < 0.001:
        print(f"  ⚠️  WARNING: Very small value range ({flat_data.max() - flat_data.min():.6f})")
    
    return outliers


def main():
    """Main function to run the detailed analysis."""
    
    # Define file paths
    output_1000m = "app/files/output/1000m_output_tifffile.tif"
    output_300m = "app/files/output/output_300m_config_minimum.tif"
    
    input_1000m = [
        "app/files/input/base/ambiental_1000m.tif",
        "app/files/input/base/socioeconomico_1000m.tif", 
        "app/files/input/base/estratégico_1000m.tif"
    ]
    
    input_300m = [
        "app/files/input/300m/ambiental_300m.tif",
        "app/files/input/300m/socioeconomico_300m.tif",
        "app/files/input/300m/estrategico_300m.tif"
    ]
    
    print("="*70)
    print("DETAILED FIS RESULTS ANALYSIS")
    print("="*70)
    
    try:
        # Create plots directory
        os.makedirs("plots", exist_ok=True)
        
        # Analyze 1000m results
        print("\n" + "="*50)
        print("ANALYZING 1000m RESULTS")
        print("="*50)
        
        with rasterio.open(output_1000m) as src:
            data_1000m = src.read(1)
        
        analyze_value_distribution(data_1000m, "1000m FIS Output", 
                                 "plots/1000m_distribution_analysis.png")
        analyze_spatial_patterns(data_1000m, "1000m FIS Output",
                               "plots/1000m_spatial_patterns.png")
        check_for_anomalies(data_1000m, "1000m FIS Output")
        compare_with_input_data(output_1000m, input_1000m,
                              "plots/1000m_input_output_comparison.png")
        
        # Analyze 300m results
        print("\n" + "="*50)
        print("ANALYZING 300m RESULTS")
        print("="*50)
        
        with rasterio.open(output_300m) as src:
            data_300m = src.read(1)
        
        analyze_value_distribution(data_300m, "300m FIS Output",
                                 "plots/300m_distribution_analysis.png")
        analyze_spatial_patterns(data_300m, "300m FIS Output",
                               "plots/300m_spatial_patterns.png")
        check_for_anomalies(data_300m, "300m FIS Output")
        compare_with_input_data(output_300m, input_300m,
                              "plots/300m_input_output_comparison.png")
        
        print(f"\n✅ Detailed analysis completed successfully!")
        print(f"📊 Analysis plots saved to plots/ directory")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 