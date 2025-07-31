#!/usr/bin/env python3
"""
Explore tifffile library directly
Read and plot input files using tifffile to understand the data structure
"""

import numpy as np
import matplotlib.pyplot as plt
import tifffile
import os
from pathlib import Path


def explore_tifffile_file(file_path):
    """
    Explore a TIFF file using tifffile library.
    
    Args:
        file_path (str): Path to the TIFF file
    """
    
    print(f"Exploring file: {file_path}")
    print(f"File exists: {os.path.exists(file_path)}")
    
    if not os.path.exists(file_path):
        print(f"❌ File does not exist: {file_path}")
        return
    
    # Get file size
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    print(f"File size: {file_size_mb:.1f} MB")
    
    try:
        # Read the file with tifffile
        print("\nReading with tifffile...")
        data = tifffile.imread(file_path)
        
        print(f"Data type: {data.dtype}")
        print(f"Data shape: {data.shape}")
        print(f"Data ndim: {data.ndim}")
        
        # If it's a 3D array, show info about each band
        if data.ndim == 3:
            print(f"Number of bands: {data.shape[0]}")
            for i in range(data.shape[0]):
                band_data = data[i]
                print(f"  Band {i}: shape {band_data.shape}, range {band_data.min():.3f} to {band_data.max():.3f}")
        else:
            # Single band or 2D array
            print(f"Data range: {data.min():.3f} to {data.max():.3f}")
            print(f"Data mean: {data.mean():.3f}")
            print(f"Data std: {data.std():.3f}")
            print(f"Data median: {np.median(data):.3f}")
            print(f"Unique values: {len(np.unique(data))}")
            
            # Show some sample values
            unique_vals, counts = np.unique(data, return_counts=True)
            print(f"Most common values:")
            for i in range(min(10, len(unique_vals))):
                print(f"  {unique_vals[i]:.3f}: {counts[i]:,} pixels ({counts[i]/data.size*100:.1f}%)")
        
        return data
        
    except Exception as e:
        print(f"❌ Error reading file with tifffile: {e}")
        import traceback
        traceback.print_exc()
        return None


def plot_tifffile_data(data, title, output_file=None):
    """
    Plot the tifffile data.
    
    Args:
        data: numpy array from tifffile
        title: title for the plot
        output_file: optional output file path
    """
    
    if data is None:
        print("❌ No data to plot")
        return
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Main data plot
    if data.ndim == 3:
        # Multi-band data - show first band
        im1 = axes[0, 0].imshow(data[0], cmap='viridis')
        axes[0, 0].set_title(f'{title} - Band 1\n{data.shape[1]}x{data.shape[2]} pixels')
    else:
        # Single band data
        im1 = axes[0, 0].imshow(data, cmap='viridis')
        axes[0, 0].set_title(f'{title}\n{data.shape[0]}x{data.shape[1]} pixels')
    
    axes[0, 0].set_xlabel('Column')
    axes[0, 0].set_ylabel('Row')
    plt.colorbar(im1, ax=axes[0, 0], label='Value')
    
    # Histogram
    if data.ndim == 3:
        flat_data = data[0].flatten()
    else:
        flat_data = data.flatten()
    
    axes[0, 1].hist(flat_data, bins=50, alpha=0.7, edgecolor='black')
    axes[0, 1].set_title('Value Distribution')
    axes[0, 1].set_xlabel('Value')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].axvline(flat_data.mean(), color='red', linestyle='--', label=f'Mean: {flat_data.mean():.3f}')
    axes[0, 1].axvline(np.median(flat_data), color='green', linestyle='--', label=f'Median: {np.median(flat_data):.3f}')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Statistics text
    stats_text = f"""Statistics:
Min: {flat_data.min():.3f}
Max: {flat_data.max():.3f}
Mean: {flat_data.mean():.3f}
Median: {np.median(flat_data):.3f}
Std: {flat_data.std():.3f}
Unique values: {len(np.unique(flat_data))}
Total pixels: {len(flat_data):,}"""
    
    axes[1, 0].text(0.1, 0.9, stats_text, transform=axes[1, 0].transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', 
                    facecolor='white', alpha=0.9), fontsize=10)
    axes[1, 0].set_title('Data Statistics')
    axes[1, 0].axis('off')
    
    # Box plot
    axes[1, 1].boxplot(flat_data, vert=False)
    axes[1, 1].set_title('Box Plot')
    axes[1, 1].set_xlabel('Value')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add overall title
    fig.suptitle(f'Tifffile Data Exploration: {title}', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    # Save if output file specified
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_file}")
    
    plt.show()


def compare_with_rasterio(file_path):
    """
    Compare tifffile reading with rasterio reading.
    
    Args:
        file_path (str): Path to the TIFF file
    """
    
    print(f"\n{'='*60}")
    print(f"COMPARING TIFFFILE vs RASTERIO")
    print(f"{'='*60}")
    
    try:
        import rasterio
        
        # Read with tifffile
        print("Reading with tifffile...")
        data_tifffile = tifffile.imread(file_path)
        
        # Read with rasterio
        print("Reading with rasterio...")
        with rasterio.open(file_path) as src:
            data_rasterio = src.read(1)  # Read first band
        
        print(f"\nComparison:")
        print(f"Tifffile shape: {data_tifffile.shape}")
        print(f"Rasterio shape: {data_rasterio.shape}")
        print(f"Tifffile dtype: {data_tifffile.dtype}")
        print(f"Rasterio dtype: {data_rasterio.dtype}")
        
        # Check if they're the same
        if data_tifffile.shape == data_rasterio.shape:
            print(f"Shapes match: ✅")
            
            # Compare data
            if np.array_equal(data_tifffile, data_rasterio):
                print(f"Data is identical: ✅")
            else:
                print(f"Data differs: ❌")
                diff = data_tifffile - data_rasterio
                print(f"Max difference: {np.abs(diff).max():.6f}")
                print(f"Mean difference: {np.abs(diff).mean():.6f}")
                print(f"Non-zero differences: {np.count_nonzero(diff):,}")
        else:
            print(f"Shapes don't match: ❌")
            
        return data_tifffile, data_rasterio
        
    except Exception as e:
        print(f"❌ Error in comparison: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def main():
    """Main function to explore tifffile."""
    
    # Define input files
    input_files = {
        'social': "app/files/input/base/socioeconomico_1000m.tif",
        'environmental': "app/files/input/base/ambiental_1000m.tif",
        'strategic': "app/files/input/base/estratégico_1000m.tif"
    }
    
    print("="*80)
    print("TIFFFILE LIBRARY EXPLORATION")
    print("="*80)
    
    # Create plots directory
    os.makedirs("plots", exist_ok=True)
    
    # Explore each file
    for input_type, file_path in input_files.items():
        print(f"\n{'='*60}")
        print(f"EXPLORING {input_type.upper()} INPUT")
        print(f"{'='*60}")
        
        # Explore with tifffile
        data = explore_tifffile_file(file_path)
        
        if data is not None:
            # Plot the data
            output_plot = f"plots/tifffile_{input_type}_exploration.png"
            plot_tifffile_data(data, f"{input_type.title()} Input", output_plot)
            
            # Compare with rasterio
            data_tifffile, data_rasterio = compare_with_rasterio(file_path)
            
            if data_tifffile is not None and data_rasterio is not None:
                # Create comparison plot
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
                
                # Tifffile plot
                im1 = ax1.imshow(data_tifffile, cmap='viridis')
                ax1.set_title(f'Tifffile - {input_type.title()}')
                plt.colorbar(im1, ax=ax1)
                
                # Rasterio plot
                im2 = ax2.imshow(data_rasterio, cmap='viridis')
                ax2.set_title(f'Rasterio - {input_type.title()}')
                plt.colorbar(im2, ax=ax2)
                
                # Difference plot
                diff = data_tifffile - data_rasterio
                im3 = ax3.imshow(diff, cmap='RdBu_r')
                ax3.set_title(f'Difference (Tifffile - Rasterio)\nMax: {np.abs(diff).max():.6f}')
                plt.colorbar(im3, ax=ax3)
                
                plt.tight_layout()
                plt.savefig(f"plots/tifffile_vs_rasterio_{input_type}.png", dpi=300, bbox_inches='tight')
                plt.show()
        
        # Only process the first file for now
        break


if __name__ == "__main__":
    main() 