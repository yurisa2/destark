#!/usr/bin/env python3
"""
Explore FIS output files directly
Read and analyze the FIS output files to understand the differences
"""

import numpy as np
import matplotlib.pyplot as plt
import tifffile
import rasterio
import os
from pathlib import Path


def explore_fis_output(file_path, library_name):
    """
    Explore a FIS output file.
    
    Args:
        file_path (str): Path to the FIS output file
        library_name (str): Name of the library used (rasterio/tifffile)
    """
    
    print(f"\n{'='*60}")
    print(f"EXPLORING {library_name.upper()} FIS OUTPUT")
    print(f"{'='*60}")
    
    print(f"File: {file_path}")
    print(f"File exists: {os.path.exists(file_path)}")
    
    if not os.path.exists(file_path):
        print(f"❌ File does not exist: {file_path}")
        return None
    
    # Get file size
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    print(f"File size: {file_size_mb:.1f} MB")
    
    try:
        # Read with tifffile
        print("\nReading with tifffile...")
        data_tifffile = tifffile.imread(file_path)
        
        print(f"Tifffile - Data type: {data_tifffile.dtype}")
        print(f"Tifffile - Data shape: {data_tifffile.shape}")
        print(f"Tifffile - Data ndim: {data_tifffile.ndim}")
        print(f"Tifffile - Data range: {data_tifffile.min():.3f} to {data_tifffile.max():.3f}")
        print(f"Tifffile - Data mean: {data_tifffile.mean():.3f}")
        print(f"Tifffile - Data std: {data_tifffile.std():.3f}")
        print(f"Tifffile - Data median: {np.median(data_tifffile):.3f}")
        print(f"Tifffile - Unique values: {len(np.unique(data_tifffile))}")
        
        # Read with rasterio
        print("\nReading with rasterio...")
        with rasterio.open(file_path) as src:
            data_rasterio = src.read(1)  # Read first band
        
        print(f"Rasterio - Data type: {data_rasterio.dtype}")
        print(f"Rasterio - Data shape: {data_rasterio.shape}")
        print(f"Rasterio - Data range: {data_rasterio.min():.3f} to {data_rasterio.max():.3f}")
        print(f"Rasterio - Data mean: {data_rasterio.mean():.3f}")
        print(f"Rasterio - Data std: {data_rasterio.std():.3f}")
        print(f"Rasterio - Data median: {np.median(data_rasterio):.3f}")
        print(f"Rasterio - Unique values: {len(np.unique(data_rasterio))}")
        
        # Compare tifffile vs rasterio reading
        print(f"\nComparison (Tifffile vs Rasterio reading):")
        if np.array_equal(data_tifffile, data_rasterio):
            print(f"  Data is identical: ✅")
        else:
            print(f"  Data differs: ❌")
            diff = data_tifffile - data_rasterio
            print(f"  Max difference: {np.abs(diff).max():.6f}")
            print(f"  Mean difference: {np.abs(diff).mean():.6f}")
            print(f"  Non-zero differences: {np.count_nonzero(diff):,}")
        
        return data_tifffile, data_rasterio
        
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def plot_fis_output_comparison(data_tifffile, data_rasterio, library_name, output_file=None):
    """
    Plot FIS output comparison.
    
    Args:
        data_tifffile: data read with tifffile
        data_rasterio: data read with rasterio
        library_name: name of the library used for processing
        output_file: optional output file path
    """
    
    if data_tifffile is None or data_rasterio is None:
        print("❌ No data to plot")
        return
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Determine global range for consistent color scaling
    global_min = min(data_tifffile.min(), data_rasterio.min())
    global_max = max(data_tifffile.max(), data_rasterio.max())
    
    # Tifffile reading
    im1 = axes[0, 0].imshow(data_tifffile, cmap='viridis', vmin=global_min, vmax=global_max)
    axes[0, 0].set_title(f'Tifffile Reading\n{data_tifffile.shape[0]}x{data_tifffile.shape[1]} pixels')
    axes[0, 0].set_xlabel('Column')
    axes[0, 0].set_ylabel('Row')
    plt.colorbar(im1, ax=axes[0, 0], label='Value')
    
    # Rasterio reading
    im2 = axes[0, 1].imshow(data_rasterio, cmap='viridis', vmin=global_min, vmax=global_max)
    axes[0, 1].set_title(f'Rasterio Reading\n{data_rasterio.shape[0]}x{data_rasterio.shape[1]} pixels')
    axes[0, 1].set_xlabel('Column')
    axes[0, 1].set_ylabel('Row')
    plt.colorbar(im2, ax=axes[0, 1], label='Value')
    
    # Difference
    diff = data_tifffile - data_rasterio
    im3 = axes[0, 2].imshow(diff, cmap='RdBu_r')
    axes[0, 2].set_title(f'Difference (Tifffile - Rasterio)\nMax: {np.abs(diff).max():.6f}')
    axes[0, 2].set_xlabel('Column')
    axes[0, 2].set_ylabel('Row')
    plt.colorbar(im3, ax=axes[0, 2], label='Difference')
    
    # Histograms
    axes[1, 0].hist(data_tifffile.flatten(), bins=50, alpha=0.7, label='Tifffile', color='blue')
    axes[1, 0].hist(data_rasterio.flatten(), bins=50, alpha=0.7, label='Rasterio', color='red')
    axes[1, 0].set_title('Value Distribution')
    axes[1, 0].set_xlabel('Value')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Statistics comparison
    stats_text = f"""Tifffile Reading:
Min: {data_tifffile.min():.3f}
Max: {data_tifffile.max():.3f}
Mean: {data_tifffile.mean():.3f}
Std: {data_tifffile.std():.3f}
Unique: {len(np.unique(data_tifffile))}

Rasterio Reading:
Min: {data_rasterio.min():.3f}
Max: {data_rasterio.max():.3f}
Mean: {data_rasterio.mean():.3f}
Std: {data_rasterio.std():.3f}
Unique: {len(np.unique(data_rasterio))}

Difference:
Max abs: {np.abs(diff).max():.6f}
Mean abs: {np.abs(diff).mean():.6f}
Non-zero: {np.count_nonzero(diff):,}"""
    
    axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', 
                    facecolor='white', alpha=0.9), fontsize=8)
    axes[1, 1].set_title('Statistics Comparison')
    axes[1, 1].axis('off')
    
    # Box plots
    axes[1, 2].boxplot([data_tifffile.flatten(), data_rasterio.flatten()], 
                       labels=['Tifffile', 'Rasterio'])
    axes[1, 2].set_title('Box Plot Comparison')
    axes[1, 2].set_ylabel('Value')
    axes[1, 2].grid(True, alpha=0.3)
    
    # Add overall title
    fig.suptitle(f'FIS Output Analysis: {library_name} Library', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    # Save if output file specified
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_file}")
    
    plt.show()


def main():
    """Main function to explore FIS outputs."""
    
    # Define FIS output files
    fis_outputs = {
        'rasterio': "app/files/output/output_1000m_config_max_rasterio.tif",
        'tifffile': "app/files/output/output_1000m_config_max_tifffile.tif"
    }
    
    print("="*80)
    print("FIS OUTPUT EXPLORATION")
    print("="*80)
    
    # Create plots directory
    os.makedirs("plots", exist_ok=True)
    
    # Explore each FIS output
    for library_name, file_path in fis_outputs.items():
        data_tifffile, data_rasterio = explore_fis_output(file_path, library_name)
        
        if data_tifffile is not None and data_rasterio is not None:
            # Plot the comparison
            output_plot = f"plots/fis_output_{library_name}_analysis.png"
            plot_fis_output_comparison(data_tifffile, data_rasterio, library_name, output_plot)
    
    # Now compare the two FIS outputs directly
    print(f"\n{'='*80}")
    print("DIRECT FIS OUTPUT COMPARISON")
    print("="*80)
    
    # Read both FIS outputs with tifffile
    rasterio_output = tifffile.imread(fis_outputs['rasterio'])
    tifffile_output = tifffile.imread(fis_outputs['tifffile'])
    
    print(f"Rasterio FIS output:")
    print(f"  Shape: {rasterio_output.shape}")
    print(f"  Range: {rasterio_output.min():.3f} to {rasterio_output.max():.3f}")
    print(f"  Mean: {rasterio_output.mean():.3f}")
    print(f"  Std: {rasterio_output.std():.3f}")
    print(f"  Unique values: {len(np.unique(rasterio_output))}")
    
    print(f"\nTifffile FIS output:")
    print(f"  Shape: {tifffile_output.shape}")
    print(f"  Range: {tifffile_output.min():.3f} to {tifffile_output.max():.3f}")
    print(f"  Mean: {tifffile_output.mean():.3f}")
    print(f"  Std: {tifffile_output.std():.3f}")
    print(f"  Unique values: {len(np.unique(tifffile_output))}")
    
    # Create direct comparison plot
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Rasterio FIS output
    im1 = ax1.imshow(rasterio_output, cmap='viridis')
    ax1.set_title(f'Rasterio FIS Output\nRange: {rasterio_output.min():.3f} to {rasterio_output.max():.3f}')
    plt.colorbar(im1, ax=ax1, label='Value')
    
    # Tifffile FIS output
    im2 = ax2.imshow(tifffile_output, cmap='viridis')
    ax2.set_title(f'Tifffile FIS Output\nRange: {tifffile_output.min():.3f} to {tifffile_output.max():.3f}')
    plt.colorbar(im2, ax=ax2, label='Value')
    
    # Direct difference
    direct_diff = rasterio_output - tifffile_output
    im3 = ax3.imshow(direct_diff, cmap='RdBu_r')
    ax3.set_title(f'Direct Difference (Rasterio - Tifffile)\nMax: {np.abs(direct_diff).max():.6f}')
    plt.colorbar(im3, ax=ax3, label='Difference')
    
    plt.tight_layout()
    plt.savefig("plots/fis_output_direct_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nDirect comparison:")
    print(f"  Max absolute difference: {np.abs(direct_diff).max():.6f}")
    print(f"  Mean absolute difference: {np.abs(direct_diff).mean():.6f}")
    print(f"  Non-zero differences: {np.count_nonzero(direct_diff):,}")


if __name__ == "__main__":
    main() 