#!/usr/bin/env python3
"""
Script to plot all FIS model outputs in a single comparison plot.
Shows all six FIS models plus the original output for comparison.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import show
import os
from pathlib import Path


def plot_all_models(output_files, titles=None, output_plot=None, 
                   colormap='viridis', dpi=300, figsize=(20, 12)):
    """
    Plot all model outputs in a grid layout.
    
    Args:
        output_files (list): List of output TIFF file paths
        titles (list): List of titles for each plot
        output_plot (str): Path to save the comparison plot
        colormap (str): Matplotlib colormap name
        dpi (int): DPI for saved image
        figsize (tuple): Figure size (width, height)
    """
    
    # Check if all files exist
    for file_path in output_files:
        if not os.path.exists(file_path):
            print(f"Error: File {file_path} does not exist.")
            return
    
    # Default titles
    if titles is None:
        titles = [
            'Original Output',
            'Round Up',
            'Round Down', 
            'Mode',
            'Minimum',
            'Maximum',
            'Median'
        ]
    
    # Create subplots - 3 rows, 3 columns (7 plots total)
    fig, axes = plt.subplots(3, 3, figsize=figsize)
    axes = axes.flatten()
    
    # First pass: collect all data to find global min/max for consistent scaling
    all_data = []
    for file_path in output_files:
        with rasterio.open(file_path) as src:
            data = src.read(1)
            nodata = src.nodata
            if nodata is not None:
                valid_data = data[data != nodata]
            else:
                valid_data = data.flatten()
            if len(valid_data) > 0:
                all_data.extend(valid_data)
    
    # Calculate global min/max for consistent scaling
    if all_data:
        global_min = min(all_data)
        global_max = max(all_data)
        # Use 0-10 scale if data is in that range, otherwise use actual min/max
        if global_min >= 0 and global_max <= 10:
            vmin, vmax = 0, 10
        else:
            vmin, vmax = global_min, global_max
    else:
        vmin, vmax = 0, 10
    
    # Plot each raster with consistent scaling
    for i, (file_path, title) in enumerate(zip(output_files, titles)):
        if i >= len(axes):
            break
            
        with rasterio.open(file_path) as src:
            data = src.read(1)
            transform = src.transform
            nodata = src.nodata
            
            # Plot using imshow with consistent scaling
            im = axes[i].imshow(data, cmap=colormap, extent=[0, data.shape[1], 0, data.shape[0]], 
                               vmin=vmin, vmax=vmax)
            axes[i].set_title(title, fontsize=12, fontweight='bold')
            
            # Add colorbar
            plt.colorbar(im, ax=axes[i], shrink=0.8)
            
            # Add basic stats
            if nodata is not None:
                valid_data = data[data != nodata]
            else:
                valid_data = data.flatten()
            
            if len(valid_data) > 0:
                stats_text = f"Min: {valid_data.min():.2f}\nMax: {valid_data.max():.2f}\nMean: {valid_data.mean():.2f}"
                axes[i].text(0.02, 0.98, stats_text, transform=axes[i].transAxes, 
                           verticalalignment='top', bbox=dict(boxstyle='round', 
                           facecolor='white', alpha=0.8), fontsize=8)
    
    # Hide unused subplots
    for i in range(len(output_files), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    # Add overall title
    fig.suptitle('FIS Model Comparison - All Outputs', fontsize=16, fontweight='bold', y=0.98)
    
    # Save if output plot specified
    if output_plot:
        plt.savefig(output_plot, dpi=dpi, bbox_inches='tight')
        print(f"Comparison plot saved to: {output_plot}")
    
    plt.show()


def create_statistics_table(output_files, titles=None):
    """
    Create a statistics table for all models.
    
    Args:
        output_files (list): List of output TIFF file paths
        titles (list): List of titles for each model
    """
    
    if titles is None:
        titles = [
            'Original Output',
            'Round Up',
            'Round Down', 
            'Mode',
            'Minimum',
            'Maximum',
            'Median'
        ]
    
    print("\n=== Model Statistics Comparison ===")
    print(f"{'Model':<15} {'Min':<8} {'Max':<8} {'Mean':<8} {'Std':<8} {'Valid Pixels':<12}")
    print("-" * 70)
    
    for file_path, title in zip(output_files, titles):
        if os.path.exists(file_path):
            with rasterio.open(file_path) as src:
                data = src.read(1)
                nodata = src.nodata
                
                if nodata is not None:
                    valid_data = data[data != nodata]
                else:
                    valid_data = data.flatten()
                
                if len(valid_data) > 0:
                    print(f"{title:<15} {valid_data.min():<8.2f} {valid_data.max():<8.2f} "
                          f"{valid_data.mean():<8.2f} {valid_data.std():<8.2f} {len(valid_data):<12,}")
                else:
                    print(f"{title:<15} {'N/A':<8} {'N/A':<8} {'N/A':<8} {'N/A':<8} {'N/A':<12}")
        else:
            print(f"{title:<15} {'File not found':<8} {'N/A':<8} {'N/A':<8} {'N/A':<8} {'N/A':<12}")


def main():
    parser = argparse.ArgumentParser(description='Plot all FIS model outputs')
    parser.add_argument('--output-files', nargs='+', 
                       help='List of output TIFF files to plot')
    parser.add_argument('--titles', nargs='+',
                       help='Titles for each plot')
    parser.add_argument('--output-plot', 
                       help='Output file for the comparison plot (optional, no file saved by default)')
    parser.add_argument('--colormap', default='viridis', 
                       help='Matplotlib colormap (default: viridis)')
    parser.add_argument('--dpi', type=int, default=300, 
                       help='DPI for saved image (default: 300)')
    parser.add_argument('--figsize', nargs=2, type=float, default=[20, 12],
                       help='Figure size width height (default: 20 12)')
    parser.add_argument('--stats-only', action='store_true',
                       help='Only show statistics table, no plot')
    
    args = parser.parse_args()
    
    # Default output files if not specified
    if not args.output_files:
        args.output_files = [
            'output_max.tif',  # Original output
            'output_round_up.tif',
            'output_round_down.tif',
            'output_mode.tif',
            'output_minimum.tif',
            'output_maximum.tif',
            'output_median.tif'
        ]
    
    # Check which files exist
    existing_files = []
    existing_titles = []
    
    for i, file_path in enumerate(args.output_files):
        if os.path.exists(file_path):
            existing_files.append(file_path)
            if args.titles and i < len(args.titles):
                existing_titles.append(args.titles[i])
            else:
                # Use filename as title if no custom title provided
                existing_titles.append(os.path.basename(file_path))
        else:
            print(f"Warning: File {file_path} not found, skipping...")
    
    if not existing_files:
        print("Error: No output files found!")
        return
    
    # Create statistics table
    create_statistics_table(existing_files, existing_titles)
    
    # Create plot if not stats-only
    if not args.stats_only:
        plot_all_models(
            output_files=existing_files,
            titles=existing_titles,
            output_plot=args.output_plot,
            colormap=args.colormap,
            dpi=args.dpi,
            figsize=tuple(args.figsize)
        )


if __name__ == "__main__":
    main() 