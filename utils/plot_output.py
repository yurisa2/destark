#!/usr/bin/env python3
"""
Script to plot TIFF raster files with visualization options.
Supports plotting input rasters and output results from FIS processing.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import show
import os
from pathlib import Path


def plot_raster(file_path, title=None, output_file=None, colormap='viridis', 
                show_stats=True, dpi=300, figsize=(10, 8)):
    """
    Plot a raster file with optional statistics and save to file.
    
    Args:
        file_path (str): Path to the TIFF file
        title (str): Title for the plot
        output_file (str): Path to save the plot (optional)
        colormap (str): Matplotlib colormap name
        show_stats (bool): Whether to display statistics
        dpi (int): DPI for saved image
        figsize (tuple): Figure size (width, height)
    """
    
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist.")
        return
    
    # Read the raster
    with rasterio.open(file_path) as src:
        data = src.read(1)  # Read first band
        transform = src.transform
        crs = src.crs
        
        # Get NoData value
        nodata = src.nodata
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot the raster using imshow for better control
        im = ax.imshow(data, cmap=colormap, extent=[0, data.shape[1], 0, data.shape[0]])
        ax.set_title(title or os.path.basename(file_path))
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Value')
        
        # Add statistics if requested
        if show_stats:
            # Calculate statistics excluding NoData
            if nodata is not None:
                valid_data = data[data != nodata]
            else:
                valid_data = data.flatten()
            
            if len(valid_data) > 0:
                stats_text = f"""
Statistics:
Min: {valid_data.min():.3f}
Max: {valid_data.max():.3f}
Mean: {valid_data.mean():.3f}
Std: {valid_data.std():.3f}
Valid pixels: {len(valid_data):,}
                """.strip()
                
                # Add text box with statistics
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                       verticalalignment='top', bbox=dict(boxstyle='round', 
                       facecolor='white', alpha=0.8), fontsize=9)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if output file specified
        if output_file:
            plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
            print(f"Plot saved to: {output_file}")
        
        # Show plot
        plt.show()


def plot_comparison(input_files, output_file, titles=None, output_plot=None, 
                   colormap='viridis', dpi=300, figsize=(15, 10)):
    """
    Plot multiple input rasters and the output raster for comparison.
    
    Args:
        input_files (list): List of input TIFF file paths
        output_file (str): Path to output TIFF file
        titles (list): List of titles for each plot
        output_plot (str): Path to save the comparison plot
        colormap (str): Matplotlib colormap name
        dpi (int): DPI for saved image
        figsize (tuple): Figure size (width, height)
    """
    
    # Check if all files exist
    all_files = input_files + [output_file]
    for file_path in all_files:
        if not os.path.exists(file_path):
            print(f"Error: File {file_path} does not exist.")
            return
    
    # Default titles
    if titles is None:
        titles = ['Environmental', 'Social', 'Strategic', 'FIS Output']
    
    # Create subplots
    n_plots = len(all_files)
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    # Plot each raster
    for i, (file_path, title) in enumerate(zip(all_files, titles)):
        if i >= len(axes):
            break
            
        with rasterio.open(file_path) as src:
            data = src.read(1)
            transform = src.transform
            nodata = src.nodata
            
            # Plot using imshow for better control
            im = axes[i].imshow(data, cmap=colormap, extent=[0, data.shape[1], 0, data.shape[0]])
            axes[i].set_title(title)
            
            # Add colorbar
            plt.colorbar(im, ax=axes[i])
            
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
    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    # Save if output plot specified
    if output_plot:
        plt.savefig(output_plot, dpi=dpi, bbox_inches='tight')
        print(f"Comparison plot saved to: {output_plot}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Plot TIFF raster files')
    parser.add_argument('input_file', help='Path to the TIFF file to plot')
    parser.add_argument('-o', '--output', help='Output file for the plot (PNG/PDF)')
    parser.add_argument('-t', '--title', help='Title for the plot')
    parser.add_argument('-c', '--colormap', default='viridis', 
                       help='Matplotlib colormap (default: viridis)')
    parser.add_argument('--no-stats', action='store_true', 
                       help='Hide statistics on plot')
    parser.add_argument('--dpi', type=int, default=300, 
                       help='DPI for saved image (default: 300)')
    parser.add_argument('--figsize', nargs=2, type=float, default=[10, 8],
                       help='Figure size width height (default: 10 8)')
    
    # Comparison mode
    parser.add_argument('--comparison', action='store_true',
                       help='Plot comparison of input files and output')
    parser.add_argument('--input-files', nargs='+', 
                       help='Input files for comparison (environmental social strategic)')
    parser.add_argument('--output-file', 
                       help='Output file for comparison')
    parser.add_argument('--titles', nargs='+',
                       help='Titles for comparison plots')
    parser.add_argument('--output-plot',
                       help='Output file for comparison plot')
    
    args = parser.parse_args()
    
    if args.comparison:
        if not args.input_files or not args.output_file:
            print("Error: --comparison mode requires --input-files and --output-file")
            return
        
        plot_comparison(
            input_files=args.input_files,
            output_file=args.output_file,
            titles=args.titles,
            output_plot=args.output_plot,
            colormap=args.colormap,
            dpi=args.dpi,
            figsize=tuple(args.figsize)
        )
    else:
        plot_raster(
            file_path=args.input_file,
            title=args.title,
            output_file=args.output,
            colormap=args.colormap,
            show_stats=not args.no_stats,
            dpi=args.dpi,
            figsize=tuple(args.figsize)
        )


if __name__ == "__main__":
    main() 