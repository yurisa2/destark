#!/usr/bin/env python3
"""
Example script showing how to use the plot_output.py script.
This demonstrates plotting individual rasters and comparison plots.
"""

import subprocess
import sys
import os

def run_plot_command(cmd):
    """Run a plotting command and print the command."""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    else:
        print("Command completed successfully")
    return result.returncode == 0

def main():
    print("=== TIFF Raster Plotting Examples ===\n")
    
    # Example 1: Plot a single output file
    print("1. Plotting a single output TIFF file:")
    print("   python utils/plot_output.py output.tif -t 'FIS Output' -o output_plot.png")
    print()
    
    # Example 2: Plot with custom colormap
    print("2. Plotting with custom colormap:")
    print("   python utils/plot_output.py output.tif -c 'plasma' -t 'FIS Result' -o output_plasma.png")
    print()
    
    # Example 3: Plot without statistics
    print("3. Plotting without statistics:")
    print("   python utils/plot_output.py output.tif --no-stats -o output_clean.png")
    print()
    
    # Example 4: Comparison plot
    print("4. Comparison plot (input files + output):")
    print("   python utils/plot_output.py dummy.tif --comparison \\")
    print("     --input-files environmental.tif social.tif strategic.tif \\")
    print("     --output-file output.tif \\")
    print("     --output-plot comparison.png")
    print()
    
    # Example 5: High resolution plot
    print("5. High resolution plot:")
    print("   python utils/plot_output.py output.tif --dpi 600 --figsize 12 10 -o output_hires.png")
    print()
    
    print("=== Usage Notes ===")
    print("- Use -o to save plots to files")
    print("- Use -c to change colormap (viridis, plasma, inferno, etc.)")
    print("- Use --comparison for side-by-side comparison")
    print("- Use --dpi for higher resolution output")
    print("- Use --figsize to control plot dimensions")
    print()
    
    print("=== Available Colormaps ===")
    print("Common options: viridis, plasma, inferno, magma, cividis")
    print("Sequential: Blues, Greens, Reds, Oranges, Purples")
    print("Diverging: RdYlBu, RdYlGn, PuOr, BrBG")
    print("Qualitative: Set1, Set2, Set3, tab10, tab20")

if __name__ == "__main__":
    main() 