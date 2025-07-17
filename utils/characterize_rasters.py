#!/usr/bin/env python3
"""
Raster Characterization Script
Analyzes raster files and provides basic statistics for understanding data ranges.
"""

import rasterio
import numpy as np
import argparse
import sys
import os
from pathlib import Path


def analyze_raster(file_path: str, title: str = None):
    """
    Analyze a raster file and return basic statistics.
    
    Args:
        file_path: Path to the raster file
        title: Title for the analysis (defaults to filename)
    
    Returns:
        Dictionary with raster statistics
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return None
    
    try:
        with rasterio.open(file_path) as src:
            # Read the first band
            data = src.read(1)
            
            # Get basic info
            height, width = data.shape
            nodata = src.nodata
            
            # Calculate statistics (excluding NoData values)
            if nodata is not None:
                valid_data = data[data != nodata]
            else:
                valid_data = data[~np.isnan(data)]
            
            if len(valid_data) == 0:
                print(f"Warning: No valid data found in {file_path}")
                return None
            
            # Calculate statistics
            stats = {
                'file': file_path,
                'title': title or os.path.basename(file_path),
                'dimensions': f"{width} x {height}",
                'total_pixels': height * width,
                'valid_pixels': len(valid_data),
                'nodata_pixels': (height * width) - len(valid_data),
                'nodata_value': nodata,
                'min': float(np.min(valid_data)),
                'max': float(np.max(valid_data)),
                'mean': float(np.mean(valid_data)),
                'median': float(np.median(valid_data)),
                'std': float(np.std(valid_data)),
                'percentiles': {
                    '25%': float(np.percentile(valid_data, 25)),
                    '50%': float(np.percentile(valid_data, 50)),
                    '75%': float(np.percentile(valid_data, 75)),
                    '90%': float(np.percentile(valid_data, 90)),
                    '95%': float(np.percentile(valid_data, 95)),
                    '99%': float(np.percentile(valid_data, 99))
                }
            }
            
            return stats
            
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None


def print_statistics(stats):
    """
    Print raster statistics in a formatted way.
    
    Args:
        stats: Dictionary with raster statistics
    """
    if stats is None:
        return
    
    print(f"\n{'='*60}")
    print(f"RASTER ANALYSIS: {stats['title']}")
    print(f"{'='*60}")
    print(f"File: {stats['file']}")
    print(f"Dimensions: {stats['dimensions']}")
    print(f"Total pixels: {stats['total_pixels']:,}")
    print(f"Valid pixels: {stats['valid_pixels']:,}")
    print(f"NoData pixels: {stats['nodata_pixels']:,}")
    print(f"NoData value: {stats['nodata_value']}")
    
    print(f"\n{'='*40}")
    print("BASIC STATISTICS")
    print(f"{'='*40}")
    print(f"Minimum:     {stats['min']:>10.4f}")
    print(f"Maximum:     {stats['max']:>10.4f}")
    print(f"Mean:        {stats['mean']:>10.4f}")
    print(f"Median:      {stats['median']:>10.4f}")
    print(f"Std Dev:     {stats['std']:>10.4f}")
    
    print(f"\n{'='*40}")
    print("PERCENTILES")
    print(f"{'='*40}")
    for pct, value in stats['percentiles'].items():
        print(f"{pct:>6}: {value:>10.4f}")
    
    print(f"\n{'='*40}")
    print("DATA RANGE ANALYSIS")
    print(f"{'='*40}")
    range_val = stats['max'] - stats['min']
    print(f"Data range:  {range_val:>10.4f}")
    print(f"Range %:     {(range_val/stats['max']*100):>10.2f}% of max")
    
    # Check if data is in expected ranges for FIS
    print(f"\n{'='*40}")
    print("FIS COMPATIBILITY CHECK")
    print(f"{'='*40}")
    
    # Check if data is in 0-4 range (as per your FIS models)
    if stats['min'] >= 0 and stats['max'] <= 4:
        print("✅ Data range (0-4): Compatible with FIS models")
    else:
        print("⚠️  Data range: May need scaling for FIS models")
        print(f"   Current range: {stats['min']:.2f} to {stats['max']:.2f}")
        print(f"   Expected range: 0 to 4")
    
    # Check for reasonable data distribution
    if stats['std'] > 0:
        cv = stats['std'] / stats['mean']  # Coefficient of variation
        if 0.1 < cv < 2.0:
            print("✅ Data distribution: Good variability")
        else:
            print("⚠️  Data distribution: May have limited variability")
        print(f"   Coefficient of variation: {cv:.3f}")
    
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Characterize raster files with basic statistics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a single raster
  python utils/characterize_rasters.py social.tif

  # Analyze multiple rasters
  python utils/characterize_rasters.py social.tif environmental.tif strategic.tif

  # Analyze with custom titles
  python utils/characterize_rasters.py --titles "Social Factor" "Environmental Factor" "Strategic Factor" social.tif environmental.tif strategic.tif
        """
    )
    
    parser.add_argument('raster_files', nargs='+', help='Raster files to analyze')
    parser.add_argument('--titles', nargs='+', help='Custom titles for each raster file')
    parser.add_argument('--summary', action='store_true', help='Show summary comparison')
    
    args = parser.parse_args()
    
    # Check if number of titles matches number of files
    if args.titles and len(args.titles) != len(args.raster_files):
        print("Error: Number of titles must match number of raster files")
        sys.exit(1)
    
    print("RASTER CHARACTERIZATION TOOL")
    print("=" * 60)
    print(f"Analyzing {len(args.raster_files)} raster file(s)...")
    
    all_stats = []
    
    # Analyze each raster
    for i, raster_file in enumerate(args.raster_files):
        title = args.titles[i] if args.titles else None
        stats = analyze_raster(raster_file, title)
        if stats:
            all_stats.append(stats)
            print_statistics(stats)
    
    # Show summary comparison if requested
    if args.summary and len(all_stats) > 1:
        print_summary_comparison(all_stats)


def print_summary_comparison(stats_list):
    """
    Print a summary comparison of multiple rasters.
    
    Args:
        stats_list: List of statistics dictionaries
    """
    print(f"\n{'='*80}")
    print("SUMMARY COMPARISON")
    print(f"{'='*80}")
    
    # Create comparison table
    print(f"{'File':<20} {'Min':<8} {'Max':<8} {'Mean':<8} {'Std':<8} {'Valid%':<8}")
    print("-" * 80)
    
    for stats in stats_list:
        valid_pct = (stats['valid_pixels'] / stats['total_pixels']) * 100
        print(f"{stats['title'][:19]:<20} {stats['min']:<8.3f} {stats['max']:<8.3f} "
              f"{stats['mean']:<8.3f} {stats['std']:<8.3f} {valid_pct:<8.1f}")
    
    print(f"\n{'='*80}")
    print("RECOMMENDATIONS")
    print(f"{'='*80}")
    
    # Check if all rasters have same dimensions
    dimensions = [stats['dimensions'] for stats in stats_list]
    if len(set(dimensions)) == 1:
        print("✅ All rasters have same dimensions")
    else:
        print("⚠️  Rasters have different dimensions:")
        for stats in stats_list:
            print(f"   {stats['title']}: {stats['dimensions']}")
    
    # Check data ranges
    ranges = [(stats['min'], stats['max']) for stats in stats_list]
    min_range = min([r[0] for r in ranges])
    max_range = max([r[1] for r in ranges])
    
    print(f"\nData range across all rasters: {min_range:.3f} to {max_range:.3f}")
    
    if min_range >= 0 and max_range <= 4:
        print("✅ All rasters compatible with FIS models (0-4 range)")
    else:
        print("⚠️  Some rasters may need scaling for FIS models")
        print("   Consider normalizing data to 0-4 range")
    
    # Check for outliers
    for stats in stats_list:
        if stats['std'] > 0:
            cv = stats['std'] / stats['mean']
            if cv > 1.5:
                print(f"⚠️  {stats['title']}: High variability (CV={cv:.3f})")
            elif cv < 0.1:
                print(f"⚠️  {stats['title']}: Low variability (CV={cv:.3f})")


if __name__ == "__main__":
    main() 