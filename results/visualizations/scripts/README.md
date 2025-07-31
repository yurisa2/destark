# Visualization Scripts

This directory contains scripts for creating visualizations and plots of FIS processing results.

## Scripts Overview

### Core Visualization Scripts
- **`plot_individual.py`** - Plot individual raster files
- **`better_plot.py`** - Enhanced plotting with better styling
- **`simple_plot.py`** - Simple plotting script
- **`plot_results.py`** - Plot FIS processing results

## Visualization Types

### Individual Raster Plots
- Single raster visualization
- Custom color schemes
- Statistical information display
- Export to various formats

### Comparison Plots
- Side-by-side comparisons
- Difference plots
- Statistical analysis visualization
- Multi-resolution comparisons

### Result Analysis Plots
- FIS output visualization
- Performance metrics plots
- Error analysis visualization
- Quality assessment plots

## Usage Examples

### Plot Individual Raster
```bash
python plot_individual.py input_raster.tif -o output_plot.png
```

### Enhanced Plotting
```bash
python better_plot.py input_raster.tif --title "FIS Output" --colormap viridis
```

### Simple Plot
```bash
python simple_plot.py input_raster.tif
```

### Plot Results
```bash
python plot_results.py results_directory/ -o results_summary.png
```

## Output Formats
- PNG (high resolution for publications)
- PDF (vector graphics)
- JPG (web-friendly)
- Interactive HTML (for web display)

## Configuration
- Customizable color schemes
- Configurable plot sizes
- Statistical information display
- Export quality settings 