# TIFF Raster Plotting Script

This script provides functionality to plot TIFF raster files with various visualization options, including statistics, custom colormaps, and comparison plots.

## Features

- **Single raster plotting** with statistics and custom colormaps
- **Comparison plots** showing input files and output side-by-side
- **High-resolution output** with customizable DPI and figure size
- **Statistics display** including min, max, mean, std, and valid pixel count
- **NoData handling** - automatically excludes NoData values from statistics
- **Multiple output formats** - PNG, PDF, etc.

## Usage

### Basic Plotting

```bash
# Plot a single TIFF file
python plot_output.py output.tif

# Plot with custom title and save to file
python plot_output.py output.tif -t "FIS Output" -o output_plot.png

# Plot with custom colormap
python plot_output.py output.tif -c "plasma" -t "FIS Result" -o output_plasma.png
```

### Comparison Plotting

```bash
# Plot comparison of input files and output
python plot_output.py dummy.tif --comparison \
  --input-files environmental.tif social.tif strategic.tif \
  --output-file output.tif \
  --output-plot comparison.png
```

### Advanced Options

```bash
# High resolution plot
python plot_output.py output.tif --dpi 600 --figsize 12 10 -o output_hires.png

# Plot without statistics
python plot_output.py output.tif --no-stats -o output_clean.png

# Custom titles for comparison
python plot_output.py dummy.tif --comparison \
  --input-files env.tif soc.tif strat.tif \
  --output-file output.tif \
  --titles "Environmental" "Social" "Strategic" "FIS Output" \
  --output-plot comparison.png
```

## Command Line Options

### Basic Options
- `input_file`: Path to the TIFF file to plot
- `-o, --output`: Output file for the plot (PNG/PDF)
- `-t, --title`: Title for the plot
- `-c, --colormap`: Matplotlib colormap (default: viridis)
- `--no-stats`: Hide statistics on plot
- `--dpi`: DPI for saved image (default: 300)
- `--figsize`: Figure size width height (default: 10 8)

### Comparison Mode Options
- `--comparison`: Enable comparison mode
- `--input-files`: Input files for comparison (environmental social strategic)
- `--output-file`: Output file for comparison
- `--titles`: Titles for comparison plots
- `--output-plot`: Output file for comparison plot

## Available Colormaps

### Sequential (good for continuous data)
- `viridis`, `plasma`, `inferno`, `magma`, `cividis`
- `Blues`, `Greens`, `Reds`, `Oranges`, `Purples`

### Diverging (good for data with center point)
- `RdYlBu`, `RdYlGn`, `PuOr`, `BrBG`, `RdBu`

### Qualitative (good for categorical data)
- `Set1`, `Set2`, `Set3`, `tab10`, `tab20`

## Examples

### Example 1: Plot FIS Output
```bash
python plot_output.py files/output/base/output.tif \
  -t "FIS Output - Maximum Configuration" \
  -c "plasma" \
  -o fis_output_plot.png
```

### Example 2: Comparison Plot
```bash
python plot_output.py dummy.tif --comparison \
  --input-files files/input/base/environmental.tif \
                files/input/base/social.tif \
                files/input/base/strategic.tif \
  --output-file files/output/base/output.tif \
  --titles "Environmental" "Social" "Strategic" "FIS Output" \
  --output-plot comparison_plot.png
```

### Example 3: High-Resolution Publication Plot
```bash
python plot_output.py files/output/base/output.tif \
  -t "Fuzzy Inference System Output" \
  -c "RdYlBu" \
  --dpi 600 \
  --figsize 12 10 \
  -o publication_plot.png
```

## Statistics Displayed

When statistics are enabled (default), the plot shows:
- **Min**: Minimum value (excluding NoData)
- **Max**: Maximum value (excluding NoData)
- **Mean**: Average value (excluding NoData)
- **Std**: Standard deviation (excluding NoData)
- **Valid pixels**: Count of non-NoData pixels

## Notes

- The script automatically handles NoData values and excludes them from statistics
- For comparison plots, all input files must exist
- The script uses rasterio for reading TIFF files and matplotlib for plotting
- Output files are saved with high quality settings suitable for publications
- Interactive plots are shown in addition to saved files (if specified)

## Dependencies

- `rasterio`: For reading TIFF files
- `matplotlib`: For plotting
- `numpy`: For data manipulation

These are already included in the project's requirements.txt. 