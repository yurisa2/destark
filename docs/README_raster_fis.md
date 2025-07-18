# Raster Fuzzy Inference System

A specialized fuzzy inference system designed to process raster TIFF files. This system takes 3 input TIFF files (social, environmental, strategic factors) and outputs a single TIFF file using configurable fuzzy logic rules.

## Features

- **Raster Processing**: Directly processes GeoTIFF files
- **External Configuration**: All fuzzy logic rules and membership functions defined in JSON files
- **Flexible Input**: Supports any 3-factor assessment system
- **Real-time Progress Tracking**: Shows live progress bars for both regular and parallel processing
- **Parallel Processing**: Multi-core processing for faster execution on large rasters
- **Error Handling**: Robust error handling for NoData pixels and processing errors
- **Command Line Interface**: Easy-to-use CLI for batch processing

## Requirements

```bash
pip install rasterio scikit-fuzzy numpy pandas
```

## Quick Start

### 1. Create Configuration Template

```bash
python run_raster_fis.py --create-config
```

This creates a `raster_fis_config.json` file with default fuzzy logic rules.

### 2. Process Your TIFF Files

```bash
python run_raster_fis.py social.tif environmental.tif strategic.tif output.tif
```

### 3. Custom Configuration

```bash
python run_raster_fis.py social.tif environmental.tif strategic.tif output.tif --config my_config.json
```

## Configuration File Structure

The configuration file defines the fuzzy logic system:

```json
{
  "description": "Raster fuzzy inference system for environmental assessment",
  "input_variables": {
    "social": {
      "min": 0,
      "max": 10,
      "step": 0.1,
      "membership_functions": {
        "low": {
          "type": "trapmf",
          "params": [0, 0, 2, 4]
        },
        "medium": {
          "type": "trapmf",
          "params": [2, 4, 6, 7]
        },
        "high": {
          "type": "trapmf",
          "params": [6, 7, 10, 10]
        }
      }
    },
    "environmental": {
      "min": 0,
      "max": 10,
      "step": 0.1,
      "membership_functions": {
        "low": {
          "type": "trapmf",
          "params": [0, 0, 2, 5]
        },
        "medium": {
          "type": "trapmf",
          "params": [2, 5, 6, 8]
        },
        "high": {
          "type": "trapmf",
          "params": [6, 8, 10, 10]
        }
      }
    },
    "strategic": {
      "min": 0,
      "max": 10,
      "step": 0.1,
      "membership_functions": {
        "low": {
          "type": "trapmf",
          "params": [0, 0, 3, 5]
        },
        "medium": {
          "type": "trapmf",
          "params": [3, 5, 7, 8]
        },
        "high": {
          "type": "trapmf",
          "params": [7, 8, 10, 10]
        }
      }
    }
  },
  "output_variable": {
    "name": "priority",
    "min": 0,
    "max": 10,
    "step": 0.1,
    "membership_functions": {
      "very_low": {
        "type": "trimf",
        "params": [0, 0, 2.5]
      },
      "low": {
        "type": "trimf",
        "params": [0, 2.5, 5]
      },
      "medium": {
        "type": "trimf",
        "params": [2.5, 5, 7.5]
      },
      "high": {
        "type": "trimf",
        "params": [5, 5.5, 10]
      },
      "very_high": {
        "type": "trimf",
        "params": [7.5, 10, 10]
      }
    }
  },
  "rules": [
    {
      "antecedent": [
        {"variable": "social", "membership": "low"},
        {"variable": "environmental", "membership": "low"},
        {"variable": "strategic", "membership": "high"}
      ],
      "consequent": "very_low"
    }
    // ... more rules
  ]
}
```

## Input Variables

Each input variable defines:
- **min/max**: Value range for the variable
- **step**: Resolution for fuzzy logic calculations
- **membership_functions**: Fuzzy sets (low, medium, high, etc.)

## Membership Functions

### Trapezoidal (`trapmf`)
- Parameters: `[a, b, c, d]` where a ≤ b ≤ c ≤ d
- Creates trapezoidal membership function

### Triangular (`trimf`)
- Parameters: `[a, b, c]` where a ≤ b ≤ c
- Creates triangular membership function

## Rules

Rules define the fuzzy logic relationships:
- **antecedent**: List of conditions (variable + membership)
- **consequent**: Output membership function

## Command Line Options

```bash
python run_raster_fis.py [OPTIONS] social.tif environmental.tif strategic.tif output.tif
```

### Options:
- `--config, -c`: Configuration file path (default: raster_fis_config.json)
- `--create-config`: Create a template configuration file
- `--nodata`: NoData value for output raster (default: 5.0)
- `--verbose, -v`: Enable verbose output

## Progress Tracking

The system provides real-time progress tracking for both regular and parallel processing:

### Regular Processing
- Shows pixel-by-pixel progress with estimated time remaining
- Displays processing rate (pixels per second)
- Updates in real-time during processing

### Parallel Processing
- Shows chunk-by-chunk progress across multiple CPU cores
- Displays overall processing rate and estimated completion time
- Shows progress for both processing and result combination phases

Example output:
```
Processing 1000000 pixels...
Processing pixels: 100%|██████████| 1000000/1000000 [02:30<00:00, 6666.67pixel/s]
Pixel processing completed in 150.23 seconds
```

## Examples

### Basic Usage
```bash
# Create configuration template
python run_raster_fis.py --create-config

# Process files with default config
python run_raster_fis.py social.tif environmental.tif strategic.tif output.tif
```

### Custom Configuration
```bash
# Use custom configuration
python run_raster_fis.py social.tif environmental.tif strategic.tif output.tif --config my_rules.json
```

### Parallel Processing
```bash
# Use parallel processing with 8 CPU cores
python run_raster_fis_parallel.py social.tif environmental.tif strategic.tif output.tif --cores 8

# Use parallel processing with custom chunk size
python run_raster_fis_parallel.py social.tif environmental.tif strategic.tif output.tif --cores 4 --chunk-size 200
```

### Custom NoData Value
```bash
# Use different NoData value
python run_raster_fis.py social.tif environmental.tif strategic.tif output.tif --nodata -32768
```

## Programmatic Usage

```python
from raster_fuzzy_system import RasterFuzzyInferenceSystem

# Initialize system
fis = RasterFuzzyInferenceSystem('config.json')

# Process rasters
fis.process_rasters(
    social_tiff='social.tif',
    environmental_tiff='environmental.tif',
    strategic_tiff='strategic.tif',
    output_tiff='output.tif'
)
```

## Input Requirements

### TIFF File Requirements:
- All input TIFFs must have the same dimensions
- Same coordinate reference system (CRS)
- Same pixel size and extent
- Values should be within the range defined in configuration

### Data Requirements:
- Values should match the min/max ranges in configuration
- NoData pixels are automatically handled
- NaN values are treated as NoData

## Output

The system produces:
- **Single TIFF file**: Combined fuzzy logic result
- **Same georeference**: Maintains spatial properties
- **Float32 data type**: High precision output
- **Configurable NoData**: Default 5.0

## Performance

- **Progress tracking**: Shows processing percentage
- **Memory efficient**: Processes pixel by pixel
- **Error handling**: Continues processing on individual pixel errors
- **Large file support**: Handles large raster files

## Customization

### Modifying Membership Functions
Edit the `membership_functions` section in your config:

```json
"membership_functions": {
  "low": {
    "type": "trapmf",
    "params": [0, 0, 2, 4]  // Adjust these values
  }
}
```

### Adding New Rules
Add rules to the `rules` array:

```json
{
  "antecedent": [
    {"variable": "social", "membership": "high"},
    {"variable": "environmental", "membership": "low"},
    {"variable": "strategic", "membership": "medium"}
  ],
  "consequent": "high"
}
```

### Changing Value Ranges
Modify `min`, `max`, and `step` values in input variables:

```json
"social": {
  "min": 0,
  "max": 100,  // Change from 10 to 100
  "step": 1.0  // Adjust resolution
}
```

## Troubleshooting

### Common Issues:

1. **"All input rasters must have the same dimensions"**
   - Ensure all TIFF files have the same width and height
   - Check that they cover the same geographic area

2. **"Configuration file not found"**
   - Run `python run_raster_fis.py --create-config` first
   - Check the file path in your command

3. **"Error processing pixel"**
   - Check that input values are within the configured ranges
   - Verify NoData values are consistent

4. **Memory issues with large files**
   - The system processes pixel by pixel, so memory usage is minimal
   - For very large files, processing may take time

### Debug Mode:
```bash
python run_raster_fis.py social.tif environmental.tif strategic.tif output.tif --verbose
```

## File Structure

```
app/
├── raster_fuzzy_system.py    # Main system class
├── run_raster_fis.py        # Command line interface
├── README_raster_fis.md     # This documentation
└── raster_fis_config.json   # Configuration template (created)
```

## Advanced Usage

### Custom Fuzzy Logic
You can implement any fuzzy logic system by modifying the configuration:

1. **Different membership functions**: Change `type` and `params`
2. **More input variables**: Add new variables to `input_variables`
3. **Complex rules**: Add more sophisticated rule combinations
4. **Different output scales**: Modify output variable ranges

### Integration with GIS
The output TIFF can be directly used in:
- QGIS
- ArcGIS
- GRASS GIS
- Any GIS software that supports GeoTIFF

### Batch Processing
For multiple datasets:

```bash
#!/bin/bash
for area in area1 area2 area3; do
    python run_raster_fis.py \
        "${area}_social.tif" \
        "${area}_environmental.tif" \
        "${area}_strategic.tif" \
        "${area}_output.tif"
done
``` 