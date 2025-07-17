# Project Structure

This document describes the organized structure of the destark project.

## Directory Structure

```
destark/
├── docs/                          # Documentation files
│   ├── README_FIS_Models.md      # FIS model configurations
│   ├── README_Plotting.md        # Plotting utilities guide
│   ├── README_fuzzy_system.md    # Fuzzy system documentation
│   ├── README_raster_fis.md      # Raster FIS documentation
│   └── README_Project_Structure.md # This file
├── utils/                         # Utility scripts
│   ├── characterize_rasters.py   # Raster characterization
│   ├── plot_all_models.py        # Plot all FIS models
│   └── plot_output.py            # Single raster plotting
├── scripts/                       # Bash scripts
│   └── run_all_models.sh         # Run all FIS models
├── app/                          # Main application
│   ├── config/                   # Configuration files
│   │   ├── config_max.json
│   │   ├── config_median.json
│   │   ├── config_minimum.json
│   │   ├── config_mode.json
│   │   ├── config_round_down.json
│   │   └── config_round_up.json
│   ├── files/                    # Input/output files
│   │   ├── input/
│   │   └── output/
│   ├── run_raster_fis.py         # Serial FIS processing
│   ├── run_raster_fis_parallel.py # Parallel FIS processing
│   └── raster_fuzzy_system.py    # Core FIS implementation
└── requirements.txt               # Python dependencies
```

## Updated Commands

### Running FIS Models

**Run all models:**
```bash
# From root directory
./scripts/run_all_models.sh
```

**Run single model:**
```bash
# From root directory
python app/run_raster_fis_parallel.py app/files/input/base/socioeconomico_1000m.tif app/files/input/base/ambiental_1000m.tif app/files/input/base/estrategico_1000m.tif output_max.tif --config app/config/config_max.json --nodata 5
```

### Plotting

**Plot all models comparison:**
```bash
# From root directory
python utils/plot_all_models.py
```

**Plot single raster:**
```bash
# From root directory
python utils/plot_output.py output_max.tif -t "FIS Output" -o output_plot.png
```

**Characterize rasters:**
```bash
# From root directory
python utils/characterize_rasters.py app/files/input/base/
```

### Documentation

All documentation is now in the `docs/` directory:
- `README_FIS_Models.md` - FIS model configurations and usage
- `README_Plotting.md` - Plotting utilities guide
- `README_fuzzy_system.md` - Fuzzy system documentation
- `README_raster_fis.md` - Raster FIS documentation

## Benefits of New Structure

1. **Better Organization**: Related files are grouped together
2. **Easier Navigation**: Clear separation of concerns
3. **Maintainability**: Configuration files are separate from code
4. **Reusability**: Utility scripts are easily accessible
5. **Documentation**: All docs are centralized

## File Types by Directory

- **docs/**: All `.md` files (documentation)
- **utils/**: All plotting and characterization scripts (`.py`)
- **scripts/**: All bash scripts (`.sh`)
- **app/config/**: All configuration files (`.json`)
- **app/**: Core application code (`.py`)

This structure makes the project more professional and easier to maintain. 