# Phase 1: Foundation Scripts

This directory contains execution scripts for Phase 1 of our research - the foundational FIS algorithm development.

## Scripts Overview

### Core Execution Scripts
- **`run_1000m_config_max.py`** - Execute FIS processing with maximum aggregation on 1000m data
- **`test_working_system.py`** - Test the basic FIS system functionality
- **`simple_fuzzy_test.py`** - Simple fuzzy logic testing script
- **`debug_fuzzy_issue.py`** - Debugging script for FIS issues

## Usage Examples

### Run 1000m Processing
```bash
python run_1000m_config_max.py
```

### Test System
```bash
python test_working_system.py
```

### Debug Issues
```bash
python debug_fuzzy_issue.py
```

## Configuration
- Uses `raster_fis_config.json` from parent directory
- Processes 1000m resolution data
- Single-threaded processing
- Baseline performance metrics

## Expected Outputs
- Processed raster files in `data/` directory
- Performance metrics and timing information
- Validation results and error reports 