# Resolution Scaling Studies

This directory contains experiments and analysis for scaling FIS processing across different resolutions.

## Contents

### Analysis Files
- **`300m_PROCESSING_SUMMARY.md`** - Summary of 300m resolution processing
- **`300m_RUNTIME_PREDICTIONS.md`** - Runtime predictions for 300m data
- **`runtime_prediction_analysis.py`** - Analysis script for runtime predictions

## Resolution Progression

### Scaling Analysis
- **1000m → 300m**: 10.82x pixel increase
- **300m → 30m**: 100x pixel increase
- **Total scaling**: 1000x from 1000m to 30m

### Performance Scaling Laws
- **Processing Time**: T ∝ N^1.2 (where N = pixel count)
- **Memory Usage**: M ∝ N^1.0 (linear scaling)
- **Cost**: C ∝ N^1.1 (slight super-linear)

## Key Findings

### 300m Resolution Results
- **Dimensions**: 14479 × 15187 pixels
- **Total Pixels**: 219.8 million
- **Processing Time**: 17.5-52 minutes (depending on configuration)
- **Memory Usage**: 8-40GB (configurable)

### Optimal Configurations
| Configuration | Runtime | Memory | Cores | Chunk Size | Use Case |
|---------------|---------|--------|-------|------------|----------|
| **Conservative** | 52 min | ~8GB | 2 | 100 | Testing, limited RAM |
| **Memory-Optimized** | 17.5 min | ~12GB | 6 | 100 | Limited RAM systems ⭐ |
| **Balanced** | 26 min | ~16GB | 4 | 200 | Most systems ⭐ |
| **Performance** | 13 min | ~32GB | 8 | 300 | High-end systems |
| **Maximum** | 10.5 min | ~40GB | 10 | 500 | Maximum speed |

## Running Analysis

### Runtime Prediction Analysis
```bash
python runtime_prediction_analysis.py
```

## Expected Outputs
- Scaling law derivations
- Performance predictions for different resolutions
- Memory usage analysis
- Configuration recommendations 