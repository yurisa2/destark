# Research Data Management

This directory contains all research datasets used throughout the three phases of our FIS development and scaling research.

## 📊 Dataset Overview

### Resolution Progression

| Phase | Resolution | Pixels | File Size | Use Case |
|-------|------------|--------|-----------|----------|
| **Phase 1** | 1000m | 20.3M | ~81MB | Algorithm development |
| **Phase 2** | 300m | 219.8M | ~877MB | Scaling studies |
| **Phase 3** | 30m | ~22B | ~87GB | Production processing |

## 📁 Directory Structure

```
data/
├── 1000m/                    # Phase 1: Foundation datasets
│   ├── socioeconomico_1000m.tif
│   ├── ambiental_1000m.tif
│   ├── estratégico_1000m.tif
│   └── README.md
├── 300m/                     # Phase 2: Scaling datasets
│   ├── socioeconomico_300m.tif
│   ├── ambiental_300m.tif
│   ├── estratégico_300m.tif
│   └── README.md
├── 30m/                      # Phase 3: Production datasets
│   ├── socioeconomico_30m.tif
│   ├── ambiental_30m.tif
│   ├── estratégico_30m.tif
│   └── README.md
├── metadata/                 # Data documentation
│   ├── data_provenance.md
│   ├── quality_assessment.md
│   └── validation_reports/
└── README.md                 # This file
```

## 🔬 Dataset Characteristics

### Input Variables

All datasets contain three input variables for environmental assessment:

1. **Social Factor** (`socioeconomico_*.tif`)
   - Represents socioeconomic considerations
   - Value range: 0-10
   - Higher values indicate higher social priority

2. **Environmental Factor** (`ambiental_*.tif`)
   - Represents environmental considerations
   - Value range: 0-10
   - Higher values indicate higher environmental priority

3. **Strategic Factor** (`estratégico_*.tif`)
   - Represents strategic considerations
   - Value range: 0-10
   - Higher values indicate higher strategic priority

### Data Quality Metrics

| Resolution | Data Density | NoData % | Min Value | Max Value | Mean | Std Dev |
|------------|--------------|----------|-----------|-----------|------|---------|
| **1000m** | 90.5% | 9.5% | 0.000 | 10.000 | 1.487 | 2.828 |
| **300m** | 87.8% | 12.2% | 0.000 | 10.000 | 1.421 | 2.580 |
| **30m** | TBD | TBD | TBD | TBD | TBD | TBD |

## 📋 Data Provenance

### Source Information
- **Original Data**: Environmental assessment datasets
- **Processing**: Resampled to different resolutions for research purposes
- **Coordinate System**: UTM projection
- **Format**: GeoTIFF with compression

### Processing History
1. **Phase 1**: Original 1000m resolution data
2. **Phase 2**: Resampled to 300m for scaling studies
3. **Phase 3**: Resampled to 30m for production testing

## 🔍 Data Validation

### Quality Checks Performed
- ✅ Spatial consistency across resolutions
- ✅ Value range validation (0-10)
- ✅ NoData handling
- ✅ Coordinate system verification
- ✅ File integrity checks

### Validation Reports
- [1000m Validation Report](metadata/validation_reports/1000m_validation.md)
- [300m Validation Report](metadata/validation_reports/300m_validation.md)
- [30m Validation Report](metadata/validation_reports/30m_validation.md)

## 🚀 Usage Examples

### Loading Data in Python
```python
import rasterio

# Load 1000m data
with rasterio.open('data/1000m/socioeconomico_1000m.tif') as src:
    social_1000m = src.read(1)

# Load 300m data
with rasterio.open('data/300m/socioeconomico_300m.tif') as src:
    social_300m = src.read(1)
```

### Data Comparison
```python
# Compare resolutions
print(f"1000m shape: {social_1000m.shape}")
print(f"300m shape: {social_300m.shape}")
print(f"Scaling factor: {social_300m.shape[0] / social_1000m.shape[0]:.1f}x")
```

## 📚 Research Applications

### Phase 1: Algorithm Development
- **Dataset**: 1000m resolution
- **Purpose**: FIS algorithm development and validation
- **Processing**: Single-threaded

### Phase 2: Scaling Studies
- **Dataset**: 300m resolution
- **Purpose**: Multiprocessing performance evaluation
- **Processing**: Parallel processing

### Phase 3: Production Testing
- **Dataset**: 30m resolution
- **Purpose**: Distributed computing validation
- **Processing**: Cloud-based distributed

## 🔒 Data Access

### Local Access
- All datasets are stored locally for research purposes
- Compressed GeoTIFF format for efficient storage
- Backup copies maintained in secure location

### Cloud Storage
- Datasets uploaded to S3 for cloud processing
- Organized by resolution in separate buckets
- Access controlled through AWS IAM

## 📖 Related Documentation

- [Data Processing Pipeline](../research/methodology/data_processing.md)
- [Quality Assessment](../data/metadata/quality_assessment.md)
- [Validation Procedures](../data/metadata/validation_procedures.md)
- [Storage Management](../infrastructure/storage_management.md) 