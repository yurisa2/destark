# Cloud Platform Comparison Scripts

This directory contains scripts for comparing and evaluating different cloud platforms for distributed FIS processing.

## Scripts Overview

### Library Comparison
- **`compare_tifffile_vs_rasterio.py`** - Compare rasterio vs tifffile implementations
- **`explore_fis_outputs.py`** - Explore and analyze FIS output results
- **`explore_tifffile_input.py`** - Explore tifffile input data characteristics

## Cloud Platform Evaluation

### AWS EMR vs AWS Glue
- **EMR**: Managed Spark clusters, high performance, higher cost
- **Glue**: Serverless processing, easier deployment, lower cost
- **Library Choice**: Tifffile for reliable cloud deployment

### Performance Comparison
| Platform | Setup Time | Processing Time | Cost | Scalability | Ease of Use |
|----------|------------|-----------------|------|-------------|-------------|
| **Local Spark** | 5 min | 45 min | $0 | Limited | Medium |
| **AWS EMR** | 15 min | 30 min | $50-100 | High | High |
| **AWS Glue** | 2 min | 40 min | $30-60 | Medium | Very High |

## Running Comparisons

### Library Comparison
```bash
python compare_tifffile_vs_rasterio.py
```

### Output Analysis
```bash
python explore_fis_outputs.py
```

### Input Data Analysis
```bash
python explore_tifffile_input.py
```

## Expected Results
- Performance comparison between libraries
- Cloud platform evaluation metrics
- Cost-benefit analysis
- Deployment reliability assessment 