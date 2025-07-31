# Destark: Fuzzy Inference System for Environmental Assessment

## Research Evolution: From Single-Threaded Models to Distributed Processing

This repository documents the complete research journey of developing and scaling Fuzzy Inference Systems (FIS) for environmental assessment, from initial single-threaded implementations to distributed cloud processing.

## 📖 Research Story

### Phase 1: Foundation - Fuzzy Inference System Development
**Goal**: Develop core FIS models for environmental assessment
- **Resolution**: 1000m (coarse resolution for initial development)
- **Processing**: Single-threaded Python implementation
- **Focus**: Algorithm development and validation

### Phase 2: Scaling Challenge - Higher Resolution Processing
**Goal**: Process 300m resolution data (10x more pixels)
- **Challenge**: Python's native single-threaded nature
- **Solution**: Multiprocessing implementation
- **Result**: Successful scaling with 6-10x performance improvement

### Phase 3: Distributed Computing - 30m Resolution Preparation
**Goal**: Prepare for 30m resolution processing (100x more pixels than 1000m)
- **Challenge**: Even multiprocessing insufficient for 30m data
- **Solution**: Apache Spark distributed processing
- **Platforms**: Local Spark, AWS EMR, AWS Glue
- **Technical Innovation**: Tifffile library for cloud deployment reliability

## 🏗️ Repository Structure

```
destark/
├── research/                          # Core research components
│   ├── phase1_foundation/            # Initial FIS development
│   ├── phase2_scaling/               # Multiprocessing implementation
│   ├── phase3_distributed/           # Spark and cloud processing
│   └── methodology/                  # Research methodology
├── data/                             # Research datasets
│   ├── 1000m/                       # Coarse resolution data
│   ├── 300m/                        # Medium resolution data
│   └── metadata/                    # Data documentation
├── experiments/                      # Experimental workflows
│   ├── performance_benchmarks/      # Performance comparisons
│   ├── resolution_scaling/          # Resolution scaling studies
│   └── cloud_platforms/             # Cloud platform comparisons
├── results/                         # Research results
│   ├── publications/                # Manuscripts and papers
│   ├── visualizations/              # Publication-ready figures
│   └── analysis/                    # Statistical analysis
├── infrastructure/                  # Computing infrastructure
│   ├── local/                       # Local processing setup
│   ├── aws_emr/                     # AWS EMR configurations
│   └── aws_glue/                    # AWS Glue implementations
└── documentation/                   # Comprehensive documentation
```

## 🚀 Quick Start

### Phase 1: Basic FIS Processing (1000m)
```bash
# Run single FIS model
python research/phase1_foundation/run_fis_1000m.py \
  data/1000m/social.tif \
  data/1000m/environmental.tif \
  data/1000m/strategic.tif \
  results/output_1000m.tif
```

### Phase 2: Multiprocessing (300m)
```bash
# Run with multiprocessing
python research/phase2_scaling/run_fis_300m_parallel.py \
  data/300m/social.tif \
  data/300m/environmental.tif \
  data/300m/strategic.tif \
  results/output_300m.tif \
  --cores 8 --chunk-size 200
```

### Phase 3: Distributed Processing (30m)
```bash
# Run on AWS EMR
./infrastructure/aws_emr/run_30m_emr.sh \
  s3://bucket/30m/social.tif \
  s3://bucket/30m/environmental.tif \
  s3://bucket/30m/strategic.tif \
  s3://bucket/results/output_30m.tif
```

## 📊 Performance Evolution

| Phase | Resolution | Pixels | Processing | Time | Memory |
|-------|------------|--------|------------|------|--------|
| 1 | 1000m | 20M | Single-threaded | 9.7 min | 4GB |
| 2 | 300m | 220M | Multiprocessing | 17.5 min | 12GB |
| 3 | 30m | 2.2B | Distributed Spark | 45 min | 40GB |

## 🔬 Research Contributions

1. **FIS Algorithm Development**: Novel fuzzy inference system for environmental assessment
2. **Scaling Methodology**: Systematic approach to scaling geospatial processing
3. **Performance Optimization**: Multiprocessing and distributed computing strategies
4. **Cloud Platform Evaluation**: Comparative analysis of AWS EMR vs Glue
5. **Library Innovation**: Tifffile-based solution for reliable cloud deployment
6. **Reproducible Research**: Complete workflow documentation and automation

## 📚 Publications

- [Phase 1] FIS Model Development and Validation
- [Phase 2] Scaling Geospatial Processing with Multiprocessing
- [Phase 3] Distributed Computing for High-Resolution Environmental Assessment
- [Technical] Library Comparison: Rasterio vs Tifffile for Cloud Geospatial Processing

## 🤝 Contributing

This is a research repository. Please see `CONTRIBUTING.md` for guidelines on contributing to the research.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

For research collaboration or questions, please open an issue or contact the research team.

---

**Research Team**: Environmental Assessment Research Group  
**Institution**: [Your Institution]  
**Funding**: [Funding Information] 