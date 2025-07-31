# Experimental Workflows

This directory contains all experimental workflows and studies conducted throughout our three-phase research journey. Each experiment is designed to validate our approaches and provide quantitative evidence for our research contributions.

## 🔬 Experimental Design

### Research Questions

1. **Phase 1**: How do different FIS aggregation methods affect environmental assessment results?
2. **Phase 2**: How does multiprocessing scale with increasing data resolution?
3. **Phase 3**: How do different cloud platforms perform for distributed geospatial processing?

## 📁 Directory Structure

```
experiments/
├── performance_benchmarks/     # Performance evaluation studies
│   ├── phase1_baseline/       # Single-threaded baseline
│   ├── phase2_scaling/        # Multiprocessing scaling
│   ├── phase3_distributed/    # Distributed performance
│   └── cross_platform/        # Platform comparisons
├── resolution_scaling/         # Resolution scaling studies
│   ├── 1000m_analysis/        # Coarse resolution analysis
│   ├── 300m_analysis/         # Medium resolution analysis
│   ├── 30m_analysis/          # Fine resolution analysis
│   └── scaling_laws/          # Scaling law derivation
└── cloud_platforms/           # Cloud platform evaluations
    ├── aws_emr/               # AWS EMR studies
    ├── aws_glue/              # AWS Glue studies
    └── cost_analysis/         # Cost-benefit analysis
```

## 🎯 Experimental Categories

### 1. Performance Benchmarks

#### Phase 1: Baseline Performance
- **Objective**: Establish baseline performance metrics
- **Metrics**: Processing time, memory usage, CPU utilization
- **Results**: 9.7 minutes for 20.3M pixels (1000m resolution)

#### Phase 2: Scaling Performance
- **Objective**: Evaluate multiprocessing scaling
- **Metrics**: Speedup, efficiency, memory scaling
- **Results**: 6-10x performance improvement with multiprocessing

#### Phase 3: Distributed Performance
- **Objective**: Compare distributed computing platforms
- **Metrics**: Processing time, cost, scalability
- **Results**: AWS EMR vs Glue performance comparison

### 2. Resolution Scaling Studies

#### Scaling Analysis
- **1000m → 300m**: 10.82x pixel increase
- **300m → 30m**: 100x pixel increase
- **Total scaling**: 1000x from 1000m to 30m

#### Performance Scaling Laws
- **Linear scaling**: Processing time vs pixel count
- **Memory scaling**: Memory usage vs resolution
- **Cost scaling**: Cloud costs vs data size

### 3. Cloud Platform Evaluations

#### AWS EMR Studies
- **Setup time**: 15 minutes
- **Processing time**: 30 minutes for 30m data
- **Cost**: $50-100 per job
- **Scalability**: High

#### AWS Glue Studies
- **Setup time**: 2 minutes
- **Processing time**: 40 minutes for 30m data
- **Cost**: $30-60 per job
- **Scalability**: Medium

## 📊 Key Experiments

### Experiment 1: FIS Model Comparison
```bash
# Run all FIS models on 1000m data
python experiments/performance_benchmarks/phase1_baseline/run_fis_comparison.py \
  data/1000m/ \
  results/experiments/fis_comparison/
```

### Experiment 2: Multiprocessing Scaling
```bash
# Test different core configurations
python experiments/performance_benchmarks/phase2_scaling/run_scaling_test.py \
  data/300m/ \
  --cores 2,4,6,8,10 \
  --chunk-sizes 100,200,300,500
```

### Experiment 3: Cloud Platform Comparison
```bash
# Compare EMR vs Glue
python experiments/cloud_platforms/run_platform_comparison.py \
  s3://bucket/30m/ \
  --platforms emr,glue \
  --iterations 3
```

## 📈 Experimental Results

### Performance Evolution

| Phase | Resolution | Processing | Time | Improvement |
|-------|------------|------------|------|-------------|
| **1** | 1000m | Single-threaded | 9.7 min | Baseline |
| **2** | 300m | Multiprocessing | 17.5 min | 6-10x speedup |
| **3** | 30m | Distributed | 45 min | Cloud scaling |

### Scaling Laws Derived

1. **Processing Time**: T ∝ N^1.2 (where N = pixel count)
2. **Memory Usage**: M ∝ N^1.0 (linear scaling)
3. **Cost**: C ∝ N^1.1 (slight super-linear)

## 🔍 Validation Experiments

### Statistical Validation
- **Cross-validation**: 5-fold cross-validation of FIS models
- **Sensitivity analysis**: Parameter sensitivity studies
- **Robustness tests**: Noise and error tolerance

### Quality Assurance
- **Result consistency**: Cross-platform result validation
- **Error analysis**: Systematic error quantification
- **Performance regression**: Continuous performance monitoring

## 📚 Research Contributions

### Novel Experimental Designs
1. **Multi-resolution scaling study**: First systematic study of FIS scaling
2. **Cloud platform comparison**: Comprehensive cloud evaluation
3. **Performance regression testing**: Automated performance validation

### Quantitative Results
1. **Scaling laws**: Derived mathematical scaling relationships
2. **Cost models**: Economic analysis of cloud computing
3. **Performance benchmarks**: Comprehensive performance database

## 🚀 Running Experiments

### Prerequisites
```bash
# Install experiment dependencies
pip install -r requirements-experiments.txt

# Setup experiment environment
python experiments/setup_experiments.py
```

### Example Experiment
```bash
# Run complete experiment suite
python experiments/run_all_experiments.py \
  --data-dir data/ \
  --results-dir results/experiments/ \
  --config experiments/config/experiment_config.json
```

## 📖 Documentation

### Experiment Reports
- [Phase 1 Results](performance_benchmarks/phase1_baseline/results.md)
- [Phase 2 Results](performance_benchmarks/phase2_scaling/results.md)
- [Phase 3 Results](performance_benchmarks/phase3_distributed/results.md)

### Analysis Reports
- [Scaling Analysis](resolution_scaling/scaling_analysis.md)
- [Platform Comparison](cloud_platforms/platform_comparison.md)
- [Cost Analysis](cloud_platforms/cost_analysis.md)

## 🔄 Reproducibility

### Reproducible Workflows
- All experiments are fully automated
- Configuration files capture all parameters
- Results are version-controlled
- Environment specifications included

### Data Management
- Raw data preserved
- Intermediate results stored
- Final results documented
- Statistical analysis scripts included 