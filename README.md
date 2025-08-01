# Destark: Fuzzy Inference System for Environmental Assessment

## Research Evolution: From Single-Threaded Models to Distributed Processing

This repository documents the complete research journey of developing and scaling Fuzzy Inference Systems (FIS) for environmental assessment, from initial single-threaded implementations to distributed cloud processing, now enhanced with a comprehensive web-based frontend system.

## 🆕 NEW: Complete Web Application

The Destark FIS system now includes a **comprehensive, didactic web frontend** that provides:

- **🏗️ Visual FIS Builder**: Drag-and-drop interface for building fuzzy logic systems
- **📋 Rule Base Management**: Auto-generate and edit all 27 fuzzy rules
- **📥 Configuration Export**: Download JSON configurations for the FIS engine
- **🚀 System Execution**: Upload GeoTIFF files and execute FIS processing
- **📚 Educational Content**: Extensive help system and tutorials

### Quick Start

```bash
# Clone the repository
git clone <repository-url>
cd destark

# Start the complete system (Docker required)
./quick_start.sh

# Or start manually
docker-compose up -d
```

**Access the application:**
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Spark Web UI**: http://localhost:8080

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

### Phase 4: Web Application - User-Friendly Interface
**Goal**: Create accessible, educational interface for FIS systems
- **Challenge**: Make complex fuzzy logic accessible to non-experts
- **Solution**: Comprehensive React-based web application
- **Features**: Visual builders, educational content, real-time validation
- **Innovation**: Patent-worthy didactic interface for environmental assessment

## 🏗️ Repository Structure

```
destark/
├── frontend/                     # NEW: React web application
│   ├── src/components/          # FIS Builder components
│   ├── src/utils/               # Utility functions
│   └── package.json            # Frontend dependencies
├── backend/                      # NEW: FastAPI backend
│   ├── main.py                 # API endpoints
│   └── requirements.txt        # Backend dependencies
├── research/                     # Core research components
│   ├── phase1_foundation/      # Initial FIS development
│   ├── phase2_scaling/         # Multiprocessing implementation
│   ├── phase3_distributed/     # Spark and cloud processing
│   └── methodology/            # Research methodology
├── app/                         # Core FIS engine
│   ├── raster_fuzzy_lib.py     # Main FIS implementation
│   ├── raster_fuzzy_spark.py   # Spark-based processing
│   └── config/                 # FIS configurations
├── data/                        # Research datasets
├── experiments/                 # Experimental workflows
├── results/                     # Research results
├── infrastructure/              # Computing infrastructure
├── documentation/               # Comprehensive documentation
├── docker-compose.yml          # NEW: Complete system orchestration
├── quick_start.sh              # NEW: Easy deployment script
└── README_FRONTEND.md          # NEW: Frontend documentation
```

## 🚀 Quick Start

### Option 1: Complete System (Recommended)
```bash
# Start everything with one command
./quick_start.sh
```

### Option 2: Manual Setup
```bash
# Start the complete system
docker-compose up -d

# Access the web application
open http://localhost:3000
```

### Option 3: Development Mode
```bash
# Frontend development
cd frontend
npm install
npm start

# Backend development
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

## 📊 Performance Evolution

| Phase | Resolution | Pixels | Processing | Time | Memory | Interface |
|-------|------------|--------|------------|------|--------|-----------|
| 1 | 1000m | 20M | Single-threaded | 9.7 min | 4GB | CLI |
| 2 | 300m | 220M | Multiprocessing | 17.5 min | 12GB | CLI |
| 3 | 30m | 2.2B | Distributed Spark | 45 min | 40GB | CLI |
| 4 | All | All | Web Application | Variable | Variable | **Web UI** |

## 🔬 Research Contributions

1. **FIS Algorithm Development**: Novel fuzzy inference system for environmental assessment
2. **Scaling Methodology**: Systematic approach to scaling geospatial processing
3. **Performance Optimization**: Multiprocessing and distributed computing strategies
4. **Cloud Platform Evaluation**: Comparative analysis of AWS EMR vs Glue
5. **Library Innovation**: Tifffile-based solution for reliable cloud deployment
6. **Web Interface Innovation**: Patent-worthy didactic interface for FIS systems
7. **Reproducible Research**: Complete workflow documentation and automation

## 🌐 Web Application Features

### FIS Builder Interface
- **Visual Configuration**: Drag-and-drop interface for building fuzzy logic systems
- **Input Variables**: Configure social, environmental, and strategic factors
- **Output Variables**: Define priority assessment with 5 membership levels
- **Membership Functions**: Visual editor for trapezoidal and triangular functions
- **Real-time Validation**: Instant feedback on configuration errors

### Rule Base Management
- **Auto-Generation**: Generate all 27 possible fuzzy rules automatically
- **Editable Rules**: Click-to-edit interface for customizing rules
- **Rule Validation**: Real-time syntax and logic checking
- **Visual Rule Builder**: Drag-and-drop antecedent/consequent selection
- **Rule Statistics**: Live count of valid/invalid rules

### System Execution
- **File Upload**: Drag-and-drop GeoTIFF upload interface
- **Processing Status**: Real-time execution progress
- **Results Display**: Map visualization and statistics
- **Download Results**: Export processed GeoTIFF files

### Educational Content
- **Interactive Tutorials**: Step-by-step guidance
- **Fuzzy Logic Education**: Clear explanations of concepts
- **Context-Sensitive Help**: Help that appears based on user actions
- **Tooltips**: Hover explanations for all interface elements

## 📚 Documentation

- **[Frontend Guide](README_FRONTEND.md)**: Comprehensive frontend documentation
- **[Deployment Guide](DEPLOYMENT_GUIDE.md)**: Complete deployment instructions
- **[API Documentation](http://localhost:8000/docs)**: Interactive API docs (when running)

## 🚀 Deployment Options

### Local Development
```bash
./quick_start.sh
```

### Production Deployment
- **AWS**: ECS Fargate + EMR for Spark processing
- **Google Cloud**: GKE + Cloud Run
- **Azure**: Container Instances + AKS
- **Docker**: Any platform with Docker support

See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for detailed instructions.

## 🔧 Development

### Frontend Development
```bash
cd frontend
npm install
npm start
npm test
```

### Backend Development
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
pytest test_main.py
```

### Testing
```bash
# Frontend tests
cd frontend && npm test

# Backend tests
cd backend && pytest test_main.py

# Integration tests
docker-compose up -d
# Run tests against running system
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

This is a research repository with an active web application component. Please see `CONTRIBUTING.md` for guidelines on contributing to the research and development.

## 📞 Contact

For research collaboration, questions, or technical support:
- Open an issue in the repository
- Contact the research team
- Check the documentation and help guides

---

**Research Team**: Environmental Assessment Research Group  
**Institution**: [Your Institution]  
**Funding**: [Funding Information] 