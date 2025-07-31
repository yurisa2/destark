#!/bin/bash

echo "=== INSTALLING DEVELOPMENT TOOLS AND LIBRARIES ==="

# Update system
echo "Updating system packages..."
sudo yum update -y

# Install basic development tools
echo "Installing basic development tools..."
sudo yum install -y git python3-pip python3-devel gcc gcc-c++ make java-17-amazon-corretto-devel unzip wget curl

# Try to install geospatial libraries (with fallbacks)
echo "Installing geospatial libraries..."
sudo yum install -y gdal-devel proj-devel geos-devel hdf5-devel netcdf-devel libspatialindex-devel 2>/dev/null || \
sudo yum install -y gdal-devel proj-devel geos-devel hdf5-devel netcdf-devel 2>/dev/null || \
sudo yum install -y gdal-devel proj-devel geos-devel 2>/dev/null || \
echo "Some geospatial packages not available via yum, will install via pip"

# Set up Java environment
echo "Setting up Java environment..."
echo 'export JAVA_HOME=/usr/lib/jvm/jre-17' >> /etc/profile
echo 'export PATH=$JAVA_HOME/bin:$PATH' >> /etc/profile
source /etc/profile

# Upgrade pip
echo "Upgrading pip..."
python3 -m pip install --upgrade pip

# Install Python packages
echo "Installing Python packages for adveng processing..."
python3 -m pip install numpy>=1.21.0 scipy>=1.7.0 pandas>=1.3.0 python-dateutil>=2.8.0 \
    rasterio>=1.3.0 fiona>=1.8.0 shapely>=1.8.0 pyproj>=3.2.0 scikit-fuzzy>=0.4.2 \
    boto3>=1.26.0 s3fs>=2022.11.0 matplotlib>=3.5.0 seaborn>=0.11.0 tqdm>=4.62.0 \
    pyspark>=3.4.0 networkx>=2.8.0

# Verify installations
echo "Verifying installations..."
git --version
java -version
python3 -c "import numpy, rasterio, skfuzzy, boto3, s3fs, dateutil, pyspark; print('✓ All development packages installed successfully')"

echo "=== DEVELOPMENT ENVIRONMENT READY ===" 