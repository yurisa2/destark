#!/bin/bash

# Bootstrap script to install Python packages on EMR nodes
# This script runs on all nodes (master and workers) during cluster startup

echo "=== Installing Python packages on EMR nodes ==="

# Update package lists
yum update -y

# Install system dependencies
yum install -y python3-pip python3-devel gcc gcc-c++ make

# Upgrade pip
python3 -m pip install --upgrade pip

# Install Python packages
echo "Installing Python packages..."

# Core scientific packages
python3 -m pip install numpy scipy pandas

# Date utilities (often required by other packages)
python3 -m pip install python-dateutil

# Geospatial packages
python3 -m pip install rasterio fiona shapely pyproj

# Fuzzy logic package
python3 -m pip install scikit-fuzzy

# AWS packages
python3 -m pip install boto3 s3fs

# Additional useful packages
python3 -m pip install matplotlib seaborn

echo "=== Python packages installation complete ==="

# Verify installations
echo "=== Verifying installations ==="
python3 -c "
import numpy, rasterio, skfuzzy, boto3, s3fs, dateutil
print('✓ All required packages installed successfully')
"

echo "=== Bootstrap script completed ===" 