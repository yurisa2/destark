#!/bin/bash

# Install Python packages system-wide on EMR
# This script installs packages in the system Python environment

set -e

echo "=== Installing Python packages system-wide ==="

# Update package lists
echo "Updating package lists..."
sudo yum update -y

# Install system dependencies
echo "Installing system dependencies..."
sudo yum install -y python3-pip python3-devel gcc gcc-c++ make

# Upgrade pip
echo "Upgrading pip..."
sudo python3 -m pip install --upgrade pip

# Install Python packages system-wide
echo "Installing Python packages system-wide..."
sudo python3 -m pip install python-dateutil numpy scipy pandas rasterio fiona shapely pyproj scikit-fuzzy boto3 s3fs matplotlib seaborn

# Verify installations
echo "Verifying installations..."
python3 -c "
try:
    import dateutil, numpy, rasterio, skfuzzy, boto3, s3fs
    print('✓ All required packages installed system-wide successfully')
    print('✓ Packages are available to all users')
except ImportError as e:
    print(f'✗ Package import failed: {e}')
    exit(1)
"

# Test with different users
echo "Testing package availability for different users..."
sudo -u hadoop python3 -c "import dateutil, numpy, rasterio, skfuzzy, boto3, s3fs; print('✓ Packages available to hadoop user')" || echo "✗ Packages not available to hadoop user"

echo ""
echo "=== System-wide installation completed ==="
echo "✓ Packages installed in system Python"
echo "✓ Available to all users on this node"
echo "✓ Ready for Spark applications" 