#!/bin/bash

echo "=== MANUALLY FIXING EMR CLUSTER ==="
echo "This script will install all missing packages and fix the environment"

# Update system
echo "Updating system packages..."
sudo yum update -y

# Install basic development tools
echo "Installing basic development tools..."
sudo yum install -y git python3-pip python3-devel gcc gcc-c++ make java-17-amazon-corretto-devel unzip wget curl

# Set up Java environment (force Java 17)
echo "Setting up Java 17 environment..."
sudo alternatives --install /usr/bin/java java /usr/lib/jvm/jre-17/bin/java 1
sudo alternatives --set java /usr/lib/jvm/jre-17/bin/java
export JAVA_HOME=/usr/lib/jvm/jre-17
echo 'export JAVA_HOME=/usr/lib/jvm/jre-17' >> /etc/profile
echo 'export PATH=$JAVA_HOME/bin:$PATH' >> /etc/profile
source /etc/profile

# Verify Java version
echo "Verifying Java version..."
java -version

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

echo "=== CLUSTER FIXED SUCCESSFULLY ==="
echo "You can now run your tests again!" 