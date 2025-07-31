#!/bin/bash

# Simple command to run Spark FIS processing on EMR master node
# Copy this script to your master node and run it

echo "=== SPARK FIS PROCESSING - MASTER NODE ==="
echo "This will process all FIS configurations using distributed Spark"
echo ""

# Install required packages
echo "Installing required packages..."
pip3 install rasterio scikit-fuzzy boto3 networkx pyspark

# Run the Spark FIS processing
echo "Starting distributed Spark FIS processing..."
python3 /tmp/spark_fis_master_command.py

echo "=== COMPLETED ===" 