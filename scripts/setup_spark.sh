#!/bin/bash

# Setup script for Spark-based raster fuzzy inference system
# This script starts the Spark cluster and prepares the environment

echo "=== Setting up Spark Cluster for Raster Fuzzy Inference ==="

# Create necessary directories
echo "Creating directories..."
mkdir -p data
mkdir -p logs
mkdir -p notebooks
mkdir -p app/config

# Check if Docker and Docker Compose are installed
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if docker compose is available
if ! docker compose version &> /dev/null; then
    echo "Error: Docker Compose is not available. Please install Docker Compose first."
    exit 1
fi

# Stop any existing containers
echo "Stopping existing containers..."
docker compose down

# Start the Spark cluster
echo "Starting Spark cluster..."
docker compose up -d

# Wait for services to be ready
echo "Waiting for services to be ready..."
sleep 30

# Check if services are running
echo "Checking service status..."
docker compose ps

# Test Spark connectivity
echo "Testing Spark connectivity..."
docker exec jupyter-spark python -c "
import findspark
findspark.init()
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName('TestConnection') \
    .master('spark://spark-master:7077') \
    .getOrCreate()

print('Spark session created successfully!')
print(f'Spark version: {spark.version}')
print(f'Available executors: {spark.sparkContext.getExecutorMemoryStatus()}')

spark.stop()
print('Test completed successfully!')
"

if [ $? -eq 0 ]; then
    echo "✓ Spark cluster is ready!"
    echo ""
    echo "=== Access Information ==="
    echo "Spark Web UI: http://localhost:8080"
    echo "Jupyter Lab: http://localhost:8888"
    echo "Spark Master: spark://localhost:7077"
    echo ""
    echo "=== Usage Examples ==="
    echo "# Test with local mode:"
    echo "python app/raster_fuzzy_spark.py test_social.tif test_env.tif test_strat.tif output.tif --local"
    echo ""
    echo "# Test with cluster mode:"
    echo "python app/raster_fuzzy_spark.py test_social.tif test_env.tif test_strat.tif output.tif"
    echo ""
    echo "# Access Jupyter for development:"
    echo "Open http://localhost:8888 in your browser"
else
    echo "✗ Spark cluster setup failed. Check the logs:"
    echo "docker compose logs"
    exit 1
fi 