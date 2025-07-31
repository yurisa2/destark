#!/bin/bash

# Build and Run Simple EMR Environment
# Uses Amazon Linux 2 with EMR components installed

echo "=== BUILDING AND RUNNING SIMPLE EMR ENVIRONMENT ==="

# Set AWS credentials
export AWS_ACCESS_KEY_ID="YOUR_AWS_ACCESS_KEY_ID"
export AWS_SECRET_ACCESS_KEY="YOUR_AWS_SECRET_ACCESS_KEY"
export AWS_DEFAULT_REGION="us-east-2"

echo "AWS credentials set for region: $AWS_DEFAULT_REGION"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ ERROR: Docker is not running. Please start Docker and try again."
    exit 1
fi

# Build the Docker image using simple EMR setup
echo "Building Docker image with simple EMR..."
docker build -f Dockerfile.emr-simple -t emr-simple:7.9.0 .

if [ $? -ne 0 ]; then
    echo "❌ ERROR: Docker build failed"
    exit 1
fi

echo "✅ Docker image built successfully with simple EMR"

# Create necessary directories
mkdir -p data logs

# Run the container with distributed Spark setup
echo "Starting simple EMR environment with distributed Spark..."
docker run -it --rm \
    --name emr-simple-spark \
    -p "4040:4040" \
    -p "8080:8080" \
    -p "7077:7077" \
    -v "$(pwd)/scripts:/opt/destark/scripts" \
    -v "$(pwd)/data:/opt/destark/data" \
    -v "$(pwd)/logs:/opt/destark/logs" \
    -e AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID" \
    -e AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY" \
    -e AWS_DEFAULT_REGION="$AWS_DEFAULT_REGION" \
    emr-simple:7.9.0

echo "✅ Simple EMR environment stopped" 