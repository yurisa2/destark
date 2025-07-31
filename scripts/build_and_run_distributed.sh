#!/bin/bash

# Build and Run Distributed EMR Environment
# Tests distributed Spark processing with S3 integration

echo "=== BUILDING AND RUNNING DISTRIBUTED EMR ENVIRONMENT ==="

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

# Build the Docker image
echo "Building Docker image..."
docker build -f Dockerfile.emr-local -t emr-local:7.9.0 .

if [ $? -ne 0 ]; then
    echo "❌ ERROR: Docker build failed"
    exit 1
fi

echo "✅ Docker image built successfully"

# Create necessary directories
mkdir -p data logs

# Run the container with distributed Spark setup
echo "Starting EMR local environment with distributed Spark..."
docker run -it --rm \
    --name emr-local-spark \
    -p "4040:4040" \
    -p "8080:8080" \
    -p "7077:7077" \
    -v "$(pwd)/scripts:/opt/destark/scripts" \
    -v "$(pwd)/data:/opt/destark/data" \
    -v "$(pwd)/logs:/opt/destark/logs" \
    -e AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID" \
    -e AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY" \
    -e AWS_DEFAULT_REGION="$AWS_DEFAULT_REGION" \
    emr-local:7.9.0

echo "✅ EMR local environment stopped" 