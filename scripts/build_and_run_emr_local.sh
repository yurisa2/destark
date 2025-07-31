#!/bin/bash

# Build and Run EMR 7.9.0 Local Environment
echo "=== BUILDING EMR 7.9.0 LOCAL ENVIRONMENT ==="

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

# Run the container
echo "Starting EMR local environment..."
docker run -it --rm \
    --name emr-local-spark \
    -p 4040:4040 \
    -p 8080:8080 \
    -p 7077:7077 \
    -v "$(pwd)/scripts:/opt/destark/scripts" \
    -v "$(pwd)/data:/opt/destark/data" \
    -v "$(pwd)/logs:/opt/destark/logs" \
    -e AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID" \
    -e AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY" \
    -e AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-2}" \
    emr-local:7.9.0

echo "✅ EMR local environment stopped" 