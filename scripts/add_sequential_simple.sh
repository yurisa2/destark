#!/bin/bash

# Add Sequential FIS Processing

set -e

# Configuration
CLUSTER_ID="${1:-}"
REGION="${2:-us-east-1}"

if [ -z "$CLUSTER_ID" ]; then
    echo "Usage: $0 <cluster-id> [region]"
    echo "Example: $0 j-1234567890ABCD us-east-1"
    exit 1
fi

echo "Adding sequential FIS processing to cluster: $CLUSTER_ID in region: $REGION"
echo "This will:"
echo "1. Download input files once"
echo "2. Process each config sequentially (one by one)"
echo "3. Use boto3 for S3 operations (no subprocess)"
echo "4. Continue even if one config fails"
echo "5. Keep the cluster running when done"
echo ""

# Add the sequential processing step
aws emr add-steps \
    --cluster-id "$CLUSTER_ID" \
    --region "$REGION" \
    --steps file://scripts/emr_step_sequential_simple.json

echo "Sequential FIS processing step added successfully!"
echo "This will process configs one by one for maximum reliability."
echo "Monitor progress with: aws emr describe-cluster --cluster-id $CLUSTER_ID --region $REGION" 