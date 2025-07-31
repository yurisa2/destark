#!/bin/bash

# Add Pure Python FIS Workflow

set -e

# Configuration
CLUSTER_ID="${1:-}"
REGION="${2:-us-east-1}"

if [ -z "$CLUSTER_ID" ]; then
    echo "Usage: $0 <cluster-id> [region]"
    echo "Example: $0 j-1234567890ABCD us-east-1"
    exit 1
fi

echo "Adding pure Python FIS workflow to cluster: $CLUSTER_ID in region: $REGION"
echo "This will:"
echo "1. Download input files"
echo "2. Process all 5 FIS configs using pure Python (no Spark)"
echo "3. Keep the cluster running when done"
echo ""

# Add the pure Python workflow step
aws emr add-steps \
    --cluster-id "$CLUSTER_ID" \
    --region "$REGION" \
    --steps file://scripts/emr_step_pure_python_workflow.json

echo "Pure Python FIS workflow step added successfully!"
echo "This step will process all configs without Spark to avoid memory issues."
echo "Monitor progress with: aws emr describe-cluster --cluster-id $CLUSTER_ID --region $REGION" 