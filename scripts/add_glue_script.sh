#!/bin/bash

# Add Glue Script

set -e

# Configuration
CLUSTER_ID="${1:-}"
REGION="${2:-us-east-1}"

if [ -z "$CLUSTER_ID" ]; then
    echo "Usage: $0 <cluster-id> [region]"
    echo "Example: $0 j-1234567890ABCD us-east-1"
    exit 1
fi

echo "Adding Glue script to cluster: $CLUSTER_ID in region: $REGION"

# Add the Glue script step
aws emr add-steps \
    --cluster-id "$CLUSTER_ID" \
    --region "$REGION" \
    --steps file://scripts/emr_step_run_glue_script.json

echo "Glue script step added successfully!"
echo "This will run your existing raster_fuzzy_glue.py script"
echo "Monitor progress with: aws emr describe-cluster --cluster-id $CLUSTER_ID --region $REGION" 