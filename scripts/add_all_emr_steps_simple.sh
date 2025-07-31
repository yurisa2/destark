#!/bin/bash

# Simple EMR Steps Addition Script
# Adds all steps at once using the combined configuration

set -e

# Configuration
CLUSTER_ID="${1:-}"
REGION="${2:-us-east-1}"

if [ -z "$CLUSTER_ID" ]; then
    echo "Usage: $0 <cluster-id> [region]"
    echo "Example: $0 j-1234567890ABCD us-east-1"
    exit 1
fi

echo "Adding all EMR steps to cluster: $CLUSTER_ID in region: $REGION"

# Add all steps at once
aws emr add-steps \
    --cluster-id "$CLUSTER_ID" \
    --region "$REGION" \
    --steps file://scripts/emr_all_steps_combined.json

echo "All steps added successfully!"
echo "Monitor progress with: aws emr describe-cluster --cluster-id $CLUSTER_ID --region $REGION"
echo "Check step status with: aws emr list-steps --cluster-id $CLUSTER_ID --region $REGION"
echo "View logs with: aws emr ssh --cluster-id $CLUSTER_ID --key-pair-file your-key.pem" 