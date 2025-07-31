#!/bin/bash

# Add Simple FIS Workflow

set -e

# Configuration
CLUSTER_ID="${1:-}"
REGION="${2:-us-east-1}"

if [ -z "$CLUSTER_ID" ]; then
    echo "Usage: $0 <cluster-id> [region]"
    echo "Example: $0 j-1234567890ABCD us-east-1"
    exit 1
fi

echo "Adding simple FIS workflow to cluster: $CLUSTER_ID in region: $REGION"
echo "This will:"
echo "1. Download input files"
echo "2. Process all 5 FIS configs (median, minimum, mode, round_down, round_up)"
echo "3. Terminate the cluster when done"
echo ""
echo "WARNING: This will terminate the cluster when finished!"
echo ""

read -p "Continue? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 1
fi

# Add the simple workflow step
aws emr add-steps \
    --cluster-id "$CLUSTER_ID" \
    --region "$REGION" \
    --steps file://scripts/emr_step_simple_workflow.json

echo "Simple FIS workflow step added successfully!"
echo "This step will process all configs and terminate the cluster."
echo "Monitor progress with: aws emr describe-cluster --cluster-id $CLUSTER_ID --region $REGION" 