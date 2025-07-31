#!/bin/bash

# Add Complete FIS Workflow

set -e

# Configuration
CLUSTER_ID="${1:-}"
REGION="${2:-us-east-1}"

if [ -z "$CLUSTER_ID" ]; then
    echo "Usage: $0 <cluster-id> [region]"
    echo "Example: $0 j-1234567890ABCD us-east-1"
    exit 1
fi

echo "Adding complete FIS workflow to cluster: $CLUSTER_ID in region: $REGION"
echo "This will:"
echo "1. Wait for the current config_max.json job to complete"
echo "2. Process all remaining FIS configs (median, minimum, mode, round_down, round_up)"
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

# Add the complete workflow step
aws emr add-steps \
    --cluster-id "$CLUSTER_ID" \
    --region "$REGION" \
    --steps file://scripts/emr_step_complete_workflow.json

echo "Complete FIS workflow step added successfully!"
echo "This step will:"
echo "- Wait up to 2 hours for config_max.json to complete"
echo "- Process 5 additional FIS configs"
echo "- Terminate cluster <EMR-CLUSTER-ID> when done"
echo ""
echo "Monitor progress with: aws emr describe-cluster --cluster-id $CLUSTER_ID --region $REGION" 