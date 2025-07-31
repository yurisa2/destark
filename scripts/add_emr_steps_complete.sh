#!/bin/bash

# Complete EMR Steps Addition Script
# This script adds all necessary steps to an EMR cluster in the correct order

set -e

# Configuration
CLUSTER_ID="${1:-}"
REGION="${2:-us-east-1}"

if [ -z "$CLUSTER_ID" ]; then
    echo "Usage: $0 <cluster-id> [region]"
    echo "Example: $0 j-1234567890ABCD us-east-1"
    exit 1
fi

echo "Adding EMR steps to cluster: $CLUSTER_ID in region: $REGION"

# Step 1: Install System Dependencies
echo "Adding Step 1: Install System Dependencies..."
aws emr add-steps \
    --cluster-id "$CLUSTER_ID" \
    --region "$REGION" \
    --steps file://scripts/emr_step_install_dependencies.json

echo "Waiting for Step 1 to complete..."
aws emr wait step-complete \
    --cluster-id "$CLUSTER_ID" \
    --step-id $(aws emr list-steps --cluster-id "$CLUSTER_ID" --region "$REGION" --query 'Steps[-1].Id' --output text)

# Step 2: Clone Repository
echo "Adding Step 2: Clone Repository..."
aws emr add-steps \
    --cluster-id "$CLUSTER_ID" \
    --region "$REGION" \
    --steps file://scripts/emr_step_clone_repository.json

echo "Waiting for Step 2 to complete..."
aws emr wait step-complete \
    --cluster-id "$CLUSTER_ID" \
    --step-id $(aws emr list-steps --cluster-id "$CLUSTER_ID" --region "$REGION" --query 'Steps[-1].Id' --output text)

# Step 3: Install Python Dependencies
echo "Adding Step 3: Install Python Dependencies..."
aws emr add-steps \
    --cluster-id "$CLUSTER_ID" \
    --region "$REGION" \
    --steps file://scripts/emr_step_install_python_dependencies.json

echo "Waiting for Step 3 to complete..."
aws emr wait step-complete \
    --cluster-id "$CLUSTER_ID" \
    --step-id $(aws emr list-steps --cluster-id "$CLUSTER_ID" --region "$REGION" --query 'Steps[-1].Id' --output text)

# Step 4: Run FIS Models
echo "Adding Step 4: Run FIS Models..."
aws emr add-steps \
    --cluster-id "$CLUSTER_ID" \
    --region "$REGION" \
    --steps file://scripts/emr_step_run_fis_models.json

echo "All steps added successfully!"
echo "Monitor progress with: aws emr describe-cluster --cluster-id $CLUSTER_ID --region $REGION"
echo "Check step status with: aws emr list-steps --cluster-id $CLUSTER_ID --region $REGION" 