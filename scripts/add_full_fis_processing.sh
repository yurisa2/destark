#!/bin/bash

# Add Full FIS Processing

set -e

# Configuration
CLUSTER_ID="${1:-}"
REGION="${2:-us-east-1}"

if [ -z "$CLUSTER_ID" ]; then
    echo "Usage: $0 <cluster-id> [region]"
    echo "Example: $0 j-1234567890ABCD us-east-1"
    exit 1
fi

echo "Adding full FIS processing to cluster: $CLUSTER_ID in region: $REGION"
echo "This will:"
echo "1. Download all input files"
echo "2. Process all 5 FIS configs with REAL FIS logic"
echo "3. Use block processing (500 rows) to avoid memory issues"
echo "4. Process sequentially for reliability"
echo "5. Keep the cluster running when done"
echo ""

# Add the full FIS processing step
aws emr add-steps \
    --cluster-id "$CLUSTER_ID" \
    --region "$REGION" \
    --steps file://scripts/emr_step_full_fis_processing.json

echo "Full FIS processing step added successfully!"
echo "This will use your REAL FIS logic with proper fuzzy inference processing."
echo "Monitor progress with: aws emr describe-cluster --cluster-id $CLUSTER_ID --region $REGION" 