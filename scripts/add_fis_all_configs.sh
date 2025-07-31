#!/bin/bash

# Add All FIS Configs Processing Script

set -e

# Configuration
CLUSTER_ID="${1:-}"
REGION="${2:-us-east-1}"

if [ -z "$CLUSTER_ID" ]; then
    echo "Usage: $0 <cluster-id> [region]"
    echo "Example: $0 j-1234567890ABCD us-east-1"
    exit 1
fi

echo "Adding all FIS configs processing to cluster: $CLUSTER_ID in region: $REGION"

# Add the all FIS configs processing step
aws emr add-steps \
    --cluster-id "$CLUSTER_ID" \
    --region "$REGION" \
    --steps file://scripts/emr_step_run_all_fis_configs_proper.json

echo "All FIS configs processing step added successfully!"
echo "This will process all 6 configs and create properly named output files."
echo "Output files will be: result_config_max.tif, result_config_median.tif, etc."
echo "Monitor progress with: aws emr describe-cluster --cluster-id $CLUSTER_ID --region $REGION" 