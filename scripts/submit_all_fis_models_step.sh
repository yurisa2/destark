#!/bin/bash

# Submit All FIS Models EMR Step
# This script submits a step to run all FIS models on an EMR cluster

set -e

# Configuration
CLUSTER_ID="${1:-<EMR-CLUSTER-ID>}"
SOCIAL_TIFF="${2:-s3://<AWS-BUCKET>/unifile_test/so300m.in}"
ENVIRONMENTAL_TIFF="${3:-s3://<AWS-BUCKET>/unifile_test/e300m.in}"
STRATEGIC_TIFF="${4:-s3://<AWS-BUCKET>/unifile_test/s300m.in}"
OUTPUT_PREFIX="${5:-s3://<AWS-BUCKET>/unifile_test/result_300m}"

# Validate cluster ID
if [[ -z "$CLUSTER_ID" ]]; then
    echo "Usage: $0 [cluster_id] [social_tiff] [environmental_tiff] [strategic_tiff] [output_prefix]"
    echo ""
    echo "Examples:"
    echo "  # Use default parameters"
    echo "  $0"
    echo ""
    echo "  # Use custom cluster ID"
    echo "  $0 j-1234567890"
    echo ""
    echo "  # Use custom input/output files"
    echo "  $0 <EMR-CLUSTER-ID> s3://bucket/input/social.tif s3://bucket/input/env.tif s3://bucket/input/strat.tif s3://bucket/output/result"
    exit 1
fi

echo "=== Submitting All FIS Models EMR Step ==="
echo "Cluster ID: $CLUSTER_ID"
echo "Input files:"
echo "  Social: $SOCIAL_TIFF"
echo "  Environmental: $ENVIRONMENTAL_TIFF"
echo "  Strategic: $STRATEGIC_TIFF"
echo "Output prefix: $OUTPUT_PREFIX"
echo ""

# Check if cluster exists and is running
echo "Checking cluster status..."
CLUSTER_STATUS=$(aws emr describe-cluster --cluster-id "$CLUSTER_ID" --query 'Cluster.Status.State' --output text 2>/dev/null || echo "NOT_FOUND")

if [[ "$CLUSTER_STATUS" == "NOT_FOUND" ]]; then
    echo "Error: Cluster $CLUSTER_ID not found or access denied"
    exit 1
elif [[ "$CLUSTER_STATUS" != "RUNNING" ]]; then
    echo "Error: Cluster $CLUSTER_ID is not running (status: $CLUSTER_STATUS)"
    echo "Please wait for the cluster to be in RUNNING state"
    exit 1
fi

echo "✓ Cluster is running"

# Create step configuration
echo "Creating step configuration..."
cat > temp_step.json << EOF
[
  {
    "Name": "Run All FIS Models - Raster Fuzzy Inference",
    "ActionOnFailure": "CONTINUE",
    "HadoopJarStep": {
      "Jar": "command-runner.jar",
      "Args": [
        "bash",
        "-c",
        "cd /mnt/destark && git pull origin main && chmod +x scripts/run_all_fis_models_emr.sh && ./scripts/run_all_fis_models_emr.sh \"$SOCIAL_TIFF\" \"$ENVIRONMENTAL_TIFF\" \"$STRATEGIC_TIFF\" \"$OUTPUT_PREFIX\""
      ]
    }
  }
]
EOF

# Submit the step
echo "Submitting step to EMR cluster..."
STEP_ID=$(aws emr add-steps \
    --cluster-id "$CLUSTER_ID" \
    --steps file://temp_step.json \
    --query 'StepIds[0]' \
    --output text)

echo "✓ Step submitted with ID: $STEP_ID"

# Clean up temporary file
rm -f temp_step.json

# Monitor step progress
echo ""
echo "Monitoring step progress..."
echo "You can also monitor in the EMR Console:"
echo "https://console.aws.amazon.com/elasticmapreduce/home?region=us-east-1#/clusterDetails/$CLUSTER_ID"
echo ""

while true; do
    STEP_STATUS=$(aws emr describe-step \
        --cluster-id "$CLUSTER_ID" \
        --step-id "$STEP_ID" \
        --query 'Step.Status.State' \
        --output text)
    
    echo "$(date): Step status: $STEP_STATUS"
    
    if [[ "$STEP_STATUS" == "COMPLETED" ]]; then
        echo ""
        echo "✓ All FIS models processing completed successfully!"
        echo ""
        echo "Generated output files:"
        echo "  ${OUTPUT_PREFIX}_max.tif"
        echo "  ${OUTPUT_PREFIX}_minimum.tif"
        echo "  ${OUTPUT_PREFIX}_median.tif"
        echo "  ${OUTPUT_PREFIX}_mode.tif"
        echo "  ${OUTPUT_PREFIX}_round_up.tif"
        echo "  ${OUTPUT_PREFIX}_round_down.tif"
        echo "  ${OUTPUT_PREFIX}_default.tif"
        break
    elif [[ "$STEP_STATUS" == "FAILED" ]]; then
        echo ""
        echo "✗ Step failed!"
        echo "Check the EMR Console for detailed logs:"
        echo "https://console.aws.amazon.com/elasticmapreduce/home?region=us-east-1#/clusterDetails/$CLUSTER_ID"
        echo ""
        echo "Step details:"
        aws emr describe-step --cluster-id "$CLUSTER_ID" --step-id "$STEP_ID"
        exit 1
    elif [[ "$STEP_STATUS" == "CANCELLED" ]]; then
        echo ""
        echo "Step was cancelled"
        exit 1
    fi
    
    # Wait before checking again
    sleep 30
done

echo ""
echo "=== Step Execution Complete ==="
echo "Step ID: $STEP_ID"
echo "Cluster ID: $CLUSTER_ID"
echo "All FIS models have been processed successfully!" 