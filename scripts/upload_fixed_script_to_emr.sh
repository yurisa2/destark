#!/bin/bash

# Upload the fixed spark submit script to EMR cluster
# Usage: ./scripts/upload_fixed_script_to_emr.sh <cluster_id> [master_dns]

set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 <cluster_id> [master_dns]"
    echo ""
    echo "Example:"
    echo "  $0 j-1234567890abcdef"
    echo "  $0 j-1234567890abcdef ec2-123-456-789-10.compute-1.amazonaws.com"
    exit 1
fi

CLUSTER_ID="$1"
MASTER_DNS="$2"

echo "=== Uploading Fixed Spark Submit Script to EMR Cluster ==="
echo "Cluster ID: $CLUSTER_ID"

# Get master DNS if not provided
if [ -z "$MASTER_DNS" ]; then
    echo "Getting master DNS from EMR cluster..."
    MASTER_DNS=$(aws emr describe-cluster --cluster-id "$CLUSTER_ID" --query 'Cluster.MasterPublicDnsName' --output text)
    if [ $? -ne 0 ]; then
        echo "Error: Could not get master DNS for cluster $CLUSTER_ID"
        exit 1
    fi
fi

echo "Master DNS: $MASTER_DNS"

# Check if the fixed script exists
if [ ! -f "scripts/run_all_fis_models_spark_submit_fixed.sh" ]; then
    echo "Error: Fixed script not found at scripts/run_all_fis_models_spark_submit_fixed.sh"
    exit 1
fi

# Upload the fixed script to the cluster
echo "Uploading fixed script to EMR cluster..."
scp -i ~/.ssh/emr-key.pem -o StrictHostKeyChecking=no \
    scripts/run_all_fis_models_spark_submit_fixed.sh \
    hadoop@"$MASTER_DNS":/mnt/destark/scripts/

if [ $? -eq 0 ]; then
    echo "✓ Successfully uploaded fixed script to EMR cluster"
    echo ""
    echo "To use the fixed script on the EMR cluster, run:"
    echo "  chmod +x /mnt/destark/scripts/run_all_fis_models_spark_submit_fixed.sh"
    echo "  ./scripts/run_all_fis_models_spark_submit_fixed.sh \\"
    echo "    s3://<AWS-BUCKET>/unifile_test/so300m.in \\"
    echo "    s3://<AWS-BUCKET>/unifile_test/e300m.in \\"
    echo "    s3://<AWS-BUCKET>/unifile_test/s300m.in \\"
    echo "    s3://<AWS-BUCKET>/unifile_test/result_300m"
else
    echo "✗ Failed to upload script to EMR cluster"
    echo ""
    echo "Alternative: You can manually copy the script content and create it on the cluster"
    echo "Or use the existing test_java_version.sh script first:"
    echo "  ./scripts/test_java_version.sh"
fi 