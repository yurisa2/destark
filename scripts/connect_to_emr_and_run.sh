#!/bin/bash

# Connect to EMR master node and run the optimized FIS job
# Usage: ./scripts/connect_to_emr_and_run.sh <cluster_id> [master_dns]

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

echo "=== Connecting to EMR Master Node and Running FIS Job ==="
echo "Cluster ID: $CLUSTER_ID"

# Get master DNS if not provided
if [ -z "$MASTER_DNS" ]; then
    echo "Getting master DNS from EMR cluster..."
    MASTER_DNS=$(aws emr describe-cluster --cluster-id "$CLUSTER_ID" --query 'Cluster.MasterPublicDnsName' --output text)
    if [ $? -ne 0 ]; then
        echo "Error: Could not get master DNS for cluster $CLUSTER_ID"
        echo "Please provide the master DNS manually:"
        echo "  $0 $CLUSTER_ID <master_dns>"
        exit 1
    fi
fi

echo "Master DNS: $MASTER_DNS"

# Check if EMR key exists
if [ ! -f "~/.ssh/emr-key.pem" ]; then
    echo "Error: EMR key not found at ~/.ssh/emr-key.pem"
    echo "Please ensure you have the correct SSH key for your EMR cluster"
    exit 1
fi

echo "Connecting to EMR master node..."
echo "Once connected, run these commands:"
echo ""
echo "  cd /mnt/destark"
echo "  mkdir -p logs"
echo "  ./scripts/run_all_fis_models_spark_submit_optimized.sh \\"
echo "    s3://adveng-pipeline/unifile_test/so300m.in \\"
echo "    s3://adveng-pipeline/unifile_test/e300m.in \\"
echo "    s3://adveng-pipeline/unifile_test/s300m.in \\"
echo "    s3://adveng-pipeline/unifile_test/result_300m"
echo ""
echo "To monitor jobs:"
echo "  yarn application -list"
echo "  tail -f logs/round_up_nohup.log"
echo ""

# Connect to EMR master node
ssh -i ~/.ssh/emr-key.pem -o StrictHostKeyChecking=no hadoop@"$MASTER_DNS" 