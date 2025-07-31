#!/bin/bash

# List EMR clusters and their status
# This script helps identify available clusters for testing

set -e

REGION="${1:-us-east-2}"

echo "=========================================="
echo "EMR Clusters in Region: $REGION"
echo "=========================================="

echo ""
echo "Active Clusters:"
echo "================"
aws emr list-clusters \
    --region "$REGION" \
    --active \
    --query 'Clusters[*].[Id,Name,Status.State,Status.Timeline.CreationDateTime,InstanceGroups[0].InstanceType]' \
    --output table

echo ""
echo "All Clusters (Last 30 days):"
echo "============================"
aws emr list-clusters \
    --region "$REGION" \
    --created-after $(date -d '30 days ago' +%Y-%m-%d) \
    --query 'Clusters[*].[Id,Name,Status.State,Status.Timeline.CreationDateTime,InstanceGroups[0].InstanceType]' \
    --output table

echo ""
echo "Cluster Details (if you have a specific cluster ID):"
echo "==================================================="
echo "To get detailed information about a specific cluster:"
echo "aws emr describe-cluster --cluster-id <CLUSTER_ID> --region $REGION"
echo ""
echo "To check cluster steps:"
echo "aws emr list-steps --cluster-id <CLUSTER_ID> --region $REGION"
echo ""
echo "To connect to master node:"
echo "aws emr ssh --cluster-id <CLUSTER_ID> --key-pair-file <KEY_FILE> --region $REGION" 