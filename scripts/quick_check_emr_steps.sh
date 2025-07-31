#!/bin/bash

# Quick EMR Steps Status Checker

CLUSTER_ID="${1:-}"
REGION="${2:-us-east-1}"

if [ -z "$CLUSTER_ID" ]; then
    echo "Usage: $0 <cluster-id> [region]"
    echo "Example: $0 j-1234567890ABCD us-east-1"
    exit 1
fi

echo "=== Quick EMR Steps Status ==="
aws emr list-steps --cluster-id "$CLUSTER_ID" --region "$REGION" --query 'Steps[*].[Name,Status.State,Status.StateChangeReason.Message]' --output table 