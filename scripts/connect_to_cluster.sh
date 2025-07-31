#!/bin/bash

# Script to connect to EMR cluster for development
# Usage: ./connect_to_cluster.sh <cluster-id>

if [ $# -eq 0 ]; then
    echo "Usage: $0 <cluster-id>"
    echo "Example: $0 <EMR-CLUSTER-ID>"
    exit 1
fi

CLUSTER_ID=$1

echo "Getting cluster details for: $CLUSTER_ID"

# Get cluster status and master node DNS
CLUSTER_INFO=$(aws emr describe-cluster --cluster-id $CLUSTER_ID --region us-east-2)

# Extract status
STATUS=$(echo "$CLUSTER_INFO" | jq -r '.Cluster.Status.State')

if [ "$STATUS" != "WAITING" ] && [ "$STATUS" != "RUNNING" ]; then
    echo "Cluster is not ready. Current status: $STATUS"
    echo "Please wait for the cluster to be in WAITING or RUNNING state."
    exit 1
fi

# Extract master public DNS
MASTER_DNS=$(echo "$CLUSTER_INFO" | jq -r '.Cluster.MasterPublicDnsName')

if [ "$MASTER_DNS" == "null" ] || [ -z "$MASTER_DNS" ]; then
    echo "Could not get master node DNS. Cluster might still be starting up."
    exit 1
fi

echo "Cluster Status: $STATUS"
echo "Master Node DNS: $MASTER_DNS"
echo ""
echo "Connecting to master node..."
echo "SSH Command: ssh -i /path/to/<KEY-PAIR-NAME>.pem hadoop@$MASTER_DNS"
echo ""

# Try to connect if key file exists
KEY_FILE="/path/to/<KEY-PAIR-NAME>.pem"
if [ -f "$KEY_FILE" ]; then
    echo "Found key file at $KEY_FILE"
    echo "Connecting to cluster..."
    ssh -i "$KEY_FILE" hadoop@$MASTER_DNS
else
    echo "Key file not found at $KEY_FILE"
    echo "Please update the KEY_FILE variable in this script with the correct path to your .pem file"
    echo "Then run: ssh -i $KEY_FILE hadoop@$MASTER_DNS"
fi 