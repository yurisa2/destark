#!/bin/bash

# Script to check EMR cluster logs and troubleshoot issues
# Updated to use the new S3 bucket for logs

set -e

echo "=== EMR Cluster Log Checker ==="

# Configuration
LOG_URI="s3://adveng-pipeline-unifile/logs"
REGION="us-east-2"

# Check if cluster ID is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <cluster_id>"
    echo ""
    echo "To find your cluster ID:"
    echo "aws emr list-clusters --active --region $REGION"
    exit 1
fi

CLUSTER_ID="$1"

echo "Checking logs for cluster: $CLUSTER_ID"
echo "Log URI: $LOG_URI"
echo ""

# Check cluster status
echo "=== Cluster Status ==="
aws emr describe-cluster --cluster-id "$CLUSTER_ID" --region "$REGION" --query 'Cluster.{Status:Status.State,Name:Name,ReleaseLabel:ReleaseLabel,MasterPublicDnsName:MasterPublicDnsName}' --output table

echo ""

# List log files in S3
echo "=== Available Log Files ==="
aws s3 ls "$LOG_URI/" --recursive --human-readable --summarize || echo "No logs found or access denied"

echo ""

# Check specific log directories
echo "=== Checking Specific Log Directories ==="

# Check for the specific cluster log directory
CLUSTER_LOG_DIR="$LOG_URI/$CLUSTER_ID"
echo "Looking for cluster logs in: $CLUSTER_LOG_DIR"

if aws s3 ls "$CLUSTER_LOG_DIR/" > /dev/null 2>&1; then
    echo "✓ Found cluster logs"
    echo "Available log directories:"
    aws s3 ls "$CLUSTER_LOG_DIR/" --recursive --human-readable
else
    echo "✗ No cluster logs found in $CLUSTER_LOG_DIR"
fi

echo ""

# Check for step logs
echo "=== Step Logs ==="
STEPS_LOG_DIR="$LOG_URI/$CLUSTER_ID/steps"
if aws s3 ls "$STEPS_LOG_DIR/" > /dev/null 2>&1; then
    echo "✓ Found step logs"
    echo "Step log directories:"
    aws s3 ls "$STEPS_LOG_DIR/" --recursive --human-readable
else
    echo "✗ No step logs found"
fi

echo ""

# Check for container logs
echo "=== Container Logs ==="
CONTAINER_LOG_DIR="$LOG_URI/$CLUSTER_ID/containers"
if aws s3 ls "$CONTAINER_LOG_DIR/" > /dev/null 2>&1; then
    echo "✓ Found container logs"
    echo "Container log directories:"
    aws s3 ls "$CONTAINER_LOG_DIR/" --recursive --human-readable
else
    echo "✗ No container logs found"
fi

echo ""

# Check for application logs
echo "=== Application Logs ==="
APP_LOG_DIR="$LOG_URI/$CLUSTER_ID/applications"
if aws s3 ls "$APP_LOG_DIR/" > /dev/null 2>&1; then
    echo "✓ Found application logs"
    echo "Application log directories:"
    aws s3 ls "$APP_LOG_DIR/" --recursive --human-readable
else
    echo "✗ No application logs found"
fi

echo ""

# Show recent log entries if available
echo "=== Recent Log Entries (if available) ==="
echo "To view specific log files, use:"
echo "aws s3 cp s3://adveng-pipeline-unifile/logs/$CLUSTER_ID/<log_path> -"
echo ""
echo "Example commands:"
echo "aws s3 cp s3://adveng-pipeline-unifile/logs/$CLUSTER_ID/steps/*/controller -"
echo "aws s3 cp s3://adveng-pipeline-unifile/logs/$CLUSTER_ID/steps/*/stderr -"
echo "aws s3 cp s3://adveng-pipeline-unifile/logs/$CLUSTER_ID/steps/*/stdout -"

echo ""
echo "=== Troubleshooting Tips ==="
echo "1. If logs are not accessible, check IAM permissions for the EMR role"
echo "2. Ensure the S3 bucket s3://adveng-pipeline-unifile exists and is accessible"
echo "3. Check if the cluster has proper permissions to write to the log URI"
echo "4. Verify the cluster is in a state where logs are being generated"
echo ""
echo "To check cluster permissions:"
echo "aws emr describe-cluster --cluster-id $CLUSTER_ID --region $REGION --query 'Cluster.ServiceRole'"
echo ""
echo "To check EMR role permissions:"
echo "aws iam get-role --role-name Dpe-EMR-Role" 