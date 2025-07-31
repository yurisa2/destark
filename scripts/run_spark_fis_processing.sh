#!/bin/bash

# Run Spark FIS Processing

set -e

CLUSTER_ID="${1:-}"
REGION="${2:-us-east-2}"

if [ -z "$CLUSTER_ID" ]; then
    echo "Usage: $0 <cluster-id> [region]"
    echo "Example: $0 j-1234567890ABCD us-east-2"
    exit 1
fi

echo "=== SPARK FIS PROCESSING SETUP ==="
echo "Cluster: $CLUSTER_ID"
echo "Region: $REGION"
echo ""

# Upload Spark script to S3
echo "Step 1: Uploading Spark FIS script to S3..."
aws s3 cp scripts/spark_fis_emr.py s3://<AWS-BUCKET>-unifile/unifile_test/spark_fis_emr.py

if [ $? -eq 0 ]; then
    echo "✅ Script uploaded successfully"
else
    echo "❌ Failed to upload script"
    exit 1
fi

# Create EMR step
echo "Step 2: Creating EMR step configuration..."
cat > /tmp/spark_fis_step.json << 'EOF'
[
  {
    "Name": "Spark FIS Processing",
    "ActionOnFailure": "CONTINUE",
    "Type": "CUSTOM_JAR",
    "Jar": "command-runner.jar",
    "Args": [
      "bash", "-c",
      "cd /mnt/destark && echo '=== SPARK FIS PROCESSING STARTED ===' && echo 'Installing packages...' && pip3 install rasterio scikit-fuzzy boto3 networkx pyspark && echo 'Downloading input files...' && aws s3 cp s3://<AWS-BUCKET>-unifile/unifile_test/so300m.in /tmp/social.tif && aws s3 cp s3://<AWS-BUCKET>-unifile/unifile_test/e300m.in /tmp/environmental.tif && aws s3 cp s3://<AWS-BUCKET>-unifile/unifile_test/s300m.in /tmp/strategic.tif && echo 'Downloading and running Spark FIS processing...' && aws s3 cp s3://<AWS-BUCKET>-unifile/unifile_test/spark_fis_emr.py /tmp/spark_fis_emr.py && chmod +x /tmp/spark_fis_emr.py && python3 /tmp/spark_fis_emr.py && echo '=== SPARK FIS PROCESSING COMPLETED ==='"
    ]
  }
]
EOF

# Add EMR step
echo "Step 3: Adding EMR step..."
aws emr add-steps \
    --cluster-id "$CLUSTER_ID" \
    --region "$REGION" \
    --steps file:///tmp/spark_fis_step.json

if [ $? -eq 0 ]; then
    echo "✅ EMR step added successfully!"
    echo ""
    echo "=== SPARK FIS PROCESSING ==="
    echo "This will:"
    echo "1. ✅ Install packages (rasterio, scikit-fuzzy, boto3, pyspark)"
    echo "2. ✅ Download input files"
    echo "3. ✅ Process all 5 configs with SPARK FIS:"
    echo "   - config_median.json → result_spark_config_median.tif"
    echo "   - config_minimum.json → result_spark_config_minimum.tif"
    echo "   - config_mode.json → result_spark_config_mode.tif"
    echo "   - config_round_down.json → result_spark_config_round_down.tif"
    echo "   - config_round_up.json → result_spark_config_round_up.tif"
    echo "4. ✅ Use distributed processing across all nodes"
    echo "5. ✅ Much faster than single-node processing"
    echo ""
    echo "Check results in: s3://<AWS-BUCKET>-unifile/unifile_test/"
    echo "Expected runtime: 15-45 minutes (much faster!)"
else
    echo "❌ Failed to add EMR step"
    exit 1
fi

# Cleanup
rm -f /tmp/spark_fis_step.json

echo ""
echo "=== SPARK FIS PROCESSING SETUP COMPLETED ===" 