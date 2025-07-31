#!/bin/bash

# Run Full FIS Processing - CLI Version

set -e

# Configuration
CLUSTER_ID="${1:-}"
REGION="${2:-us-east-2}"

if [ -z "$CLUSTER_ID" ]; then
    echo "Usage: $0 <cluster-id> [region]"
    echo "Example: $0 j-1234567890ABCD us-east-2"
    exit 1
fi

echo "=== FULL FIS PROCESSING SETUP ==="
echo "Cluster: $CLUSTER_ID"
echo "Region: $REGION"
echo ""

# Step 1: Upload the Python script to S3
echo "Step 1: Uploading FIS processing script to S3..."
aws s3 cp scripts/full_fis_processing.py s3://<AWS-BUCKET>-unifile/unifile_test/full_fis_processing.py

if [ $? -eq 0 ]; then
    echo "✅ Script uploaded successfully"
else
    echo "❌ Failed to upload script"
    exit 1
fi

# Step 2: Create a temporary step file with the uploaded script
echo "Step 2: Creating EMR step configuration..."
cat > /tmp/fis_step.json << 'EOF'
[
  {
    "Name": "Full FIS Processing",
    "ActionOnFailure": "CONTINUE",
    "Type": "CUSTOM_JAR",
    "Jar": "command-runner.jar",
    "Args": [
      "bash", "-c",
      "echo '=== FULL FIS PROCESSING STARTED ===' && echo 'Installing packages...' && pip3 install rasterio scikit-fuzzy boto3 networkx && echo 'Downloading input files...' && aws s3 cp s3://<AWS-BUCKET>-unifile/unifile_test/so300m.in /tmp/social.tif && aws s3 cp s3://<AWS-BUCKET>-unifile/unifile_test/e300m.in /tmp/environmental.tif && aws s3 cp s3://<AWS-BUCKET>-unifile/unifile_test/s300m.in /tmp/strategic.tif && echo 'Downloading and running FIS processing script...' && aws s3 cp s3://<AWS-BUCKET>-unifile/unifile_test/full_fis_processing.py /tmp/full_fis_processing.py && chmod +x /tmp/full_fis_processing.py && python3 /tmp/full_fis_processing.py && echo '=== FULL FIS PROCESSING COMPLETED ==='"
    ]
  }
]
EOF

# Step 3: Add the step using the temporary file
echo "Step 3: Adding EMR step..."
aws emr add-steps \
    --cluster-id "$CLUSTER_ID" \
    --region "$REGION" \
    --steps file:///tmp/fis_step.json

if [ $? -eq 0 ]; then
    echo "✅ EMR step added successfully!"
    echo ""
    echo "=== WHAT WILL HAPPEN ==="
    echo "1. ✅ Install packages (rasterio, scikit-fuzzy, boto3, networkx)"
    echo "2. ✅ Download input files (social, environmental, strategic)"
    echo "3. ✅ Download and run the FIS processing script"
    echo "4. ✅ Process all 5 configs with REAL FIS logic:"
    echo "   - config_median.json → result_config_median.tif"
    echo "   - config_minimum.json → result_config_minimum.tif"
    echo "   - config_mode.json → result_config_mode.tif"
    echo "   - config_round_down.json → result_config_round_down.tif"
    echo "   - config_round_up.json → result_config_round_up.tif"
    echo "5. ✅ Upload results to S3"
    echo ""
    echo "Check results in: s3://<AWS-BUCKET>-unifile/unifile_test/"
    echo "Expected runtime: 1.5-3 hours"
else
    echo "❌ Failed to add EMR step"
    exit 1
fi

# Cleanup
rm -f /tmp/fis_step.json

echo ""
echo "=== FULL FIS PROCESSING SETUP COMPLETED ===" 