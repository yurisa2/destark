#!/bin/bash

# Run Spark Submit FIS Processing - Properly Distributed

set -e

CLUSTER_ID="${1:-}"
REGION="${2:-us-east-2}"

if [ -z "$CLUSTER_ID" ]; then
    echo "Usage: $0 <cluster-id> [region]"
    echo "Example: $0 j-1234567890ABCD us-east-2"
    exit 1
fi

echo "=== SPARK SUBMIT FIS PROCESSING SETUP ==="
echo "Cluster: $CLUSTER_ID"
echo "Region: $REGION"
echo ""

# Upload Spark script to S3
echo "Step 1: Uploading Spark FIS script to S3..."
aws s3 cp scripts/spark_fis_emr_submit.py s3://<AWS-BUCKET>-unifile/unifile_test/spark_fis_emr_submit.py

if [ $? -eq 0 ]; then
    echo "✅ Script uploaded successfully"
else
    echo "❌ Failed to upload script"
    exit 1
fi

# Create EMR step with spark-submit
echo "Step 2: Creating EMR step with spark-submit..."
cat > /tmp/spark_submit_step.json << 'EOF'
[
  {
    "Name": "Spark Submit FIS Processing",
    "ActionOnFailure": "CONTINUE",
    "Type": "CUSTOM_JAR",
    "Jar": "command-runner.jar",
    "Args": [
      "bash", "-c",
      "cd /mnt/destark && echo '=== SPARK SUBMIT FIS PROCESSING STARTED ===' && echo 'Installing packages...' && pip3 install rasterio scikit-fuzzy boto3 networkx pyspark && echo 'Downloading input files...' && aws s3 cp s3://<AWS-BUCKET>-unifile/unifile_test/so300m.in /tmp/social.tif && aws s3 cp s3://<AWS-BUCKET>-unifile/unifile_test/e300m.in /tmp/environmental.tif && aws s3 cp s3://<AWS-BUCKET>-unifile/unifile_test/s300m.in /tmp/strategic.tif && echo 'Downloading and running Spark Submit FIS processing...' && aws s3 cp s3://<AWS-BUCKET>-unifile/unifile_test/spark_fis_emr_submit.py /tmp/spark_fis_emr_submit.py && chmod +x /tmp/spark_fis_emr_submit.py && spark-submit --master yarn --deploy-mode cluster --executor-memory 4g --executor-cores 2 --driver-memory 2g --conf spark.sql.adaptive.enabled=true --conf spark.sql.adaptive.coalescePartitions.enabled=true --conf spark.serializer=org.apache.spark.serializer.KryoSerializer --conf spark.kryo.registrationRequired=false --conf spark.sql.adaptive.advisoryPartitionSizeInBytes=128m --conf spark.sql.adaptive.maxShuffledHashJoinLocalMapThreshold=0 --conf spark.sql.adaptive.skewJoin.enabled=true --conf spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes=256m --conf spark.sql.adaptive.skewJoin.skewedPartitionFactor=5 --conf spark.driver.maxResultSize=1g /tmp/spark_fis_emr_submit.py && echo '=== SPARK SUBMIT FIS PROCESSING COMPLETED ==='"
    ]
  }
]
EOF

# Add EMR step
echo "Step 3: Adding EMR step..."
aws emr add-steps \
    --cluster-id "$CLUSTER_ID" \
    --region "$REGION" \
    --steps file:///tmp/spark_submit_step.json

if [ $? -eq 0 ]; then
    echo "✅ EMR step added successfully!"
    echo ""
    echo "=== SPARK SUBMIT FIS PROCESSING ==="
    echo "This will:"
    echo "1. ✅ Install packages (rasterio, scikit-fuzzy, boto3, pyspark)"
    echo "2. ✅ Download input files"
    echo "3. ✅ Use spark-submit with --master yarn --deploy-mode cluster"
    echo "4. ✅ Distribute work across ALL nodes (not just master)"
    echo "5. ✅ Process all 5 configs with proper Spark distribution:"
    echo "   - config_median.json → result_spark_submit_config_median.tif"
    echo "   - config_minimum.json → result_spark_submit_config_minimum.tif"
    echo "   - config_mode.json → result_spark_submit_config_mode.tif"
    echo "   - config_round_down.json → result_spark_submit_config_round_down.tif"
    echo "   - config_round_up.json → result_spark_submit_config_round_up.tif"
    echo "6. ✅ Use 4g executor memory, 2 cores per executor"
    echo "7. ✅ Proper cluster mode deployment"
    echo ""
    echo "Check results in: s3://<AWS-BUCKET>-unifile/unifile_test/"
    echo "Expected runtime: 10-30 minutes (properly distributed!)"
    echo ""
    echo "You should now see ALL nodes working, not just the master!"
else
    echo "❌ Failed to add EMR step"
    exit 1
fi

# Cleanup
rm -f /tmp/spark_submit_step.json

echo ""
echo "=== SPARK SUBMIT FIS PROCESSING SETUP COMPLETED ===" 