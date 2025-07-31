#!/bin/bash

# Add Spark Submit FIS Processing (Direct)

set -e

CLUSTER_ID="${1:-}"
REGION="${2:-us-east-2}"

if [ -z "$CLUSTER_ID" ]; then
    echo "Usage: $0 <cluster-id> [region]"
    echo "Example: $0 j-1234567890ABCD us-east-2"
    exit 1
fi

echo "=== SPARK SUBMIT FIS PROCESSING (DIRECT) SETUP ==="
echo "Cluster: $CLUSTER_ID"
echo "Region: $REGION"
echo ""

# Add EMR step
echo "Adding EMR step with direct script creation..."
aws emr add-steps \
    --cluster-id "$CLUSTER_ID" \
    --region "$REGION" \
    --steps file://scripts/emr_step_spark_submit_direct.json

if [ $? -eq 0 ]; then
    echo "✅ EMR step added successfully!"
    echo ""
    echo "=== SPARK SUBMIT FIS PROCESSING (DIRECT) ==="
    echo "This will:"
    echo "1. ✅ Install packages (rasterio, scikit-fuzzy, boto3, pyspark)"
    echo "2. ✅ Download input files"
    echo "3. ✅ Create Spark FIS script directly on cluster (no S3 upload needed)"
    echo "4. ✅ Use spark-submit with --master yarn --deploy-mode cluster"
    echo "5. ✅ Distribute work across ALL nodes (not just master)"
    echo "6. ✅ Process all 5 configs with proper Spark distribution:"
    echo "   - config_median.json → result_spark_submit_config_median.tif"
    echo "   - config_minimum.json → result_spark_submit_config_minimum.tif"
    echo "   - config_mode.json → result_spark_submit_config_mode.tif"
    echo "   - config_round_down.json → result_spark_submit_config_round_down.tif"
    echo "   - config_round_up.json → result_spark_submit_config_round_up.tif"
    echo "7. ✅ Use 4g executor memory, 2 cores per executor"
    echo "8. ✅ Proper cluster mode deployment"
    echo ""
    echo "Check results in: s3://<AWS-BUCKET>-unifile/unifile_test/"
    echo "Expected runtime: 10-30 minutes (properly distributed!)"
    echo ""
    echo "You should now see ALL nodes working, not just the master!"
else
    echo "❌ Failed to add EMR step"
    exit 1
fi

echo ""
echo "=== SPARK SUBMIT FIS PROCESSING (DIRECT) SETUP COMPLETED ===" 