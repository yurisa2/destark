#!/bin/bash

# Quick script to run raster processing with the new S3 bucket
# Updated to use s3://<AWS-BUCKET>-unifile

echo "=== Running Raster Processing with New S3 Bucket ==="
echo "Using: s3://<AWS-BUCKET>-unifile"

# Create logs directory if it doesn't exist
mkdir -p logs

# Run the archive script with the new S3 bucket paths
./scripts/run_all_fis_models_spark_submit_archive.sh \
  s3://<AWS-BUCKET>-unifile/unifile_test/so300m.in \
  s3://<AWS-BUCKET>-unifile/unifile_test/e300m.in \
  s3://<AWS-BUCKET>-unifile/unifile_test/s300m.in \
  s3://<AWS-BUCKET>-unifile/unifile_test/result_300m

echo ""
echo "=== Jobs Submitted Successfully ==="
echo "Monitor progress with:"
echo "  yarn application -list"
echo "  tail -f logs/*_archive.log"
echo ""
echo "Expected output files:"
echo "  s3://<AWS-BUCKET>-unifile/unifile_test/result_300m_max.tif"
echo "  s3://<AWS-BUCKET>-unifile/unifile_test/result_300m_median.tif"
echo "  s3://<AWS-BUCKET>-unifile/unifile_test/result_300m_minimum.tif"
echo "  s3://<AWS-BUCKET>-unifile/unifile_test/result_300m_mode.tif"
echo "  s3://<AWS-BUCKET>-unifile/unifile_test/result_300m_round_up.tif"
echo "  s3://<AWS-BUCKET>-unifile/unifile_test/result_300m_round_down.tif" 