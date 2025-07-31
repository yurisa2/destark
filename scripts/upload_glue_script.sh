#!/bin/bash

# Upload Glue Script to S3

set -e

echo "Uploading raster_fuzzy_glue.py to S3..."

# Upload the script to S3
aws s3 cp app/raster_fuzzy_glue.py s3://<AWS-BUCKET>-unifile/unifile_test/raster_fuzzy_glue.py

echo "Script uploaded successfully!"
echo "Now you can run the EMR step that uses this script." 