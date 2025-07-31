#!/bin/bash

# Upload bootstrap script to S3
echo "Uploading bootstrap script to S3..."

# Upload the bootstrap script
aws s3 cp scripts/bootstrap_install_packages.sh s3://<AWS-BUCKET>-unifile/emr-bootstrap/bootstrap_install_packages.sh

echo "Bootstrap script uploaded to: s3://<AWS-BUCKET>-unifile/emr-bootstrap/bootstrap_install_packages.sh"
echo "You can now use this as a bootstrap action in your EMR cluster." 