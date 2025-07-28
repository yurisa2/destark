#!/bin/bash

# Test script to verify config file access

echo "=== Testing Config File Access ==="

# Test the original config
echo "Testing original config:"
aws s3 ls s3://adveng-pipeline/unifile_test/config_round_up.json

# Test the Spark-accessible config
echo "Testing Spark-accessible config:"
aws s3 ls s3://adveng-pipeline/unifile_test/spark_config_round_up.json

# Download and check content
echo "Downloading and checking content:"
aws s3 cp s3://adveng-pipeline/unifile_test/spark_config_round_up.json /tmp/test_config.json
cat /tmp/test_config.json | head -10

echo "=== Test Complete ===" 