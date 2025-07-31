#!/bin/bash

# Complete FIS Workflow Script for EMR
# This script installs packages and runs all FIS models

set -e  # Exit on any error

echo "=== Starting Complete FIS Workflow ==="
echo "Timestamp: $(date)"

# Step 1: Navigate to code directory
echo "Step 1: Navigating to code directory..."
cd /mnt/destark
echo "Current directory: $(pwd)"

# Step 2: Pull latest code
echo "Step 2: Pulling latest code..."
git pull origin main
echo "✓ Code updated"

# Step 3: Install Python packages
echo "Step 3: Installing Python packages..."
python3 -m pip install --user python-dateutil numpy scipy pandas rasterio scikit-fuzzy boto3 s3fs

# Step 4: Verify package installation
echo "Step 4: Verifying package installation..."
python3 -c "
try:
    import dateutil, numpy, rasterio, skfuzzy, boto3, s3fs
    print('✓ All required packages installed successfully')
except ImportError as e:
    print(f'✗ Package import failed: {e}')
    exit(1)
"
echo "✓ Packages verified"

# Step 5: Make scripts executable
echo "Step 5: Making scripts executable..."
chmod +x scripts/run_all_fis_models_local_optimized.sh
echo "✓ Scripts made executable"

# Step 6: Run all FIS models
echo "Step 6: Running all FIS models..."
echo "Input files:"
echo "  Social: s3://<AWS-BUCKET>/unifile_test/so300m.in"
echo "  Environmental: s3://<AWS-BUCKET>/unifile_test/e300m.in"
echo "  Strategic: s3://<AWS-BUCKET>/unifile_test/s300m.in"
echo "Output prefix: s3://<AWS-BUCKET>/unifile_test/result_300m"
echo ""

# Run the local optimized script
./scripts/run_all_fis_models_local_optimized.sh \
    s3://<AWS-BUCKET>/unifile_test/so300m.in \
    s3://<AWS-BUCKET>/unifile_test/e300m.in \
    s3://<AWS-BUCKET>/unifile_test/s300m.in \
    s3://<AWS-BUCKET>/unifile_test/result_300m

echo ""
echo "=== Complete FIS Workflow Finished Successfully ==="
echo "Timestamp: $(date)"
echo "Output files created:"
echo "  s3://<AWS-BUCKET>/unifile_test/result_300m_max.tif"
echo "  s3://<AWS-BUCKET>/unifile_test/result_300m_median.tif"
echo "  s3://<AWS-BUCKET>/unifile_test/result_300m_minimum.tif"
echo "  s3://<AWS-BUCKET>/unifile_test/result_300m_mode.tif"
echo "  s3://<AWS-BUCKET>/unifile_test/result_300m_round_up.tif"
echo "  s3://<AWS-BUCKET>/unifile_test/result_300m_round_down.tif" 