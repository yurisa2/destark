#!/bin/bash

# Upload Input Files to S3 for Glue FIS Job
# This script uploads the necessary input files to S3

set -e

# Configuration
AWS_REGION="${1:-us-east-2}"
S3_BUCKET="${2:-<AWS-BUCKET>-unifile}"
S3_PREFIX="${3:-unifile_test}"
RESOLUTION="${4:-1000m}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    print_error "AWS CLI is not installed. Please install it first."
    exit 1
fi

print_status "Upload Configuration:"
echo "  AWS Region: $AWS_REGION"
echo "  S3 Bucket: $S3_BUCKET"
echo "  S3 Prefix: $S3_PREFIX"
echo "  Resolution: $RESOLUTION"

# Define file paths
INPUT_DIR="app/files/input"
CONFIG_DIR="app/config"

# Check if input files exist
print_status "Checking input files..."

# Check config files
if [ ! -d "$CONFIG_DIR" ]; then
    print_error "Config directory not found: $CONFIG_DIR"
    exit 1
fi

# Check input files based on resolution
if [ "$RESOLUTION" = "1000m" ]; then
    INPUT_SUBDIR="$INPUT_DIR/base"
    SOCIAL_FILE="$INPUT_SUBDIR/socioeconomico_1000m.tif"
    ENVIRONMENTAL_FILE="$INPUT_SUBDIR/ambiental_1000m.tif"
    STRATEGIC_FILE="$INPUT_SUBDIR/estratégico_1000m.tif"
elif [ "$RESOLUTION" = "300m" ]; then
    INPUT_SUBDIR="$INPUT_DIR/300m"
    SOCIAL_FILE="$INPUT_SUBDIR/socioeconomico_300m.tif"
    ENVIRONMENTAL_FILE="$INPUT_SUBDIR/ambiental_300m.tif"
    STRATEGIC_FILE="$INPUT_SUBDIR/estrategico_300m.tif"
else
    print_error "Unsupported resolution: $RESOLUTION. Supported: 1000m, 300m"
    exit 1
fi

# Verify input files exist
INPUT_FILES=("$SOCIAL_FILE" "$ENVIRONMENTAL_FILE" "$STRATEGIC_FILE")
for file in "${INPUT_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        print_error "Input file not found: $file"
        exit 1
    fi
    file_size_mb=$(du -m "$file" | cut -f1)
    print_status "Found: $file (${file_size_mb} MB)"
done

# Step 1: Upload config files
print_status "Step 1: Uploading config files to S3..."
aws s3 cp "$CONFIG_DIR/" "s3://$S3_BUCKET/$S3_PREFIX/config/" --recursive --region "$AWS_REGION"

if [ $? -ne 0 ]; then
    print_error "Failed to upload config files"
    exit 1
fi
print_success "Config files uploaded"

# Step 2: Create input directory structure
print_status "Step 2: Creating input directory structure..."
aws s3api put-object --bucket "$S3_BUCKET" --key "$S3_PREFIX/input/$RESOLUTION/" --region "$AWS_REGION" > /dev/null 2>&1 || true

# Step 3: Upload input files
print_status "Step 3: Uploading input files to S3..."

# Upload social file
print_status "Uploading social file..."
aws s3 cp "$SOCIAL_FILE" "s3://$S3_BUCKET/$S3_PREFIX/input/$RESOLUTION/socioeconomico_$RESOLUTION.tif" --region "$AWS_REGION"
if [ $? -ne 0 ]; then
    print_error "Failed to upload social file"
    exit 1
fi
print_success "Social file uploaded"

# Upload environmental file
print_status "Uploading environmental file..."
aws s3 cp "$ENVIRONMENTAL_FILE" "s3://$S3_BUCKET/$S3_PREFIX/input/$RESOLUTION/ambiental_$RESOLUTION.tif" --region "$AWS_REGION"
if [ $? -ne 0 ]; then
    print_error "Failed to upload environmental file"
    exit 1
fi
print_success "Environmental file uploaded"

# Upload strategic file
print_status "Uploading strategic file..."
aws s3 cp "$STRATEGIC_FILE" "s3://$S3_BUCKET/$S3_PREFIX/input/$RESOLUTION/estrategico_$RESOLUTION.tif" --region "$AWS_REGION"
if [ $? -ne 0 ]; then
    print_error "Failed to upload strategic file"
    exit 1
fi
print_success "Strategic file uploaded"

# Step 4: Create output directory
print_status "Step 4: Creating output directory..."
aws s3api put-object --bucket "$S3_BUCKET" --key "$S3_PREFIX/output/" --region "$AWS_REGION" > /dev/null 2>&1 || true

# Step 5: Verify uploads
print_status "Step 5: Verifying uploads..."

# List uploaded files
print_status "Uploaded files:"
aws s3 ls "s3://$S3_BUCKET/$S3_PREFIX/config/" --region "$AWS_REGION" --recursive
aws s3 ls "s3://$S3_BUCKET/$S3_PREFIX/input/$RESOLUTION/" --region "$AWS_REGION" --recursive

# Step 6: Create upload summary
print_status "Step 6: Creating upload summary..."

cat > upload_summary.md << EOF
# S3 Input Files Upload Summary

## Upload Information
- **Date**: $(date)
- **AWS Region**: $AWS_REGION
- **S3 Bucket**: $S3_BUCKET
- **S3 Prefix**: $S3_PREFIX
- **Resolution**: $RESOLUTION

## Files Uploaded

### Config Files
- **Location**: s3://$S3_BUCKET/$S3_PREFIX/config/
- **Files**: All JSON configuration files

### Input Files
- **Location**: s3://$S3_BUCKET/$S3_PREFIX/input/$RESOLUTION/
- **Files**:
  - socioeconomico_$RESOLUTION.tif
  - ambiental_$RESOLUTION.tif
  - estrategico_$RESOLUTION.tif

### Output Directory
- **Location**: s3://$S3_BUCKET/$S3_PREFIX/output/
- **Status**: Created (ready for job output)

## S3 Structure
\`\`\`
s3://$S3_BUCKET/$S3_PREFIX/
├── config/
│   ├── config_max.json
│   ├── config_median.json
│   └── config_minimum.json
├── input/
│   └── $RESOLUTION/
│       ├── socioeconomico_$RESOLUTION.tif
│       ├── ambiental_$RESOLUTION.tif
│       └── estrategico_$RESOLUTION.tif
└── output/
    └── (will contain job output)
\`\`\`

## Expected Job Output
- **File**: output_${RESOLUTION}_config_max_glue.tif
- **Location**: s3://$S3_BUCKET/$S3_PREFIX/output/

## Next Steps
1. Deploy the Glue job using: \`./deploy_glue_tifffile.sh $AWS_REGION\`
2. Run the job using: \`aws glue start-job-run --job-name fis-tifffile-processor --region $AWS_REGION\`
3. Monitor the job in AWS Glue console
4. Download results from S3 output directory

## Verification Commands

### List uploaded files
\`\`\`bash
aws s3 ls s3://$S3_BUCKET/$S3_PREFIX/ --recursive --region $AWS_REGION
\`\`\`

### Check specific file
\`\`\`bash
aws s3 ls s3://$S3_BUCKET/$S3_PREFIX/input/$RESOLUTION/socioeconomico_$RESOLUTION.tif --region $AWS_REGION
\`\`\`

EOF

print_success "Upload summary created: upload_summary.md"

print_success "🎉 Input files uploaded to S3 successfully!"
print_status "Summary:"
echo "  Config files: s3://$S3_BUCKET/$S3_PREFIX/config/"
echo "  Input files: s3://$S3_BUCKET/$S3_PREFIX/input/$RESOLUTION/"
echo "  Output directory: s3://$S3_BUCKET/$S3_PREFIX/output/"
echo ""
print_status "Next steps:"
echo "  1. Deploy the Glue job: ./deploy_glue_tifffile.sh $AWS_REGION"
echo "  2. Run the job: aws glue start-job-run --job-name fis-tifffile-processor --region $AWS_REGION"
echo "  3. Monitor in AWS Glue console"
echo "  4. Check output at: s3://$S3_BUCKET/$S3_PREFIX/output/"

print_status "Upload summary saved to: upload_summary.md" 