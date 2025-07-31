# EMR Step: Run All FIS Models

This guide explains how to submit an EMR step to run all fuzzy inference system (FIS) models on your EMR cluster.

## Quick Start

### **Submit the EMR Step:**

```bash
# Use default parameters (your cluster and files)
./scripts/submit_all_fis_models_step.sh

# Or specify custom parameters
./scripts/submit_all_fis_models_step.sh \
    <EMR-CLUSTER-ID> \
    s3://<AWS-BUCKET>/unifile_test/so300m.in \
    s3://<AWS-BUCKET>/unifile_test/e300m.in \
    s3://<AWS-BUCKET>/unifile_test/s300m.in \
    s3://<AWS-BUCKET>/unifile_test/result_300m
```

## What the Step Does

The EMR step will:

1. **Navigate to your code directory** (`/mnt/destark`)
2. **Pull latest changes** from git
3. **Run all 7 FIS models** sequentially:
   - Maximum aggregation (`config_max.json`)
   - Minimum aggregation (`config_minimum.json`)
   - Median aggregation (`config_median.json`)
   - Mode aggregation (`config_mode.json`)
   - Round up (`config_round_up.json`)
   - Round down (`config_round_down.json`)
   - Default (`raster_fis_config.json`)
4. **Generate 7 output files** in S3

## Expected Output Files

After completion, you'll have these files in S3:

```
s3://<AWS-BUCKET>/unifile_test/result_300m_max.tif
s3://<AWS-BUCKET>/unifile_test/result_300m_minimum.tif
s3://<AWS-BUCKET>/unifile_test/result_300m_median.tif
s3://<AWS-BUCKET>/unifile_test/result_300m_mode.tif
s3://<AWS-BUCKET>/unifile_test/result_300m_round_up.tif
s3://<AWS-BUCKET>/unifile_test/result_300m_round_down.tif
s3://<AWS-BUCKET>/unifile_test/result_300m_default.tif
```

## Monitoring

### **Command Line Monitoring:**
The script will automatically monitor the step progress and show status updates every 30 seconds.

### **EMR Console Monitoring:**
You can also monitor in the AWS EMR Console:
```
https://console.aws.amazon.com/elasticmapreduce/home?region=us-east-1#/clusterDetails/<EMR-CLUSTER-ID>
```

### **Step Status:**
- **PENDING**: Step is queued
- **RUNNING**: Step is executing
- **COMPLETED**: All models processed successfully
- **FAILED**: Step failed (check logs)
- **CANCELLED**: Step was cancelled

## Estimated Processing Time

For your dataset (14,479 × 15,187 pixels):
- **Total time**: 3-7 hours (depending on cluster size)
- **Per model**: 30-60 minutes
- **Sequential processing**: Models run one after another

## Troubleshooting

### **Step Fails:**
1. Check EMR Console for detailed logs
2. Verify S3 file paths are correct
3. Ensure cluster has enough resources
4. Check if config files exist in S3

### **Long Processing Time:**
- Monitor cluster resource usage
- Check if other steps are running
- Consider increasing cluster size

### **S3 Access Issues:**
- Verify IAM roles have S3 permissions
- Check S3 bucket policies
- Ensure file paths are correct

## Manual Step Submission

If you prefer to submit manually:

```bash
# Create step configuration
cat > step.json << 'EOF'
[
  {
    "Name": "Run All FIS Models - Raster Fuzzy Inference",
    "ActionOnFailure": "CONTINUE",
    "HadoopJarStep": {
      "Jar": "command-runner.jar",
      "Args": [
        "bash",
        "-c",
        "cd /mnt/destark && git pull origin main && chmod +x scripts/run_all_fis_models_emr.sh && ./scripts/run_all_fis_models_emr.sh s3://<AWS-BUCKET>/unifile_test/so300m.in s3://<AWS-BUCKET>/unifile_test/e300m.in s3://<AWS-BUCKET>/unifile_test/s300m.in s3://<AWS-BUCKET>/unifile_test/result_300m"
      ]
    }
  }
]
EOF

# Submit step
aws emr add-steps --cluster-id <EMR-CLUSTER-ID> --steps file://step.json
```

## Cluster Requirements

- **Status**: Must be in RUNNING state
- **Memory**: At least 8GB per node recommended
- **Storage**: Sufficient space for temporary files
- **IAM**: Proper S3 access permissions

## Cost Considerations

- **Processing time**: 3-7 hours of cluster usage
- **Storage**: 7 × input file size for outputs
- **Data transfer**: S3 read/write costs
- **Consider**: Use spot instances to reduce costs 