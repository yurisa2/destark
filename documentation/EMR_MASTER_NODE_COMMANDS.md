# EMR Master Node Commands

## Quick Start (Run these on EMR master node)

```bash
# 1. Navigate to project directory
cd /mnt/destark

# 2. Create logs directory (fixes nohup issue)
mkdir -p logs

# 3. Run the optimized job
./scripts/run_all_fis_models_spark_submit_optimized.sh \
  s3://<AWS-BUCKET>/unifile_test/so300m.in \
  s3://<AWS-BUCKET>/unifile_test/e300m.in \
  s3://<AWS-BUCKET>/unifile_test/s300m.in \
  s3://<AWS-BUCKET>/unifile_test/result_300m
```

## Monitor Jobs

```bash
# Check YARN applications
yarn application -list

# Monitor specific model logs
tail -f logs/round_up_nohup.log
tail -f logs/round_down_nohup.log
tail -f logs/max_nohup.log
tail -f logs/minimum_nohup.log
tail -f logs/median_nohup.log
tail -f logs/mode_nohup.log

# Check cluster resources
yarn node -list
yarn queue -status default
```

## Check Job Status

```bash
# View application details
yarn application -status <application_id>

# View detailed logs
yarn logs -applicationId <application_id>

# Kill job if needed
yarn application -kill <application_id>
```

## Expected Output Files

After completion, check S3 for:
- `s3://<AWS-BUCKET>/unifile_test/result_300m_max.tif`
- `s3://<AWS-BUCKET>/unifile_test/result_300m_median.tif`
- `s3://<AWS-BUCKET>/unifile_test/result_300m_minimum.tif`
- `s3://<AWS-BUCKET>/unifile_test/result_300m_mode.tif`
- `s3://<AWS-BUCKET>/unifile_test/result_300m_round_up.tif`
- `s3://<AWS-BUCKET>/unifile_test/result_300m_round_down.tif`

## Troubleshooting

```bash
# Check if logs directory exists
ls -la logs/

# Check running processes
ps aux | grep spark-submit

# Check Java version
java -version

# Check Spark version
spark-submit --version
``` 