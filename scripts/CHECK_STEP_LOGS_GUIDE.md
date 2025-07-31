# How to Check EMR Step Logs

## Step ID: s-071992378ESOU7F4NSB
## Cluster: <EMR-CLUSTER-ID>
## Region: us-east-2

## Option 1: AWS Console (Recommended)

### Steps to Check Logs:
1. **Go to EMR Console**
   - Open AWS Console
   - Navigate to EMR service in us-east-2 region
   - Find cluster `<EMR-CLUSTER-ID>`

2. **Find the Step**
   - Click on the cluster
   - Go to "Steps" tab
   - Look for step ID `s-071992378ESOU7F4NSB`
   - Click on the step name

3. **View Logs**
   - Look for "Logs" section
   - Click on "stdout" or "stderr" to see output
   - Check "Step Details" for execution time and status

## Option 2: SSH to EMR Cluster

If you have SSH access to the cluster:

```bash
# SSH to the master node
aws emr ssh --cluster-id <EMR-CLUSTER-ID> --key-pair-file your-key.pem

# Check step logs
sudo cat /var/log/hadoop/steps/s-071992378ESOU7F4NSB/stdout
sudo cat /var/log/hadoop/steps/s-071992378ESOU7F4NSB/stderr
```

## Option 3: Fix IAM Permissions

Add this policy to your user `Yuri_Sa`:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "elasticmapreduce:DescribeStep",
                "elasticmapreduce:ListSteps"
            ],
            "Resource": "*"
        }
    ]
}
```

Then run:
```bash
./scripts/check_step_logs.sh <EMR-CLUSTER-ID> s-071992378ESOU7F4NSB us-east-2
```

## Why the Step Finished Quickly

If the step finished very quickly, it might be because:

1. **Simple Test Script** - The step might have just run a basic test
2. **No Heavy Processing** - If it was just creating files/imports
3. **Error or Exception** - Check stderr logs for errors
4. **Expected Behavior** - Some setup steps are meant to be quick

## What to Look For in Logs:

### ✅ Success Indicators:
- "FIS Models Ready for Processing!"
- "Real FIS models deployed and tested successfully!"
- No error messages

### ❌ Failure Indicators:
- Import errors
- Permission denied messages
- File not found errors
- Python exceptions

## Next Steps:

1. **Check the logs** using AWS Console
2. **Verify the step completed successfully**
3. **If successful**, you can proceed with your actual FIS processing
4. **If failed**, check the error and fix the issue 