# EMR Step Monitoring Guide

## Current Situation
You have EMR cluster `<EMR-CLUSTER-ID>` in `us-east-2` but CLI access is limited due to IAM permissions.

## How to Check Step Results

### Option 1: AWS Console (Recommended)
1. **Go to EMR Console**
   - Open AWS Console
   - Navigate to EMR service in us-east-2 region
   - Find cluster `<EMR-CLUSTER-ID>`

2. **Check Steps Tab**
   - Click on the cluster
   - Go to "Steps" tab
   - View status of each step:
     - ✅ COMPLETED
     - ❌ FAILED  
     - 🔄 RUNNING
     - ⏳ PENDING

3. **View Step Details**
   - Click on any step to see:
     - Start/End times
     - Error messages
     - Logs

### Option 2: Fix AWS CLI Permissions

#### Required IAM Permissions
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "elasticmapreduce:ListClusters",
                "elasticmapreduce:DescribeCluster", 
                "elasticmapreduce:ListSteps",
                "elasticmapreduce:DescribeStep",
                "elasticmapreduce:AddSteps"
            ],
            "Resource": "*"
        }
    ]
}
```

#### Steps to Add Permissions
1. Go to IAM Console
2. Find user `Yuri_Sa`
3. Add the above policy or attach `AmazonElasticMapReduceFullAccess`

### Option 3: Use AWS CLI with Different Profile
```bash
# Check if you have other profiles
aws configure list-profiles

# Use a different profile if available
aws emr list-steps --cluster-id <EMR-CLUSTER-ID> --region us-east-2 --profile <profile-name>
```

## Expected Step Results

Based on the fixed configuration, you should see these steps:

1. **Install System Dependencies** - Should complete successfully
2. **Create Repository Structure** - Should complete successfully  
3. **Create Requirements Files** - Should complete successfully
4. **Install Python Dependencies** - Should complete successfully
5. **Create Simple FIS Script** - Should complete successfully
6. **Run FIS Models Test** - Should complete successfully

## Common Issues and Solutions

### If Steps Failed:
1. **Check error messages** in the console
2. **Look at logs** for specific failure reasons
3. **Verify cluster state** - ensure it's RUNNING
4. **Check instance types** - ensure sufficient resources

### If Cluster Terminated:
1. **Check termination reason** in cluster details
2. **Verify billing** - ensure account has sufficient funds
3. **Check service limits** - ensure not hitting EMR limits

## Quick Status Check Commands

Once permissions are fixed:
```bash
# Quick status
./scripts/quick_check_emr_steps.sh <EMR-CLUSTER-ID> us-east-2

# Detailed results
./scripts/check_emr_step_results.sh <EMR-CLUSTER-ID> us-east-2
```

## Next Steps

1. **Check console first** - Most reliable method
2. **Fix IAM permissions** if needed
3. **Re-run steps** if any failed using the fixed configuration
4. **Monitor logs** for any issues 