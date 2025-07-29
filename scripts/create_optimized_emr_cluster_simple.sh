#!/bin/bash

# Optimized EMR Cluster for Raster Processing Workload
# This script creates an EMR cluster optimized for raster processing with Spark

set -e

echo "=== Creating Optimized EMR Cluster for Raster Processing ==="

# Configuration variables
CLUSTER_NAME="emr-raster-processing-optimized"
LOG_URI="s3://adveng-pipeline-unifile/logs"
RELEASE_LABEL="emr-7.9.0"
SERVICE_ROLE="arn:aws:iam::475136118191:role/Dpe-EMR-Role"
INSTANCE_PROFILE="Dpe-EMR-Ec2-Role"
SECURITY_GROUP="sg-08820671de936fff3"
SERVICE_ACCESS_SG="sg-08a6d24ab86f818d0"
SUBNET_ID="subnet-0f8d451af7c42c7ef"
KEY_NAME="DATA_INFRA_EMR_KEYPAIR"
REGION="us-east-2"

# Create temporary JSON files
echo "Creating configuration files..."

# Instance groups configuration
cat > /tmp/instance-groups.json << 'EOF'
[
  {
    "InstanceCount": 1,
    "InstanceGroupType": "MASTER",
    "Name": "Primary",
    "InstanceType": "m5.2xlarge",
    "EbsConfiguration": {
      "EbsBlockDeviceConfigs": [
        {
          "VolumeSpecification": {
            "VolumeType": "gp3",
            "SizeInGB": 100
          },
          "VolumesPerInstance": 2
        }
      ]
    }
  },
  {
    "InstanceCount": 2,
    "InstanceGroupType": "CORE",
    "Name": "Core",
    "InstanceType": "m5.2xlarge",
    "EbsConfiguration": {
      "EbsBlockDeviceConfigs": [
        {
          "VolumeSpecification": {
            "VolumeType": "gp3",
            "SizeInGB": 100
          },
          "VolumesPerInstance": 2
        }
      ]
    }
  },
  {
    "InstanceCount": 3,
    "InstanceGroupType": "TASK",
    "Name": "Task",
    "InstanceType": "m5.2xlarge",
    "EbsConfiguration": {
      "EbsBlockDeviceConfigs": [
        {
          "VolumeSpecification": {
            "VolumeType": "gp3",
            "SizeInGB": 100
          },
          "VolumesPerInstance": 2
        }
      ]
    }
  }
]
EOF

# Steps configuration
cat > /tmp/steps.json << 'EOF'
[
  {
    "Name": "Install System Dependencies and Git",
    "ActionOnFailure": "CONTINUE",
    "Jar": "command-runner.jar",
    "Args": [
      "bash",
      "-c",
      "yum update -y && yum install -y git python3-pip python3-devel gcc gcc-c++ make gdal-devel proj-devel geos-devel hdf5-devel netcdf-devel libspatialindex-devel java-17-amazon-corretto-devel && echo 'export JAVA_HOME=/usr/lib/jvm/jre-17' >> /etc/profile && echo 'export PATH=$JAVA_HOME/bin:$PATH' >> /etc/profile && git --version && java -version"
    ]
  },
  {
    "Name": "Install Python Packages for Raster Processing",
    "ActionOnFailure": "CONTINUE",
    "Jar": "command-runner.jar",
    "Args": [
      "bash",
      "-c",
      "python3 -m pip install --upgrade pip && python3 -m pip install numpy>=1.21.0 scipy>=1.7.0 pandas>=1.3.0 python-dateutil>=2.8.0 rasterio>=1.3.0 fiona>=1.8.0 shapely>=1.8.0 pyproj>=3.2.0 scikit-fuzzy>=0.4.2 boto3>=1.26.0 s3fs>=2022.11.0 matplotlib>=3.5.0 seaborn>=0.11.0 tqdm>=4.62.0 && python3 -c \"import numpy, rasterio, skfuzzy, boto3, s3fs, dateutil; print('✓ All raster processing packages installed successfully')\""
    ]
  },
  {
    "Name": "Clone Repository and Setup Environment",
    "ActionOnFailure": "CONTINUE",
    "Jar": "command-runner.jar",
    "Args": [
      "bash",
      "-c",
      "cd /mnt && git clone https://github.com/your-username/destark.git || echo 'Repository already exists or clone failed' && cd /mnt/destark && ls -la && echo '✓ Repository setup complete'"
    ]
  },
  {
    "Name": "Verify Raster Processing Environment",
    "ActionOnFailure": "CONTINUE",
    "Jar": "command-runner.jar",
    "Args": [
      "bash",
      "-c",
      "python3 -c \"import rasterio; print(f'✓ Rasterio version: {rasterio.__version__}')\" && python3 -c \"import skfuzzy; print('✓ Scikit-fuzzy imported successfully')\" && python3 -c \"import numpy; print(f'✓ NumPy version: {numpy.__version__}')\" && python3 -c \"import boto3; print('✓ Boto3 imported successfully')\" && echo '✓ Raster processing environment ready!'"
    ]
  }
]
EOF

# Configurations
cat > /tmp/configurations.json << 'EOF'
[
  {
    "Classification": "hdfs-site",
    "Properties": {
      "dfs.namenode.kerberos.principal.pattern": "*",
      "dfs.replication": "2"
    }
  },
  {
    "Classification": "hadoop-env",
    "Configurations": [
      {
        "Classification": "export",
        "Properties": {
          "JAVA_HOME": "/usr/lib/jvm/jre-17"
        }
      }
    ],
    "Properties": {}
  },
  {
    "Classification": "yarn-site",
    "Properties": {
      "yarn.node-labels.am.default-node-label-expression": "CORE",
      "yarn.node-labels.enabled": "true",
      "yarn.scheduler.minimum-allocation-mb": "32",
      "yarn.nodemanager.resource.memory-mb": "28672",
      "yarn.scheduler.maximum-allocation-mb": "24576"
    }
  },
  {
    "Classification": "spark-env",
    "Configurations": [
      {
        "Classification": "export",
        "Properties": {
          "JAVA_HOME": "/usr/lib/jvm/jre-17",
          "SPARK_HISTORY_OPTS": "-Dspark.ui.filters=org.apache.spark.deploy.yarn.YarnProxyRedirectFilter"
        }
      }
    ],
    "Properties": {}
  },
  {
    "Classification": "spark-defaults",
    "Properties": {
      "spark.driver.memory": "8g",
      "spark.driver.cores": "2",
      "spark.executor.memory": "20g",
      "spark.executor.cores": "4",
      "spark.dynamicAllocation.enabled": "true",
      "spark.dynamicAllocation.maxExecutors": "8",
      "spark.dynamicAllocation.minExecutors": "2",
      "spark.dynamicAllocation.shuffleTracking.enabled": "true",
      "spark.sql.shuffle.partitions": "200",
      "spark.default.parallelism": "200",
      "spark.sql.adaptive.enabled": "true",
      "spark.sql.adaptive.coalescePartitions.enabled": "true",
      "spark.sql.adaptive.skewJoin.enabled": "true",
      "spark.sql.adaptive.localShuffleReader.enabled": "true",
      "spark.sql.adaptive.advisoryPartitionSizeInBytes": "128m",
      "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max": "512m",
      "spark.kryoserializer.buffer": "128m",
      "spark.rpc.message.maxSize": "512",
      "spark.hadoop.aws.catalog.credentials.provider.factory.class": "com.indeed.emr.glue.multi.catalog.hive2.core.metastore.CredentialsProviderFactory",
      "spark.hadoop.fs.s3.customAWSCredentialsProvider": "com.indeed.spark.hivesupport.HadoopConfEMRFSCredentialsProvider",
      "spark.hadoop.fs.s3.impl": "com.amazon.ws.emr.hadoop.fs.EmrFileSystem",
      "spark.hadoop.fs.s3a.fast.upload": "true",
      "spark.hadoop.fs.s3a.impl": "com.amazon.ws.emr.hadoop.fs.EmrFileSystem",
      "spark.hadoop.glue.catalog.datalake": "052300729316",
      "spark.hadoop.hive.imetastoreclient.factory.class": "com.indeed.emr.glue.multi.catalog.hive2.client.AWSMultiGlueDataCatalogHiveClientFactory",
      "spark.hadoop.hive.metastore.client.factory.class": "com.indeed.emr.glue.multi.catalog.hive2.client.AWSMultiGlueDataCatalogHiveClientFactory",
      "spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version": "2",
      "spark.hadoop.mapreduce.fileoutputcommitter.cleanup-failures.ignored": "true",
      "spark.history.fs.cleaner.enabled": "true",
      "spark.history.fs.cleaner.interval": "24h",
      "spark.history.fs.cleaner.maxAge": "72h",
      "spark.sql.catalog.spark_catalog": "org.apache.iceberg.spark.SparkSessionCatalog",
      "spark.sql.catalog.spark_catalog.catalog-impl": "org.apache.iceberg.aws.glue.GlueCatalog",
      "spark.sql.catalog.spark_catalog.io-impl": "org.apache.iceberg.aws.s3.S3FileIO",
      "spark.sql.extensions": "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions",
      "spark.sql.legacy.allowNonEmptyLocationInCTAS": "true",
      "spark.sql.orc.filterPushdown": "true",
      "spark.sql.parquet.filterPushdown": "true",
      "spark.sql.sources.partitionOverwriteMode": "dynamic",
      "spark.sql.splits.include.file.footer": "true",
      "spark.submit.deployMode": "cluster",
      "spark.yarn.dist.files": "/etc/spark/conf.dist/hive-site.xml",
      "spark.yarn.queue": "default",
      "spark.yarn.security.tokens.hive.enabled": "false"
    }
  }
]
EOF

# Create the cluster
echo "Creating EMR cluster: $CLUSTER_NAME"
echo "Log URI: $LOG_URI"

aws emr create-cluster \
  --name "$CLUSTER_NAME" \
  --log-uri "$LOG_URI" \
  --release-label "$RELEASE_LABEL" \
  --service-role "$SERVICE_ROLE" \
  --managed-scaling-policy '{"ComputeLimits":{"UnitType":"Instances","MinimumCapacityUnits":3,"MaximumCapacityUnits":20,"MaximumOnDemandCapacityUnits":20,"MaximumCoreCapacityUnits":2}}' \
  --ec2-attributes "{\"InstanceProfile\":\"$INSTANCE_PROFILE\",\"EmrManagedMasterSecurityGroup\":\"$SECURITY_GROUP\",\"EmrManagedSlaveSecurityGroup\":\"$SECURITY_GROUP\",\"KeyName\":\"$KEY_NAME\",\"ServiceAccessSecurityGroup\":\"$SERVICE_ACCESS_SG\",\"SubnetIds\":[\"$SUBNET_ID\"]}" \
  --applications Name=Hadoop Name=Hive Name=JupyterEnterpriseGateway Name=Livy Name=Spark \
  --configurations file:///tmp/configurations.json \
  --instance-groups file:///tmp/instance-groups.json \
  --steps file:///tmp/steps.json \
  --scale-down-behavior "TERMINATE_AT_TASK_COMPLETION" \
  --ebs-root-volume-size "30" \
  --region "$REGION" \
  --output json

# Clean up temporary files
rm -f /tmp/instance-groups.json /tmp/steps.json /tmp/configurations.json

echo ""
echo "=== EMR Cluster Creation Initiated ==="
echo "Cluster Name: $CLUSTER_NAME"
echo "Release Label: $RELEASE_LABEL"
echo "Log URI: $LOG_URI"
echo "Master Instance: m5.2xlarge"
echo "Core Instances: 2 x m5.2xlarge"
echo "Task Instances: 3 x m5.2xlarge"
echo "EBS Configuration: 2 x 100GB gp3 per instance"
echo ""
echo "Monitor cluster creation with:"
echo "aws emr describe-cluster --cluster-id <CLUSTER_ID> --region $REGION"
echo ""
echo "View logs in S3:"
echo "aws s3 ls $LOG_URI/"
echo ""
echo "Connect to master node with:"
echo "aws emr ssh --cluster-id <CLUSTER_ID> --key-pair-file <KEY_FILE> --region $REGION" 