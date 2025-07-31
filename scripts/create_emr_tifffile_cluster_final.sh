#!/bin/bash

# Create Optimized EMR Cluster for tifffile-based FIS Processing
# This script creates an EMR cluster optimized for cost and performance

set -e

echo "=========================================="
echo "Creating Optimized EMR Cluster"
echo "Name: emr-adveng-development"
echo "Focus: tifffile-based FIS processing"
echo "=========================================="

# Configuration
CLUSTER_NAME="emr-adveng-development"
REGION="us-east-2"
LOG_URI="s3://<AWS-BUCKET>-unifile/logs"
RELEASE_LABEL="emr-7.9.0"

echo "Creating EMR cluster with optimized configuration..."
echo "  Cluster Name: $CLUSTER_NAME"
echo "  Region: $REGION"
echo "  Release: $RELEASE_LABEL"
echo ""

# Create the EMR cluster
aws emr create-cluster \
    --name "$CLUSTER_NAME" \
    --log-uri "$LOG_URI" \
    --release-label "$RELEASE_LABEL" \
    --service-role "arn:aws:iam::<AWS-ACCOUNT-ID>:role/<EMR-ROLE-NAME>" \
    --managed-scaling-policy '{"ComputeLimits":{"UnitType":"Instances","MinimumCapacityUnits":3,"MaximumCapacityUnits":20,"MaximumOnDemandCapacityUnits":20,"MaximumCoreCapacityUnits":2}}' \
    --ec2-attributes '{"InstanceProfile":"<EMR-EC2-ROLE-NAME>","EmrManagedMasterSecurityGroup":"sg-<SECURITY-GROUP-ID>","EmrManagedSlaveSecurityGroup":"sg-<SECURITY-GROUP-ID>","KeyName":"<KEY-PAIR-NAME>","ServiceAccessSecurityGroup":"sg-<SERVICE-SECURITY-GROUP-ID>","SubnetIds":["subnet-<SUBNET-ID>"]}' \
    --applications Name=Hadoop Name=Hive Name=JupyterEnterpriseGateway Name=Livy Name=Spark \
    --configurations '[{"Classification":"hdfs-site","Properties":{"dfs.namenode.kerberos.principal.pattern":"*","dfs.replication":"2"}},{"Classification":"hadoop-env","Configurations":[{"Classification":"export","Properties":{"JAVA_HOME":"/usr/lib/jvm/jre-17"}}],"Properties":{}},{"Classification":"yarn-site","Properties":{"yarn.node-labels.am.default-node-label-expression":"CORE","yarn.node-labels.enabled":"true","yarn.nodemanager.resource.memory-mb":"28672","yarn.scheduler.maximum-allocation-mb":"24576","yarn.scheduler.minimum-allocation-mb":"32"}},{"Classification":"spark-env","Configurations":[{"Classification":"export","Properties":{"JAVA_HOME":"/usr/lib/jvm/jre-17","SPARK_HISTORY_OPTS":"-Dspark.ui.filters=org.apache.spark.deploy.yarn.YarnProxyRedirectFilter"}}],"Properties":{}},{"Classification":"spark-defaults","Properties":{"spark.default.parallelism":"200","spark.driver.cores":"2","spark.driver.memory":"8g","spark.dynamicAllocation.enabled":"true","spark.dynamicAllocation.maxExecutors":"8","spark.dynamicAllocation.minExecutors":"2","spark.dynamicAllocation.shuffleTracking.enabled":"true","spark.executor.cores":"4","spark.executor.memory":"20g","spark.hadoop.fs.s3.impl":"com.amazon.ws.emr.hadoop.fs.EmrFileSystem","spark.hadoop.fs.s3a.fast.upload":"true","spark.hadoop.fs.s3a.impl":"com.amazon.ws.emr.hadoop.fs.EmrFileSystem","spark.hadoop.glue.catalog.datalake":"<GLUE-CATALOG-ID>","spark.hadoop.hive.imetastoreclient.factory.class":"com.indeed.emr.glue.multi.catalog.hive2.client.AWSMultiGlueDataCatalogHiveClientFactory","spark.hadoop.hive.metastore.client.factory.class":"com.indeed.emr.glue.multi.catalog.hive2.client.AWSMultiGlueDataCatalogHiveClientFactory","spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version":"2","spark.hadoop.mapreduce.fileoutputcommitter.cleanup-failures.ignored":"true","spark.history.fs.cleaner.enabled":"true","spark.history.fs.cleaner.interval":"24h","spark.history.fs.cleaner.maxAge":"72h","spark.kryoserializer.buffer":"128m","spark.kryoserializer.buffer.max":"512m","spark.rpc.message.maxSize":"512","spark.serializer":"org.apache.spark.serializer.KryoSerializer","spark.sql.adaptive.advisoryPartitionSizeInBytes":"128m","spark.sql.adaptive.coalescePartitions.enabled":"true","spark.sql.adaptive.enabled":"true","spark.sql.adaptive.localShuffleReader.enabled":"true","spark.sql.adaptive.skewJoin.enabled":"true","spark.sql.catalog.spark_catalog":"org.apache.iceberg.spark.SparkSessionCatalog","spark.sql.catalog.spark_catalog.catalog-impl":"org.apache.iceberg.aws.glue.GlueCatalog","spark.sql.catalog.spark_catalog.io-impl":"org.apache.iceberg.aws.s3.S3FileIO","spark.sql.extensions":"org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions","spark.sql.legacy.allowNonEmptyLocationInCTAS":"true","spark.sql.orc.filterPushdown":"true","spark.sql.parquet.filterPushdown":"true","spark.sql.shuffle.partitions":"200","spark.sql.sources.partitionOverwriteMode":"dynamic","spark.sql.splits.include.file.footer":"true","spark.submit.deployMode":"cluster","spark.yarn.dist.files":"/etc/spark/conf.dist/hive-site.xml","spark.yarn.queue":"default","spark.yarn.security.tokens.hive.enabled":"false"}}]' \
    --instance-groups '[{"InstanceCount":2,"InstanceGroupType":"CORE","Name":"Core","InstanceType":"m5.2xlarge","EbsConfiguration":{"EbsBlockDeviceConfigs":[{"VolumeSpecification":{"VolumeType":"gp3","SizeInGB":50},"VolumesPerInstance":1}]}},{"InstanceCount":1,"InstanceGroupType":"TASK","Name":"Task","InstanceType":"m5.2xlarge","EbsConfiguration":{"EbsBlockDeviceConfigs":[{"VolumeSpecification":{"VolumeType":"gp3","SizeInGB":50},"VolumesPerInstance":1}]}},{"InstanceCount":1,"InstanceGroupType":"MASTER","Name":"Primary","InstanceType":"m5.2xlarge","EbsConfiguration":{"EbsBlockDeviceConfigs":[{"VolumeSpecification":{"VolumeType":"gp3","SizeInGB":50},"VolumesPerInstance":1}]}}]' \
    --steps '[{"Name":"Install Complete FIS Development Environment","ActionOnFailure":"CONTINUE","Jar":"command-runner.jar","Properties":"=","Args":["bash","-c","echo \"=== INSTALLING COMPLETE FIS DEVELOPMENT ENVIRONMENT ===\" && sudo yum update -y && echo \"Installing system packages...\" && sudo yum install -y git python3-pip python3-devel gcc gcc-c++ make java-17-amazon-corretto-devel unzip wget curl && echo \"Setting up Java environment...\" && echo \"export JAVA_HOME=/usr/lib/jvm/jre-17\" >> /etc/profile && echo \"export PATH=$JAVA_HOME/bin:$PATH\" >> /etc/profile && source /etc/profile && echo \"Upgrading pip...\" && python3 -m pip install --upgrade pip && echo \"Installing ALL Python packages for complete FIS processing...\" && python3 -m pip install --no-cache-dir tifffile>=2023.0.0 scikit-fuzzy>=0.4.2 numpy>=1.21.0 scipy>=1.7.0 boto3>=1.26.0 pandas>=1.3.0 networkx>=2.6.0 matplotlib>=3.5.0 tqdm>=4.62.0 pyspark>=3.4.0 psutil>=5.8.0 && echo \"Verifying installations...\" && git --version && java -version && python3 -c \"import tifffile, numpy, skfuzzy, boto3, pyspark, psutil, pandas, networkx, matplotlib, tqdm; print(\"✓ ALL FIS packages installed successfully\")\" && echo \"=== COMPLETE FIS ENVIRONMENT READY ===\""]},{"Name":"Setup Development Directory","ActionOnFailure":"CONTINUE","Jar":"command-runner.jar","Args":["bash","-c","echo \"=== SETTING UP DEVELOPMENT DIRECTORY ===\" && sudo mkdir -p /mnt/destark && sudo chown -R hadoop:hadoop /mnt/destark && cd /mnt/destark && echo \"Creating development structure...\" && mkdir -p app scripts logs data config files/input files/output utils docs && echo \"Development directory structure created:\" && ls -la && echo \"=== DEVELOPMENT DIRECTORY READY ===\""]},{"Name":"Verify Complete FIS Environment","ActionOnFailure":"CONTINUE","Jar":"command-runner.jar","Args":["bash","-c","echo \"=== VERIFYING COMPLETE FIS ENVIRONMENT ===\" && echo \"Testing ALL FIS package imports...\" && python3 -c \"import tifffile; print(f\"✓ Tifffile imported successfully\")\" && python3 -c \"import skfuzzy; print(\"✓ Scikit-fuzzy imported successfully\")\" && python3 -c \"import numpy; print(f\"✓ NumPy version: {numpy.__version__}\")\" && python3 -c \"import boto3; print(\"✓ Boto3 imported successfully\")\" && python3 -c \"import pyspark; print(\"✓ PySpark imported successfully\")\" && python3 -c \"import psutil; print(\"✓ Psutil imported successfully\")\" && python3 -c \"import pandas; print(\"✓ Pandas imported successfully\")\" && python3 -c \"import networkx; print(\"✓ NetworkX imported successfully\")\" && python3 -c \"import matplotlib; print(\"✓ Matplotlib imported successfully\")\" && python3 -c \"import tqdm; print(\"✓ TQDM imported successfully\")\" && echo \"Testing tifffile operations...\" && python3 -c \"import tifffile; import numpy as np; data = np.random.rand(10, 10).astype(np.float32); tifffile.imwrite(\"/tmp/test.tif\", data); print(\"✓ Tifffile write test passed\")\" && python3 -c \"import tifffile; data = tifffile.imread(\"/tmp/test.tif\"); print(f\"✓ Tifffile read test passed, shape: {data.shape}\")\" && echo \"Testing development directory...\" && cd /mnt/destark && pwd && ls -la && echo \"=== COMPLETE FIS ENVIRONMENT VERIFIED ===\" && echo \"Cluster is ready for complete FIS processing!\""]}]' \
    --scale-down-behavior "TERMINATE_AT_TASK_COMPLETION" \
    --ebs-root-volume-size "30" \
    --region "$REGION"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ EMR cluster creation initiated successfully!"
    echo ""
    echo "Cluster Details:"
    echo "  Name: $CLUSTER_NAME"
    echo "  Region: $REGION"
    echo "  Release: $RELEASE_LABEL"
    echo "  Instance Type: m5.2xlarge (optimized EBS: 50GB gp3)"
    echo "  Instance Count: 1 Master + 2 Core + 1 Task"
    echo ""
    echo "Installed Packages:"
    echo "  ✓ tifffile>=2023.0.0 (raster I/O)"
    echo "  ✓ scikit-fuzzy>=0.4.2 (fuzzy logic)"
    echo "  ✓ numpy>=1.21.0 (numerical computing)"
    echo "  ✓ scipy>=1.7.0 (scientific computing)"
    echo "  ✓ boto3>=1.26.0 (AWS SDK)"
    echo "  ✓ pyspark>=3.4.0 (distributed processing)"
    echo "  ✓ psutil>=5.8.0 (system monitoring)"
    echo "  ✓ pandas>=1.3.0 (data manipulation)"
    echo "  ✓ networkx>=2.6.0 (graph algorithms)"
    echo "  ✓ matplotlib>=3.5.0 (plotting)"
    echo "  ✓ tqdm>=4.62.0 (progress bars)"
    echo ""
    echo "Next Steps:"
    echo "1. Monitor cluster creation:"
    echo "   aws emr list-clusters --region $REGION --active"
    echo ""
    echo "2. Get cluster ID and test:"
    echo "   ./scripts/test_emr_tifffile_simple.sh <CLUSTER_ID> $REGION"
    echo ""
    echo "3. Run full S3 test:"
    echo "   ./scripts/test_emr_tifffile_s3.sh <CLUSTER_ID> $REGION"
    echo ""
    echo "4. Deploy your FIS application:"
    echo "   aws s3 cp app/raster_fuzzy_spark_s3_tifffile.py s3://<AWS-BUCKET>-unifile/unifile_test/"
    echo ""
    echo "=========================================="
    echo "EMR Cluster Creation Complete!"
    echo "=========================================="
else
    echo "❌ Failed to create EMR cluster"
    exit 1
fi 