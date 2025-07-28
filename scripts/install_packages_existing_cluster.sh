#!/bin/bash

# Install Python packages on existing EMR cluster
# This script connects to all nodes and installs packages

set -e

CLUSTER_ID="${1:-j-2XV0JESM33Y1E}"

if [ -z "$CLUSTER_ID" ]; then
    echo "Usage: $0 <cluster-id>"
    echo "Example: $0 j-2XV0JESM33Y1E"
    exit 1
fi

echo "=== Installing Python packages on EMR cluster $CLUSTER_ID ==="

# Get cluster information
echo "Getting cluster information..."
CLUSTER_INFO=$(aws emr describe-cluster --cluster-id "$CLUSTER_ID" --query 'Cluster.Status.State' --output text)

if [ "$CLUSTER_INFO" != "WAITING" ] && [ "$CLUSTER_INFO" != "RUNNING" ]; then
    echo "Error: Cluster is not in WAITING or RUNNING state. Current state: $CLUSTER_INFO"
    exit 1
fi

# Get master node public DNS
MASTER_DNS=$(aws emr describe-cluster --cluster-id "$CLUSTER_ID" --query 'Cluster.MasterPublicDnsName' --output text)
echo "Master node: $MASTER_DNS"

# Get all instances
INSTANCES=$(aws emr list-instances --cluster-id "$CLUSTER_ID" --query 'Instances[*].[InstanceId,InstanceType,InstanceGroupType,PublicDnsName]' --output text)

echo "Found instances:"
echo "$INSTANCES"

# Function to install packages on a node
install_packages_on_node() {
    local node_dns="$1"
    local node_type="$2"
    
    echo "Installing packages on $node_type node: $node_dns"
    
    ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 hadoop@"$node_dns" << 'EOF'
        echo "=== Installing Python packages on $(hostname) ==="
        
        # Update package lists
        sudo yum update -y
        
        # Install system dependencies
        sudo yum install -y python3-pip python3-devel gcc gcc-c++ make
        
        # Upgrade pip
        python3 -m pip install --upgrade pip --user
        
        # Install Python packages
        echo "Installing Python packages..."
        python3 -m pip install --user numpy scipy pandas python-dateutil rasterio fiona shapely pyproj scikit-fuzzy boto3 s3fs matplotlib seaborn
        
        # Verify installations
        echo "Verifying installations..."
        python3 -c "
import numpy, rasterio, skfuzzy, boto3, s3fs, dateutil
print('✓ All required packages installed successfully on ' + __import__('socket').gethostname())
"
        
        echo "=== Package installation complete on $(hostname) ==="
EOF
}

# Install on master node
echo "Installing packages on master node..."
install_packages_on_node "$MASTER_DNS" "master"

# Install on core nodes
echo "Installing packages on core nodes..."
CORE_NODES=$(aws emr list-instances --cluster-id "$CLUSTER_ID" --instance-group-types CORE --query 'Instances[*].PublicDnsName' --output text)

for core_node in $CORE_NODES; do
    if [ -n "$core_node" ] && [ "$core_node" != "None" ] && [ "$core_node" != "$MASTER_DNS" ]; then
        install_packages_on_node "$core_node" "core"
    fi
done

# Install on task nodes (if any)
echo "Installing packages on task nodes..."
TASK_NODES=$(aws emr list-instances --cluster-id "$CLUSTER_ID" --instance-group-types TASK --query 'Instances[*].PublicDnsName' --output text)

for task_node in $TASK_NODES; do
    if [ -n "$task_node" ] && [ "$task_node" != "None" ]; then
        install_packages_on_node "$task_node" "task"
    fi
done

echo "=== Package installation completed on all nodes ==="
echo "You can now run your Spark applications with cluster mode!" 