#!/bin/bash

# Simple Python Package Installation for EMR
# This script installs packages on current node and other nodes if accessible

set -e

CLUSTER_ID="${1:-<EMR-CLUSTER-ID>}"

if [ -z "$CLUSTER_ID" ]; then
    echo "Usage: $0 <cluster-id>"
    echo "Example: $0 <EMR-CLUSTER-ID>"
    exit 1
fi

echo "=== Installing Python packages on EMR cluster $CLUSTER_ID ==="

# Step 1: Install packages on current node
echo "Step 1: Installing packages on current node ($(hostname))..."
python3 -m pip install --user python-dateutil numpy scipy pandas rasterio scikit-fuzzy boto3 s3fs

# Verify installation on current node
echo "Verifying packages on current node..."
python3 -c "
try:
    import dateutil, numpy, rasterio, skfuzzy, boto3, s3fs
    print('✓ All packages installed successfully on ' + __import__('socket').gethostname())
except ImportError as e:
    print(f'✗ Package import failed: {e}')
    exit(1)
"

echo "✓ Packages installed on current node"

# Step 2: Try to install on other nodes
echo ""
echo "Step 2: Attempting to install packages on other nodes..."

# Get master node public DNS
MASTER_DNS=$(aws emr describe-cluster --cluster-id "$CLUSTER_ID" --query 'Cluster.MasterPublicDnsName' --output text)
CURRENT_HOSTNAME=$(hostname)

echo "Master node: $MASTER_DNS"
echo "Current node: $CURRENT_HOSTNAME"

# Function to install packages on a node
install_packages_on_node() {
    local node_dns="$1"
    local node_type="$2"
    
    # Skip if it's the current node
    if [ "$node_dns" = "$CURRENT_HOSTNAME" ] || [ "$node_dns" = "$(hostname)" ]; then
        echo "Skipping current node: $node_dns"
        return 0
    fi
    
    echo "Installing packages on $node_type node: $node_dns"
    
    # Try SSH with timeout and continue on failure
    timeout 30 ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 hadoop@"$node_dns" << 'EOF' || echo "Failed to install on $node_dns, continuing..."
        echo "=== Installing Python packages on $(hostname) ==="
        
        # Install Python packages
        python3 -m pip install --user python-dateutil numpy scipy pandas rasterio scikit-fuzzy boto3 s3fs
        
        # Verify installations
        echo "Verifying installations..."
        python3 -c "
import dateutil, numpy, rasterio, skfuzzy, boto3, s3fs
print('✓ All required packages installed successfully on ' + __import__('socket').gethostname())
"
        
        echo "=== Package installation complete on $(hostname) ==="
EOF
}

# Install on core nodes
echo "Installing packages on core nodes..."
CORE_NODES=$(aws emr list-instances --cluster-id "$CLUSTER_ID" --instance-group-types CORE --query 'Instances[*].PublicDnsName' --output text)

for core_node in $CORE_NODES; do
    if [ -n "$core_node" ] && [ "$core_node" != "None" ]; then
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

echo ""
echo "=== Package installation completed ==="
echo "✓ Current node packages installed"
echo "✓ Attempted installation on other nodes"
echo "You can now run your Spark applications!" 