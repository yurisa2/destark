# EMR Cluster Viability Testing Commands

## Quick Test Commands

After connecting to your EMR master node, run these commands to verify everything is working:

### 1. Basic System Tools
```bash
# Test Git
git --version

# Test Java
java -version

# Test Python
python3 --version

# Test pip
pip3 --version
```

### 2. Python Package Imports
```bash
# Test core packages
python3 -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python3 -c "import pandas; print(f'Pandas: {pandas.__version__}')"
python3 -c "import scipy; print(f'SciPy: {scipy.__version__}')"

# Test geospatial packages
python3 -c "import rasterio; print(f'Rasterio: {rasterio.__version__}')"
python3 -c "import fiona; print(f'Fiona: {fiona.__version__}')"
python3 -c "import shapely; print(f'Shapely: {shapely.__version__}')"
python3 -c "import pyproj; print(f'PyProj: {pyproj.__version__}')"

# Test fuzzy logic
python3 -c "import skfuzzy; print('Scikit-fuzzy: OK')"

# Test AWS packages
python3 -c "import boto3; print(f'Boto3: {boto3.__version__}')"
python3 -c "import s3fs; print('S3fs: OK')"

# Test Spark
python3 -c "import pyspark; print(f'PySpark: {pyspark.__version__}')"
python3 -c "import networkx; print(f'NetworkX: {networkx.__version__}')"
```

### 3. AWS Integration
```bash
# Test AWS credentials
aws sts get-caller-identity

# Test S3 access
aws s3 ls s3://<AWS-BUCKET>-unifile/ --max-items 5
```

### 4. Spark Environment
```bash
# Test Spark installation
spark-submit --version

# Test Spark session creation
python3 -c "
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('Test').getOrCreate()
print(f'Spark version: {spark.version}')
spark.stop()
"
```

### 5. Development Directory
```bash
# Check development directory
cd /mnt/destark
pwd
ls -la
```

### 6. System Resources
```bash
# Check memory
free -h

# Check disk space
df -h

# Check CPU cores
nproc
```

### 7. Network Connectivity
```bash
# Test internet
ping -c 3 google.com

# Test AWS endpoints
curl -s https://sts.us-east-2.amazonaws.com/ | head -5
```

### 8. Sample Data Processing Tests

#### Test Raster Processing
```bash
python3 -c "
import numpy as np
import rasterio
from rasterio.transform import from_origin

# Create sample data
data = np.random.random((100, 100)).astype(np.float32)
transform = from_origin(0, 0, 1, 1)

# Write test file
with rasterio.open('/tmp/test_raster.tif', 'w', 
                   driver='GTiff', height=100, width=100, count=1, 
                   dtype=np.float32, crs='EPSG:4326', transform=transform) as dst:
    dst.write(data, 1)

print('Sample raster file created: /tmp/test_raster.tif')
"
```

#### Test FIS Processing
```bash
python3 -c "
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Create simple FIS
x = np.arange(0, 11, 1)
y = np.arange(0, 11, 1)

# Create fuzzy variables
input_var = ctrl.Antecedent(x, 'input')
output_var = ctrl.Consequent(y, 'output')

# Define membership functions
input_var['low'] = fuzz.trimf(input_var.universe, [0, 0, 5])
input_var['high'] = fuzz.trimf(input_var.universe, [5, 10, 10])

output_var['low'] = fuzz.trimf(output_var.universe, [0, 0, 5])
output_var['high'] = fuzz.trimf(output_var.universe, [5, 10, 10])

# Create rules
rule1 = ctrl.Rule(input_var['low'], output_var['low'])
rule2 = ctrl.Rule(input_var['high'], output_var['high'])

# Create control system
control_system = ctrl.ControlSystem([rule1, rule2])
simulation = ctrl.ControlSystemSimulation(control_system)

# Test with input
simulation.input['input'] = 3.0
simulation.compute()

print(f'FIS test successful! Input: 3.0, Output: {simulation.output[\"output\"]:.2f}')
"
```

### 9. Cluster Information
```bash
# Get cluster ID
echo $CLUSTER_ID

# Get instance type
curl -s http://169.254.169.254/latest/meta-data/instance-type

# Get instance ID
curl -s http://169.254.169.254/latest/meta-data/instance-id
```

## Complete Test Script

For a comprehensive test, run the full test script:

```bash
# Upload the test script to the cluster
scp -i /path/to/<KEY-PAIR-NAME>.pem scripts/test_cluster_viability.sh hadoop@<master-public-dns>:/home/hadoop/

# Connect to the cluster and run the test
ssh -i /path/to/<KEY-PAIR-NAME>.pem hadoop@<master-public-dns>
chmod +x test_cluster_viability.sh
./test_cluster_viability.sh
```

## Expected Results

✅ **All tests should pass** if the cluster is properly configured  
✅ **Python packages** should import without errors  
✅ **AWS credentials** should be available  
✅ **Spark session** should create successfully  
✅ **Sample data processing** should work  
✅ **Network connectivity** should be available  

## Troubleshooting

If any tests fail:

1. **Check bootstrap logs**: Look at the bootstrap action logs in the EMR console
2. **Verify IAM roles**: Ensure the EMR role has proper permissions
3. **Check package versions**: Some packages might need version adjustments
4. **Network issues**: Verify security groups allow proper access

## Next Steps After Testing

Once all tests pass:

1. **Clone your repository**:
   ```bash
   cd /mnt/destark
   git clone https://github.com/your-username/destark.git .
   ```

2. **Start developing**:
   ```bash
   # Test your existing code
   python3 app/raster_fuzzy_spark.py
   
   # Submit Spark jobs
   spark-submit app/raster_fuzzy_spark.py
   ```

3. **Monitor resources**:
   ```bash
   # Check YARN applications
   yarn application -list
   
   # Check Spark history
   # Access via: http://<master-public-dns>:18080
   ``` 