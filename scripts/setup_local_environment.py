#!/usr/bin/env python3
"""
Local Environment Setup and Test for Spark FIS Processing
This script checks dependencies and sets up the local environment
"""

import subprocess
import sys
import os

def run_command(command):
    """Run a command and return the result"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def check_python_version():
    """Check Python version"""
    print("=== CHECKING PYTHON VERSION ===")
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 7:
        print("✅ Python version is compatible")
        return True
    else:
        print("❌ Python 3.7+ is required")
        return False

def install_package(package):
    """Install a Python package"""
    print(f"Installing {package}...")
    success, stdout, stderr = run_command(f"pip3 install {package}")
    if success:
        print(f"✅ {package} installed successfully")
    else:
        print(f"❌ Failed to install {package}: {stderr}")
    return success

def check_package(package, import_name=None):
    """Check if a package is installed"""
    if import_name is None:
        import_name = package
    
    try:
        __import__(import_name)
        print(f"✅ {package} is available")
        return True
    except ImportError:
        print(f"❌ {package} is not available")
        return False

def check_aws_credentials():
    """Check AWS credentials"""
    print("\n=== CHECKING AWS CREDENTIALS ===")
    
    access_key = os.environ.get('AWS_ACCESS_KEY_ID')
    secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
    region = os.environ.get('AWS_DEFAULT_REGION', 'us-east-2')
    
    if access_key and secret_key:
        print("✅ AWS credentials are set")
        print(f"Region: {region}")
        return True
    else:
        print("❌ AWS credentials are not set")
        print("Please set them with:")
        print("export AWS_ACCESS_KEY_ID='your-access-key'")
        print("export AWS_SECRET_ACCESS_KEY='your-secret-key'")
        print("export AWS_DEFAULT_REGION='us-east-2'")
        return False

def test_fuzzy_system():
    """Test the fuzzy system with a simple example"""
    print("\n=== TESTING FUZZY SYSTEM ===")
    
    test_code = '''
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Create a simple fuzzy system
x = ctrl.Antecedent(np.arange(0, 11, 1), 'x')
y = ctrl.Consequent(np.arange(0, 11, 1), 'y')

x['low'] = fuzz.trimf(x.universe, [0, 0, 5])
x['high'] = fuzz.trimf(x.universe, [5, 10, 10])

y['low'] = fuzz.trimf(y.universe, [0, 0, 5])
y['high'] = fuzz.trimf(y.universe, [5, 10, 10])

rule1 = ctrl.Rule(x['low'], y['low'])
rule2 = ctrl.Rule(x['high'], y['high'])

system = ctrl.ControlSystem([rule1, rule2])
sim = ctrl.ControlSystemSimulation(system)

sim.x = 3
sim.compute()
print("Fuzzy test result: " + str(sim.y))
'''
    
    success, stdout, stderr = run_command(f"python3 -c \"{test_code}\"")
    if success:
        print("✅ Fuzzy system test passed")
        print(f"Output: {stdout.strip()}")
        return True
    else:
        print("❌ Fuzzy system test failed")
        print(f"Error: {stderr}")
        return False

def main():
    """Main function to set up and test the environment"""
    print("=== LOCAL ENVIRONMENT SETUP AND TEST ===")
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Check and install required packages
    print("\n=== CHECKING REQUIRED PACKAGES ===")
    
    packages = [
        ('numpy', 'numpy'),
        ('scipy', 'scipy'),
        ('rasterio', 'rasterio'),
        ('scikit-fuzzy', 'skfuzzy'),
        ('boto3', 'boto3'),
        ('pyspark', 'pyspark')
    ]
    
    missing_packages = []
    for package, import_name in packages:
        if not check_package(package, import_name):
            missing_packages.append(package)
    
    # Install missing packages
    if missing_packages:
        print(f"\n=== INSTALLING MISSING PACKAGES ===")
        print(f"Missing packages: {', '.join(missing_packages)}")
        
        for package in missing_packages:
            install_package(package)
        
        # Check again after installation
        print(f"\n=== RE-CHECKING PACKAGES ===")
        for package, import_name in packages:
            check_package(package, import_name)
    
    # Check AWS credentials
    aws_ok = check_aws_credentials()
    
    # Test fuzzy system
    fuzzy_ok = test_fuzzy_system()
    
    # Summary
    print("\n=== SUMMARY ===")
    if aws_ok and fuzzy_ok:
        print("✅ Environment is ready for Spark FIS processing!")
        print("\nNext steps:")
        print("1. Run: python3 scripts/test_spark_fis_local.py")
        print("2. If that works, run: python3 scripts/spark_fis_local.py")
        return True
    else:
        print("❌ Environment setup incomplete")
        if not aws_ok:
            print("- AWS credentials need to be set")
        if not fuzzy_ok:
            print("- Fuzzy system test failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 