#!/bin/bash

# Test Java versions and fix compatibility issues

echo "=== Testing Java Versions ==="

# Check available Java versions
echo "Available Java versions:"
java -version 2>&1
echo ""

# Check JAVA_HOME
echo "JAVA_HOME: $JAVA_HOME"
echo ""

# Check if Java 17 is available
if command -v java17 &> /dev/null; then
    echo "Java 17 is available"
    java17 -version 2>&1
    echo ""
    echo "Setting JAVA_HOME to Java 17..."
    export JAVA_HOME=/usr/lib/jvm/java-17-amazon-corretto
    export PATH=$JAVA_HOME/bin:$PATH
    echo "New JAVA_HOME: $JAVA_HOME"
    echo "New Java version:"
    java -version 2>&1
    echo ""
else
    echo "Java 17 not found, checking system Java..."
    ls -la /usr/lib/jvm/ 2>/dev/null || echo "No /usr/lib/jvm/ directory"
    echo ""
fi

# Check EMR Java setup
echo "EMR Java setup:"
if [ -f /etc/emr-release ]; then
    echo "EMR release file found"
    cat /etc/emr-release
    echo ""
fi

# Check if we can use system Java
echo "Trying to use system Java for Spark..."
export JAVA_HOME=/usr/lib/jvm/jre-17
export PATH=$JAVA_HOME/bin:$PATH

echo "Updated JAVA_HOME: $JAVA_HOME"
echo "Updated Java version:"
java -version 2>&1
echo ""

# Test if this fixes the Spark issue
echo "=== Testing Spark with updated Java ==="
python3 -c "
import os
os.environ['JAVA_HOME'] = '/usr/lib/jvm/jre-17'
try:
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.appName('Test').master('local[1]').getOrCreate()
    print('✓ Spark session created successfully with updated Java')
    spark.stop()
except Exception as e:
    print(f'✗ Spark still fails: {e}')
" 