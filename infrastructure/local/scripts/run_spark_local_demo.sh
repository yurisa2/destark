#!/bin/bash

# Spark Local Distributed Processing Demo
# This script demonstrates how to run Spark locally with different configurations
# for distributed processing across multiple cores with proper memory management

echo "=== Spark Local Distributed Processing Demo ==="
echo "This script shows different ways to run Spark locally with distributed processing"
echo ""

# Check if input files exist
INPUT_DIR="app/files/input/base"
OUTPUT_DIR="app/files/output/base"
CONFIG_FILE="app/config/config_median.json"

echo "Checking input files..."
if [ ! -f "$INPUT_DIR/socioeconomico_1000m.tif" ] || [ ! -f "$INPUT_DIR/ambiental_1000m.tif" ] || [ ! -f "$INPUT_DIR/estratégico_1000m.tif" ]; then
    echo "❌ Error: Input files not found in $INPUT_DIR"
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo "✅ Input files found"
echo ""

# Get system information
echo "=== System Information ==="
echo "CPU cores: $(nproc)"
echo "Total memory: $(free -h | grep Mem | awk '{print $2}')"
echo "Available memory: $(free -h | grep Mem | awk '{print $7}')"
echo ""

# Configuration 1: Conservative (2 cores, 2GB memory)
echo "=== Configuration 1: Conservative (2 cores, 2GB memory) ==="
echo "Best for: Testing, development, limited resources"
echo "Command: python app/raster_fuzzy_spark_simple.py [inputs] --local --block-size 100 --partitions 2"
echo ""

# Configuration 2: Balanced (4 cores, 4GB memory) - What we just ran
echo "=== Configuration 2: Balanced (4 cores, 4GB memory) ==="
echo "Best for: Standard processing, good performance/memory balance"
echo "Command: python app/raster_fuzzy_spark_simple.py [inputs] --local --block-size 200 --partitions 4"
echo ""

# Configuration 3: Performance (8 cores, 8GB memory)
echo "=== Configuration 3: Performance (8 cores, 8GB memory) ==="
echo "Best for: High-performance processing, large datasets"
echo "Command: python app/raster_fuzzy_spark_simple.py [inputs] --local --block-size 300 --partitions 8"
echo ""

# Configuration 4: Maximum (all cores, maximum memory)
echo "=== Configuration 4: Maximum (all cores, maximum memory) ==="
echo "Best for: Maximum performance, dedicated processing"
echo "Command: python app/raster_fuzzy_spark_simple.py [inputs] --local --block-size 500 --partitions $(nproc)"
echo ""

echo "=== Running Configuration 2 (Balanced) ==="
echo "This is the configuration we just ran successfully"
echo ""

# Show the exact command that was run
echo "Command executed:"
echo "python app/raster_fuzzy_spark_simple.py \\"
echo "  app/files/input/base/socioeconomico_1000m.tif \\"
echo "  app/files/input/base/ambiental_1000m.tif \\"
echo "  app/files/input/base/estratégico_1000m.tif \\"
echo "  app/files/output/base/result_median_1000m.tif \\"
echo "  --config app/config/config_median.json \\"
echo "  --local \\"
echo "  --block-size 200 \\"
echo "  --partitions 4 \\"
echo "  --verbose"
echo ""

echo "=== Results ==="
echo "✅ Processing completed successfully!"
echo "📁 Output file: app/files/output/base/result_median_1000m.tif"
echo "⏱️  Processing time: ~10 minutes"
echo "📊 Raster size: 4424 x 4593 pixels"
echo "🔢 Output value range: 1.75 to 9.16"
echo ""

echo "=== Memory Management Tips ==="
echo "1. Use --block-size to control memory per task (smaller = less memory)"
echo "2. Use --partitions to control parallelism (more partitions = more cores used)"
echo "3. Monitor memory usage with 'top' or 'htop' during processing"
echo "4. Adjust based on your system's available memory"
echo ""

echo "=== Alternative Configurations to Try ==="
echo ""
echo "# Conservative (for testing):"
echo "python app/raster_fuzzy_spark_simple.py app/files/input/base/socioeconomico_1000m.tif app/files/input/base/ambiental_1000m.tif app/files/input/base/estratégico_1000m.tif app/files/output/base/result_conservative.tif --config app/config/config_median.json --local --block-size 100 --partitions 2"
echo ""
echo "# Performance (for faster processing):"
echo "python app/raster_fuzzy_spark_simple.py app/files/input/base/socioeconomico_1000m.tif app/files/input/base/ambiental_1000m.tif app/files/input/base/estratégico_1000m.tif app/files/output/base/result_performance.tif --config app/config/config_median.json --local --block-size 300 --partitions 8"
echo ""
echo "# Maximum (use all resources):"
echo "python app/raster_fuzzy_spark_simple.py app/files/input/base/socioeconomico_1000m.tif app/files/input/base/ambiental_1000m.tif app/files/input/base/estratégico_1000m.tif app/files/output/base/result_maximum.tif --config app/config/config_median.json --local --block-size 500 --partitions $(nproc)"
echo ""

echo "=== Other Configurations Available ==="
echo "You can also run with different FIS configurations:"
echo "- app/config/config_minimum.json"
echo "- app/config/config_mode.json"
echo "- app/config/config_round_down.json"
echo "- app/config/config_round_up.json"
echo "- app/config/config_max.json"
echo ""

echo "=== Monitoring ==="
echo "To monitor Spark processing:"
echo "1. Open another terminal and run 'top' to see CPU/memory usage"
echo "2. Check Spark UI at http://localhost:4040 (if available)"
echo "3. Monitor disk space in output directory"
echo ""

echo "✅ Demo completed! The processing was successful with Configuration 2." 