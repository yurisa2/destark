# Spark-based Raster Fuzzy Inference System

This document describes the Spark-based implementation of the raster fuzzy inference system for distributed processing of large raster datasets.

## Overview

The Spark-based system (`raster_fuzzy_spark.py`) extends the unified raster fuzzy inference system to leverage Apache Spark for distributed processing. This enables:

- **Scalable processing** of very large raster datasets
- **Fault tolerance** with automatic recovery from node failures
- **Resource optimization** with dynamic allocation of computing resources
- **Real-time monitoring** through Spark Web UI

## Architecture

### Components

1. **Spark Cluster**: Docker-based Spark cluster with master and worker nodes
2. **SparkRasterFuzzyInferenceSystem**: Main processing class for distributed raster operations
3. **DataFrame Operations**: Efficient data manipulation using Spark DataFrames
4. **RDD Processing**: Distributed fuzzy logic computation using Spark RDDs

### Processing Flow

```
Input Rasters → DataFrame Conversion → Join Operations → RDD Processing → Fuzzy Logic → Output Raster
```

## Setup

### Prerequisites

- Docker and Docker Compose
- Python 3.8+ with required dependencies
- At least 4GB RAM available for the cluster

### Quick Start

1. **Start the Spark cluster:**
   ```bash
   sh scripts/setup_spark.sh
   ```

2. **Install Spark dependencies:**
   ```bash
   pip install -r requirements-spark.txt
   ```

3. **Test the system:**
   ```bash
   python scripts/test_spark_system.py
   ```

### Access Points

- **Spark Web UI**: http://localhost:8080
- **Jupyter Lab**: http://localhost:8888
- **Spark Master**: spark://localhost:7077

## Usage

### Command Line Interface

```bash
# Local mode (for testing)
python app/raster_fuzzy_spark.py social.tif env.tif strat.tif output.tif --local

# Cluster mode (for production)
python app/raster_fuzzy_spark.py social.tif env.tif strat.tif output.tif

# With custom parameters
python app/raster_fuzzy_spark.py social.tif env.tif strat.tif output.tif \
    --chunk-size 1000 \
    --partitions 8 \
    --config my_config.json
```

### Programmatic Usage

```python
from app.raster_fuzzy_spark import SparkRasterFuzzyInferenceSystem, create_spark_session

# Create Spark session
spark = create_spark_session(
    app_name="MyRasterProcessing",
    master_url="spark://spark-master:7077",
    local_mode=False
)

# Initialize Spark-based system
spark_fis = SparkRasterFuzzyInferenceSystem(spark, 'config.json')

# Process rasters
spark_fis.process_rasters_spark(
    social_tiff='social.tif',
    environmental_tiff='env.tif',
    strategic_tiff='strat.tif',
    output_tiff='output.tif',
    chunk_size=1000,
    num_partitions=8
)

# Stop Spark session
spark.stop()
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--config, -c` | Configuration JSON file path | `raster_fis_config.json` |
| `--nodata` | NoData value for output raster | `5.0` |
| `--chunk-size` | Number of pixels per chunk | `1000` |
| `--partitions` | Number of Spark partitions | Auto |
| `--local` | Run in local mode for testing | `False` |
| `--master` | Spark master URL | `spark://spark-master:7077` |
| `--verbose, -v` | Enable verbose output | `False` |

## Performance Considerations

### Chunk Size Optimization

- **Small chunks (100-500)**: Better for memory-constrained environments
- **Medium chunks (500-2000)**: Good balance for most datasets
- **Large chunks (2000+)**: Better for very large datasets with sufficient memory

### Partition Strategy

- **Auto**: Let Spark determine optimal partitioning
- **Manual**: Set based on available cores and data size
- **Rule of thumb**: 2-4 partitions per executor core

### Memory Configuration

The Docker setup includes:
- **Worker Memory**: 2GB per worker
- **Driver Memory**: 2GB
- **Executor Cores**: 2 per worker

For production, adjust these in `docker-compose.yml`:

```yaml
environment:
  - SPARK_WORKER_MEMORY=4G
  - SPARK_WORKER_CORES=4
```

## Monitoring and Debugging

### Spark Web UI

Access http://localhost:8080 to monitor:
- Cluster status and resource usage
- Running applications
- Worker node status

### Application Monitoring

When running applications, access http://localhost:4040 to see:
- Job progress and DAG visualization
- Stage and task details
- Performance metrics

### Logs

```bash
# View cluster logs
docker-compose logs spark-master
docker-compose logs spark-worker-1

# View application logs
docker exec jupyter-spark tail -f /opt/bitnami/spark/logs/
```

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**
   - Reduce chunk size
   - Increase worker memory in docker-compose.yml
   - Use fewer partitions

2. **Slow Processing**
   - Increase number of workers
   - Optimize chunk size
   - Check network connectivity between nodes

3. **Serialization Errors**
   - Ensure all functions are serializable
   - Avoid using non-serializable objects in worker functions

4. **Connection Issues**
   - Check if all containers are running: `docker-compose ps`
   - Restart cluster: `docker-compose restart`
   - Check network connectivity: `docker network ls`

### Performance Tuning

1. **Data Skew**
   - Use `repartition()` to balance data distribution
   - Consider data partitioning strategy

2. **Memory Management**
   - Monitor memory usage in Spark Web UI
   - Adjust executor memory settings
   - Use broadcast variables for small datasets

3. **Network Optimization**
   - Ensure low-latency network between nodes
   - Consider data locality for large datasets

## Scaling

### Horizontal Scaling

Add more worker nodes to `docker-compose.yml`:

```yaml
spark-worker-3:
  image: bitnami/spark:3.5.0
  container_name: spark-worker-3
  environment:
    - SPARK_MODE=worker
    - SPARK_MASTER_URL=spark://spark-master:7077
    - SPARK_WORKER_MEMORY=2G
    - SPARK_WORKER_CORES=2
  volumes:
    - ./app:/opt/bitnami/spark/app
    - ./data:/opt/bitnami/spark/data
  depends_on:
    - spark-master
  networks:
    - spark-network
```

### Vertical Scaling

Increase resources per worker:

```yaml
environment:
  - SPARK_WORKER_MEMORY=4G
  - SPARK_WORKER_CORES=4
```

## Integration with Existing Systems

### Migration from Sequential/Parallel Systems

The Spark system maintains compatibility with existing configurations:

```python
# Same configuration format
config = {
    "input_variables": {...},
    "output_variable": {...},
    "rules": [...]
}

# Same fuzzy logic processing
fuzzy_system = UnifiedRasterFuzzyInferenceSystem(config_path)
```

### Batch Processing Integration

Update your batch scripts to use Spark:

```bash
# Replace sequential/parallel calls with Spark
python app/raster_fuzzy_spark.py "$SOCIAL" "$ENVIRONMENTAL" "$STRATEGIC" "$output_file" \
    --config "app/config/$config_file" \
    --chunk-size 1000 \
    --partitions 8
```

## Development

### Jupyter Notebook

Access Jupyter Lab at http://localhost:8888 for:
- Interactive development and testing
- Performance analysis
- Data exploration

### Adding Custom Processing

Extend the system by adding custom RDD transformations:

```python
def custom_processing(rdd):
    return rdd.map(lambda x: process_pixel(x))

# Use in processing pipeline
processed_rdd = pixel_rdd.transform(custom_processing)
```

## File Structure

```
├── docker-compose.yml              # Spark cluster configuration
├── app/
│   ├── raster_fuzzy_spark.py      # Spark-based processing system
│   ├── raster_fuzzy_lib.py        # Core fuzzy logic library
│   └── config/                    # Configuration files
├── scripts/
│   ├── setup_spark.sh            # Cluster setup script
│   └── test_spark_system.py      # System testing
├── notebooks/
│   └── spark_raster_fuzzy_demo.ipynb  # Development notebook
├── data/                          # Input/output data
├── logs/                          # Spark logs
└── requirements-spark.txt         # Spark dependencies
```

## Future Enhancements

1. **GPU Acceleration**: Integrate with RAPIDS for GPU-accelerated processing
2. **Streaming Processing**: Real-time raster processing with Spark Streaming
3. **Machine Learning**: Integrate with Spark MLlib for advanced analytics
4. **Cloud Deployment**: Kubernetes-based deployment for cloud environments
5. **Advanced Monitoring**: Integration with Prometheus/Grafana for metrics

## Support

For issues and questions:
1. Check Spark Web UI for cluster status
2. Review application logs for error details
3. Test with smaller datasets first
4. Verify network connectivity between nodes
5. Check resource allocation and memory settings 