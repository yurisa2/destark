# AWS Glue 5 with Rasterio Setup Guide

This guide provides two approaches to set up AWS Glue 5 with rasterio and other geospatial libraries for raster fuzzy processing.

## Approach 1: Custom Docker Image (Recommended)

This approach creates a custom Docker image based on the official AWS Glue 5 image and pushes it to Amazon ECR.

### Prerequisites

- AWS CLI configured with appropriate permissions
- Docker installed and running
- AWS account with ECR access

### Steps

1. **Build and Push Docker Image**

```bash
# Make the script executable
chmod +x scripts/build_and_push_glue5_docker.sh

# Run the script (it will automatically detect your AWS account ID)
./scripts/build_and_push_glue5_docker.sh
```

You can customize the build by setting environment variables:

```bash
export AWS_REGION="us-west-2"
export ECR_REPOSITORY_NAME="my-glue5-rasterio"
export IMAGE_TAG="v1.0"
./scripts/build_and_push_glue5_docker.sh
```

2. **Use in AWS Glue**

After the script completes, you'll get an ECR image URI like:
```
123456789012.dkr.ecr.us-east-1.amazonaws.com/glue5-rasterio:latest
```

To use this in AWS Glue:
1. Go to AWS Glue Console
2. Create a new Glue job
3. In the job parameters, set the Image URI to the ECR URI above
4. Ensure your Glue job has IAM permissions to pull from ECR

### Testing the Docker Image Locally

```bash
# Test the environment
docker run -it 123456789012.dkr.ecr.us-east-1.amazonaws.com/glue5-rasterio:latest /opt/amazon/glue/test_environment.sh

# Or run interactively
docker run -it 123456789012.dkr.ecr.us-east-1.amazonaws.com/glue5-rasterio:latest /bin/bash
```

## Approach 2: S3 Python Libraries

This approach uploads Python wheel files to S3 and references them in your Glue job.

### Prerequisites

- AWS CLI configured with appropriate permissions
- pip3 installed

### Steps

1. **Upload Libraries to S3**

```bash
# Make the script executable
chmod +x scripts/upload_glue5_libraries.sh

# Run the script
./scripts/upload_glue5_libraries.sh
```

You can customize the upload by setting environment variables:

```bash
export S3_BUCKET="my-glue-assets-bucket"
export S3_PREFIX="python-libs-glue5"
export AWS_REGION="us-west-2"
./scripts/upload_glue5_libraries.sh
```

2. **Use in AWS Glue**

After the script completes, you'll get S3 paths like:
```
s3://aws-glue-assets-475136118191-us-east-1/python-libs-glue5/numpy-*.whl
s3://aws-glue-assets-475136118191-us-east-1/python-libs-glue5/rasterio-*.whl
```

To use these in AWS Glue:
1. Go to AWS Glue Console
2. Create a new Glue job
3. In the job parameters, add the S3 paths to the "Python library path" field
4. Ensure your Glue job has IAM permissions to access S3

## Included Libraries

Both approaches include the following libraries:

- **rasterio** (>=1.3.0) - Geospatial raster processing
- **fiona** (>=1.8.0) - Vector data access
- **shapely** (>=1.8.0) - Geometric operations
- **scikit-fuzzy** (>=0.4.2) - Fuzzy logic processing
- **numpy** (>=1.21.0) - Numerical computing
- **scipy** (>=1.7.0) - Scientific computing
- **boto3** (>=1.26.0) - AWS SDK
- **numba** (>=0.56.0) - Performance optimization
- **Additional dependencies**: click, cligj, attrs, certifi, affine, pyparsing

## System Dependencies

The Docker approach automatically installs these system dependencies:
- gcc, gcc-c++ (compilers)
- gdal-devel (GDAL development libraries)
- proj-devel (PROJ projection library)
- geos-devel (GEOS geometry library)
- hdf5-devel, netcdf-devel (scientific data formats)
- libspatialindex-devel (spatial indexing)
- python3-devel, python3-pip (Python development tools)

## IAM Permissions

### For Docker Approach (ECR)
Your AWS Glue job needs these permissions:
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "ecr:GetAuthorizationToken",
                "ecr:BatchCheckLayerAvailability",
                "ecr:GetDownloadUrlForLayer",
                "ecr:BatchGetImage"
            ],
            "Resource": "*"
        }
    ]
}
```

### For S3 Approach
Your AWS Glue job needs these permissions:
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject"
            ],
            "Resource": "arn:aws:s3:::your-bucket-name/python-libs-glue5/*"
        }
    ]
}
```

## Troubleshooting

### Docker Build Issues
- Ensure Docker has enough memory (at least 4GB recommended)
- Check that you have AWS credentials configured
- Verify ECR repository creation permissions

### S3 Upload Issues
- Check S3 bucket permissions
- Ensure AWS credentials are configured
- Verify the bucket exists in the specified region

### Glue Job Issues
- Check IAM permissions for ECR/S3 access
- Verify the image URI or S3 paths are correct
- Check Glue job logs for specific error messages

## Comparison of Approaches

| Aspect | Docker Approach | S3 Libraries Approach |
|--------|----------------|----------------------|
| **Setup Complexity** | Medium | Low |
| **Build Time** | Longer (builds full image) | Faster (just uploads wheels) |
| **Customization** | High (full control over environment) | Low (limited to Python packages) |
| **Size** | Larger (includes system dependencies) | Smaller (just Python packages) |
| **Maintenance** | Requires rebuilding image for updates | Just upload new wheels |
| **Performance** | Potentially better (optimized environment) | Standard Glue environment |

## Recommendations

- **Use Docker approach** if you need custom system dependencies or want full control over the environment
- **Use S3 libraries approach** if you only need Python packages and want simpler maintenance
- **For production**, the Docker approach is recommended for better reproducibility and control 