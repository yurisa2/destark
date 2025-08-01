# Destark FIS System Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying the Destark FIS (Fuzzy Inference System) frontend and backend components. The system is designed to be deployed both locally for development and on cloud platforms for production use.

## System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   React Frontend │    │   FastAPI Backend│    │  Spark Cluster  │
│   (Port 3000)   │◄──►│   (Port 8000)   │◄──►│   (Port 8080)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Prerequisites

### System Requirements
- **CPU**: 4+ cores recommended
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 20GB available space
- **OS**: Linux, macOS, or Windows with Docker support

### Software Requirements
- **Docker**: 20.10+ with Docker Compose
- **Node.js**: 18+ (for local development)
- **Python**: 3.9+ (for local development)
- **Git**: Latest version

## Local Development Deployment

### Option 1: Docker Compose (Recommended)

#### Step 1: Clone Repository
```bash
git clone <repository-url>
cd destark
```

#### Step 2: Start All Services
```bash
# Start the complete system
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f
```

#### Step 3: Access Services
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **Spark Web UI**: http://localhost:8080
- **API Documentation**: http://localhost:8000/docs

#### Step 4: Stop Services
```bash
# Stop all services
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

### Option 2: Local Development

#### Frontend Setup
```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm start
```

#### Backend Setup
```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start API server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

#### Spark Setup (Optional)
```bash
# Start Spark cluster
docker-compose up spark-master spark-worker -d
```

## Production Deployment

### AWS Deployment

#### Option 1: AWS ECS with Fargate

##### Step 1: Prepare Docker Images
```bash
# Build and tag images
docker build -t destark-fis-frontend:latest ./frontend
docker build -t destark-fis-backend:latest ./backend

# Push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

docker tag destark-fis-frontend:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/destark-fis-frontend:latest
docker tag destark-fis-backend:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/destark-fis-backend:latest

docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/destark-fis-frontend:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/destark-fis-backend:latest
```

##### Step 2: Create ECS Cluster
```bash
# Create cluster
aws ecs create-cluster --cluster-name destark-fis-cluster

# Create task definitions (see AWS console or CLI documentation)
# Create services for frontend and backend
```

##### Step 3: Configure Load Balancer
```bash
# Create Application Load Balancer
# Configure target groups for frontend (port 3000) and backend (port 8000)
# Set up health checks and routing rules
```

#### Option 2: AWS EMR for Spark Processing

##### Step 1: Create EMR Cluster
```bash
aws emr create-cluster \
  --name "Destark FIS EMR Cluster" \
  --release-label emr-6.15.0 \
  --applications Name=Spark \
  --ec2-attributes KeyName=your-key-pair \
  --instance-groups InstanceGroupType=MASTER,InstanceCount=1,InstanceType=m5.xlarge \
  InstanceGroupType=CORE,InstanceCount=2,InstanceType=m5.xlarge \
  --use-default-roles
```

##### Step 2: Configure EMR Steps
```bash
# Upload application files to S3
aws s3 cp ./app s3://your-bucket/destark/app --recursive

# Create EMR step for FIS processing
aws emr add-steps \
  --cluster-id <cluster-id> \
  --steps Type=Spark,Name="FIS Processing",ActionOnFailure=CONTINUE,Args=[--class,com.example.FISProcessor,s3://your-bucket/destark/app/fis-processor.jar]
```

### Google Cloud Platform Deployment

#### Option 1: Google Kubernetes Engine (GKE)

##### Step 1: Create GKE Cluster
```bash
# Create cluster
gcloud container clusters create destark-fis-cluster \
  --zone us-central1-a \
  --num-nodes 3 \
  --machine-type n1-standard-4

# Get credentials
gcloud container clusters get-credentials destark-fis-cluster --zone us-central1-a
```

##### Step 2: Deploy Applications
```bash
# Apply Kubernetes manifests
kubectl apply -f k8s/frontend-deployment.yaml
kubectl apply -f k8s/backend-deployment.yaml
kubectl apply -f k8s/services.yaml

# Check deployment status
kubectl get pods
kubectl get services
```

#### Option 2: Cloud Run (Serverless)

##### Step 1: Build and Deploy Backend
```bash
# Build container
gcloud builds submit --tag gcr.io/PROJECT_ID/destark-fis-backend

# Deploy to Cloud Run
gcloud run deploy destark-fis-backend \
  --image gcr.io/PROJECT_ID/destark-fis-backend \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

##### Step 2: Deploy Frontend
```bash
# Build frontend
cd frontend
npm run build

# Deploy to Firebase Hosting or Cloud Storage
firebase deploy
```

### Azure Deployment

#### Option 1: Azure Container Instances

##### Step 1: Create Container Registry
```bash
# Create ACR
az acr create --resource-group destark-rg --name destarkacr --sku Basic

# Build and push images
az acr build --registry destarkacr --image destark-fis-frontend:latest ./frontend
az acr build --registry destarkacr --image destark-fis-backend:latest ./backend
```

##### Step 2: Deploy Containers
```bash
# Deploy backend
az container create \
  --resource-group destark-rg \
  --name destark-fis-backend \
  --image destarkacr.azurecr.io/destark-fis-backend:latest \
  --dns-name-label destark-fis-backend \
  --ports 8000

# Deploy frontend
az container create \
  --resource-group destark-rg \
  --name destark-fis-frontend \
  --image destarkacr.azurecr.io/destark-fis-frontend:latest \
  --dns-name-label destark-fis-frontend \
  --ports 3000
```

## Configuration Management

### Environment Variables

#### Frontend Configuration
```bash
# .env file for frontend
REACT_APP_API_URL=http://localhost:8000
REACT_APP_ENVIRONMENT=development
REACT_APP_VERSION=1.0.0
```

#### Backend Configuration
```bash
# .env file for backend
DATABASE_URL=postgresql://user:pass@localhost:5432/destark
REDIS_URL=redis://localhost:6379
LOG_LEVEL=INFO
CORS_ORIGINS=http://localhost:3000
```

### Configuration Files

#### Docker Compose Configuration
```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  frontend:
    build: ./frontend
    environment:
      - REACT_APP_API_URL=https://api.yourdomain.com
    ports:
      - "3000:3000"
  
  backend:
    build: ./backend
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - REDIS_URL=${REDIS_URL}
    ports:
      - "8000:8000"
```

## Monitoring and Logging

### Application Monitoring

#### Frontend Monitoring
```bash
# Install monitoring tools
npm install --save @sentry/react @sentry/tracing

# Configure Sentry
import * as Sentry from "@sentry/react";
Sentry.init({
  dsn: "your-sentry-dsn",
  environment: process.env.REACT_APP_ENVIRONMENT,
});
```

#### Backend Monitoring
```python
# Add monitoring to FastAPI app
from prometheus_client import Counter, Histogram
import time

# Metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests')
REQUEST_LATENCY = Histogram('http_request_duration_seconds', 'HTTP request latency')

# Middleware
@app.middleware("http")
async def monitor_requests(request, call_next):
    REQUEST_COUNT.inc()
    start_time = time.time()
    response = await call_next(request)
    REQUEST_LATENCY.observe(time.time() - start_time)
    return response
```

### Logging Configuration

#### Structured Logging
```python
# backend/logging_config.py
import logging
import json
from pythonjsonlogger import jsonlogger

class CustomJsonFormatter(jsonlogger.JsonFormatter):
    def add_fields(self, log_record, record, message_dict):
        super(CustomJsonFormatter, self).add_fields(log_record, record, message_dict)
        log_record['timestamp'] = record.created
        log_record['level'] = record.levelname
        log_record['logger'] = record.name

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

## Security Considerations

### SSL/TLS Configuration
```bash
# Generate SSL certificates
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes

# Configure nginx for SSL termination
server {
    listen 443 ssl;
    server_name yourdomain.com;
    
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    
    location / {
        proxy_pass http://localhost:3000;
    }
    
    location /api {
        proxy_pass http://localhost:8000;
    }
}
```

### Authentication and Authorization
```python
# Add JWT authentication to FastAPI
from fastapi_jwt_auth import AuthJWT
from fastapi_jwt_auth.exceptions import AuthJWTException

@app.post('/login')
async def login(user_credentials: UserCredentials, Authorize: AuthJWT = Depends()):
    # Validate credentials
    access_token = Authorize.create_access_token(subject=user_credentials.username)
    return {"access_token": access_token}
```

## Performance Optimization

### Frontend Optimization
```bash
# Enable production optimizations
npm run build

# Configure CDN for static assets
# Enable gzip compression
# Implement lazy loading for components
```

### Backend Optimization
```python
# Add caching
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend

@app.on_event("startup")
async def startup():
    redis = aioredis.from_url("redis://localhost", encoding="utf8")
    FastAPICache.init(RedisBackend(redis), prefix="fastapi-cache")

# Add response caching
@router.get("/api/data")
@cache(expire=60)
async def get_data():
    return {"data": "cached_response"}
```

## Backup and Recovery

### Database Backup
```bash
# PostgreSQL backup
pg_dump -h localhost -U username -d destark > backup.sql

# Automated backup script
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
pg_dump -h localhost -U username -d destark > backup_$DATE.sql
aws s3 cp backup_$DATE.sql s3://your-backup-bucket/
```

### File Storage Backup
```bash
# Backup uploaded files
aws s3 sync s3://your-upload-bucket/ s3://your-backup-bucket/uploads/

# Automated backup with versioning
aws s3api put-bucket-versioning --bucket your-backup-bucket --versioning-configuration Status=Enabled
```

## Troubleshooting

### Common Issues

#### Docker Issues
```bash
# Check container logs
docker-compose logs frontend
docker-compose logs backend

# Restart services
docker-compose restart frontend
docker-compose restart backend

# Clean up containers
docker-compose down -v
docker system prune -a
```

#### Network Issues
```bash
# Check connectivity
curl http://localhost:8000/health
curl http://localhost:3000

# Check ports
netstat -tulpn | grep :3000
netstat -tulpn | grep :8000
```

#### Performance Issues
```bash
# Monitor resource usage
docker stats

# Check memory usage
free -h
df -h

# Monitor CPU usage
top
htop
```

### Health Checks
```python
# Backend health check
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "version": "1.0.0",
        "services": {
            "database": check_database_connection(),
            "redis": check_redis_connection(),
            "spark": check_spark_connection()
        }
    }
```

## Maintenance

### Regular Maintenance Tasks
```bash
# Update dependencies
npm update  # Frontend
pip install --upgrade -r requirements.txt  # Backend

# Clean up old files
find /tmp/fis_jobs -mtime +7 -delete

# Monitor disk space
df -h

# Check for security updates
docker-compose pull
docker-compose up -d
```

### Scaling Considerations
```bash
# Scale backend services
docker-compose up -d --scale backend=3

# Scale Spark workers
docker-compose up -d --scale spark-worker=4

# Monitor performance
docker stats
```

## Support and Documentation

### Getting Help
- Check the application logs for error messages
- Review the troubleshooting section
- Contact the development team
- Submit issues through the project repository

### Additional Resources
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React Documentation](https://reactjs.org/docs/)
- [Docker Documentation](https://docs.docker.com/)
- [AWS ECS Documentation](https://docs.aws.amazon.com/ecs/)
- [Google Cloud Run Documentation](https://cloud.google.com/run/docs) 