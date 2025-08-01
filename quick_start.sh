#!/bin/bash

# Destark FIS System Quick Start Script
# This script sets up and starts the complete FIS system

set -e

echo "🚀 Starting Destark FIS System..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is installed
check_docker() {
    print_status "Checking Docker installation..."
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! docker compose version &> /dev/null; then
        print_error "Docker Compose is not available. Please install Docker Compose first."
        exit 1
    fi
    
    print_success "Docker and Docker Compose are installed"
}

# Check if ports are available
check_ports() {
    print_status "Checking port availability..."
    
    local ports=(3000 8000 8080)
    local unavailable_ports=()
    
    for port in "${ports[@]}"; do
        if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
            unavailable_ports+=($port)
        fi
    done
    
    if [ ${#unavailable_ports[@]} -ne 0 ]; then
        print_warning "The following ports are already in use: ${unavailable_ports[*]}"
        print_warning "Please stop the services using these ports or modify the docker-compose.yml file"
        read -p "Do you want to continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    
    print_success "Ports are available"
}

# Create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    
    mkdir -p logs
    mkdir -p data
    mkdir -p tmp
    
    print_success "Directories created"
}

# Build and start services
start_services() {
    print_status "Building and starting services..."
    
    # Pull latest images
    docker compose pull
    
    # Build and start services
    docker compose up -d --build
    
    print_success "Services started successfully"
}

# Wait for services to be ready
wait_for_services() {
    print_status "Waiting for services to be ready..."
    
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s http://localhost:8000/health >/dev/null 2>&1; then
            print_success "Backend API is ready"
            break
        fi
        
        if [ $attempt -eq $max_attempts ]; then
            print_error "Backend API failed to start within the expected time"
            exit 1
        fi
        
        print_status "Waiting for backend API... (attempt $attempt/$max_attempts)"
        sleep 2
        ((attempt++))
    done
    
    # Wait a bit more for frontend
    sleep 5
}

# Display service information
show_service_info() {
    echo
    echo "🎉 Destark FIS System is now running!"
    echo
    echo "📱 Service URLs:"
    echo "   Frontend (React):     http://localhost:3000"
    echo "   Backend API:          http://localhost:8000"
    echo "   API Documentation:    http://localhost:8000/docs"
    echo "   Spark Web UI:         http://localhost:8080"
    echo
    echo "🔧 Useful Commands:"
    echo "   View logs:            docker-compose logs -f"
    echo "   Stop services:        docker-compose down"
    echo "   Restart services:     docker-compose restart"
    echo "   Check status:         docker-compose ps"
    echo
    echo "📚 Documentation:"
    echo "   Frontend Guide:       README_FRONTEND.md"
    echo "   Deployment Guide:     DEPLOYMENT_GUIDE.md"
    echo
    echo "🚀 Getting Started:"
    echo "   1. Open http://localhost:3000 in your browser"
    echo "   2. Navigate to the FIS Builder tab"
    echo "   3. Configure your input variables and rules"
    echo "   4. Export your configuration"
    echo "   5. Go to Execute System to process your data"
    echo
}

# Check service health
check_service_health() {
    print_status "Checking service health..."
    
    local services=("frontend" "backend" "spark-master" "spark-worker")
    local failed_services=()
    
    for service in "${services[@]}"; do
        if ! docker compose ps $service | grep -q "Up"; then
            failed_services+=($service)
        fi
    done
    
    if [ ${#failed_services[@]} -ne 0 ]; then
        print_warning "The following services failed to start: ${failed_services[*]}"
        print_status "Check the logs with: docker compose logs ${failed_services[*]}"
    else
        print_success "All services are running"
    fi
}

# Main execution
main() {
    echo "=========================================="
    echo "    Destark FIS System Quick Start"
    echo "=========================================="
    echo
    
    # Check prerequisites
    check_docker
    check_ports
    
    # Create directories
    create_directories
    
    # Start services
    start_services
    
    # Wait for services
    wait_for_services
    
    # Check health
    check_service_health
    
    # Show information
    show_service_info
}

# Handle script interruption
trap 'echo -e "\n${RED}Script interrupted. Stopping services...${NC}"; docker compose down; exit 1' INT

# Run main function
main "$@" 