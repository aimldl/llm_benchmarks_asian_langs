#!/bin/bash

# KLUE STS Benchmark Setup Script
# This script sets up the environment for running KLUE STS benchmarks with Gemini 2.5 Flash on Vertex AI

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
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

# Function to check prerequisites
check_prerequisites() {
    print_info "Checking prerequisites..."
    
    # Check if Python 3.8+ is available
    if command -v python3 &> /dev/null; then
        python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        print_success "Python $python_version found"
    else
        print_error "Python 3.8+ is required but not found"
        exit 1
    fi
    
    # Check if pip is available
    if command -v pip &> /dev/null; then
        print_success "pip found"
    else
        print_error "pip is required but not found"
        exit 1
    fi
    
    # Check if Google Cloud SDK is installed
    if command -v gcloud &> /dev/null; then
        print_success "Google Cloud SDK found"
    else
        print_warning "Google Cloud SDK not found. Please install it for authentication."
    fi
    
    print_success "Prerequisites check passed"
}

# Function to install dependencies
install_dependencies() {
    print_info "Installing Python dependencies..."
    
    # Redirect pip output to /dev/null to suppress verbose installation messages
    if pip install -r requirements.txt > /dev/null 2>&1; then
        print_success "Dependencies installed successfully!"
    else
        print_error "Failed to install dependencies"
        exit 1
    fi
}

# Function to create directories
create_directories() {
    print_info "Creating necessary directories..."
    
    mkdir -p benchmark_results
    mkdir -p logs
    mkdir -p result_analysis
    
    print_success "Directories created successfully!"
}

# Function to verify dataset
verify_dataset() {
    print_info "Verifying KLUE STS dataset..."
    
    if [ -f "eval_dataset/klue-sts-v1.1_dev.json" ]; then
        print_success "KLUE STS dataset found"
    else
        print_warning "KLUE STS dataset not found. You may need to download it manually."
    fi
    
    if [ -f "eval_dataset/klue-sts-v1.1_dev_extracted.csv" ]; then
        print_success "KLUE STS extracted dataset found"
    else
        print_warning "KLUE STS extracted dataset not found. You may need to extract it manually."
    fi
}

# Function to check Google Cloud authentication
check_gcloud_auth() {
    print_info "Checking Google Cloud authentication..."
    
    if command -v gcloud &> /dev/null; then
        if gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
            print_success "Google Cloud authentication found"
        else
            print_warning "No active Google Cloud authentication found"
            print_info "Run 'gcloud auth login' to authenticate"
        fi
    else
        print_warning "Google Cloud SDK not found, skipping authentication check"
    fi
}

# Function to run tests
run_tests() {
    print_info "Running setup tests..."
    
    if python3 test_setup.py; then
        print_success "Setup tests passed!"
    else
        print_error "Setup tests failed"
        exit 1
    fi
}

# Function to show help
show_help() {
    echo "KLUE STS Benchmark Setup Script"
    echo ""
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  install    Install Python dependencies only"
    echo "  test       Run setup tests only"
    echo "  full       Complete setup (default)"
    echo "  help       Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0          # Complete setup"
    echo "  $0 install  # Install dependencies only"
    echo "  $0 test     # Run tests only"
}

# Main execution
main() {
    case "${1:-full}" in
        "install")
            print_info "Installing dependencies only..."
            check_prerequisites
            install_dependencies
            print_success "Installation completed!"
            ;;
        "test")
            print_info "Running tests only..."
            run_tests
            ;;
        "full")
            print_info "Running complete setup..."
            check_prerequisites
            install_dependencies
            create_directories
            verify_dataset
            check_gcloud_auth
            run_tests
            print_success "Setup completed successfully!"
            ;;
        "help"|"-h"|"--help")
            show_help
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@" 