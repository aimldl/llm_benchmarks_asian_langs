#!/bin/bash

# KLUE STS Dependencies Installation Script
# This script installs the required Python packages for the KLUE STS benchmark

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

# Function to check Python version
check_python_version() {
    print_info "Checking Python version..."
    
    if command -v python3 &> /dev/null; then
        python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        print_success "Python $python_version found"
        
        # Check if version is 3.8 or higher
        python3 -c "
import sys
if sys.version_info < (3, 8):
    print('Python 3.8+ is required')
    sys.exit(1)
" || {
            print_error "Python 3.8+ is required, but found $python_version"
            exit 1
        }
    else
        print_error "Python 3.8+ is required but not found"
        exit 1
    fi
}

# Function to check pip
check_pip() {
    print_info "Checking pip..."
    
    if command -v pip &> /dev/null; then
        print_success "pip found"
    else
        print_error "pip is required but not found"
        print_info "Please install pip first"
        exit 1
    fi
}

# Function to install dependencies
install_dependencies() {
    print_info "Installing Python packages..."
    
    # Redirect pip output to /dev/null to suppress verbose installation messages
    if pip install -r requirements.txt > /dev/null 2>&1; then
        print_success "Dependencies installed successfully!"
    else
        print_error "Failed to install dependencies"
        print_info "Trying with verbose output for debugging..."
        pip install -r requirements.txt
        exit 1
    fi
}

# Function to verify installation
verify_installation() {
    print_info "Verifying installation..."
    
    # Test imports
    python3 -c "
try:
    import google.genai
    import datasets
    import pandas
    import tqdm
    import sklearn
    import numpy
    print('✓ All packages imported successfully')
except ImportError as e:
    print(f'✗ Import error: {e}')
    exit(1)
" || {
        print_error "Some packages failed to import"
        exit 1
    }
    
    print_success "Installation verification passed!"
}

# Main execution
main() {
    print_info "Starting KLUE STS dependencies installation..."
    
    check_python_version
    check_pip
    install_dependencies
    verify_installation
    
    print_success "Installation completed successfully!"
    print_info "You can now run the benchmark with: ./run test"
}

# Run main function
main "$@" 