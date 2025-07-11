#!/bin/bash

# KLUE DST Setup Script
# This script sets up the environment for running KLUE Dialogue State Tracking benchmarks

set -e  # Exit on any error

echo "=========================================="
echo "KLUE DST (Dialogue State Tracking) Setup"
echo "=========================================="

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

# Check if we're in the right directory
if [ ! -f "klue_dst-gemini2_5flash.py" ]; then
    print_error "klue_dst-gemini2_5flash.py not found in current directory"
    print_error "Please run this script from the klue_dst directory"
    exit 1
fi

print_status "Starting KLUE DST setup..."

# Check Python version
print_status "Checking Python version..."
python_version=$(python3 --version 2>&1)
print_success "Found: $python_version"

# Check if pip is available
print_status "Checking pip availability..."
if command -v pip3 &> /dev/null; then
    print_success "pip3 is available"
    PIP_CMD="pip3"
elif command -v pip &> /dev/null; then
    print_success "pip is available"
    PIP_CMD="pip"
else
    print_error "Neither pip3 nor pip found. Please install pip first."
    exit 1
fi

# Check if virtual environment exists
if [ -d "venv" ]; then
    print_warning "Virtual environment 'venv' already exists"
    read -p "Do you want to recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Removing existing virtual environment..."
        rm -rf venv
        print_success "Removed existing virtual environment"
    else
        print_status "Using existing virtual environment"
    fi
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    print_status "Creating virtual environment..."
    python3 -m venv venv
    print_success "Virtual environment created"
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source venv/bin/activate
print_success "Virtual environment activated"

# Upgrade pip
print_status "Upgrading pip..."
$PIP_CMD install --upgrade pip
print_success "pip upgraded"

# Install requirements
print_status "Installing Python dependencies..."
if [ -f "requirements.txt" ]; then
    $PIP_CMD install -r requirements.txt
    print_success "Dependencies installed"
else
    print_error "requirements.txt not found"
    exit 1
fi

# Create necessary directories
print_status "Creating necessary directories..."
mkdir -p logs
mkdir -p benchmark_results
mkdir -p result_analysis
mkdir -p eval_dataset
print_success "Directories created"

# Make scripts executable
print_status "Making scripts executable..."
chmod +x run
chmod +x setup.sh
chmod +x install_dependencies.sh
chmod +x test_setup.py
chmod +x get_errors.sh
chmod +x test_logging.sh
chmod +x verify_scripts.sh
print_success "Scripts made executable"

# Check Google Cloud setup
print_status "Checking Google Cloud setup..."
if command -v gcloud &> /dev/null; then
    print_success "gcloud CLI found"
    
    # Check if user is authenticated
    if gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
        print_success "Google Cloud authentication found"
    else
        print_warning "No active Google Cloud authentication found"
        print_status "You may need to run: gcloud auth login"
    fi
    
    # Check if project is set
    if [ -n "$GOOGLE_CLOUD_PROJECT" ]; then
        print_success "GOOGLE_CLOUD_PROJECT environment variable is set: $GOOGLE_CLOUD_PROJECT"
    else
        print_warning "GOOGLE_CLOUD_PROJECT environment variable is not set"
        print_status "You will need to set it before running benchmarks:"
        print_status "export GOOGLE_CLOUD_PROJECT='your-project-id'"
    fi
else
    print_warning "gcloud CLI not found"
    print_status "You may need to install Google Cloud SDK"
fi

# Test basic imports
print_status "Testing Python imports..."
python3 -c "
import sys
print('Python version:', sys.version)

try:
    import google.genai
    print('✓ google.genai imported successfully')
except ImportError as e:
    print('✗ Failed to import google.genai:', e)
    sys.exit(1)

try:
    from datasets import load_dataset
    print('✓ datasets imported successfully')
except ImportError as e:
    print('✗ Failed to import datasets:', e)
    sys.exit(1)

try:
    import pandas as pd
    print('✓ pandas imported successfully')
except ImportError as e:
    print('✗ Failed to import pandas:', e)
    sys.exit(1)

try:
    from tqdm import tqdm
    print('✓ tqdm imported successfully')
except ImportError as e:
    print('✗ Failed to import tqdm:', e)
    sys.exit(1)

print('All required packages imported successfully!')
"

if [ $? -eq 0 ]; then
    print_success "All Python imports successful"
else
    print_error "Some Python imports failed"
    exit 1
fi

# Test dataset loading
print_status "Testing dataset loading..."
python3 -c "
from datasets import load_dataset
try:
    print('Loading KLUE DST dataset...')
    dataset = load_dataset('klue', 'dst', split='validation')
    print(f'✓ Dataset loaded successfully: {len(dataset)} samples')
    print(f'✓ Dataset features: {list(dataset.features.keys())}')
except Exception as e:
    print(f'✗ Failed to load dataset: {e}')
    exit(1)
"

if [ $? -eq 0 ]; then
    print_success "Dataset loading test successful"
else
    print_error "Dataset loading test failed"
    exit 1
fi

print_status "Setup completed successfully!"
echo ""
print_success "KLUE DST environment is ready for benchmarking"
echo ""
print_status "Next steps:"
echo "1. Set your Google Cloud project ID:"
echo "   export GOOGLE_CLOUD_PROJECT='your-project-id'"
echo ""
echo "2. Run a test benchmark:"
echo "   ./run test"
echo ""
echo "3. Run the full benchmark:"
echo "   ./run full"
echo ""
echo "4. Run with custom number of samples:"
echo "   ./run custom 100"
echo ""
print_status "Log files will be saved to the 'logs' directory"
print_status "Results will be saved to the 'benchmark_results' directory" 