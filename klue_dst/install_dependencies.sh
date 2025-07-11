#!/bin/bash

# KLUE DST Dependencies Installation Script
# This script installs the required dependencies for KLUE DST benchmarking

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

echo "=========================================="
echo "KLUE DST Dependencies Installation"
echo "=========================================="

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    print_error "requirements.txt not found in current directory"
    print_error "Please run this script from the klue_dst directory"
    exit 1
fi

print_status "Starting dependency installation..."

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

# Upgrade pip
print_status "Upgrading pip..."
$PIP_CMD install --upgrade pip
print_success "pip upgraded"

# Install requirements
print_status "Installing Python dependencies from requirements.txt..."
$PIP_CMD install -r requirements.txt
print_success "Dependencies installed"

# Verify installations
print_status "Verifying installations..."

# Test google.genai
python3 -c "
try:
    import google.genai
    print('✓ google.genai imported successfully')
except ImportError as e:
    print(f'✗ Failed to import google.genai: {e}')
    exit(1)
"

# Test datasets
python3 -c "
try:
    from datasets import load_dataset
    print('✓ datasets imported successfully')
except ImportError as e:
    print(f'✗ Failed to import datasets: {e}')
    exit(1)
"

# Test pandas
python3 -c "
try:
    import pandas as pd
    print('✓ pandas imported successfully')
except ImportError as e:
    print(f'✗ Failed to import pandas: {e}')
    exit(1)
"

# Test tqdm
python3 -c "
try:
    from tqdm import tqdm
    print('✓ tqdm imported successfully')
except ImportError as e:
    print(f'✗ Failed to import tqdm: {e}')
    exit(1)
"

print_success "All dependencies verified successfully!"

print_status "Installation completed successfully!"
echo ""
print_success "KLUE DST dependencies are ready"
echo ""
print_status "Next steps:"
echo "1. Set your Google Cloud project ID:"
echo "   export GOOGLE_CLOUD_PROJECT='your-project-id'"
echo ""
echo "2. Test the setup:"
echo "   python3 test_setup.py"
echo ""
echo "3. Run a test benchmark:"
echo "   ./run test" 