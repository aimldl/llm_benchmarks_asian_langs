#!/bin/bash

# KLUE RE Setup Script
# This script sets up the environment for running KLUE Relation Extraction benchmarks

set -e  # Exit on any error

echo "=========================================="
echo "KLUE RE (Relation Extraction) Setup Script"
echo "=========================================="
echo ""

# Check if we're in the right directory
if [ ! -f "klue_re-gemini2_5flash.py" ]; then
    echo "Error: This script must be run from the klue_re directory"
    echo "Please navigate to the klue_re directory and run this script again"
    exit 1
fi

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    echo "Error: pip3 is not installed or not in PATH"
    echo "Please install pip3"
    exit 1
fi

echo "✅ pip3 found: $(pip3 --version)"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo "✅ Virtual environment activated"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip
echo "✅ pip upgraded"

# Install requirements
echo "Installing required packages..."
if [ -f "requirements.txt" ]; then
    # Redirect pip output to /dev/null to suppress verbose installation messages
    pip install -r requirements.txt > /dev/null 2>&1
    echo "✅ Requirements installed from requirements.txt"
else
    echo "Warning: requirements.txt not found, installing basic packages..."
    # Redirect pip output to /dev/null to suppress verbose installation messages
    pip install google-genai datasets pandas tqdm google-cloud-aiplatform > /dev/null 2>&1
    echo "✅ Basic packages installed"
fi

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p logs
mkdir -p benchmark_results
mkdir -p result_analysis
mkdir -p eval_dataset
echo "✅ Directories created"

# Make scripts executable
echo "Making scripts executable..."
chmod +x run
chmod +x setup.sh
chmod +x install_dependencies.sh
chmod +x test_setup.py
chmod +x get_errors.sh
chmod +x test_logging.sh
chmod +x verify_scripts.sh
echo "✅ Scripts made executable"

# Check Google Cloud setup
echo ""
echo "Checking Google Cloud setup..."
if ! command -v gcloud &> /dev/null; then
    echo "⚠️  Warning: gcloud CLI not found"
    echo "   You may need to install Google Cloud SDK"
    echo "   Visit: https://cloud.google.com/sdk/docs/install"
else
    echo "✅ gcloud CLI found: $(gcloud --version | head -n 1)"
    
    # Check if user is authenticated
    if gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
        echo "✅ User is authenticated with gcloud"
        echo "   Active account: $(gcloud auth list --filter=status:ACTIVE --format='value(account)')"
    else
        echo "⚠️  Warning: No active gcloud authentication found"
        echo "   Run: gcloud auth login"
    fi
    
    # Check if project is set
    if [ -n "$(gcloud config get-value project 2>/dev/null)" ]; then
        echo "✅ Google Cloud project is set"
        echo "   Project: $(gcloud config get-value project)"
    else
        echo "⚠️  Warning: No Google Cloud project set"
        echo "   Run: gcloud config set project YOUR_PROJECT_ID"
    fi
fi

# Check environment variables
echo ""
echo "Checking environment variables..."
if [ -n "$GOOGLE_CLOUD_PROJECT" ]; then
    echo "✅ GOOGLE_CLOUD_PROJECT is set: $GOOGLE_CLOUD_PROJECT"
else
    echo "⚠️  Warning: GOOGLE_CLOUD_PROJECT environment variable is not set"
    echo "   You can set it with: export GOOGLE_CLOUD_PROJECT='your-project-id'"
    echo "   Or use the --project-id flag when running the benchmark"
fi

# Test basic imports
echo ""
echo "Testing Python imports..."
python3 -c "
try:
    import google.genai
    print('✅ google.genai imported successfully')
except ImportError as e:
    print(f'❌ Failed to import google.genai: {e}')
    exit(1)

try:
    from datasets import load_dataset
    print('✅ datasets imported successfully')
except ImportError as e:
    print(f'❌ Failed to import datasets: {e}')
    exit(1)

try:
    import pandas as pd
    print('✅ pandas imported successfully')
except ImportError as e:
    print(f'❌ Failed to import pandas: {e}')
    exit(1)

try:
    from tqdm import tqdm
    print('✅ tqdm imported successfully')
except ImportError as e:
    print(f'❌ Failed to import tqdm: {e}')
    exit(1)

print('✅ All required packages imported successfully')
"

# Test dataset loading
echo ""
echo "Testing dataset loading..."
python3 -c "
try:
    from datasets import load_dataset
    print('Loading KLUE RE dataset...')
    dataset = load_dataset('klue', 're', split='validation')
    print(f'✅ Dataset loaded successfully: {len(dataset)} samples')
    print(f'   Sample keys: {list(dataset[0].keys())}')
except Exception as e:
    print(f'❌ Failed to load dataset: {e}')
    exit(1)
"

echo ""
echo "=========================================="
echo "Setup completed successfully!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Make sure you have a Google Cloud project with Vertex AI enabled"
echo "2. Set your project ID: export GOOGLE_CLOUD_PROJECT='your-project-id'"
echo "3. Run a test: ./run test"
echo "4. Run full benchmark: ./run full"
echo "5. Run custom benchmark: ./run custom 100"
echo ""
echo "For more information, see README.md"
echo "" 