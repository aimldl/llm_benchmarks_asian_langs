#!/bin/bash

# KLUE NLI Benchmark - Install Dependencies Script
# This script installs the required Python dependencies for the KLUE NLI benchmark

set -e  # Exit on any error

echo "Installing KLUE NLI benchmark dependencies..."

# Check if requirements.txt exists
if [ ! -f "requirements.txt" ]; then
    echo "Error: requirements.txt not found in current directory"
    exit 1
fi

# Install dependencies
echo "Installing Python packages from requirements.txt..."
pip install -r requirements.txt

echo "✅ Dependencies installed successfully!"
echo ""
echo "Next steps:"
echo "1. Set up Google Cloud authentication"
echo "2. Set your project ID: export GOOGLE_CLOUD_PROJECT='your-project-id'"
echo "3. Run the benchmark: python klue_nli-gemini2_5flash.py --project-id 'your-project-id'" 