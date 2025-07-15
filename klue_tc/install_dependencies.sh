#!/bin/bash

echo "Installing dependencies for KLUE TC Benchmark (Vertex AI)..."

# Check if pip is available
if ! command -v pip &> /dev/null; then
    echo "Error: pip is not installed. Please install Python and pip first."
    exit 1
fi

# Install dependencies
echo "Installing Python packages..."
# Redirect pip output to /dev/null to suppress verbose installation messages
pip install -r requirements.txt > /dev/null 2>&1

# Check if installation was successful
if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed successfully!"
    echo ""
    echo "Next steps for Vertex AI setup:"
    echo "1. Install Google Cloud CLI: https://cloud.google.com/sdk/docs/install"
    echo "2. Authenticate with gcloud: gcloud auth login"
    echo "3. Set up application default credentials: gcloud auth application-default login"
    echo "4. Set your project ID: export GOOGLE_CLOUD_PROJECT='your-project-id'"
    echo "5. Enable Vertex AI API: gcloud services enable aiplatform.googleapis.com"
    echo "6. Test the setup: python test_setup.py"
    echo "7. Run the benchmark: python klue_tc-gemini2_5flash.py --project-id 'your-project-id'"
else
    echo "❌ Failed to install dependencies. Please check the error messages above."
    exit 1
fi 