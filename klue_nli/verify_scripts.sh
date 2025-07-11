#!/bin/bash

# KLUE NLI Benchmark - Verify Scripts Script
# This script verifies that all required scripts are present and executable

set -e  # Exit on any error

echo "Verifying KLUE NLI benchmark scripts..."

# List of required files
required_files=(
    "klue_nli-gemini2_5flash.py"
    "requirements.txt"
    "setup.sh"
    "run"
    "test_setup.py"
    "install_dependencies.sh"
    "README.md"
    "get_errors.sh"
)

# Check if all required files exist
missing_files=()
for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        missing_files+=("$file")
    fi
done

if [ ${#missing_files[@]} -gt 0 ]; then
    echo "❌ Missing required files:"
    for file in "${missing_files[@]}"; do
        echo "  - $file"
    done
    exit 1
fi

echo "✅ All required files found!"

# Make scripts executable
echo "Making scripts executable..."
chmod +x setup.sh
chmod +x run
chmod +x install_dependencies.sh
chmod +x verify_scripts.sh
chmod +x get_errors.sh

echo "✅ Scripts are now executable!"
echo ""
echo "You can now run:"
echo "  ./setup.sh full    # Complete setup"
echo "  ./run test         # Run small test"
echo "  ./run full         # Run full benchmark"
echo "  ./get_errors.sh    # Extract error details from results" 