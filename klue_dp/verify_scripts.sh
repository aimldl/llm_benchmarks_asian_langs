#!/bin/bash

# KLUE DP Benchmark Script Verification
# This script verifies that all Bash scripts are properly saved and executable

echo "Verifying KLUE DP Benchmark Bash scripts..."
echo "=========================================="

# List of expected scripts
scripts=("run" "install_dependencies.sh" "setup.sh" "verify_scripts.sh" "get_errors.sh" "test_logging.sh")

# Check each script
for script in "${scripts[@]}"; do
    if [ -f "$script" ]; then
        if [ -x "$script" ]; then
            echo "✅ $script - Found and executable"
        else
            echo "⚠️  $script - Found but not executable"
            chmod +x "$script"
            echo "   Made executable"
        fi
    else
        echo "❌ $script - Not found"
    fi
done

echo ""
echo "Script verification complete!"
echo ""
echo "Available scripts:"
echo "  ./run                    - Quick benchmark runner"
echo "  ./install_dependencies.sh - Install Python dependencies"
echo "  ./setup.sh               - Complete setup process"
echo "  ./verify_scripts.sh      - This verification script"
echo "  ./get_errors.sh          - Extract error details from results"
echo "  ./test_logging.sh        - Test logging functionality" 