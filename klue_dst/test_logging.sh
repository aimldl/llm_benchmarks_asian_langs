#!/bin/bash

# KLUE DST Logging Test Script
# This script tests the logging functionality of the KLUE DST benchmark

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

# Function to show usage
show_help() {
    echo "KLUE DST Logging Test Script"
    echo ""
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  test     Test logging with a small sample (10 samples)"
    echo "  custom N Test logging with N samples"
    echo "  verify   Verify existing log files"
    echo "  clean    Clean up test log files"
    echo "  help     Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 test        # Test with 10 samples"
    echo "  $0 custom 5    # Test with 5 samples"
    echo "  $0 verify      # Verify existing logs"
    echo "  $0 clean       # Clean up test logs"
    echo ""
    echo "This script tests the logging functionality by running small benchmarks"
    echo "and verifying that log files are created correctly with proper error extraction."
}

# Function to check if GOOGLE_CLOUD_PROJECT is set
check_environment() {
    if [ -z "$GOOGLE_CLOUD_PROJECT" ]; then
        print_error "GOOGLE_CLOUD_PROJECT environment variable is not set"
        print_status "Please set it with: export GOOGLE_CLOUD_PROJECT='your-project-id'"
        return 1
    fi
    
    print_success "GOOGLE_CLOUD_PROJECT is set: $GOOGLE_CLOUD_PROJECT"
    return 0
}

# Function to check if required files exist
check_files() {
    if [ ! -f "klue_dst-gemini2_5flash.py" ]; then
        print_error "klue_dst-gemini2_5flash.py not found"
        return 1
    fi
    
    if [ ! -f "run" ]; then
        print_error "run script not found"
        return 1
    fi
    
    if [ ! -x "run" ]; then
        print_warning "run script is not executable, making it executable"
        chmod +x run
    fi
    
    print_success "Required files found and executable"
    return 0
}

# Function to create test log files
create_test_logs() {
    local samples=$1
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local test_log="logs/test_logging_${timestamp}.log"
    local test_err="logs/test_logging_${timestamp}.err"
    
    print_status "Creating test log files with $samples samples..."
    
    # Create logs directory if it doesn't exist
    mkdir -p logs
    
    # Create a mock log file with typical DST benchmark output
    {
        echo "./run test"
        echo "Timestamp: $(date)"
        echo "Working Directory: $(pwd)"
        echo "========================================"
        echo ""
        echo "Running small test with $samples samples..."
        echo "Loading KLUE DST dataset for dialogue state tracking..."
        echo "✅ Successfully loaded $samples samples."
        echo "Starting benchmark..."
        echo "Processing samples: 100%|████████████████████████████████████████| $samples/$samples [00:30<00:00,  3.33it/s]"
        echo ""
        echo "Benchmark completed!"
        echo "Intent Accuracy: 0.8500"
        echo "Requested Slots F1: 0.7200"
        echo "Slot Values F1: 0.6800"
        echo "Overall F1: 0.7000"
        echo "Total time: 30.50 seconds"
        echo "Average time per sample: 3.050 seconds"
        echo ""
        echo "=" * 60
        echo "KLUE Dialogue State Tracking Benchmark Results"
        echo "=" * 60
        echo "Model: gemini-2.5-flash"
        echo "Platform: Google Cloud Vertex AI"
        echo "Project: $GOOGLE_CLOUD_PROJECT"
        echo "Location: us-central1"
        echo "Intent Accuracy: 0.8500"
        echo "Requested Slots F1: 0.7200"
        echo "Slot Values F1: 0.6800"
        echo "Overall F1: 0.7000"
        echo "Total Samples: $samples"
        echo "Total Time: 30.50 seconds"
        echo "Average Time per Sample: 3.050 seconds"
        echo "Samples per Second: 0.33"
        echo ""
        echo "Per-Domain Performance:"
        echo "  restaurant: F1 = 0.7500 (n=5)"
        echo "  hotel: F1 = 0.6500 (n=3)"
        echo "  movie: F1 = 0.7000 (n=2)"
        echo ""
        echo "Error Analysis (showing first 2 errors):"
        echo "  1. Sample ID: dst_001"
        echo "     Turn ID: 3"
        echo "     Ground Truth Intent: request"
        echo "     Predicted Intent: inform"
        echo "     Overall F1: 0.2500"
        echo ""
        echo "  2. Sample ID: dst_002"
        echo "     Turn ID: 5"
        echo "     Ground Truth Intent: book"
        echo "     Predicted Intent: request"
        echo "     Overall F1: 0.4000"
        echo ""
        echo "2024-12-01 12:00:00,123 - INFO - Benchmark completed successfully"
        echo "2024-12-01 12:00:00,124 - INFO - Results saved to benchmark_results/"
        echo "2024-12-01 12:00:00,125 - ERROR - Failed to process sample dst_001: API timeout"
        echo "2024-12-01 12:00:00,126 - ERROR - Failed to process sample dst_002: Invalid response format"
        echo "2024-12-01 12:00:00,127 - WARNING - Some samples had low confidence scores"
    } > "$test_log"
    
    # Create corresponding .err file
    {
        echo "./run test"
        echo "Timestamp: $(date)"
        echo "Working Directory: $(pwd)"
        echo "========================================"
        echo ""
        echo "Error Analysis (showing first 2 errors):"
        echo "  1. Sample ID: dst_001"
        echo "     Turn ID: 3"
        echo "     Ground Truth Intent: request"
        echo "     Predicted Intent: inform"
        echo "     Overall F1: 0.2500"
        echo ""
        echo "  2. Sample ID: dst_002"
        echo "     Turn ID: 5"
        echo "     Ground Truth Intent: book"
        echo "     Predicted Intent: request"
        echo "     Overall F1: 0.4000"
        echo ""
        echo "2024-12-01 12:00:00,125 - ERROR - Failed to process sample dst_001: API timeout"
        echo "2024-12-01 12:00:00,126 - ERROR - Failed to process sample dst_002: Invalid response format"
    } > "$test_err"
    
    print_success "Test log files created:"
    echo "  Full log: $test_log"
    echo "  Error log: $test_err"
    
    return 0
}

# Function to verify log files
verify_log_files() {
    local log_file="$1"
    local err_file="$2"
    
    print_status "Verifying log files..."
    
    # Check if files exist
    if [ ! -f "$log_file" ]; then
        print_error "Log file not found: $log_file"
        return 1
    fi
    
    if [ ! -f "$err_file" ]; then
        print_error "Error file not found: $err_file"
        return 1
    fi
    
    print_success "Both log and error files exist"
    
    # Check log file content
    echo "Log file content check:"
    
    # Check for command header
    if grep -q "^\./run" "$log_file"; then
        print_success "✓ Command header found"
    else
        print_error "✗ Command header missing"
        return 1
    fi
    
    # Check for timestamp
    if grep -q "Timestamp:" "$log_file"; then
        print_success "✓ Timestamp found"
    else
        print_error "✗ Timestamp missing"
        return 1
    fi
    
    # Check for benchmark results
    if grep -q "Intent Accuracy:" "$log_file"; then
        print_success "✓ Intent Accuracy metric found"
    else
        print_error "✗ Intent Accuracy metric missing"
        return 1
    fi
    
    if grep -q "Requested Slots F1:" "$log_file"; then
        print_success "✓ Requested Slots F1 metric found"
    else
        print_error "✗ Requested Slots F1 metric missing"
        return 1
    fi
    
    if grep -q "Slot Values F1:" "$log_file"; then
        print_success "✓ Slot Values F1 metric found"
    else
        print_error "✗ Slot Values F1 metric missing"
        return 1
    fi
    
    if grep -q "Overall F1:" "$log_file"; then
        print_success "✓ Overall F1 metric found"
    else
        print_error "✗ Overall F1 metric missing"
        return 1
    fi
    
    # Check for error analysis section
    if grep -q "Error Analysis" "$log_file"; then
        print_success "✓ Error analysis section found"
    else
        print_error "✗ Error analysis section missing"
        return 1
    fi
    
    # Check error file content
    echo ""
    echo "Error file content check:"
    
    # Check for command header in error file
    if grep -q "^\./run" "$err_file"; then
        print_success "✓ Command header found in error file"
    else
        print_error "✗ Command header missing in error file"
        return 1
    fi
    
    # Check for error analysis in error file
    if grep -q "Error Analysis" "$err_file"; then
        print_success "✓ Error analysis found in error file"
    else
        print_error "✗ Error analysis missing in error file"
        return 1
    fi
    
    # Check for ERROR logs in error file
    if grep -q "ERROR -" "$err_file"; then
        print_success "✓ ERROR logs found in error file"
    else
        print_error "✗ ERROR logs missing in error file"
        return 1
    fi
    
    # Check that error file is smaller than log file
    log_size=$(wc -c < "$log_file")
    err_size=$(wc -c < "$err_file")
    
    if [ "$err_size" -lt "$log_size" ]; then
        print_success "✓ Error file is smaller than log file (as expected)"
    else
        print_warning "⚠ Error file is not smaller than log file"
    fi
    
    return 0
}

# Function to test actual logging with run script
test_actual_logging() {
    local samples=$1
    
    print_status "Testing actual logging with $samples samples..."
    
    # Check environment and files first
    if ! check_environment; then
        return 1
    fi
    
    if ! check_files; then
        return 1
    fi
    
    # Run the benchmark with the specified number of samples
    print_status "Running benchmark with $samples samples..."
    
    # Capture the output to see what files are created
    output=$(./run custom "$samples" 2>&1)
    
    # Find the created log files
    log_files=$(find logs -name "klue_dst_custom_${samples}samples_*.log" -type f 2>/dev/null || true)
    err_files=$(find logs -name "klue_dst_custom_${samples}samples_*.err" -type f 2>/dev/null || true)
    
    if [ -n "$log_files" ]; then
        print_success "Log files created:"
        for log_file in $log_files; do
            echo "  $log_file"
        done
    else
        print_error "No log files created"
        return 1
    fi
    
    if [ -n "$err_files" ]; then
        print_success "Error files created:"
        for err_file in $err_files; do
            echo "  $err_file"
        done
    else
        print_error "No error files created"
        return 1
    fi
    
    # Verify the most recent log file
    latest_log=$(find logs -name "klue_dst_custom_${samples}samples_*.log" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
    latest_err=$(find logs -name "klue_dst_custom_${samples}samples_*.err" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
    
    if [ -n "$latest_log" ] && [ -n "$latest_err" ]; then
        verify_log_files "$latest_log" "$latest_err"
        return $?
    else
        print_error "Could not find latest log files for verification"
        return 1
    fi
}

# Function to verify existing log files
verify_existing_logs() {
    print_status "Verifying existing log files..."
    
    if [ ! -d "logs" ]; then
        print_error "Logs directory not found"
        return 1
    fi
    
    log_files=$(find logs -name "klue_dst_*.log" -type f 2>/dev/null || true)
    
    if [ -z "$log_files" ]; then
        print_warning "No KLUE DST log files found in logs directory"
        return 0
    fi
    
    print_status "Found $(echo "$log_files" | wc -l) log files"
    
    for log_file in $log_files; do
        echo ""
        print_status "Verifying: $log_file"
        
        # Find corresponding error file
        err_file="${log_file%.log}.err"
        
        if [ -f "$err_file" ]; then
            verify_log_files "$log_file" "$err_file"
            if [ $? -eq 0 ]; then
                print_success "✓ Verification passed for $log_file"
            else
                print_error "✗ Verification failed for $log_file"
            fi
        else
            print_warning "⚠ No corresponding error file found for $log_file"
        fi
    done
    
    return 0
}

# Function to clean up test log files
clean_test_logs() {
    print_status "Cleaning up test log files..."
    
    if [ ! -d "logs" ]; then
        print_warning "Logs directory not found, nothing to clean"
        return 0
    fi
    
    # Find and remove test logging files
    test_files=$(find logs -name "test_logging_*.log" -o -name "test_logging_*.err" 2>/dev/null || true)
    
    if [ -n "$test_files" ]; then
        print_status "Removing test files:"
        for file in $test_files; do
            echo "  $file"
            rm -f "$file"
        done
        print_success "Test files cleaned up"
    else
        print_warning "No test files found to clean"
    fi
    
    return 0
}

# Main script logic
case "${1:-help}" in
    "test")
        create_test_logs 10
        if [ $? -eq 0 ]; then
            # Find the created test files
            test_log=$(find logs -name "test_logging_*.log" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
            test_err=$(find logs -name "test_logging_*.err" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
            
            if [ -n "$test_log" ] && [ -n "$test_err" ]; then
                verify_log_files "$test_log" "$test_err"
                if [ $? -eq 0 ]; then
                    print_success "Logging test completed successfully!"
                else
                    print_error "Logging test failed verification"
                    exit 1
                fi
            else
                print_error "Could not find created test files"
                exit 1
            fi
        else
            print_error "Failed to create test log files"
            exit 1
        fi
        ;;
    "custom")
        if [ -z "$2" ] || ! [[ "$2" =~ ^[0-9]+$ ]]; then
            print_error "Please provide a valid number of samples for custom mode"
            echo "Usage: $0 custom N (where N is a number)"
            exit 1
        fi
        
        test_actual_logging "$2"
        if [ $? -eq 0 ]; then
            print_success "Actual logging test completed successfully!"
        else
            print_error "Actual logging test failed"
            exit 1
        fi
        ;;
    "verify")
        verify_existing_logs
        ;;
    "clean")
        clean_test_logs
        ;;
    "help"|*)
        show_help
        ;;
esac 