#!/bin/bash

# KLUE RE Logging Test Script
# This script tests the logging functionality by running a small benchmark

set -e

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

# Function to show usage
show_help() {
    echo "KLUE RE Logging Test Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -s, --samples N    Number of samples to test (default: 5)"
    echo "  -c, --clean        Clean up test files after testing"
    echo "  -v, --verbose      Show verbose output"
    echo "  -h, --help         Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                 # Test with 5 samples"
    echo "  $0 -s 10           # Test with 10 samples"
    echo "  $0 -s 3 -c         # Test with 3 samples and clean up"
    echo ""
    echo "This script will:"
    echo "1. Run a small benchmark test"
    echo "2. Verify log files are created correctly"
    echo "3. Test error extraction functionality"
    echo "4. Validate log file structure"
}

# Function to clean up test files
cleanup_test_files() {
    print_info "Cleaning up test files..."
    
    # Remove test log files
    if [ -d "logs" ]; then
        find logs -name "*test_logging*" -type f -delete
        print_success "Removed test log files"
    fi
    
    # Remove test error files
    if [ -d "result_analysis" ]; then
        find result_analysis -name "*test_logging*" -type f -delete
        print_success "Removed test error files"
    fi
    
    # Remove test benchmark results
    if [ -d "benchmark_results" ]; then
        find benchmark_results -name "*test_logging*" -type f -delete
        print_success "Removed test benchmark results"
    fi
}

# Function to validate log file structure
validate_log_file() {
    local log_file="$1"
    local errors=0
    
    print_info "Validating log file: $log_file"
    
    # Check if file exists
    if [ ! -f "$log_file" ]; then
        print_error "Log file not found: $log_file"
        return 1
    fi
    
    # Check file size
    local file_size=$(wc -c < "$log_file")
    if [ "$file_size" -eq 0 ]; then
        print_error "Log file is empty: $log_file"
        ((errors++))
    else
        print_success "Log file has content ($file_size bytes)"
    fi
    
    # Check for command header
    if grep -q "^\./run" "$log_file"; then
        print_success "Command header found"
    else
        print_error "Command header not found"
        ((errors++))
    fi
    
    # Check for timestamp
    if grep -q "Timestamp:" "$log_file"; then
        print_success "Timestamp found"
    else
        print_error "Timestamp not found"
        ((errors++))
    fi
    
    # Check for working directory
    if grep -q "Working Directory:" "$log_file"; then
        print_success "Working directory found"
    else
        print_error "Working directory not found"
        ((errors++))
    fi
    
    # Check for benchmark output
    if grep -q "KLUE RE Benchmark Results" "$log_file"; then
        print_success "Benchmark results found"
    else
        print_warning "Benchmark results not found (may be normal for small tests)"
    fi
    
    # Check for error analysis
    if grep -q "Error Analysis" "$log_file"; then
        print_success "Error analysis section found"
    else
        print_info "No error analysis section (may be normal if no errors)"
    fi
    
    return $errors
}

# Function to validate error file structure
validate_error_file() {
    local error_file="$1"
    local errors=0
    
    print_info "Validating error file: $error_file"
    
    # Check if file exists
    if [ ! -f "$error_file" ]; then
        print_error "Error file not found: $error_file"
        return 1
    fi
    
    # Check file size
    local file_size=$(wc -c < "$error_file")
    if [ "$file_size" -eq 0 ]; then
        print_error "Error file is empty: $error_file"
        ((errors++))
    else
        print_success "Error file has content ($file_size bytes)"
    fi
    
    # Check for header
    if grep -q "KLUE RE Error Analysis" "$error_file"; then
        print_success "Error analysis header found"
    else
        print_error "Error analysis header not found"
        ((errors++))
    fi
    
    # Check for source file reference
    if grep -q "Source:" "$error_file"; then
        print_success "Source file reference found"
    else
        print_error "Source file reference not found"
        ((errors++))
    fi
    
    # Check for extraction timestamp
    if grep -q "Extracted:" "$error_file"; then
        print_success "Extraction timestamp found"
    else
        print_error "Extraction timestamp not found"
        ((errors++))
    fi
    
    return $errors
}

# Function to test error extraction
test_error_extraction() {
    local log_file="$1"
    
    print_info "Testing error extraction from: $log_file"
    
    # Run error extraction
    if ./get_errors.sh -f "$log_file"; then
        print_success "Error extraction completed"
        
        # Find the generated error file
        local base_name=$(basename "$log_file" .log)
        local error_file="result_analysis/${base_name}_errors.txt"
        
        if [ -f "$error_file" ]; then
            validate_error_file "$error_file"
            return $?
        else
            print_warning "No error file generated (may be normal if no errors)"
            return 0
        fi
    else
        print_error "Error extraction failed"
        return 1
    fi
}

# Function to run the main test
run_test() {
    local samples="$1"
    local verbose="$2"
    
    print_info "Starting KLUE RE logging test with $samples samples"
    
    # Check if required files exist
    if [ ! -f "klue_re-gemini2_5flash.py" ]; then
        print_error "Main script not found: klue_re-gemini2_5flash.py"
        return 1
    fi
    
    if [ ! -f "run" ]; then
        print_error "Run script not found: run"
        return 1
    fi
    
    if [ ! -f "get_errors.sh" ]; then
        print_error "Error extraction script not found: get_errors.sh"
        return 1
    fi
    
    # Check if GOOGLE_CLOUD_PROJECT is set
    if [ -z "$GOOGLE_CLOUD_PROJECT" ]; then
        print_warning "GOOGLE_CLOUD_PROJECT not set, using test mode"
        export GOOGLE_CLOUD_PROJECT="test-project"
    fi
    
    # Create necessary directories
    mkdir -p logs result_analysis benchmark_results
    
    # Run the benchmark test
    print_info "Running benchmark test..."
    if [ "$verbose" = "true" ]; then
        ./run custom "$samples" 2>&1 | tee test_output.log
    else
        ./run custom "$samples" > test_output.log 2>&1
    fi
    
    # Check if benchmark completed
    if [ $? -eq 0 ]; then
        print_success "Benchmark test completed"
    else
        print_error "Benchmark test failed"
        if [ "$verbose" = "true" ]; then
            cat test_output.log
        fi
        return 1
    fi
    
    # Find the generated log file
    local log_file=$(find logs -name "*test_logging*" -o -name "*custom*" | head -n 1)
    
    if [ -z "$log_file" ]; then
        print_error "No log file found"
        return 1
    fi
    
    print_success "Found log file: $log_file"
    
    # Validate log file
    local log_errors=0
    validate_log_file "$log_file" || log_errors=$?
    
    # Test error extraction
    local extraction_errors=0
    test_error_extraction "$log_file" || extraction_errors=$?
    
    # Summary
    print_info "Test Summary:"
    print_info "  Log file validation: $([ $log_errors -eq 0 ] && echo "PASSED" || echo "FAILED ($log_errors errors)")"
    print_info "  Error extraction: $([ $extraction_errors -eq 0 ] && echo "PASSED" || echo "FAILED ($extraction_errors errors)")"
    
    # Clean up test output
    rm -f test_output.log
    
    return $((log_errors + extraction_errors))
}

# Parse command line arguments
SAMPLES=5
CLEANUP=false
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -s|--samples)
            SAMPLES="$2"
            shift 2
            ;;
        -c|--clean)
            CLEANUP=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help|help)
            show_help
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Main execution
echo "=========================================="
echo "KLUE RE Logging Test"
echo "=========================================="
echo ""

# Run the test
if run_test "$SAMPLES" "$VERBOSE"; then
    print_success "All tests passed!"
    
    if [ "$CLEANUP" = "true" ]; then
        cleanup_test_files
    else
        print_info "Test files preserved. Use -c to clean up."
    fi
    
    echo ""
    print_success "Logging system is working correctly!"
    exit 0
else
    print_error "Some tests failed!"
    echo ""
    print_info "Check the output above for details."
    print_info "Test files are preserved for debugging."
    exit 1
fi 