#!/bin/bash

# KLUE NER Logging Test Script
# This script tests the logging functionality of the KLUE NER benchmark

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
show_usage() {
    echo "KLUE NER Logging Test Script"
    echo ""
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  test     Test logging with a small sample (recommended)"
    echo "  full     Test logging with full dataset (time-consuming)"
    echo "  help     Show this help message"
    echo ""
    echo "This script tests:"
    echo "  1. Log file creation and naming"
    echo "  2. Error extraction functionality"
    echo "  3. Command header formatting"
    echo "  4. Log file content structure"
    echo ""
    echo "Examples:"
    echo "  $0 test    # Quick test with small sample"
    echo "  $0 full    # Full test (may take a while)"
}

# Function to check prerequisites
check_prerequisites() {
    print_info "Checking prerequisites..."
    
    # Check if run script exists
    if [ ! -f "run" ]; then
        print_error "run script not found in current directory"
        return 1
    fi
    
    # Check if Python script exists
    if [ ! -f "klue_ner-gemini2_5flash.py" ]; then
        print_error "klue_ner-gemini2_5flash.py not found in current directory"
        return 1
    fi
    
    # Check if GOOGLE_CLOUD_PROJECT is set
    if [ -z "$GOOGLE_CLOUD_PROJECT" ]; then
        print_error "GOOGLE_CLOUD_PROJECT environment variable is not set"
        print_info "Please set it with: export GOOGLE_CLOUD_PROJECT='your-project-id'"
        return 1
    fi
    
    # Check if logs directory exists
    if [ ! -d "logs" ]; then
        print_warning "logs directory not found, creating it..."
        mkdir -p logs
    fi
    
    print_success "Prerequisites check passed"
    return 0
}

# Function to test log file creation
test_log_creation() {
    local test_mode="$1"
    local sample_count="$2"
    
    print_info "Testing log file creation with mode: $test_mode"
    
    # Get initial log file count
    initial_count=$(ls -1 logs/klue_ner_*.log 2>/dev/null | wc -l)
    
    # Run the benchmark
    print_info "Running benchmark: ./run $test_mode"
    if ./run "$test_mode" > /dev/null 2>&1; then
        print_success "Benchmark completed successfully"
    else
        print_error "Benchmark failed"
        return 1
    fi
    
    # Check if new log files were created
    final_count=$(ls -1 logs/klue_ner_*.log 2>/dev/null | wc -l)
    new_files=$((final_count - initial_count))
    
    if [ "$new_files" -gt 0 ]; then
        print_success "Created $new_files new log file(s)"
    else
        print_error "No new log files were created"
        return 1
    fi
    
    return 0
}

# Function to test log file structure
test_log_structure() {
    local log_file="$1"
    
    print_info "Testing log file structure: $log_file"
    
    if [ ! -f "$log_file" ]; then
        print_error "Log file not found: $log_file"
        return 1
    fi
    
    # Check command header
    if grep -q "^\./run" "$log_file"; then
        print_success "Command header found"
    else
        print_error "Command header not found"
        return 1
    fi
    
    # Check timestamp
    if grep -q "Timestamp:" "$log_file"; then
        print_success "Timestamp found"
    else
        print_error "Timestamp not found"
        return 1
    fi
    
    # Check working directory
    if grep -q "Working Directory:" "$log_file"; then
        print_success "Working directory found"
    else
        print_error "Working directory not found"
        return 1
    fi
    
    # Check for benchmark content
    if grep -q "KLUE NER" "$log_file"; then
        print_success "Benchmark content found"
    else
        print_warning "Benchmark content not found (may be normal for failed runs)"
    fi
    
    return 0
}

# Function to test error extraction
test_error_extraction() {
    local log_file="$1"
    local err_file="${log_file%.log}.err"
    
    print_info "Testing error extraction: $err_file"
    
    if [ ! -f "$err_file" ]; then
        print_error "Error file not found: $err_file"
        return 1
    fi
    
    # Check error file structure
    if grep -q "^\./run" "$err_file"; then
        print_success "Error file has command header"
    else
        print_error "Error file missing command header"
        return 1
    fi
    
    # Check if error file is smaller than log file
    log_size=$(wc -c < "$log_file")
    err_size=$(wc -c < "$err_file")
    
    if [ "$err_size" -lt "$log_size" ]; then
        print_success "Error file is smaller than log file (as expected)"
    else
        print_warning "Error file is not smaller than log file"
    fi
    
    return 0
}

# Function to test log file naming
test_log_naming() {
    print_info "Testing log file naming convention..."
    
    # Find the most recent log file
    latest_log=$(ls -t logs/klue_ner_*.log 2>/dev/null | head -1)
    
    if [ -z "$latest_log" ]; then
        print_error "No log files found"
        return 1
    fi
    
    # Extract filename without path
    filename=$(basename "$latest_log")
    
    # Check naming pattern
    if [[ "$filename" =~ ^klue_ner_[a-z]+_[0-9a-z]+_[0-9]{8}_[0-9]{6}\.log$ ]]; then
        print_success "Log file naming convention is correct: $filename"
    else
        print_error "Log file naming convention is incorrect: $filename"
        print_info "Expected pattern: klue_ner_[mode]_[samples]_[YYYYMMDD]_[HHMMSS].log"
        return 1
    fi
    
    return 0
}

# Function to run comprehensive test
run_comprehensive_test() {
    local test_mode="$1"
    local sample_count="$2"
    
    print_info "Running comprehensive logging test..."
    print_info "Test mode: $test_mode, Sample count: $sample_count"
    
    # Test 1: Log file creation
    if ! test_log_creation "$test_mode" "$sample_count"; then
        print_error "Log file creation test failed"
        return 1
    fi
    
    # Test 2: Log file naming
    if ! test_log_naming; then
        print_error "Log file naming test failed"
        return 1
    fi
    
    # Test 3: Log file structure
    latest_log=$(ls -t logs/klue_ner_*.log 2>/dev/null | head -1)
    if ! test_log_structure "$latest_log"; then
        print_error "Log file structure test failed"
        return 1
    fi
    
    # Test 4: Error extraction
    if ! test_error_extraction "$latest_log"; then
        print_error "Error extraction test failed"
        return 1
    fi
    
    print_success "All logging tests passed!"
    return 0
}

# Function to show test results
show_test_results() {
    print_info "Test Results Summary:"
    echo ""
    
    # Show recent log files
    echo "Recent log files:"
    ls -lt logs/klue_ner_*.log 2>/dev/null | head -5 | while read line; do
        echo "  $line"
    done
    
    echo ""
    
    # Show recent error files
    echo "Recent error files:"
    ls -lt logs/klue_ner_*.err 2>/dev/null | head -5 | while read line; do
        echo "  $line"
    done
    
    echo ""
    print_info "Log files are stored in the 'logs/' directory"
    print_info "Error files contain only error-related information for easier debugging"
}

# Main script logic
main() {
    case "${1:-help}" in
        "test")
            if ! check_prerequisites; then
                exit 1
            fi
            
            if run_comprehensive_test "test" "10"; then
                show_test_results
                print_success "Logging test completed successfully!"
            else
                print_error "Logging test failed!"
                exit 1
            fi
            ;;
        "full")
            if ! check_prerequisites; then
                exit 1
            fi
            
            print_warning "Full test may take a long time..."
            read -p "Continue? (y/N): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                if run_comprehensive_test "full" "all"; then
                    show_test_results
                    print_success "Full logging test completed successfully!"
                else
                    print_error "Full logging test failed!"
                    exit 1
                fi
            else
                print_info "Test cancelled"
            fi
            ;;
        "help"|"-h"|"--help")
            show_usage
            ;;
        *)
            print_error "Unknown option: $1"
            echo ""
            show_usage
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@" 