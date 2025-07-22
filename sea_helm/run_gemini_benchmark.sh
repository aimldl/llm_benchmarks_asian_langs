#!/bin/bash

# SEA-HELM Gemini 2.5 Flash Benchmark Runner
# This script provides an easy way to run the SEA-HELM benchmark with Gemini 2.5 Flash

set -e  # Exit on any error

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
    echo "SEA-HELM Gemini 2.5 Flash Benchmark Runner"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Required Options:"
    echo "  --project-id PROJECT_ID    Google Cloud project ID"
    echo ""
    echo "Optional Options:"
    echo "  --model-name MODEL         Gemini model name (default: gemini-2.5-flash)"
    echo "  --location LOCATION        Google Cloud location (default: us-central1)"
    echo "  --output-dir DIR           Output directory (default: benchmark_results)"
    echo "  --tasks-config TASKS       Tasks configuration (default: seahelm)"
    echo "  --max-tokens N             Maximum tokens to generate (default: 2048)"
    echo "  --temperature T            Sampling temperature (default: 0.1)"
    echo "  --limit N                  Limit samples per task"
    echo "  --base-model               Treat as base model"
    echo "  --test                     Run integration test only"
    echo "  --help                     Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --project-id my-project-123"
    echo "  $0 --project-id my-project-123 --limit 100 --tasks-config nlu"
    echo "  $0 --project-id my-project-123 --test"
}

# Function to check if project ID is provided
check_project_id() {
    if [ -z "$PROJECT_ID" ]; then
        print_error "Project ID is required. Use --project-id or set GOOGLE_CLOUD_PROJECT environment variable."
        echo ""
        show_usage
        exit 1
    fi
}

# Function to run integration test
run_test() {
    print_info "Running integration test..."
    python3 test_gemini_integration.py
    if [ $? -eq 0 ]; then
        print_success "Integration test passed!"
    else
        print_error "Integration test failed!"
        exit 1
    fi
}

# Function to run benchmark
run_benchmark() {
    print_info "Running SEA-HELM benchmark with Gemini 2.5 Flash..."
    
    # Build command
    cmd="python3 seahelm_gemini2_5flash.py"
    cmd="$cmd --project-id $PROJECT_ID"
    
    if [ -n "$MODEL_NAME" ]; then
        cmd="$cmd --model-name $MODEL_NAME"
    fi
    
    if [ -n "$LOCATION" ]; then
        cmd="$cmd --location $LOCATION"
    fi
    
    if [ -n "$OUTPUT_DIR" ]; then
        cmd="$cmd --output-dir $OUTPUT_DIR"
    fi
    
    if [ -n "$TASKS_CONFIG" ]; then
        cmd="$cmd --tasks-configuration $TASKS_CONFIG"
    fi
    
    if [ -n "$MAX_TOKENS" ]; then
        cmd="$cmd --max-tokens $MAX_TOKENS"
    fi
    
    if [ -n "$TEMPERATURE" ]; then
        cmd="$cmd --temperature $TEMPERATURE"
    fi
    
    if [ -n "$LIMIT" ]; then
        cmd="$cmd --limit $LIMIT"
    fi
    
    if [ "$BASE_MODEL" = "true" ]; then
        cmd="$cmd --base-model"
    fi
    
    print_info "Executing: $cmd"
    echo ""
    
    # Execute command
    eval $cmd
    
    if [ $? -eq 0 ]; then
        print_success "Benchmark completed successfully!"
    else
        print_error "Benchmark failed!"
        exit 1
    fi
}

# Parse command line arguments
PROJECT_ID=""
MODEL_NAME=""
LOCATION=""
OUTPUT_DIR=""
TASKS_CONFIG=""
MAX_TOKENS=""
TEMPERATURE=""
LIMIT=""
BASE_MODEL="false"
TEST_ONLY="false"

while [[ $# -gt 0 ]]; do
    case $1 in
        --project-id)
            PROJECT_ID="$2"
            shift 2
            ;;
        --model-name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --location)
            LOCATION="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --tasks-config)
            TASKS_CONFIG="$2"
            shift 2
            ;;
        --max-tokens)
            MAX_TOKENS="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --limit)
            LIMIT="$2"
            shift 2
            ;;
        --base-model)
            BASE_MODEL="true"
            shift
            ;;
        --test)
            TEST_ONLY="true"
            shift
            ;;
        --help|-h)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo ""
            show_usage
            exit 1
            ;;
    esac
done

# Check if project ID is provided (either via argument or environment variable)
if [ -z "$PROJECT_ID" ]; then
    PROJECT_ID="$GOOGLE_CLOUD_PROJECT"
fi

# Main execution
if [ "$TEST_ONLY" = "true" ]; then
    run_test
else
    check_project_id
    run_benchmark
fi 