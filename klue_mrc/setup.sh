#!/bin/bash

# KLUE MRC Benchmark Complete Setup Script
# This script handles the complete setup process for the KLUE Machine Reading Comprehension benchmark

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
    echo "KLUE MRC Benchmark Setup Script"
    echo ""
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  install     Install Python dependencies only"
    echo "  test        Test the setup (after installation)"
    echo "  full        Complete setup (install + test)"
    echo "  help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 install   # Install dependencies only"
    echo "  $0 test      # Test the setup"
    echo "  $0 full      # Complete setup"
}

# Function to check prerequisites
check_prerequisites() {
    print_info "Checking prerequisites..."
    
    # Check if Python is available
    if ! command -v python &> /dev/null; then
        print_error "Python is not installed or not in PATH"
        exit 1
    fi
    
    # Check if pip is available
    if ! command -v pip &> /dev/null; then
        print_error "pip is not installed. Please install Python and pip first."
        exit 1
    fi
    
    # Check if requirements.txt exists
    if [ ! -f "requirements.txt" ]; then
        print_error "requirements.txt not found in current directory"
        exit 1
    fi
    
    print_success "Prerequisites check passed"
}

# Function to install dependencies
install_dependencies() {
    print_info "Installing Python dependencies..."
    
    # Redirect pip output to /dev/null to suppress verbose installation messages
    if pip install -r requirements.txt > /dev/null 2>&1; then
        print_success "Dependencies installed successfully!"
    else
        print_error "Failed to install dependencies"
        exit 1
    fi
}

# Function to test setup
test_setup() {
    print_info "Testing the setup..."
    
    # Check if Python script exists
    if [ ! -f "klue_mrc-gemini2_5flash.py" ]; then
        print_error "klue_mrc-gemini2_5flash.py not found in current directory"
        exit 1
    fi
    
    # Check if test script exists
    if [ ! -f "test_setup.py" ]; then
        print_error "test_setup.py not found in current directory"
        exit 1
    fi
    
    # Run the test script
    if python test_setup.py; then
        print_success "Setup test completed successfully!"
    else
        print_error "Setup test failed!"
        exit 1
    fi
}

# Function to show next steps
show_next_steps() {
    echo ""
    print_info "Next steps for Vertex AI setup:"
    echo "1. Install Google Cloud CLI: https://cloud.google.com/sdk/docs/install"
    echo "2. Authenticate with gcloud: gcloud auth login"
    echo "3. Set up application default credentials: gcloud auth application-default login"
    echo "4. Set your project ID: export GOOGLE_CLOUD_PROJECT='your-project-id'"
    echo "5. Enable Vertex AI API: gcloud services enable aiplatform.googleapis.com"
    echo "6. Run the benchmark: ./run test (for small test) or ./run full (for full benchmark)"
    echo ""
    print_info "Logging features:"
    echo "- All benchmark runs are automatically logged to the 'logs/' directory"
    echo "- Log files include command headers for easy identification"
    echo "- Separate error logs (.err) are created for focused debugging"
}

# Main script logic
main() {
    case "${1:-help}" in
        "install")
            check_prerequisites
            install_dependencies
            show_next_steps
            ;;
        "test")
            test_setup
            ;;
        "full")
            check_prerequisites
            install_dependencies
            test_setup
            show_next_steps
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