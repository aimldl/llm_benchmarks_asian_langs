#!/bin/bash

# KLUE NER Dependencies Installation Script
# This script installs all required Python dependencies for the KLUE NER benchmark

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
    echo "KLUE NER Dependencies Installation Script"
    echo ""
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  install     Install dependencies (default)"
    echo "  check       Check if dependencies are installed"
    echo "  help        Show this help message"
    echo ""
    echo "This script installs:"
    echo "  - google-genai (Google Generative AI library)"
    echo "  - datasets (Hugging Face datasets)"
    echo "  - pandas (Data manipulation)"
    echo "  - tqdm (Progress bars)"
    echo "  - google-cloud-aiplatform (Vertex AI support)"
    echo ""
    echo "Examples:"
    echo "  $0 install   # Install dependencies"
    echo "  $0 check     # Check installation"
}

# Function to check prerequisites
check_prerequisites() {
    print_info "Checking prerequisites..."
    
    # Check if Python is available
    if ! command -v python &> /dev/null; then
        print_error "Python is not installed or not in PATH"
        print_info "Please install Python 3.8 or higher"
        return 1
    fi
    
    # Check Python version
    python_version=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    print_info "Python version: $python_version"
    
    # Check if pip is available
    if ! command -v pip &> /dev/null; then
        print_error "pip is not installed. Please install Python and pip first."
        return 1
    fi
    
    # Check if requirements.txt exists
    if [ ! -f "requirements.txt" ]; then
        print_error "requirements.txt not found in current directory"
        return 1
    fi
    
    print_success "Prerequisites check passed"
    return 0
}

# Function to install dependencies
install_dependencies() {
    print_info "Installing Python dependencies..."
    
    # Upgrade pip first
    print_info "Upgrading pip..."
    if pip install --upgrade pip; then
        print_success "pip upgraded successfully"
    else
        print_warning "Failed to upgrade pip, continuing with installation"
    fi
    
    # Install dependencies from requirements.txt
    print_info "Installing dependencies from requirements.txt..."
    if pip install -r requirements.txt; then
        print_success "Dependencies installed successfully!"
    else
        print_error "Failed to install dependencies"
        return 1
    fi
    
    return 0
}

# Function to check installed dependencies
check_dependencies() {
    print_info "Checking installed dependencies..."
    
    # Define required packages
    packages=(
        "google.genai"
        "datasets"
        "pandas"
        "tqdm"
        "google.cloud.aiplatform"
    )
    
    missing_packages=()
    
    for package in "${packages[@]}"; do
        if python -c "import $package" 2>/dev/null; then
            print_success "✓ $package"
        else
            print_error "✗ $package (not installed)"
            missing_packages+=("$package")
        fi
    done
    
    if [ ${#missing_packages[@]} -eq 0 ]; then
        print_success "All dependencies are installed!"
        return 0
    else
        print_error "Missing packages: ${missing_packages[*]}"
        print_info "Run '$0 install' to install missing dependencies"
        return 1
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
    echo "6. Test the setup: ./test_setup.py"
    echo "7. Run the benchmark: ./run test (for small test) or ./run full (for full benchmark)"
    echo ""
    print_info "Logging features:"
    echo "- All benchmark runs are automatically logged to the 'logs/' directory"
    echo "- Log files include command headers for easy identification"
    echo "- Separate error logs (.err) are created for focused debugging"
}

# Main script logic
main() {
    case "${1:-install}" in
        "install")
            if check_prerequisites; then
                if install_dependencies; then
                    show_next_steps
                    print_success "Installation completed successfully!"
                else
                    print_error "Installation failed!"
                    exit 1
                fi
            else
                print_error "Prerequisites check failed!"
                exit 1
            fi
            ;;
        "check")
            check_dependencies
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