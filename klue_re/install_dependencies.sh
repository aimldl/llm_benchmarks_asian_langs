#!/bin/bash

# KLUE RE Dependency Installation Script
# This script installs all required dependencies for the KLUE RE benchmark

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
    echo "KLUE RE Dependency Installation Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -v, --venv         Create and use virtual environment"
    echo "  -u, --upgrade      Upgrade existing packages"
    echo "  -f, --force        Force reinstall packages"
    echo "  -h, --help         Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                 # Install dependencies in current environment"
    echo "  $0 -v              # Create virtual environment and install"
    echo "  $0 -u              # Upgrade existing packages"
    echo "  $0 -f              # Force reinstall all packages"
}

# Function to check Python version
check_python_version() {
    local python_cmd="$1"
    
    if ! command -v "$python_cmd" &> /dev/null; then
        return 1
    fi
    
    local version=$("$python_cmd" --version 2>&1 | cut -d' ' -f2)
    local major=$(echo "$version" | cut -d'.' -f1)
    local minor=$(echo "$version" | cut -d'.' -f2)
    
    if [ "$major" -ge 3 ] && [ "$minor" -ge 8 ]; then
        return 0
    else
        return 1
    fi
}

# Function to find suitable Python command
find_python() {
    local python_commands=("python3" "python3.11" "python3.10" "python3.9" "python3.8" "python")
    
    for cmd in "${python_commands[@]}"; do
        if check_python_version "$cmd"; then
            echo "$cmd"
            return 0
        fi
    done
    
    return 1
}

# Function to install packages
install_packages() {
    local python_cmd="$1"
    local upgrade_flag="$2"
    local force_flag="$3"
    
    print_info "Installing Python packages..."
    
    # Base packages
    local packages=(
        "google-genai>=0.3.0"
        "datasets>=2.14.0"
        "pandas>=1.5.0"
        "tqdm>=4.64.0"
        "google-cloud-aiplatform>=1.35.0"
    )
    
    # Install each package
    for package in "${packages[@]}"; do
        print_info "Installing $package..."
        
        local install_cmd="$python_cmd -m pip install"
        
        if [ "$upgrade_flag" = "true" ]; then
            install_cmd="$install_cmd --upgrade"
        fi
        
        if [ "$force_flag" = "true" ]; then
            install_cmd="$install_cmd --force-reinstall"
        fi
        
        install_cmd="$install_cmd $package"
        
        # Redirect pip output to /dev/null to suppress verbose installation messages
        if $install_cmd > /dev/null 2>&1; then
            print_success "Installed $package"
        else
            print_error "Failed to install $package"
            return 1
        fi
    done
    
    return 0
}

# Function to verify installation
verify_installation() {
    local python_cmd="$1"
    
    print_info "Verifying installation..."
    
    local packages=("google.genai" "datasets" "pandas" "tqdm" "google.cloud.aiplatform")
    local failed_packages=()
    
    for package in "${packages[@]}"; do
        if "$python_cmd" -c "import $package" 2>/dev/null; then
            print_success "✓ $package imported successfully"
        else
            print_error "✗ Failed to import $package"
            failed_packages+=("$package")
        fi
    done
    
    if [ ${#failed_packages[@]} -eq 0 ]; then
        print_success "All packages installed successfully!"
        return 0
    else
        print_error "Some packages failed to install: ${failed_packages[*]}"
        return 1
    fi
}

# Function to create virtual environment
create_venv() {
    local python_cmd="$1"
    
    print_info "Creating virtual environment..."
    
    if [ -d "venv" ]; then
        print_warning "Virtual environment already exists"
        read -p "Do you want to recreate it? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_info "Removing existing virtual environment..."
            rm -rf venv
        else
            print_info "Using existing virtual environment"
            return 0
        fi
    fi
    
    if "$python_cmd" -m venv venv; then
        print_success "Virtual environment created"
        return 0
    else
        print_error "Failed to create virtual environment"
        return 1
    fi
}

# Main installation function
main_installation() {
    local use_venv="$1"
    local upgrade_flag="$2"
    local force_flag="$3"
    
    print_info "Starting KLUE RE dependency installation..."
    echo ""
    
    # Find Python
    local python_cmd
    if ! python_cmd=$(find_python); then
        print_error "No suitable Python version found (3.8+ required)"
        print_info "Please install Python 3.8 or higher"
        return 1
    fi
    
    print_success "Found Python: $($python_cmd --version)"
    
    # Create virtual environment if requested
    if [ "$use_venv" = "true" ]; then
        if ! create_venv "$python_cmd"; then
            return 1
        fi
        
        # Activate virtual environment
        print_info "Activating virtual environment..."
        source venv/bin/activate
        python_cmd="python"  # Use python from venv
        print_success "Virtual environment activated"
    fi
    
    # Upgrade pip
    print_info "Upgrading pip..."
    # Redirect pip output to /dev/null to suppress verbose installation messages
    if "$python_cmd" -m pip install --upgrade pip > /dev/null 2>&1; then
        print_success "Pip upgraded"
    else
        print_warning "Failed to upgrade pip, continuing..."
    fi
    
    # Install packages
    if ! install_packages "$python_cmd" "$upgrade_flag" "$force_flag"; then
        print_error "Package installation failed"
        return 1
    fi
    
    # Verify installation
    if ! verify_installation "$python_cmd"; then
        print_error "Installation verification failed"
        return 1
    fi
    
    # Test dataset loading
    print_info "Testing dataset loading..."
    if "$python_cmd" -c "
try:
    from datasets import load_dataset
    print('Loading KLUE RE dataset...')
    dataset = load_dataset('klue', 're', split='validation')
    print(f'Dataset loaded successfully: {len(dataset)} samples')
except Exception as e:
    print(f'Failed to load dataset: {e}')
    exit(1)
"; then
        print_success "Dataset loading test passed"
    else
        print_error "Dataset loading test failed"
        return 1
    fi
    
    print_success "All dependencies installed successfully!"
    
    if [ "$use_venv" = "true" ]; then
        echo ""
        print_info "Virtual environment is active. To activate it in the future:"
        print_info "  source venv/bin/activate"
        print_info ""
        print_info "To deactivate:"
        print_info "  deactivate"
    fi
    
    return 0
}

# Parse command line arguments
USE_VENV=false
UPGRADE=false
FORCE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--venv)
            USE_VENV=true
            shift
            ;;
        -u|--upgrade)
            UPGRADE=true
            shift
            ;;
        -f|--force)
            FORCE=true
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
echo "KLUE RE Dependency Installation"
echo "=========================================="
echo ""

if main_installation "$USE_VENV" "$UPGRADE" "$FORCE"; then
    echo ""
    print_success "Installation completed successfully!"
    print_info "You can now run the KLUE RE benchmark."
    exit 0
else
    echo ""
    print_error "Installation failed!"
    print_info "Please check the error messages above and try again."
    exit 1
fi 