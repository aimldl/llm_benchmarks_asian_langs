#!/bin/bash

# KLUE DST Script Verification Script
# This script verifies that all required scripts exist and are executable

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

echo "=========================================="
echo "KLUE DST Script Verification"
echo "=========================================="

# Check if we're in the right directory
if [ ! -f "klue_dst-gemini2_5flash.py" ]; then
    print_error "klue_dst-gemini2_5flash.py not found in current directory"
    print_error "Please run this script from the klue_dst directory"
    exit 1
fi

print_status "Starting script verification..."

# List of required scripts and their expected permissions
declare -A required_scripts=(
    ["klue_dst-gemini2_5flash.py"]="r"
    ["run"]="rx"
    ["setup.sh"]="rx"
    ["install_dependencies.sh"]="rx"
    ["test_setup.py"]="rx"
    ["get_errors.sh"]="rx"
    ["test_logging.sh"]="rx"
    ["verify_scripts.sh"]="rx"
    ["requirements.txt"]="r"
)

# List of required directories
required_directories=(
    "logs"
    "benchmark_results"
    "result_analysis"
    "eval_dataset"
)

# Function to check file permissions
check_permissions() {
    local file="$1"
    local expected_perms="$2"
    
    if [ ! -f "$file" ]; then
        print_error "File not found: $file"
        return 1
    fi
    
    # Check read permission
    if [[ "$expected_perms" == *"r"* ]] && [ ! -r "$file" ]; then
        print_error "File not readable: $file"
        return 1
    fi
    
    # Check execute permission
    if [[ "$expected_perms" == *"x"* ]] && [ ! -x "$file" ]; then
        print_warning "File not executable: $file"
        print_status "Making $file executable..."
        chmod +x "$file"
        if [ -x "$file" ]; then
            print_success "Made $file executable"
        else
            print_error "Failed to make $file executable"
            return 1
        fi
    fi
    
    print_success "✓ $file (permissions: $expected_perms)"
    return 0
}

# Function to check directory
check_directory() {
    local dir="$1"
    
    if [ ! -d "$dir" ]; then
        print_warning "Directory not found: $dir"
        print_status "Creating directory: $dir"
        mkdir -p "$dir"
        if [ -d "$dir" ]; then
            print_success "Created directory: $dir"
        else
            print_error "Failed to create directory: $dir"
            return 1
        fi
    else
        print_success "✓ Directory exists: $dir"
    fi
    
    return 0
}

# Verify all required scripts
echo ""
print_status "Verifying required scripts..."

all_scripts_ok=true
for script in "${!required_scripts[@]}"; do
    perms="${required_scripts[$script]}"
    if ! check_permissions "$script" "$perms"; then
        all_scripts_ok=false
    fi
done

# Verify all required directories
echo ""
print_status "Verifying required directories..."

all_dirs_ok=true
for dir in "${required_directories[@]}"; do
    if ! check_directory "$dir"; then
        all_dirs_ok=false
    fi
done

# Check Python script syntax
echo ""
print_status "Checking Python script syntax..."

if command -v python3 &> /dev/null; then
    if python3 -m py_compile klue_dst-gemini2_5flash.py 2>/dev/null; then
        print_success "✓ Python script syntax is valid"
    else
        print_error "✗ Python script has syntax errors"
        all_scripts_ok=false
    fi
else
    print_warning "⚠ Python3 not found, skipping syntax check"
fi

# Check if test_setup.py is runnable
echo ""
print_status "Checking test_setup.py..."

if [ -x "test_setup.py" ]; then
    if python3 test_setup.py --help &>/dev/null || python3 test_setup.py help &>/dev/null; then
        print_success "✓ test_setup.py is runnable"
    else
        print_warning "⚠ test_setup.py may have issues (but this is expected if dependencies aren't installed)"
    fi
else
    print_error "✗ test_setup.py is not executable"
    all_scripts_ok=false
fi

# Check bash script syntax
echo ""
print_status "Checking bash script syntax..."

bash_scripts=("run" "setup.sh" "install_dependencies.sh" "get_errors.sh" "test_logging.sh" "verify_scripts.sh")

for script in "${bash_scripts[@]}"; do
    if [ -f "$script" ]; then
        if bash -n "$script" 2>/dev/null; then
            print_success "✓ $script syntax is valid"
        else
            print_error "✗ $script has syntax errors"
            all_scripts_ok=false
        fi
    fi
done

# Check requirements.txt format
echo ""
print_status "Checking requirements.txt format..."

if [ -f "requirements.txt" ]; then
    if [ -s "requirements.txt" ]; then
        print_success "✓ requirements.txt exists and is not empty"
        
        # Check if it contains expected packages
        expected_packages=("google-genai" "datasets" "pandas" "tqdm")
        for package in "${expected_packages[@]}"; do
            if grep -q "$package" requirements.txt; then
                print_success "  ✓ Contains $package"
            else
                print_warning "  ⚠ Missing $package"
            fi
        done
    else
        print_error "✗ requirements.txt is empty"
        all_scripts_ok=false
    fi
else
    print_error "✗ requirements.txt not found"
    all_scripts_ok=false
fi

# Summary
echo ""
echo "=========================================="
echo "VERIFICATION SUMMARY"
echo "=========================================="

if [ "$all_scripts_ok" = true ] && [ "$all_dirs_ok" = true ]; then
    print_success "All scripts and directories verified successfully!"
    echo ""
    print_status "Next steps:"
    echo "1. Run setup: ./setup.sh"
    echo "2. Test environment: python3 test_setup.py"
    echo "3. Run benchmark: ./run test"
    echo "4. Check logs: ./test_logging.sh verify"
    exit 0
else
    print_error "Some verification checks failed"
    echo ""
    print_status "Please fix the issues above before proceeding"
    exit 1
fi 