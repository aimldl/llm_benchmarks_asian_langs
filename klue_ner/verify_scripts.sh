#!/bin/bash

# KLUE NER Script Verification Script
# This script verifies that all required scripts exist and have proper permissions

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
    echo "KLUE NER Script Verification Script"
    echo ""
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  check     Check script existence and permissions (default)"
    echo "  fix       Fix script permissions (make executable)"
    echo "  help      Show this help message"
    echo ""
    echo "This script verifies:"
    echo "  1. All required scripts exist"
    echo "  2. Scripts have proper executable permissions"
    echo "  3. Required directories exist"
    echo "  4. Configuration files are present"
    echo ""
    echo "Examples:"
    echo "  $0 check   # Check scripts (default)"
    echo "  $0 fix     # Fix permissions"
}

# Function to check script existence and permissions
check_scripts() {
    print_info "Checking script existence and permissions..."
    
    # Define required scripts
    scripts=(
        "run"
        "setup.sh"
        "install_dependencies.sh"
        "get_errors.sh"
        "test_logging.sh"
        "verify_scripts.sh"
    )
    
    missing_scripts=()
    non_executable_scripts=()
    
    for script in "${scripts[@]}"; do
        if [ -f "$script" ]; then
            if [ -x "$script" ]; then
                print_success "✓ $script (exists, executable)"
            else
                print_warning "⚠ $script (exists, not executable)"
                non_executable_scripts+=("$script")
            fi
        else
            print_error "✗ $script (missing)"
            missing_scripts+=("$script")
        fi
    done
    
    # Report results
    if [ ${#missing_scripts[@]} -eq 0 ] && [ ${#non_executable_scripts[@]} -eq 0 ]; then
        print_success "All scripts are present and executable!"
        return 0
    else
        if [ ${#missing_scripts[@]} -gt 0 ]; then
            print_error "Missing scripts: ${missing_scripts[*]}"
        fi
        if [ ${#non_executable_scripts[@]} -gt 0 ]; then
            print_warning "Non-executable scripts: ${non_executable_scripts[*]}"
        fi
        return 1
    fi
}

# Function to fix script permissions
fix_permissions() {
    print_info "Fixing script permissions..."
    
    # Define scripts that should be executable
    scripts=(
        "run"
        "setup.sh"
        "install_dependencies.sh"
        "get_errors.sh"
        "test_logging.sh"
        "verify_scripts.sh"
    )
    
    fixed_count=0
    
    for script in "${scripts[@]}"; do
        if [ -f "$script" ]; then
            if [ ! -x "$script" ]; then
                if chmod +x "$script"; then
                    print_success "✓ Made $script executable"
                    ((fixed_count++))
                else
                    print_error "✗ Failed to make $script executable"
                fi
            else
                print_info "✓ $script already executable"
            fi
        else
            print_warning "⚠ $script not found, skipping"
        fi
    done
    
    if [ $fixed_count -gt 0 ]; then
        print_success "Fixed permissions for $fixed_count script(s)"
    else
        print_info "No permissions needed to be fixed"
    fi
}

# Function to check directories
check_directories() {
    print_info "Checking required directories..."
    
    # Define required directories
    directories=(
        "logs"
        "benchmark_results"
        "result_analysis"
        "eval_dataset"
    )
    
    missing_dirs=()
    
    for dir in "${directories[@]}"; do
        if [ -d "$dir" ]; then
            print_success "✓ $dir/ (exists)"
        else
            print_warning "⚠ $dir/ (missing)"
            missing_dirs+=("$dir")
        fi
    done
    
    # Create missing directories
    if [ ${#missing_dirs[@]} -gt 0 ]; then
        print_info "Creating missing directories..."
        for dir in "${missing_dirs[@]}"; do
            if mkdir -p "$dir"; then
                print_success "✓ Created $dir/"
            else
                print_error "✗ Failed to create $dir/"
            fi
        done
    fi
}

# Function to check configuration files
check_config_files() {
    print_info "Checking configuration files..."
    
    # Define required configuration files
    config_files=(
        "requirements.txt"
        "klue_ner-gemini2_5flash.py"
        "test_setup.py"
    )
    
    missing_files=()
    
    for file in "${config_files[@]}"; do
        if [ -f "$file" ]; then
            print_success "✓ $file (exists)"
        else
            print_error "✗ $file (missing)"
            missing_files+=("$file")
        fi
    done
    
    if [ ${#missing_files[@]} -gt 0 ]; then
        print_error "Missing configuration files: ${missing_files[*]}"
        return 1
    else
        print_success "All configuration files present!"
        return 0
    fi
}

# Function to check Python script syntax
check_python_syntax() {
    print_info "Checking Python script syntax..."
    
    python_scripts=(
        "klue_ner-gemini2_5flash.py"
        "test_setup.py"
    )
    
    syntax_errors=0
    
    for script in "${python_scripts[@]}"; do
        if [ -f "$script" ]; then
            if python -m py_compile "$script" 2>/dev/null; then
                print_success "✓ $script (syntax OK)"
            else
                print_error "✗ $script (syntax error)"
                ((syntax_errors++))
            fi
        else
            print_warning "⚠ $script (not found, skipping)"
        fi
    done
    
    if [ $syntax_errors -eq 0 ]; then
        print_success "All Python scripts have valid syntax!"
        return 0
    else
        print_error "Found $syntax_errors Python script(s) with syntax errors"
        return 1
    fi
}

# Function to run comprehensive verification
run_comprehensive_check() {
    print_info "Running comprehensive script verification..."
    
    local overall_success=true
    
    # Check 1: Scripts
    if ! check_scripts; then
        overall_success=false
    fi
    
    echo ""
    
    # Check 2: Directories
    if ! check_directories; then
        overall_success=false
    fi
    
    echo ""
    
    # Check 3: Configuration files
    if ! check_config_files; then
        overall_success=false
    fi
    
    echo ""
    
    # Check 4: Python syntax
    if ! check_python_syntax; then
        overall_success=false
    fi
    
    echo ""
    
    if [ "$overall_success" = true ]; then
        print_success "All verification checks passed!"
        print_info "The KLUE NER benchmark is ready to use."
        return 0
    else
        print_error "Some verification checks failed."
        print_info "Please fix the issues above before running the benchmark."
        return 1
    fi
}

# Main script logic
main() {
    case "${1:-check}" in
        "check")
            run_comprehensive_check
            ;;
        "fix")
            fix_permissions
            echo ""
            run_comprehensive_check
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