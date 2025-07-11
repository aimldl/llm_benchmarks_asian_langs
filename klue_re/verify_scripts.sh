#!/bin/bash

# KLUE RE Script Verification Script
# This script verifies that all required scripts exist and are executable

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
    echo "KLUE RE Script Verification Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -f, --fix         Automatically fix permissions"
    echo "  -v, --verbose     Show detailed output"
    echo "  -h, --help        Show this help message"
    echo ""
    echo "This script verifies:"
    echo "1. All required scripts exist"
    echo "2. All scripts are executable"
    echo "3. Required directories exist"
    echo "4. Required files exist"
}

# Function to check if file exists and is executable
check_script() {
    local script="$1"
    local description="$2"
    local errors=0
    
    if [ ! -f "$script" ]; then
        print_error "Missing script: $script ($description)"
        ((errors++))
    elif [ ! -x "$script" ]; then
        print_warning "Script not executable: $script ($description)"
        if [ "$FIX_PERMISSIONS" = "true" ]; then
            chmod +x "$script"
            print_success "Fixed permissions for: $script"
        else
            ((errors++))
        fi
    else
        print_success "Script OK: $script ($description)"
    fi
    
    return $errors
}

# Function to check if directory exists
check_directory() {
    local dir="$1"
    local description="$2"
    local errors=0
    
    if [ ! -d "$dir" ]; then
        print_error "Missing directory: $dir ($description)"
        if [ "$FIX_PERMISSIONS" = "true" ]; then
            mkdir -p "$dir"
            print_success "Created directory: $dir"
        else
            ((errors++))
        fi
    else
        print_success "Directory OK: $dir ($description)"
    fi
    
    return $errors
}

# Function to check if file exists
check_file() {
    local file="$1"
    local description="$2"
    local errors=0
    
    if [ ! -f "$file" ]; then
        print_error "Missing file: $file ($description)"
        ((errors++))
    else
        print_success "File OK: $file ($description)"
    fi
    
    return $errors
}

# Function to verify Python script syntax
verify_python_script() {
    local script="$1"
    local errors=0
    
    if [ -f "$script" ]; then
        if python3 -m py_compile "$script" 2>/dev/null; then
            print_success "Python syntax OK: $script"
        else
            print_error "Python syntax error in: $script"
            ((errors++))
        fi
    else
        print_error "Python script not found: $script"
        ((errors++))
    fi
    
    return $errors
}

# Function to verify shell script syntax
verify_shell_script() {
    local script="$1"
    local errors=0
    
    if [ -f "$script" ]; then
        if bash -n "$script" 2>/dev/null; then
            print_success "Shell syntax OK: $script"
        else
            print_error "Shell syntax error in: $script"
            ((errors++))
        fi
    else
        print_error "Shell script not found: $script"
        ((errors++))
    fi
    
    return $errors
}

# Main verification function
verify_all() {
    local total_errors=0
    local script_errors=0
    local dir_errors=0
    local file_errors=0
    local syntax_errors=0
    
    print_info "Starting KLUE RE script verification..."
    echo ""
    
    # Check required scripts
    print_info "1. Checking required scripts..."
    check_script "klue_re-gemini2_5flash.py" "Main benchmark script" || script_errors=$?
    check_script "run" "Benchmark runner script" || script_errors=$?
    check_script "setup.sh" "Environment setup script" || script_errors=$?
    check_script "install_dependencies.sh" "Dependency installation script" || script_errors=$?
    check_script "test_setup.py" "Setup test script" || script_errors=$?
    check_script "get_errors.sh" "Error extraction script" || script_errors=$?
    check_script "test_logging.sh" "Logging test script" || script_errors=$?
    check_script "verify_scripts.sh" "This verification script" || script_errors=$?
    echo ""
    
    # Check required directories
    print_info "2. Checking required directories..."
    check_directory "logs" "Log files directory" || dir_errors=$?
    check_directory "benchmark_results" "Benchmark results directory" || dir_errors=$?
    check_directory "result_analysis" "Result analysis directory" || dir_errors=$?
    check_directory "eval_dataset" "Evaluation dataset directory" || dir_errors=$?
    echo ""
    
    # Check required files
    print_info "3. Checking required files..."
    check_file "requirements.txt" "Python dependencies" || file_errors=$?
    check_file "README.md" "Documentation" || file_errors=$?
    check_file "ABOUT_KLUE_RE.md" "Task description" || file_errors=$?
    check_file "TROUBLESHOOTING.md" "Troubleshooting guide" || file_errors=$?
    check_file "VERTEX_AI_SETUP.md" "Vertex AI setup guide" || file_errors=$?
    echo ""
    
    # Verify script syntax
    print_info "4. Verifying script syntax..."
    verify_python_script "klue_re-gemini2_5flash.py" || syntax_errors=$?
    verify_python_script "test_setup.py" || syntax_errors=$?
    verify_shell_script "run" || syntax_errors=$?
    verify_shell_script "setup.sh" || syntax_errors=$?
    verify_shell_script "install_dependencies.sh" || syntax_errors=$?
    verify_shell_script "get_errors.sh" || syntax_errors=$?
    verify_shell_script "test_logging.sh" || syntax_errors=$?
    verify_shell_script "verify_scripts.sh" || syntax_errors=$?
    echo ""
    
    # Calculate total errors
    total_errors=$((script_errors + dir_errors + file_errors + syntax_errors))
    
    # Summary
    print_info "Verification Summary:"
    print_info "  Scripts: $([ $script_errors -eq 0 ] && echo "PASSED" || echo "FAILED ($script_errors errors)")"
    print_info "  Directories: $([ $dir_errors -eq 0 ] && echo "PASSED" || echo "FAILED ($dir_errors errors)")"
    print_info "  Files: $([ $file_errors -eq 0 ] && echo "PASSED" || echo "FAILED ($file_errors errors)")"
    print_info "  Syntax: $([ $syntax_errors -eq 0 ] && echo "PASSED" || echo "FAILED ($syntax_errors errors)")"
    echo ""
    
    if [ $total_errors -eq 0 ]; then
        print_success "All verifications passed! ($total_errors errors)"
        echo ""
        print_info "Your KLUE RE benchmark environment is ready."
        print_info "Next steps:"
        print_info "1. Run setup: ./setup.sh"
        print_info "2. Test setup: ./test_setup.py"
        print_info "3. Run benchmark: ./run test"
        return 0
    else
        print_error "Verification failed! ($total_errors errors)"
        echo ""
        if [ "$FIX_PERMISSIONS" = "true" ]; then
            print_info "Some issues were automatically fixed."
            print_info "Run this script again to verify all issues are resolved."
        else
            print_info "Use -f flag to automatically fix permissions and create missing directories."
        fi
        return 1
    fi
}

# Parse command line arguments
FIX_PERMISSIONS=false
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -f|--fix)
            FIX_PERMISSIONS=true
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
echo "KLUE RE Script Verification"
echo "=========================================="
echo ""

if [ "$FIX_PERMISSIONS" = "true" ]; then
    print_info "Auto-fix mode enabled"
fi

if verify_all; then
    exit 0
else
    exit 1
fi 