#!/bin/bash

# KLUE Benchmarks Google Cloud Configuration Script
# This script sets up the Google Cloud environment for running KLUE benchmarks

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

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if user is authenticated
check_auth() {
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
        return 1
    fi
    return 0
}

# Function to get current project
get_current_project() {
    gcloud config get-value project 2>/dev/null || echo ""
}

# Function to show usage
show_usage() {
    echo "KLUE Benchmarks Google Cloud Configuration Script"
    echo ""
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  setup        Complete setup (auth + project + API + IAM)"
    echo "  auth         Authenticate with Google Cloud"
    echo "  project      Set up project configuration"
    echo "  api          Enable required APIs"
    echo "  iam          Check and set up IAM permissions"
    echo "  verify       Verify current configuration"
    echo "  help         Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0           # Complete setup (default)"
    echo "  $0 auth      # Authenticate only"
    echo "  $0 verify    # Check current setup"
}

# Function to authenticate with Google Cloud
setup_auth() {
    print_info "Setting up Google Cloud authentication..."
    
    if ! command_exists gcloud; then
        print_error "gcloud CLI is not installed. Please install it first:"
        echo "  https://cloud.google.com/sdk/docs/install"
        exit 1
    fi
    
    # Check if already authenticated
    if check_auth; then
        current_account=$(gcloud auth list --filter=status:ACTIVE --format="value(account)")
        print_success "Already authenticated as: $current_account"
    else
        print_info "Starting authentication process..."
        gcloud auth login
        gcloud auth application-default login
        print_success "Authentication completed!"
    fi
}

# Function to set up project configuration
setup_project() {
    print_info "Setting up project configuration..."
    
    current_project=$(get_current_project)
    
    if [ -n "$current_project" ]; then
        print_info "Current project: $current_project"
        read -p "Do you want to use this project? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            project_id=$current_project
        else
            read -p "Enter your Google Cloud project ID: " project_id
        fi
    else
        read -p "Enter your Google Cloud project ID: " project_id
    fi
    
    # Validate project ID format
    if [[ ! $project_id =~ ^[a-z][a-z0-9-]{4,28}[a-z0-9]$ ]]; then
        print_error "Invalid project ID format. Project ID must be 6-30 characters long and contain only lowercase letters, numbers, and hyphens."
        exit 1
    fi
    
    # Set the project
    gcloud config set project "$project_id"
    print_success "Project set to: $project_id"
    
    # Export environment variable
    export GOOGLE_CLOUD_PROJECT="$project_id"
    
    # Add to .bashrc if not already there
    if ! grep -q "export GOOGLE_CLOUD_PROJECT=" ~/.bashrc; then
        echo "export GOOGLE_CLOUD_PROJECT=\"$project_id\"" >> ~/.bashrc
        print_success "Added GOOGLE_CLOUD_PROJECT to ~/.bashrc"
    else
        # Update existing entry
        sed -i "s/export GOOGLE_CLOUD_PROJECT=.*/export GOOGLE_CLOUD_PROJECT=\"$project_id\"/" ~/.bashrc
        print_success "Updated GOOGLE_CLOUD_PROJECT in ~/.bashrc"
    fi
}

# Function to enable required APIs
setup_apis() {
    print_info "Enabling required Google Cloud APIs..."
    
    project_id=$(get_current_project)
    if [ -z "$project_id" ]; then
        print_error "No project configured. Run '$0 project' first."
        exit 1
    fi
    
    # List of required APIs
    apis=(
        "aiplatform.googleapis.com"      # Vertex AI
        "compute.googleapis.com"         # Compute Engine (for Vertex AI)
        "storage.googleapis.com"         # Cloud Storage
        "bigquery.googleapis.com"        # BigQuery (for some Vertex AI features)
    )
    
    for api in "${apis[@]}"; do
        print_info "Enabling $api..."
        if gcloud services enable "$api" --project="$project_id" 2>/dev/null; then
            print_success "Enabled $api"
        else
            print_warning "Failed to enable $api (may already be enabled)"
        fi
    done
    
    print_success "API setup completed!"
}

# Function to check and set up IAM permissions
setup_iam() {
    print_info "Checking IAM permissions..."
    
    project_id=$(get_current_project)
    if [ -z "$project_id" ]; then
        print_error "No project configured. Run '$0 project' first."
        exit 1
    fi
    
    current_account=$(gcloud auth list --filter=status:ACTIVE --format="value(account)")
    if [ -z "$current_account" ]; then
        print_error "Not authenticated. Run '$0 auth' first."
        exit 1
    fi
    
    print_info "Checking permissions for: $current_account"
    
    # Check for required roles
    required_roles=(
        "roles/aiplatform.user"
        "roles/storage.objectViewer"
        "roles/compute.viewer"
    )
    
    missing_roles=()
    
    for role in "${required_roles[@]}"; do
        if gcloud projects get-iam-policy "$project_id" --flatten="bindings[].members" --format="table(bindings.role)" --filter="bindings.members:$current_account" | grep -q "$role"; then
            print_success "✓ $role"
        else
            print_warning "✗ $role (missing)"
            missing_roles+=("$role")
        fi
    done
    
    if [ ${#missing_roles[@]} -gt 0 ]; then
        echo
        print_warning "Missing IAM roles detected!"
        echo "To add the missing roles, run the following commands:"
        echo
        for role in "${missing_roles[@]}"; do
            echo "gcloud projects add-iam-policy-binding $project_id \\"
            echo "    --member=\"user:$current_account\" \\"
            echo "    --role=\"$role\""
            echo
        done
        echo "Or ask your project administrator to grant these roles."
    else
        print_success "All required IAM permissions are set!"
    fi
}

# Function to verify configuration
verify_config() {
    print_info "Verifying Google Cloud configuration..."
    
    # Check authentication
    if check_auth; then
        current_account=$(gcloud auth list --filter=status:ACTIVE --format="value(account)")
        print_success "✓ Authenticated as: $current_account"
    else
        print_error "✗ Not authenticated"
        return 1
    fi
    
    # Check project
    current_project=$(get_current_project)
    if [ -n "$current_project" ]; then
        print_success "✓ Project set to: $current_project"
    else
        print_error "✗ No project configured"
        return 1
    fi
    
    # Check environment variable
    if [ -n "$GOOGLE_CLOUD_PROJECT" ]; then
        print_success "✓ GOOGLE_CLOUD_PROJECT set to: $GOOGLE_CLOUD_PROJECT"
    else
        print_warning "⚠ GOOGLE_CLOUD_PROJECT not set in current session"
    fi
    
    # Check APIs
    print_info "Checking API status..."
    apis=("aiplatform.googleapis.com" "compute.googleapis.com" "storage.googleapis.com")
    for api in "${apis[@]}"; do
        if gcloud services list --enabled --filter="name:$api" --format="value(name)" | grep -q "$api"; then
            print_success "✓ $api enabled"
        else
            print_warning "⚠ $api not enabled"
        fi
    done
    
    print_success "Configuration verification completed!"
}

# Main script logic
main() {
    case "${1:-setup}" in
        "setup")
            print_info "Starting complete Google Cloud setup..."
            setup_auth
            setup_project
            setup_apis
            setup_iam
            verify_config
            print_success "Setup completed! You can now run KLUE benchmarks."
            ;;
        "auth")
            setup_auth
            ;;
        "project")
            setup_project
            ;;
        "api")
            setup_apis
            ;;
        "iam")
            setup_iam
            ;;
        "verify")
            verify_config
            ;;
        "help"|"-h"|"--help")
            show_usage
            ;;
        *)
            print_error "Unknown option: $1"
            echo
            show_usage
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"