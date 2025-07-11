#!/bin/bash

# KLUE RE Error Extraction Script
# This script extracts error information from log files and saves it to result_analysis directory

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
    echo "KLUE RE Error Extraction Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -f, --file FILE     Extract errors from specific log file"
    echo "  -d, --directory DIR Extract errors from all .log files in directory"
    echo "  -a, --all           Extract errors from all .log files in logs/ directory"
    echo "  -h, --help          Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 -f logs/klue_re_test_10samples_20241201_120000.log"
    echo "  $0 -d logs/"
    echo "  $0 -a"
    echo ""
    echo "Output:"
    echo "  Error files will be saved to result_analysis/ directory"
    echo "  Format: [original_name]_errors.txt"
}

# Function to extract errors from a single log file
extract_errors_from_file() {
    local log_file="$1"
    local output_dir="result_analysis"
    
    # Check if log file exists
    if [ ! -f "$log_file" ]; then
        print_error "Log file not found: $log_file"
        return 1
    fi
    
    # Create output directory if it doesn't exist
    mkdir -p "$output_dir"
    
    # Get base name without extension
    local base_name=$(basename "$log_file" .log)
    local error_file="$output_dir/${base_name}_errors.txt"
    
    print_info "Processing: $log_file"
    
    # Extract error analysis section
    local error_analysis=$(awk '
    BEGIN { 
        in_error_analysis = 0; 
        error_analysis_done = 0;
        error_analysis_found = 0;
    }
    
    # Start of error analysis section
    /Error Analysis \(showing first [0-9]+ errors\):/ {
        in_error_analysis = 1
        error_analysis_found = 1
        print
        next
    }
    
    # Error analysis content (numbered errors)
    in_error_analysis && /^[[:space:]]*[0-9]+\. Sample ID:/ {
        print
        next
    }
    
    # Error analysis content (sentence and entity lines)
    in_error_analysis && /^[[:space:]]*Sentence:/ {
        print
        next
    }
    
    in_error_analysis && /^[[:space:]]*Subject:/ {
        print
        next
    }
    
    in_error_analysis && /^[[:space:]]*Object:/ {
        print
        next
    }
    
    in_error_analysis && /^[[:space:]]*True Relation:/ {
        print
        next
    }
    
    in_error_analysis && /^[[:space:]]*Predicted Relation:/ {
        print
        next
    }
    
    in_error_analysis && /^[[:space:]]*Error:/ {
        print
        next
    }
    
    # Empty line after error details marks end of error analysis
    in_error_analysis && /^[[:space:]]*$/ {
        if (!error_analysis_done) {
            error_analysis_done = 1
            print
        }
    }
    
    # End of error analysis section (double empty line or new section)
    in_error_analysis && /^[[:space:]]*$/ {
        if (error_analysis_done) {
            in_error_analysis = 0
            error_analysis_done = 0
            print
        }
    }
    
    # If no error analysis was found, add a note
    END {
        if (!error_analysis_found) {
            print "No error analysis section found in the log."
        }
    }
    ' "$log_file")
    
    # Extract ERROR logs
    local error_logs=$(awk '
    BEGIN { 
        in_errors = 0; 
        in_error_block = 0;
    }
    
    # Start of ERROR logs
    /^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2},[0-9]{3} - ERROR -/ {
        in_errors = 1
        in_error_block = 1
        print
        next
    }
    
    # Continue ERROR logs until we hit a non-error line
    in_errors {
        if (/^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2},[0-9]{3} - (INFO|WARNING|DEBUG)/) {
            in_errors = 0
            in_error_block = 0
        } else if (/^[[:space:]]*$/) {
            # Empty line in error section
            if (in_error_block) {
                print
            }
        } else if (/^[[:space:]]*[{}]/) {
            # JSON content in error section
            print
        } else if (/^[[:space:]]*[A-Za-z_]+:/) {
            # Key-value pairs in error section
            print
        } else if (in_error_block) {
            # Any other content in error section
            print
        }
    }
    ' "$log_file")
    
    # Write to error file
    {
        echo "KLUE RE Error Analysis"
        echo "Source: $log_file"
        echo "Extracted: $(date)"
        echo "========================================"
        echo ""
        echo "$error_analysis"
        echo ""
        echo "$error_logs"
    } > "$error_file"
    
    # Check if error file has content
    if [ -s "$error_file" ]; then
        local line_count=$(wc -l < "$error_file")
        print_success "Errors extracted to: $error_file ($line_count lines)"
    else
        print_warning "No errors found in: $log_file"
        rm -f "$error_file"
    fi
}

# Function to process all log files in a directory
process_directory() {
    local dir="$1"
    
    if [ ! -d "$dir" ]; then
        print_error "Directory not found: $dir"
        return 1
    fi
    
    local log_files=$(find "$dir" -name "*.log" -type f)
    
    if [ -z "$log_files" ]; then
        print_warning "No .log files found in: $dir"
        return 0
    fi
    
    local count=0
    for log_file in $log_files; do
        if extract_errors_from_file "$log_file"; then
            ((count++))
        fi
    done
    
    print_success "Processed $count log files"
}

# Main script logic
case "${1:-help}" in
    "-f"|"--file")
        if [ -z "$2" ]; then
            print_error "Please specify a log file"
            show_help
            exit 1
        fi
        extract_errors_from_file "$2"
        ;;
    "-d"|"--directory")
        if [ -z "$2" ]; then
            print_error "Please specify a directory"
            show_help
            exit 1
        fi
        process_directory "$2"
        ;;
    "-a"|"--all")
        process_directory "logs"
        ;;
    "-h"|"--help"|"help")
        show_help
        ;;
    *)
        print_error "Unknown option: $1"
        show_help
        exit 1
        ;;
esac 