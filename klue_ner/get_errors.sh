#!/bin/bash

# KLUE NER Error Analysis Script
# This script extracts error information from benchmark result files

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
    echo "KLUE NER Error Analysis Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -f, --file FILE     CSV result file to analyze (default: latest file)"
    echo "  -o, --output FILE   Output file for error analysis (default: result_analysis/errors.txt)"
    echo "  -h, --help          Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Analyze latest result file"
    echo "  $0 -f klue_ner_results_20250101.csv   # Analyze specific file"
    echo "  $0 -o my_errors.txt                   # Save to specific output file"
    echo ""
    echo "The script will:"
    echo "  1. Find the most recent result file if none specified"
    echo "  2. Extract samples with errors (success=false or error messages)"
    echo "  3. Show error analysis with entity comparison"
    echo "  4. Save results to the specified output file"
}

# Function to find latest result file
find_latest_result_file() {
    local latest_file=""
    local latest_time=0
    
    for file in benchmark_results/klue_ner_results_*.csv; do
        if [ -f "$file" ]; then
            local file_time=$(stat -c %Y "$file" 2>/dev/null || stat -f %m "$file" 2>/dev/null)
            if [ "$file_time" -gt "$latest_time" ]; then
                latest_time=$file_time
                latest_file="$file"
            fi
        fi
    done
    
    echo "$latest_file"
}

# Function to analyze errors
analyze_errors() {
    local input_file="$1"
    local output_file="$2"
    
    print_info "Analyzing errors in: $input_file"
    
    # Create result_analysis directory if it doesn't exist
    mkdir -p result_analysis
    
    # Check if input file exists
    if [ ! -f "$input_file" ]; then
        print_error "Input file not found: $input_file"
        return 1
    fi
    
    # Extract error samples
    print_info "Extracting error samples..."
    
    # Create output file with header
    cat > "$output_file" << EOF
KLUE NER Error Analysis
Generated: $(date)
Input File: $input_file
========================================

EOF
    
    # Extract samples with errors and format them nicely
    awk -F',' '
    BEGIN {
        error_count = 0
        max_errors = 20  # Limit to first 20 errors for readability
    }
    
    # Skip header line
    NR == 1 { next }
    
    # Check for error conditions
    $9 == "False" || $10 != "" {
        if (error_count >= max_errors) {
            next
        }
        
        error_count++
        
        print "Error Sample #" error_count ":"
        print "  Sample ID: " $1
        print "  Text: " $2
        print "  Success: " $9
        if ($10 != "") {
            print "  Error: " $10
        }
        print "  True Entities Count: " $3
        print "  Predicted Entities Count: " $4
        print "  Precision: " $5
        print "  Recall: " $6
        print "  F1 Score: " $7
        print "  Correct Entities: " $8
        print ""
    }
    
    END {
        if (error_count == 0) {
            print "No errors found in the dataset."
        } else {
            print "Total errors found: " error_count
            if (error_count > max_errors) {
                print "(Showing first " max_errors " errors)"
            }
        }
    }
    ' "$input_file" >> "$output_file"
    
    # Count total samples and error rate
    total_samples=$(tail -n +2 "$input_file" | wc -l)
    error_samples=$(tail -n +2 "$input_file" | awk -F',' '$9 == "False" || $10 != ""' | wc -l)
    error_rate=$(echo "scale=2; $error_samples * 100 / $total_samples" | bc -l 2>/dev/null || echo "0")
    
    # Add summary to output file
    cat >> "$output_file" << EOF

========================================
Summary Statistics
========================================
Total Samples: $total_samples
Error Samples: $error_samples
Error Rate: ${error_rate}%

EOF
    
    print_success "Error analysis completed!"
    print_info "Results saved to: $output_file"
    print_info "Found $error_samples errors out of $total_samples samples (${error_rate}% error rate)"
}

# Main script logic
main() {
    local input_file=""
    local output_file="result_analysis/errors.txt"
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -f|--file)
                input_file="$2"
                shift 2
                ;;
            -o|--output)
                output_file="$2"
                shift 2
                ;;
            -h|--help)
                show_usage
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    # If no input file specified, find the latest one
    if [ -z "$input_file" ]; then
        input_file=$(find_latest_result_file)
        if [ -z "$input_file" ]; then
            print_error "No result files found in benchmark_results/ directory"
            print_info "Please run the benchmark first: ./run test"
            exit 1
        fi
        print_info "Using latest result file: $input_file"
    fi
    
    # Run error analysis
    analyze_errors "$input_file" "$output_file"
}

# Run main function with all arguments
main "$@" 