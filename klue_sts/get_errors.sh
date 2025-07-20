#!/bin/bash

# KLUE STS Error Analysis Script
# This script analyzes errors from benchmark results and provides insights

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

# Function to find latest results
find_latest_results() {
    local results_dir="benchmark_results"
    
    if [ ! -d "$results_dir" ]; then
        print_error "Results directory not found: $results_dir"
        exit 1
    fi
    
    # Find the most recent results file
    local latest_results=$(find "$results_dir" -name "klue_sts_results_*.json" -type f | sort | tail -n 1)
    
    if [ -z "$latest_results" ]; then
        print_error "No results files found in $results_dir"
        exit 1
    fi
    
    echo "$latest_results"
}

# Function to analyze errors
analyze_errors() {
    local results_file="$1"
    local output_file="$2"
    
    print_info "Analyzing errors from: $results_file"
    
    # Create Python script for analysis
    cat > /tmp/error_analysis.py << 'EOF'
#!/usr/bin/env python3
import json
import sys
from collections import Counter, defaultdict

def analyze_errors(results_file, output_file):
    """Analyze errors in benchmark results."""
    
    # Load results
    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # Initialize counters
    total_samples = len(results)
    errors = []
    finish_reasons = Counter()
    score_differences = []
    
    for result in results:
        # Check for errors
        if result.get('error'):
            errors.append(result)
        
        # Count finish reasons
        finish_reason = result.get('finish_reason', 'UNKNOWN')
        finish_reasons[finish_reason] += 1
        
        # Calculate score differences for valid predictions
        if result.get('predicted_score') is not None:
            true_score = result.get('true_score', 0)
            pred_score = result.get('predicted_score', 0)
            score_diff = abs(true_score - pred_score)
            score_differences.append(score_diff)
    
    # Write analysis to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("KLUE STS Error Analysis\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Total Samples: {total_samples}\n")
        f.write(f"Errors: {len(errors)}\n")
        f.write(f"Error Rate: {len(errors)/total_samples*100:.2f}%\n\n")
        
        f.write("Finish Reasons:\n")
        f.write("-" * 20 + "\n")
        for reason, count in finish_reasons.most_common():
            f.write(f"{reason}: {count} ({count/total_samples*100:.1f}%)\n")
        
        if score_differences:
            f.write(f"\nScore Difference Statistics:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Mean Absolute Difference: {sum(score_differences)/len(score_differences):.3f}\n")
            f.write(f"Max Difference: {max(score_differences):.3f}\n")
            f.write(f"Min Difference: {min(score_differences):.3f}\n")
        
        if errors:
            f.write(f"\nDetailed Error Analysis:\n")
            f.write("-" * 30 + "\n")
            
            # Group errors by type
            error_types = defaultdict(list)
            for error in errors:
                error_msg = error.get('error', 'Unknown error')
                error_types[error_msg].append(error)
            
            for error_type, error_list in error_types.items():
                f.write(f"\nError Type: {error_type}\n")
                f.write(f"Count: {len(error_list)}\n")
                f.write("Examples:\n")
                
                for i, error in enumerate(error_list[:3]):  # Show first 3 examples
                    f.write(f"  {i+1}. ID: {error.get('id', 'N/A')}\n")
                    f.write(f"     Sentence 1: {error.get('sentence1', 'N/A')[:50]}...\n")
                    f.write(f"     Sentence 2: {error.get('sentence2', 'N/A')[:50]}...\n")
                    f.write(f"     True Score: {error.get('true_score', 'N/A')}\n")
                    f.write(f"     Predicted Score: {error.get('predicted_score', 'N/A')}\n")
                    f.write(f"     Finish Reason: {error.get('finish_reason', 'N/A')}\n")
                    f.write("\n")
                
                if len(error_list) > 3:
                    f.write(f"  ... and {len(error_list) - 3} more errors of this type\n\n")
        
        # Performance analysis
        valid_predictions = [r for r in results if r.get('predicted_score') is not None]
        if valid_predictions:
            f.write(f"\nPerformance Analysis:\n")
            f.write("-" * 25 + "\n")
            f.write(f"Valid Predictions: {len(valid_predictions)}/{total_samples}\n")
            f.write(f"Success Rate: {len(valid_predictions)/total_samples*100:.2f}%\n")
            
            # Score distribution analysis
            true_scores = [r.get('true_score', 0) for r in valid_predictions]
            pred_scores = [r.get('predicted_score', 0) for r in valid_predictions]
            
            f.write(f"\nScore Distribution:\n")
            f.write(f"True Score Range: [{min(true_scores):.2f}, {max(true_scores):.2f}]\n")
            f.write(f"Predicted Score Range: [{min(pred_scores):.2f}, {max(pred_scores):.2f}]\n")
            f.write(f"True Score Mean: {sum(true_scores)/len(true_scores):.2f}\n")
            f.write(f"Predicted Score Mean: {sum(pred_scores)/len(pred_scores):.2f}\n")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python error_analysis.py <results_file> <output_file>")
        sys.exit(1)
    
    analyze_errors(sys.argv[1], sys.argv[2])
EOF
    
    # Run analysis
    python3 /tmp/error_analysis.py "$results_file" "$output_file"
    
    # Clean up
    rm /tmp/error_analysis.py
    
    print_success "Error analysis completed: $output_file"
}

# Function to show help
show_help() {
    echo "KLUE STS Error Analysis Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --results-file FILE    Specify results file to analyze"
    echo "  --output-file FILE     Specify output file for analysis"
    echo "  --latest               Analyze the latest results file (default)"
    echo "  --help                 Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                     # Analyze latest results"
    echo "  $0 --latest            # Analyze latest results"
    echo "  $0 --results-file benchmark_results/klue_sts_results_20240101_120000.json"
    echo "  $0 --output-file my_analysis.txt"
}

# Main execution
main() {
    local results_file=""
    local output_file=""
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --results-file)
                results_file="$2"
                shift 2
                ;;
            --output-file)
                output_file="$2"
                shift 2
                ;;
            --latest)
                # Use default behavior
                shift
                ;;
            --help|-h)
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
    
    # If no results file specified, find the latest one
    if [ -z "$results_file" ]; then
        results_file=$(find_latest_results)
    fi
    
    # If no output file specified, create one with timestamp
    if [ -z "$output_file" ]; then
        timestamp=$(date +"%Y%m%d_%H%M%S")
        output_file="result_analysis/klue_sts_error_analysis_${timestamp}.txt"
    fi
    
    # Create output directory if it doesn't exist
    mkdir -p "$(dirname "$output_file")"
    
    # Run analysis
    analyze_errors "$results_file" "$output_file"
    
    # Display summary
    print_info "Analysis Summary:"
    echo "Results file: $results_file"
    echo "Output file: $output_file"
    
    # Show first few lines of analysis
    if [ -f "$output_file" ]; then
        print_info "First few lines of analysis:"
        head -20 "$output_file"
    fi
}

# Run main function with all arguments
main "$@" 