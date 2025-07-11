#!/bin/bash

# KLUE DST Error Analysis Script
# This script analyzes error patterns in benchmark results and logs

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

# Function to show usage
show_help() {
    echo "KLUE DST Error Analysis Script"
    echo ""
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  analyze [file]  Analyze errors in a specific log file"
    echo "  latest          Analyze the most recent log file"
    echo "  all             Analyze all log files in the logs directory"
    echo "  help            Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 analyze logs/klue_dst_test_20241201_120000.log"
    echo "  $0 latest"
    echo "  $0 all"
    echo ""
    echo "Output will be saved to result_analysis/ directory"
}

# Function to analyze a single log file
analyze_log_file() {
    local log_file="$1"
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local output_file="result_analysis/dst_error_analysis_${timestamp}.txt"
    
    if [ ! -f "$log_file" ]; then
        print_error "Log file not found: $log_file"
        return 1
    fi
    
    print_status "Analyzing log file: $log_file"
    
    # Create result_analysis directory if it doesn't exist
    mkdir -p result_analysis
    
    # Start analysis
    {
        echo "KLUE DST Error Analysis Report"
        echo "=============================="
        echo "Log File: $log_file"
        echo "Analysis Time: $(date)"
        echo "=============================="
        echo ""
        
        # Extract command and timestamp from log header
        echo "EXECUTION INFO:"
        echo "==============="
        head -10 "$log_file" | grep -E "^(\./run|Timestamp|Working Directory)" || echo "No execution info found"
        echo ""
        
        # Count total samples processed
        echo "PROCESSING SUMMARY:"
        echo "==================="
        total_samples=$(grep -c "Processing samples:" "$log_file" || echo "0")
        echo "Total samples processed: $total_samples"
        
        # Count successful predictions
        successful_predictions=$(grep -c "success.*True" "$log_file" || echo "0")
        echo "Successful predictions: $successful_predictions"
        
        # Count failed predictions
        failed_predictions=$(grep -c "success.*False" "$log_file" || echo "0")
        echo "Failed predictions: $failed_predictions"
        
        if [ "$total_samples" -gt 0 ]; then
            success_rate=$(echo "scale=2; $successful_predictions * 100 / $total_samples" | bc -l 2>/dev/null || echo "N/A")
            echo "Success rate: ${success_rate}%"
        fi
        echo ""
        
        # Extract error analysis section
        echo "ERROR ANALYSIS:"
        echo "==============="
        awk '
        BEGIN { 
            in_error_analysis = 0; 
            error_analysis_found = 0;
        }
        
        # Start of error analysis section
        /Error Analysis \(showing first [0-9]+ errors\):/ {
            in_error_analysis = 1
            error_analysis_found = 1
            print
            next
        }
        
        # Error analysis content
        in_error_analysis && /^[[:space:]]*[0-9]+\. Sample ID:/ {
            print
            next
        }
        
        in_error_analysis && /^[[:space:]]*Dialogue ID:/ {
            print
            next
        }
        
        in_error_analysis && /^[[:space:]]*Turn ID:/ {
            print
            next
        }
        
        in_error_analysis && /^[[:space:]]*Domains:/ {
            print
            next
        }
        
        in_error_analysis && /^[[:space:]]*Ground Truth Intent:/ {
            print
            next
        }
        
        in_error_analysis && /^[[:space:]]*Predicted Intent:/ {
            print
            next
        }
        
        in_error_analysis && /^[[:space:]]*Ground Truth Requested Slots:/ {
            print
            next
        }
        
        in_error_analysis && /^[[:space:]]*Predicted Requested Slots:/ {
            print
            next
        }
        
        in_error_analysis && /^[[:space:]]*Ground Truth Slot Values:/ {
            print
            next
        }
        
        in_error_analysis && /^[[:space:]]*Predicted Slot Values:/ {
            print
            next
        }
        
        in_error_analysis && /^[[:space:]]*Intent Accuracy:/ {
            print
            next
        }
        
        in_error_analysis && /^[[:space:]]*Requested Slots F1:/ {
            print
            next
        }
        
        in_error_analysis && /^[[:space:]]*Slot Values F1:/ {
            print
            next
        }
        
        in_error_analysis && /^[[:space:]]*Overall F1:/ {
            print
            next
        }
        
        in_error_analysis && /^[[:space:]]*Error:/ {
            print
            next
        }
        
        # Empty line after error details marks end of error analysis
        in_error_analysis && /^[[:space:]]*$/ {
            in_error_analysis = 0
            print
        }
        
        # If no error analysis was found, add a note
        END {
            if (!error_analysis_found) {
                print "No error analysis section found in the log."
            }
        }
        ' "$log_file"
        
        echo ""
        
        # Extract ERROR logs
        echo "ERROR LOGS:"
        echo "==========="
        awk '
        BEGIN { 
            in_errors = 0; 
            error_count = 0;
        }
        
        # Start of ERROR logs
        /^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2},[0-9]{3} - ERROR -/ {
            in_errors = 1
            error_count++
            print
            next
        }
        
        # Continue ERROR logs until we hit a non-error line
        in_errors {
            if (/^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2},[0-9]{3} - (INFO|WARNING|DEBUG)/) {
                in_errors = 0
            } else if (/^[[:space:]]*$/) {
                # Empty line in error section
                print
            } else if (/^[[:space:]]*[{}]/) {
                # JSON content in error section
                print
            } else if (/^[[:space:]]*[A-Za-z_]+:/) {
                # Key-value pairs in error section
                print
            } else if (in_errors) {
                # Any other content in error section
                print
            }
        }
        
        END {
            if (error_count == 0) {
                print "No ERROR logs found."
            }
        }
        ' "$log_file"
        
        echo ""
        
        # Performance metrics summary
        echo "PERFORMANCE METRICS:"
        echo "===================="
        awk '
        /Intent Accuracy:/ { intent_acc = $3 }
        /Requested Slots F1:/ { req_slots_f1 = $4 }
        /Slot Values F1:/ { slot_values_f1 = $4 }
        /Overall F1:/ { overall_f1 = $3 }
        /Total Samples:/ { total_samples = $3 }
        /Total Time:/ { total_time = $3 }
        
        END {
            if (intent_acc != "") print "Intent Accuracy: " intent_acc
            if (req_slots_f1 != "") print "Requested Slots F1: " req_slots_f1
            if (slot_values_f1 != "") print "Slot Values F1: " slot_values_f1
            if (overall_f1 != "") print "Overall F1: " overall_f1
            if (total_samples != "") print "Total Samples: " total_samples
            if (total_time != "") print "Total Time: " total_time " seconds"
        }
        ' "$log_file"
        
    } > "$output_file"
    
    print_success "Error analysis saved to: $output_file"
    echo "Analysis completed for: $log_file"
}

# Function to find the latest log file
find_latest_log() {
    if [ ! -d "logs" ]; then
        print_error "Logs directory not found"
        return 1
    fi
    
    latest_log=$(find logs -name "klue_dst_*.log" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
    
    if [ -z "$latest_log" ]; then
        print_error "No log files found in logs directory"
        return 1
    fi
    
    echo "$latest_log"
}

# Main script logic
case "${1:-help}" in
    "analyze")
        if [ -z "$2" ]; then
            print_error "Please provide a log file to analyze"
            echo "Usage: $0 analyze <log_file>"
            exit 1
        fi
        analyze_log_file "$2"
        ;;
    "latest")
        latest_log=$(find_latest_log)
        if [ $? -eq 0 ]; then
            analyze_log_file "$latest_log"
        else
            exit 1
        fi
        ;;
    "all")
        if [ ! -d "logs" ]; then
            print_error "Logs directory not found"
            exit 1
        fi
        
        log_files=$(find logs -name "klue_dst_*.log" -type f)
        
        if [ -z "$log_files" ]; then
            print_error "No log files found in logs directory"
            exit 1
        fi
        
        print_status "Analyzing all log files..."
        for log_file in $log_files; do
            echo ""
            analyze_log_file "$log_file"
        done
        print_success "All log files analyzed"
        ;;
    "help"|*)
        show_help
        ;;
esac 