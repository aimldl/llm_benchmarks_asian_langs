#!/bin/bash

# Test script for KLUE NLI logging functionality

echo "Testing KLUE NLI logging functionality..."

# Create logs directory if it doesn't exist
mkdir -p logs

# Function to generate log filename (copied from run script)
generate_log_filename() {
    local mode=$1
    local samples=$2
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    
    if [ "$mode" = "custom" ]; then
        echo "klue_nli_${mode}_${samples}samples_${timestamp}"
    else
        echo "klue_nli_${mode}_${timestamp}"
    fi
}

# Function to extract errors (copied from run script)
extract_errors() {
    local log_content="$1"
    
    # First pass: collect error analysis section
    error_analysis=$(echo "$log_content" | awk '
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
    in_error_analysis && /^[[:space:]]*[0-9]+\. True:/ {
        print
        next
    }
    
    # Error analysis content (text and prediction lines)
    in_error_analysis && /^[[:space:]]*Premise:/ {
        print
        next
    }
    
    in_error_analysis && /^[[:space:]]*Hypothesis:/ {
        print
        next
    }
    
    in_error_analysis && /^[[:space:]]*Prediction:/ {
        print
        next
    }
    
    # Empty line after prediction marks end of error analysis
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
    ')
    
    # Second pass: collect ERROR logs
    error_logs=$(echo "$log_content" | awk '
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
    ')
    
    # Output error analysis first, then error logs
    echo "$error_analysis"
    echo ""
    echo "$error_logs"
}

# Create a sample log content for testing
create_sample_log() {
    cat << 'EOF'
2025-07-06 18:59:48,058 - INFO - Initialized Vertex AI with project: vertex-workbench-notebook, location: us-central1
2025-07-06 18:59:48,059 - INFO - Initialized model: gemini-2.5-flash
2025-07-06 18:59:48,059 - INFO - Loading KLUE NLI dataset for natural language inference...
2025-07-06 19:00:02,824 - INFO - Preparing to load a subset of 100 samples.
2025-07-06 19:00:02,828 - INFO - Reached sample limit of 100. Halting data loading.
2025-07-06 19:00:02,828 - INFO - ✅ Successfully loaded 100 samples.
2025-07-06 19:00:02,828 - INFO - Starting benchmark...
2025-07-06 19:01:48,002 - ERROR - Prediction failed: Cannot get the response text.
Cannot get the Candidate text.
Response candidate content has no parts (and thus no text). The candidate is likely blocked by the safety filters.
Content:
{
  "role": "model"
}
Candidate:
{
  "content": {
    "role": "model"
  },
  "finish_reason": "MAX_TOKENS"
}
Response:
{
  "candidates": [
    {
      "content": {
        "role": "model"
      },
      "finish_reason": "MAX_TOKENS"
    }
  ],
  "usage_metadata": {
    "prompt_token_count": 594,
    "total_token_count": 1617,
    "prompt_tokens_details": [
      {
        "modality": "TEXT",
        "token_count": 594
      }
    ],
    "thoughts_token_count": 1023
  },
  "model_version": "gemini-2.5-flash",
  "create_time": "2025-07-06T19:01:40.492285Z",
  "response_id": "FMhqaP2FHqW8nvgP2IrW4As"
}
2025-07-06 19:02:56,717 - ERROR - Prediction failed: Cannot get the response text.
Cannot get the Candidate text.
Response candidate content has no parts (and thus no text). The candidate is likely blocked by the safety filters.
Content:
{
  "role": "model"
}
Candidate:
{
  "content": {
    "role": "model"
  },
  "finish_reason": "MAX_TOKENS"
}
Response:
{
  "candidates": [
    {
      "content": {
        "role": "model"
      },
      "finish_reason": "MAX_TOKENS"
    }
  ],
  "usage_metadata": {
    "prompt_token_count": 587,
    "total_token_count": 1610,
    "prompt_tokens_details": [
      {
        "modality": "TEXT",
        "token_count": 587
      }
    ],
    "thoughts_token_count": 1023
  },
  "model_version": "gemini-2.5-flash",
  "create_time": "2025-07-06T19:02:48.616287Z",
  "response_id": "WMhqaMPPJaKgnvgPiPGr4AE"
}
Processing samples: 100%|█████████████████████| 100/100 [07:25<00:00,  4.46s/it]
2025-07-06 19:07:28,421 - INFO - Benchmark completed!
2025-07-06 19:07:28,421 - INFO - Accuracy: 0.7500 (75/100)
2025-07-06 19:07:28,421 - INFO - Total time: 445.59 seconds
2025-07-06 19:07:28,421 - INFO - Average time per sample: 4.456 seconds
2025-07-06 19:07:28,421 - INFO - Metrics saved to: benchmark_results/klue_nli_metrics_20250706_190728.json
2025-07-06 19:07:28,423 - INFO - Detailed results saved to: benchmark_results/klue_nli_results_20250706_190728.json
2025-07-06 19:07:28,426 - INFO - Results saved as CSV: benchmark_results/klue_nli_results_20250706_190728.csv

============================================================
KLUE Natural Language Inference Benchmark Results
============================================================
Model: gemini-2.5-flash
Platform: Google Cloud Vertex AI
Project: vertex-workbench-notebook
Location: us-central1
Accuracy: 0.7500 (75/100)
Total Time: 445.59 seconds
Average Time per Sample: 4.456 seconds
Samples per Second: 0.22

Per-label Accuracy:
  contradiction: 0.8000 (16/20)
  entailment: 0.7000 (14/20)
  neutral: 0.7500 (45/60)

Error Analysis (showing first 5 errors):
  1. True: entailment | Predicted: neutral
     Premise: 한국은 아시아에 위치한 국가이다.
     Hypothesis: 한국은 아시아 대륙에 있다.
     Prediction: neutral

  2. True: contradiction | Predicted: entailment
     Premise: 오늘은 비가 온다.
     Hypothesis: 오늘은 맑다.
     Prediction: entailment

  3. True: entailment | Predicted: contradiction
     Premise: 그는 의사이다.
     Hypothesis: 그는 의료진이다.
     Prediction: contradiction

  4. True: neutral | Predicted: entailment
     Premise: 그녀는 학생이다.
     Hypothesis: 그녀는 대학생이다.
     Prediction: entailment

  5. True: contradiction | Predicted: neutral
     Premise: 이 음식은 맛있다.
     Hypothesis: 이 음식은 맛없다.
     Prediction: neutral
EOF
}

# Test the logging functionality
echo "Creating test log file..."
log_filename=$(generate_log_filename "test" "100")
sample_log_content=$(create_sample_log)

# Add command header to log file
echo "$0 (test script)" > "logs/${log_filename}.log"
echo "Timestamp: $(date)" >> "logs/${log_filename}.log"
echo "Working Directory: $(pwd)" >> "logs/${log_filename}.log"
echo "========================================" >> "logs/${log_filename}.log"
echo "" >> "logs/${log_filename}.log"

# Save full log content
echo "$sample_log_content" >> "logs/${log_filename}.log"

# Add command header to error file
echo "$0 (test script)" > "logs/${log_filename}.err"
echo "Timestamp: $(date)" >> "logs/${log_filename}.err"
echo "Working Directory: $(pwd)" >> "logs/${log_filename}.err"
echo "========================================" >> "logs/${log_filename}.err"
echo "" >> "logs/${log_filename}.err"

# Extract and save errors
extract_errors "$sample_log_content" >> "logs/${log_filename}.err"

echo "Test completed!"
echo "Generated files:"
echo "  Full log: logs/${log_filename}.log"
echo "  Error log: logs/${log_filename}.err"
echo ""
echo "Full log size: $(wc -l < logs/${log_filename}.log) lines"
echo "Error log size: $(wc -l < logs/${log_filename}.err) lines"
echo ""
echo "Error log content preview:"
echo "=========================="
head -20 "logs/${log_filename}.err" 