#!/bin/bash

# Test script for KLUE MRC logging functionality

echo "Testing KLUE MRC logging functionality..."

# Create logs directory if it doesn't exist
mkdir -p logs

# Function to generate log filename (copied from run script)
generate_log_filename() {
    local mode=$1
    local samples=$2
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    
    if [ "$mode" = "custom" ]; then
        echo "klue_mrc_${mode}_${samples}samples_${timestamp}"
    else
        echo "klue_mrc_${mode}_${timestamp}"
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
    in_error_analysis && /^[[:space:]]*[0-9]+\. Sample ID:/ {
        print
        next
    }
    
    # Error analysis content (question and context lines)
    in_error_analysis && /^[[:space:]]*Question:/ {
        print
        next
    }
    
    in_error_analysis && /^[[:space:]]*Context:/ {
        print
        next
    }
    
    in_error_analysis && /^[[:space:]]*Ground Truth:/ {
        print
        next
    }
    
    in_error_analysis && /^[[:space:]]*Predicted:/ {
        print
        next
    }
    
    in_error_analysis && /^[[:space:]]*Exact Match:/ {
        print
        next
    }
    
    in_error_analysis && /^[[:space:]]*F1 Score:/ {
        print
        next
    }
    
    in_error_analysis && /^[[:space:]]*Is Impossible:/ {
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
2025-07-06 18:59:48,059 - INFO - Loading KLUE MRC dataset for machine reading comprehension...
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
  "create_time": "2025-07-06T19:02:56.717Z",
  "response_id": "FMhqaP2FHqW8nvgP2IrW4As"
}

Error Analysis (showing first 5 errors):
  1. Sample ID: mrc-0001
     Question: 이 기사의 제목은 무엇인가요?
     Context: 서울시는 오늘 새로운 환경 정책을 발표했습니다...
     Ground Truth: ['서울시 새로운 환경 정책 발표']
     Predicted: 서울시가 새로운 환경 정책을 발표했다
     Exact Match: 0.0000 | F1: 0.8000
     Is Impossible: False

  2. Sample ID: mrc-0002
     Question: 언제 이 사건이 발생했나요?
     Context: 지난주 화요일에 발생한 교통사고로...
     Ground Truth: ['지난주 화요일']
     Predicted: 답을 찾을 수 없습니다
     Exact Match: 0.0000 | F1: 0.0000
     Is Impossible: True

  3. Sample ID: mrc-0003
     Question: 누가 이 프로젝트를 주도했나요?
     Context: 김철수 박사가 이번 연구 프로젝트를...
     Ground Truth: ['김철수 박사']
     Predicted: 김철수 박사
     Exact Match: 1.0000 | F1: 1.0000
     Is Impossible: False

Benchmark completed!
Exact Match: 0.3333
F1 Score: 0.6000
Impossible Accuracy: 0.0000
Total time: 120.45 seconds
Average time per sample: 1.20 seconds
EOF

# Test the logging functionality
echo "Creating test log file..."
log_filename=$(generate_log_filename "test" "5")
create_sample_log > "logs/${log_filename}.log"

echo "Testing error extraction..."
extract_errors "$(cat logs/${log_filename}.log)" > "logs/${log_filename}.err"

echo "Test completed!"
echo "Log files created:"
echo "  Full log: logs/${log_filename}.log"
echo "  Error log: logs/${log_filename}.err"

# Show the error extraction results
echo ""
echo "Error extraction results:"
echo "========================="
cat "logs/${log_filename}.err" 
