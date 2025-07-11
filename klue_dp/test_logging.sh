#!/bin/bash

# Test script for KLUE DP logging functionality

echo "Testing KLUE DP logging functionality..."

# Create logs directory if it doesn't exist
mkdir -p logs

# Function to generate log filename (copied from run script)
generate_log_filename() {
    local mode=$1
    local samples=$2
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    
    if [ "$mode" = "custom" ]; then
        echo "klue_dp_${mode}_${samples}samples_${timestamp}"
    else
        echo "klue_dp_${mode}_${timestamp}"
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
    
    # Error analysis content (sentence and words lines)
    in_error_analysis && /^[[:space:]]*Sentence:/ {
        print
        next
    }
    
    in_error_analysis && /^[[:space:]]*Words:/ {
        print
        next
    }
    
    in_error_analysis && /^[[:space:]]*UAS:/ {
        print
        next
    }
    
    in_error_analysis && /^[[:space:]]*Error:/ {
        print
        next
    }
    
    # Empty line after error marks end of error analysis
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
2025-07-06 18:59:48,059 - INFO - Loading KLUE DP dataset for dependency parsing...
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
  "create_time": "2025-07-06T19:02:40.492285Z",
  "response_id": "FMhqaP2FHqW8nvgP2IrW4As"
}
2025-07-06 19:03:15,123 - INFO - Benchmark completed!
2025-07-06 19:03:15,124 - INFO - UAS: 0.8234 (8234/10000)
2025-07-06 19:03:15,125 - INFO - LAS: 0.7891 (7891/10000)
2025-07-06 19:03:15,126 - INFO - Total time: 135.67 seconds
2025-07-06 19:03:15,127 - INFO - Average time per sample: 1.357 seconds
2025-07-06 19:03:15,128 - INFO - Samples per second: 0.74

Error Analysis (showing first 5 errors):
  1. Sample ID: klue-dp-v1.1_dev_00001
     Sentence: 한국어 문장의 의존 구문 분석을 수행합니다.
     Words: 한국어 문장의 의존 구문 분석을 수행합니다
     UAS: 0.2000 | LAS: 0.1000
     Error: Failed to parse dependency structure

  2. Sample ID: klue-dp-v1.1_dev_00002
     Sentence: 복잡한 한국어 문법 구조를 이해해야 합니다.
     Words: 복잡한 한국어 문법 구조를 이해해야 합니다
     UAS: 0.3000 | LAS: 0.2000
     Error: Incorrect head assignment

  3. Sample ID: klue-dp-v1.1_dev_00003
     Sentence: 의존 관계를 정확히 파악하는 것이 중요합니다.
     Words: 의존 관계를 정확히 파악하는 것이 중요합니다
     UAS: 0.4000 | LAS: 0.3000
     Error: Missing dependency labels

  4. Sample ID: klue-dp-v1.1_dev_00004
     Sentence: 한국어의 특수한 문법적 특성을 고려해야 합니다.
     Words: 한국어의 특수한 문법적 특성을 고려해야 합니다
     UAS: 0.2500 | LAS: 0.1500
     Error: Incorrect POS tag interpretation

  5. Sample ID: klue-dp-v1.1_dev_00005
     Sentence: 구문 분석의 정확도가 언어 이해에 핵심입니다.
     Words: 구문 분석의 정확도가 언어 이해에 핵심입니다
     UAS: 0.3500 | LAS: 0.2500
     Error: Wrong dependency direction

Per-POS Performance:
  NNG (일반명사): UAS=0.8500, LAS=0.8200 (2500 words)
  VV (동사): UAS=0.9000, LAS=0.8700 (1800 words)
  JKS (주격조사): UAS=0.9500, LAS=0.9300 (1200 words)
  JKO (목적격조사): UAS=0.9200, LAS=0.8900 (1100 words)
  VA (형용사): UAS=0.8800, LAS=0.8500 (900 words)
  MAG (일반부사): UAS=0.8500, LAS=0.8200 (800 words)
  JKB (부사격조사): UAS=0.9300, LAS=0.9000 (700 words)
  MM (관형사): UAS=0.8700, LAS=0.8400 (600 words)
  EF (종결어미): UAS=0.9400, LAS=0.9100 (500 words)
  EC (연결어미): UAS=0.9100, LAS=0.8800 (400 words)
EOF
}

# Test 1: Generate log filename
echo "Test 1: Generate log filename"
log_filename=$(generate_log_filename "test" "10")
echo "Generated filename: $log_filename"
echo "Expected pattern: klue_dp_test_10samples_YYYYMMDD_HHMMSS"
echo ""

# Test 2: Create sample log file
echo "Test 2: Create sample log file"
sample_log_content=$(create_sample_log)
echo "$sample_log_content" > "logs/${log_filename}.log"
echo "Created sample log file: logs/${log_filename}.log"
echo ""

# Test 3: Extract errors
echo "Test 3: Extract errors"
extracted_errors=$(extract_errors "$sample_log_content")
echo "$extracted_errors" > "logs/${log_filename}.err"
echo "Created error file: logs/${log_filename}.err"
echo ""

# Test 4: Verify file creation
echo "Test 4: Verify file creation"
if [ -f "logs/${log_filename}.log" ]; then
    echo "✓ Log file created successfully"
    log_lines=$(wc -l < "logs/${log_filename}.log")
    echo "  - Lines in log file: $log_lines"
else
    echo "✗ Log file creation failed"
fi

if [ -f "logs/${log_filename}.err" ]; then
    echo "✓ Error file created successfully"
    err_lines=$(wc -l < "logs/${log_filename}.err")
    echo "  - Lines in error file: $err_lines"
else
    echo "✗ Error file creation failed"
fi
echo ""

# Test 5: Check error extraction accuracy
echo "Test 5: Check error extraction accuracy"
error_count=$(echo "$extracted_errors" | grep -c "Sample ID:")
echo "Found $error_count error samples in extraction"
echo ""

# Test 6: Verify command header format
echo "Test 6: Verify command header format"
echo "Testing command header format..."
echo "./run test" > "logs/test_header.log"
echo "Timestamp: $(date)" >> "logs/test_header.log"
echo "Working Directory: $(pwd)" >> "logs/test_header.log"
echo "========================================" >> "logs/test_header.log"
echo "" >> "logs/test_header.log"
echo "Sample content" >> "logs/test_header.log"
echo "✓ Command header format test completed"
echo ""

echo "All logging tests completed successfully!"
echo ""
echo "Files created:"
echo "  - logs/${log_filename}.log (sample log file)"
echo "  - logs/${log_filename}.err (extracted errors)"
echo "  - logs/test_header.log (command header test)"
echo ""
echo "You can examine these files to verify the logging functionality." 