#!/bin/bash

# Default file name
DEFAULT_FILE="benchmark_results/klue_nli_results_009100_20250709_233534.csv"

# Use the first argument as the file name, or the default if no argument is provided
FILE="${1:-$DEFAULT_FILE}"

# Check if the input CSV file exists
if [ ! -f "$FILE" ]; then
  echo "Error: Input CSV file '$FILE' not found." >&2
  echo "Usage: $0 [csv_file_path]" >&2
  echo "  csv_file_path: Path to the CSV file to analyze (default: $DEFAULT_FILE)" >&2
  exit 1
fi

# Create result_analysis directory if it doesn't exist
mkdir -p result_analysis

# Extract filename without path for output naming
FILENAME=$(basename "$FILE")
OUTPUT_FILE="result_analysis/errors_${FILENAME%.csv}.txt"

# Create a temporary file for the awk script
# mktemp creates a unique temporary file name
AWK_SCRIPT_TEMP=$(mktemp)

# Write the awk script into the temporary file using a heredoc
# This ensures precise control over the content and line endings
cat > "$AWK_SCRIPT_TEMP" << 'EOF_AWK_SCRIPT'
BEGIN {
    error_col = 0;
}
NR==1 {
    for (i=1; i<=NF; i++) {
        if ($i == "error") {
            error_col = i;
            break;
        }
    }
    if (error_col == 0) {
        # Using FILENAME_FROM_SHELL passed via -v
        print "Error: Column \"error\" not found in the header of \x27" FILENAME_FROM_SHELL "\x27." > "/dev/stderr";
        exit 1;
    }
    next;
}
NR>1 {
    if (error_col > 0) {
        if ($error_col != "") {
             print $error_col;
        }
    }
}
EOF_AWK_SCRIPT

# Execute awk with the temporary script file and the input CSV file
# -f "$AWK_SCRIPT_TEMP" tells awk to read its script from the temporary file
# -v FILENAME_FROM_SHELL="$FILE" passes the actual input CSV filename to awk
echo "Analyzing errors in: $FILE"
echo "Output will be saved to: $OUTPUT_FILE"
echo "---"

awk -F',' -f "$AWK_SCRIPT_TEMP" -v FILENAME_FROM_SHELL="$FILE" "$FILE" > "$OUTPUT_FILE"

# Check if the output file was created and has content
if [ -s "$OUTPUT_FILE" ]; then
    ERROR_COUNT=$(wc -l < "$OUTPUT_FILE")
    echo "Analysis complete! Found $ERROR_COUNT errors."
    echo "Error details saved to: $OUTPUT_FILE"
else
    echo "No errors found in the CSV file."
    echo "Empty output file created: $OUTPUT_FILE"
fi

# Remove the temporary awk script file upon completion
rm -f "$AWK_SCRIPT_TEMP" 