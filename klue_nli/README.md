# KLUE Natural Language Inference Benchmark with Gemini 2.5 Flash on Vertex AI

This repository contains a benchmark script for evaluating Gemini 2.5 Flash on the Korean Language Understanding Evaluation (KLUE) Natural Language Inference task using Google Cloud Vertex AI.

## Overview

The KLUE Natural Language Inference (NLI) task involves determining the logical relationship between a premise and a hypothesis in Korean text. The model must classify the relationship into one of three categories:

- **entailment (함의)**: The premise logically entails the hypothesis
- **contradiction (모순)**: The premise contradicts the hypothesis  
- **neutral (중립)**: The premise neither entails nor contradicts the hypothesis

## Features

- **Comprehensive Benchmarking**: Evaluates accuracy, speed, and per-category performance
- **Detailed Analysis**: Provides error analysis and per-label accuracy breakdown
- **Flexible Configuration**: Supports various model parameters and sampling options
- **Result Export**: Saves results in JSON and CSV formats for further analysis
- **Progress Tracking**: Real-time progress bar and logging
- **Vertex AI Integration**: Uses Google Cloud Vertex AI for model inference

## Prerequisites

1. **Google Cloud Project**: You need a Google Cloud project with Vertex AI API enabled
2. **Authentication**: Set up authentication using one of the following methods:
   - Service Account Key (recommended for production)
   - Application Default Credentials (ADC) for local development
   - gcloud CLI authentication

## Installation

### Quick Setup (Recommended)

Use the provided setup script for easy installation:

```bash
# Complete setup (install dependencies + test)
./setup.sh full

# Or step by step:
./setup.sh install    # Install dependencies only
./setup.sh test       # Test the setup
```

### Manual Installation

1. Clone this repository or download the files
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Setup

### 1. Google Cloud Setup

1. **Create a Google Cloud Project** (if you don't have one):
   ```bash
   gcloud projects create YOUR_PROJECT_ID
   ```

2. **Enable Required APIs**:
   ```bash
   gcloud services enable aiplatform.googleapis.com
   ```

3. **Set up Authentication** (choose one method):

   **Method A: Service Account (Recommended)**
   ```bash
   # Create a service account
   gcloud iam service-accounts create klue-benchmark \
       --display-name="KLUE Benchmark Service Account"
   
   # Grant necessary permissions
   gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
       --member="serviceAccount:klue-benchmark@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
       --role="roles/aiplatform.user"
   
   # Create and download key
   gcloud iam service-accounts keys create key.json \
       --iam-account=klue-benchmark@YOUR_PROJECT_ID.iam.gserviceaccount.com
   
   # Set environment variable
   export GOOGLE_APPLICATION_CREDENTIALS="path/to/key.json"
   ```

   **Method B: Application Default Credentials**
   ```bash
   gcloud auth application-default login
   ```

   **Method C: gcloud CLI**
   ```bash
   gcloud auth login
   ```

4. **Set Project ID**:
   ```bash
   export GOOGLE_CLOUD_PROJECT="YOUR_PROJECT_ID"
   ```

5. **Consider Adding Project ID to .bashrc**:
For convenience, consider adding this line to your .bashrc. This environment variable will be lost every time the OS is restarted causing the following error:

In the Python code, 
```python
    def _initialize_vertex_ai(self):
        """Initialize Vertex AI with project and location."""
        try:
            project_id = self.config.project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
            if not project_id:
                raise ValueError("Google Cloud project ID must be provided via the --project-id flag or by setting the GOOGLE_CLOUD_PROJECT environment variable.")
```

In Terminal,
```bash
Error: GOOGLE_CLOUD_PROJECT environment variable is not set
Please set it with: export GOOGLE_CLOUD_PROJECT='your-project-id'
```

To fix this error, you may re-run:

```bash
echo 'export GOOGLE_CLOUD_PROJECT="YOUR_PROJECT_ID"' >> ~/.bashrc
```

But a more convenient and permanent fix is to add Project ID to `.bashrc`.

```bash
# Fetch the PROJECT_ID automatically with gcloud config get-value project
echo "export GOOGLE_CLOUD_PROJECT=\"$(gcloud config get-value project)\"" >> ~/.bashrc
echo "GOOGLE_CLOUD_PROJECT=$GOOGLE_CLOUD_PROJECT"

# Activate the change
source ~/.bashrc
```
Note the `gcloud` command automatically fetches `YOUR_PROJECT_ID`.

## Usage

### Basic Usage

```bash
python klue_nli-gemini2_5flash.py --project-id "your-project-id"
```

### Advanced Usage

```bash
# Test with limited samples (useful for quick testing)
python klue_nli-gemini2_5flash.py --project-id "your-project-id" --max-samples 100

# Custom output directory
python klue_nli-gemini2_5flash.py --project-id "your-project-id" --output-dir "my_results"

# Use different Vertex AI location
python klue_nli-gemini2_5flash.py --project-id "your-project-id" --location "us-west1"

# Adjust model parameters
python klue_nli-gemini2_5flash.py --project-id "your-project-id" --temperature 0.1 --max-tokens 512

# Skip saving detailed predictions (saves disk space)
python klue_nli-gemini2_5flash.py --project-id "your-project-id" --no-save-predictions
```

### Command Line Arguments

- `--project-id`: Google Cloud project ID (required if not set as environment variable)
- `--location`: Vertex AI location (default: "us-central1")
- `--max-samples`: Maximum number of samples to test (default: all test samples)
- `--output-dir`: Output directory for results (default: "benchmark_results")
- `--temperature`: Model temperature (default: 0.0)
- `--max-tokens`: Maximum output tokens (default: 1024)
- `--no-save-predictions`: Skip saving detailed prediction results

### Quick Start with Run Script

For convenience, a simple Bash script `run` is provided to quickly execute common benchmark scenarios:

```bash
# Make the script executable (if needed)
chmod +x run

# Run a small test with 10 samples
./run test

# Run the full benchmark (all test samples)
./run full

# Run with custom number of samples
./run custom 100

# Show help and available options
./run help
```

The run script automatically:
- Checks if the Python script exists
- Verifies that `GOOGLE_CLOUD_PROJECT` environment variable is set
- Executes the appropriate Python command with the correct parameters
- Provides clear error messages if prerequisites are not met
- **Captures and saves all output to log files** (see [Logging](#logging) section)

This is the recommended way to run the benchmark for most users.

| Consideration | Natural Language Inference (NLI) | Text Classification (TC) |
| :--- | :--- | :--- |
| **Input Length** | Longer inputs (premise + hypothesis) | Shorter inputs (e.g., news titles) |
| **Token Usage** | Higher token consumption due to paired inputs | Lower token consumption per sample |
| **Processing Time** | Slower due to complex logical reasoning | Faster due to a simpler task |

---

## Logging

The benchmark automatically captures and saves all output to log files for later review and analysis.

### Log File Locations

-   **Logs:** All log files (`.log`, `.err`) are saved in the `logs/` directory.
-   **Intermediate Results:** CSV, JSON, and TXT files with intermediate results are saved in the `benchmark_results/` directory.

#### `logs/` Directory Structure
```bash
logs/
├── klue_nli_test_10samples_20250706_185948.log
├── klue_nli_test_10samples_20250706_185948.err
├── klue_nli_custom_100samples_20250706_185948.log
└── klue_nli_custom_100samples_20250706_185948.err
```

#### `benchmark_results/` Directory Structure
```bash
benchmark_results/
├── klue_nli_results_009050_20250709_233210.csv
├── klue_nli_results_009100_20250709_233534.json
├── klue_nli_metrics_009050_20250709_233210.json
└── klue_nli_error_analysis_009100_20250709_233534.txt
```
### File Naming Convention

Log files follow a consistent naming format:
`klue_nli_[mode]_[samples]samples_[timestamp].[log|err]`

-   **`mode`**: The benchmark mode (`test`, `full`, or `custom`).
-   **`samples`**: The number of samples processed (`10`, `all`, or a specific number).
-   **`timestamp`**: The execution timestamp (`YYYYMMDD_HHMMSS`).

### File Types

#### `.log` Files
Contain the complete output from a benchmark run, including:
-   Command header with run details.
-   Initialization messages and progress updates.
-   Final results and performance metrics.
-   A summary of the error analysis.
-   All detailed error logs.

#### `.err` Files
Contain a focused summary of all errors, including:
-   Command header with run details.
-   The error analysis section showing misclassified samples.
-   Detailed error logs with timestamps.
-   JSON response details for failed predictions.
-   Error messages and stack traces.

### Command Header Format

Each log file begins with a self-documenting header for easy identification:
./run custom 100 Timestamp: Thu Jul 11 05:08:49 PDT 2025 Working Directory: /path/to/klue_nli
### Benefits of Logging
-   **Complete Audit Trail**: Every run is logged for future reference.
-   **Self-Documenting**: Each log includes the command and its context.
-   **Targeted Error Analysis**: Separate `.err` files simplify reviewing issues.
-   **Performance Tracking**: Easily compare results across different runs.
-   **Efficient Debugging**: Detailed logs help identify and resolve issues quickly.
-   **Compliance**: Maintain clear records of all benchmark executions.

### Example Usage

1.  **Run the benchmark:**
    ```bash
    # Run a quick test with 10 samples
    ./run test

    # Run the full benchmark on all samples
    ./run full

    # Run a custom benchmark with 100 samples
    ./run custom 100
    ```

2.  **Check the generated files:**
    ```bash
    # See what logs were created
    ls -la logs/

    # Review the full output of a specific run
    cat logs/klue_nli_custom_100samples_*.log

    # Review only the errors for that run
    cat logs/klue_nli_custom_100samples_*.err
    ```

### File Management
-   Log files are created automatically for each run.
-   Old log files are **not** automatically deleted.
-   Periodically archive or remove old logs to manage disk space.

## Available Scripts

The project includes several Bash scripts to simplify common tasks:

### Core Scripts

- **`./run`** - Quick benchmark runner
  - `./run test` - Run small test (10 samples)
  - `./run full` - Run full benchmark
  - `./run custom N` - Run with N samples
  - `./run help` - Show usage

- **`./setup.sh`** - Complete setup process
  - `./setup.sh install` - Install dependencies only
  - `./setup.sh test` - Test the setup
  - `./setup.sh full` - Complete setup (install + test)

- **`./install_dependencies.sh`** - Install Python dependencies
  - Simple dependency installation with next steps guidance

- **`./verify_scripts.sh`** - Verify all scripts are properly saved
  - Checks that all Bash scripts exist and are executable

- **`./get_errors.sh`** - Extract error details from benchmark results
  - `./get_errors.sh` - Analyze default results file
  - `./get_errors.sh [csv_file_path]` - Analyze specific CSV file
  - Outputs error details to `result_analysis/errors_[filename].txt`

### Usage Examples

```bash
# Complete setup and run
./setup.sh full
./run test

# Install dependencies only
./install_dependencies.sh

# Verify scripts are working
./verify_scripts.sh
```

## Error Analysis with get_errors.sh

The `get_errors.sh` script extracts and analyzes error details from benchmark result files, making it easy to review misclassified samples.

### Features

- **Automatic Error Extraction**: Finds all rows with non-empty error columns
- **Flexible Input**: Works with any CSV file containing an "error" column
- **Organized Output**: Saves error details to the `result_analysis/` directory
- **Error Counting**: Reports the total number of errors found
- **Clear Feedback**: Provides status messages during processing

### Usage

```bash
# Analyze the default results file
./get_errors.sh

# Analyze a specific CSV file
./get_errors.sh benchmark_results/klue_nli_results_009100_20250709_233534.csv

# Analyze intermediate results
./get_errors.sh benchmark_results/klue_nli_results_009050_20250709_233210.csv
```

### Output

The script creates error analysis files in the `result_analysis/` directory:

```bash
result_analysis/
├── errors_klue_nli_results_009100_20250709_233534.txt
├── errors_klue_nli_results_009050_20250709_233210.txt
└── ...
```

Each output file contains:
- One error message per line
- Detailed error descriptions from the benchmark
- JSON response details for failed predictions
- Error timestamps and context

### Example Output

```bash
$ ./get_errors.sh benchmark_results/klue_nli_results_009100_20250709_233534.csv
Analyzing errors in: benchmark_results/klue_nli_results_009100_20250709_233534.csv
Output will be saved to: result_analysis/errors_klue_nli_results_009100_20250709_233534.txt
---
Analysis complete! Found 15 errors.
Error details saved to: result_analysis/errors_klue_nli_results_009100_20250709_233534.txt
```

### Integration with Benchmark Workflow

1. **Run the benchmark** to generate results:
   ```bash
   ./run custom 100
   ```

2. **Analyze errors** from the generated results:
   ```bash
   ./get_errors.sh benchmark_results/klue_nli_results_*.csv
   ```

3. **Review error details** in the analysis files:
   ```bash
   cat result_analysis/errors_*.txt
   ```

## Task Description

### Natural Language Inference (NLI)

Natural Language Inference is a fundamental task in natural language processing that involves determining the logical relationship between two sentences: a premise and a hypothesis.

**Input Format:**
- **Premise**: A statement that serves as the context or background information
- **Hypothesis**: A statement whose truth value needs to be determined relative to the premise

**Output Categories:**
1. **Entailment (함의)**: The premise logically entails the hypothesis
   - Example: Premise "김철수는 의사다" → Hypothesis "김철수는 의료진이다" → Entailment

2. **Contradiction (모순)**: The premise contradicts the hypothesis
   - Example: Premise "김철수는 의사다" → Hypothesis "김철수는 의사가 아니다" → Contradiction

3. **Neutral (중립)**: The premise neither entails nor contradicts the hypothesis
   - Example: Premise "김철수는 의사다" → Hypothesis "오늘 날씨가 좋다" → Neutral

### KLUE-NLI Dataset

The KLUE-NLI dataset contains Korean sentence pairs with their corresponding logical relationships. The dataset is designed to evaluate a model's ability to understand Korean language semantics and logical reasoning.

## Results

The benchmark provides comprehensive evaluation metrics including:

- **Overall Accuracy**: Percentage of correctly classified samples
- **Per-label Accuracy**: Accuracy breakdown by entailment/contradiction/neutral
- **Processing Speed**: Time per sample and samples per second
- **Error Analysis**: Detailed analysis of misclassified samples

Results are saved in multiple formats:
- `klue_nli_metrics_YYYYMMDD_HHMMSS.json`: Summary metrics
- `klue_nli_results_YYYYMMDD_HHMMSS.json`: Detailed results
- `klue_nli_results_YYYYMMDD_HHMMSS.csv`: Results in CSV format for analysis

## Model Performance

The benchmark evaluates Gemini 2.5 Flash on the KLUE-NLI task, providing insights into:

- **Semantic Understanding**: How well the model understands Korean language semantics
- **Logical Reasoning**: The model's ability to perform logical inference
- **Korean Language Proficiency**: Performance on Korean-specific linguistic phenomena

## Troubleshooting

### Common Issues

1. **Authentication Errors**:
   ```bash
   # Ensure proper authentication
   gcloud auth application-default login
   export GOOGLE_CLOUD_PROJECT="your-project-id"
   ```

2. **API Not Enabled**:
   ```bash
   # Enable Vertex AI API
   gcloud services enable aiplatform.googleapis.com
   ```

3. **Dataset Loading Issues**:
   ```bash
   # Install/upgrade datasets library
   pip install --upgrade datasets
   ```

4. **Memory Issues with Large Datasets**:
   ```bash
   # Use --max-samples to limit dataset size
   python klue_nli-gemini2_5flash.py --project-id "your-project-id" --max-samples 1000
   ```

### Getting Help

If you encounter issues:

1. Check the troubleshooting section above
2. Verify your Google Cloud setup and permissions
3. Ensure all dependencies are properly installed
4. Check the logs for detailed error messages

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- KLUE dataset creators and maintainers
- Google Cloud Vertex AI team
- Hugging Face datasets library 