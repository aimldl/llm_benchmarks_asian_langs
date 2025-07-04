# KLUE Topic Classification Benchmark with Gemini 2.5 Flash on Vertex AI

This repository contains a benchmark script for evaluating Gemini 2.5 Flash on the Korean Language Understanding Evaluation (KLUE) Topic Classification task using Google Cloud Vertex AI.

## Summary
In essence, run the following commands:
```bash
$ git clone https://github.com/aimldl/llm_benchmarks_asian_langs.git
$ cd klue_tc
$ ./setup.sh full
$ ./run test
```


## Overview

The KLUE Topic Classification (TC) task involves classifying Korean news articles into 7 categories:
- 정치 (Politics)
- 경제 (Economy)
- 사회 (Society)
- 생활문화 (Lifestyle & Culture)
- 세계 (World)
- IT과학 (IT & Science)
- 스포츠 (Sports)

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
            # Hard-coded project ID for debugging/testing
            #   project_id = "vertex-workbench-notebook"  
            # Corrected: Use project ID from config or environment for better practice
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

But a more convenient and permanant fix is to add Project ID to `.bashrc`.

```bash
# Fetch the PROJECT_ID automatically with gcloud config get-value project
echo "export GOOGLE_CLOUD_PROJECT=\"$(gcloud config get-value project)\"" >> ~/.bashrc
echo "GOOGLE_CLOUD_PROJECT=$GOOGLE_CLOUD_PROJECT"

# Activate the change
source ~/.bashrc
```
Note the `gcloud` command automatically fetchs `YOUR_PROJECT_ID`.

## Usage

### Basic Usage

```bash
python klue_tc-gemini2_5flash.py --project-id "your-project-id"
```

### Advanced Usage

```bash
# Test with limited samples (useful for quick testing)
python klue_tc-gemini2_5flash.py --project-id "your-project-id" --max-samples 100

# Custom output directory
python klue_tc-gemini2_5flash.py --project-id "your-project-id" --output-dir "my_results"

# Use different Vertex AI location
python klue_tc-gemini2_5flash.py --project-id "your-project-id" --location "us-west1"

# Adjust model parameters
python klue_tc-gemini2_5flash.py --project-id "your-project-id" --temperature 0.1 --max-tokens 512

# Skip saving detailed predictions (saves disk space)
python klue_tc-gemini2_5flash.py --project-id "your-project-id" --no-save-predictions
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

This is the recommended way to run the benchmark for most users.

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

## Output

The benchmark generates several output files in the specified output directory:

1. **Metrics JSON**: `klue_tc_metrics_YYYYMMDD_HHMMSS.json`
   - Overall accuracy and performance metrics
   - Timing information

2. **Detailed Results JSON**: `klue_tc_results_YYYYMMDD_HHMMSS.json`
   - Individual predictions for each sample
   - Error analysis data

3. **CSV Results**: `klue_tc_results_YYYYMMDD_HHMMSS.csv`
   - Tabular format for easy analysis in Excel/spreadsheets

## Example Output

```
============================================================
KLUE Topic Classification Benchmark Results
============================================================
Model: gemini-2.0-flash-exp
Platform: Google Cloud Vertex AI
Project: your-project-id
Location: us-central1
Accuracy: 0.8542 (854/1000)
Total Time: 125.34 seconds
Average Time per Sample: 0.125 seconds
Samples per Second: 7.98

Per-label Accuracy:
  정치: 0.8900 (89/100)
  경제: 0.8700 (87/100)
  사회: 0.8400 (84/100)
  생활문화: 0.8200 (82/100)
  세계: 0.8800 (88/100)
  IT과학: 0.8500 (85/100)
  스포츠: 0.8600 (86/100)
```

## Model Configuration

The benchmark uses the following default configuration:
- **Model**: `gemini-2.0-flash-exp` (Gemini 2.5 Flash)
- **Platform**: Google Cloud Vertex AI
- **Temperature**: 0.0 (deterministic outputs)
- **Max Tokens**: 1024
- **Top-p**: 1.0
- **Top-k**: 1

## Dataset Information

The benchmark uses the KLUE TC test set, which contains:
- **Total samples**: ~1,000 test samples
- **Format**: Korean news articles with title and text
- **Labels**: 7 topic categories
- **Source**: [KLUE Benchmark](https://klue-benchmark.com/)

## Performance Considerations

- **Rate Limiting**: The script includes a 0.1-second delay between requests to avoid rate limiting
- **Memory Usage**: For large datasets, consider using `--max-samples` to limit memory usage
- **Vertex AI Quotas**: Monitor your Vertex AI usage and quotas in the Google Cloud Console
- **Costs**: Vertex AI charges per request; monitor your usage accordingly

## Troubleshooting

### Common Issues

1. **Authentication Error**: Make sure you have proper authentication set up
   ```bash
   # Check if authentication is working
   gcloud auth list
   gcloud config get-value project
   ```

2. **Project ID Error**: Ensure your project ID is correct and Vertex AI API is enabled
   ```bash
   # Enable Vertex AI API
   gcloud services enable aiplatform.googleapis.com
   ```

3. **Permission Error**: Make sure your service account has the necessary permissions
   ```bash
   # Grant Vertex AI User role
   gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
       --member="serviceAccount:YOUR_SERVICE_ACCOUNT" \
       --role="roles/aiplatform.user"
   ```

4. **Rate Limiting**: If you encounter rate limits, increase the delay in the script
5. **Memory Issues**: Use `--max-samples` to limit the dataset size
6. **Network Issues**: Ensure stable internet connection for API calls

### Error Messages

- `"Google Cloud project ID must be provided"`: Set your project ID via environment variable or command line
- `"Failed to initialize Vertex AI"`: Check your authentication and project setup
- `"Failed to initialize model"`: Check your project permissions and Vertex AI API status
- `"Failed to load dataset"`: Ensure you have internet access to download the KLUE dataset

### Debugging

To debug authentication issues:
```bash
# Test Vertex AI access
python -c "
from google.cloud import aiplatform
aiplatform.init(project='YOUR_PROJECT_ID')
print('Vertex AI initialized successfully')
"
```

## Cost Estimation

Vertex AI pricing for Gemini models:
- **Input tokens**: ~$0.0005 per 1K tokens
- **Output tokens**: ~$0.0015 per 1K tokens

For the full KLUE TC test set (~1,000 samples):
- Estimated cost: $5-15 USD (depending on text length)
- Use `--max-samples` for cost control during testing

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve this benchmark.

## License

This project is open source. Please check the individual library licenses for dependencies.

## References

- [KLUE Benchmark](https://klue-benchmark.com/)
- [Google Cloud Vertex AI](https://cloud.google.com/vertex-ai)
- [Vertex AI Generative AI](https://cloud.google.com/vertex-ai/docs/generative-ai)
- [Hugging Face Datasets](https://huggingface.co/datasets) 