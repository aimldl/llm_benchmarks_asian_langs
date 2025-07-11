# KLUE Dependency Parsing Benchmark with Gemini 2.5 Flash on Vertex AI

This repository contains a benchmark script for evaluating Gemini 2.5 Flash on the Korean Language Understanding Evaluation (KLUE) Dependency Parsing task using Google Cloud Vertex AI.

## Summary
In essence, run the following commands:
```bash
$ git clone https://github.com/aimldl/llm_benchmarks_asian_langs.git
$ cd klue_dp
$ ./setup.sh full
$ ./run test
```

## Overview

The KLUE Dependency Parsing (DP) task involves analyzing Korean sentences to identify the grammatical relationships between words. The task requires:
- **Part-of-Speech (POS) Tagging**: Identifying the grammatical category of each word
- **Dependency Parsing**: Determining which word each word depends on (head) and the type of dependency relationship

The benchmark evaluates two key metrics:
- **UAS (Unlabeled Attachment Score)**: Percentage of words with correctly identified heads
- **LAS (Labeled Attachment Score)**: Percentage of words with correctly identified heads and dependency labels

## Features

- **Comprehensive Benchmarking**: Evaluates UAS, LAS, and per-POS performance
- **Detailed Analysis**: Provides error analysis and per-POS accuracy breakdown
- **Flexible Configuration**: Supports various model parameters and sampling options
- **Result Export**: Saves results in JSON and CSV formats for further analysis
- **Progress Tracking**: Real-time progress bar and logging
- **Vertex AI Integration**: Uses Google Cloud Vertex AI for model inference
- **Professional Logging**: Automatic log file generation with error extraction

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
python klue_dp-gemini2_5flash.py --project-id "your-project-id"
```

### Advanced Usage

```bash
# Test with limited samples (useful for quick testing)
python klue_dp-gemini2_5flash.py --project-id "your-project-id" --max-samples 100

# Custom output directory
python klue_dp-gemini2_5flash.py --project-id "your-project-id" --output-dir "my_results"

# Use different Vertex AI location
python klue_dp-gemini2_5flash.py --project-id "your-project-id" --location "us-west1"

# Adjust model parameters
python klue_dp-gemini2_5flash.py --project-id "your-project-id" --temperature 0.1 --max-tokens 4096

# Skip saving detailed predictions (saves disk space)
python klue_dp-gemini2_5flash.py --project-id "your-project-id" --no-save-predictions
```

### Command Line Arguments

- `--project-id`: Google Cloud project ID (required if not set as environment variable)
- `--location`: Vertex AI location (default: "us-central1")
- `--max-samples`: Maximum number of samples to test (default: all validation samples)
- `--output-dir`: Output directory for results (default: "benchmark_results")
- `--temperature`: Model temperature (default: 0.1)
- `--max-tokens`: Maximum output tokens (default: 4096)
- `--no-save-predictions`: Skip saving detailed prediction results

### Quick Start with Run Script

For convenience, a simple Bash script `run` is provided to quickly execute common benchmark scenarios:

```bash
# Make the script executable (if needed)
chmod +x run

# Run a small test (10 samples)
./run test

# Run the full benchmark
./run full

# Run with custom number of samples
./run custom 50
```

The run script automatically:
- Creates log files in the `logs/` directory
- Extracts error information to separate `.err` files
- Provides progress updates during execution

## Output and Results

### Result Files

The benchmark generates several output files:

1. **Metrics JSON**: `klue_dp_metrics_[timestamp].json`
   - Overall performance metrics (UAS, LAS, timing)
   - Per-POS performance breakdown

2. **Detailed Results JSON**: `klue_dp_results_[timestamp].json`
   - Complete prediction results for each sample
   - Raw model responses and error information

3. **CSV Results**: `klue_dp_results_[timestamp].csv`
   - Tabular format for easy analysis
   - Includes sentence, words, POS tags, predictions, and metrics

4. **Error Analysis**: `klue_dp_error_analysis_[timestamp].txt`
   - Detailed analysis of failed predictions
   - Sample sentences and error patterns

### Log Files

All benchmark runs are automatically logged:

- **Full Logs**: `logs/klue_dp_[mode]_[samples]samples_[timestamp].log`
  - Complete execution output
  - Command headers for easy identification

- **Error Logs**: `logs/klue_dp_[mode]_[samples]samples_[timestamp].err`
  - Extracted error information only
  - Focused debugging information

### Performance Metrics

The benchmark reports:

- **UAS (Unlabeled Attachment Score)**: Percentage of words with correct head assignment
- **LAS (Labeled Attachment Score)**: Percentage of words with correct head and dependency label
- **Per-POS Performance**: Accuracy breakdown by part-of-speech category
- **Timing Information**: Total time, average time per sample, samples per second

## Korean POS Tags

The benchmark handles 35 Korean POS tags including:

- **Nouns**: NNG (일반명사), NNP (고유명사), NNB (의존명사), etc.
- **Verbs**: VV (동사), VA (형용사), VX (보조용언), etc.
- **Particles**: JKS (주격조사), JKO (목적격조사), JKB (부사격조사), etc.
- **Endings**: EF (종결어미), EC (연결어미), ETN (명사형전성어미), etc.
- **Others**: MM (관형사), MAG (일반부사), IC (감탄사), etc.

## Dependency Relations

The benchmark evaluates various dependency relations including:

- **Core Arguments**: nsubj (주어), obj (목적어), iobj (간접목적어)
- **Modifiers**: amod (형용사 수식어), nummod (수사 수식어), advmod (부사 수식어)
- **Function Words**: case (격조사), mark (접속조사), aux (보조동사)
- **Special Relations**: root (루트), punct (구두점), compound (복합어)

## Troubleshooting

### Common Issues

1. **Authentication Errors**:
   ```bash
   # Verify credentials
   gcloud auth list
   gcloud config get-value project
   ```

2. **API Not Enabled**:
   ```bash
   # Enable Vertex AI API
   gcloud services enable aiplatform.googleapis.com
   ```

3. **Memory Issues**:
   - Reduce `--max-samples` for testing
   - Use `--no-save-predictions` to save disk space

4. **Rate Limiting**:
   - The script includes built-in delays between API calls
   - Adjust `sleep_interval_between_api_calls` in the code if needed

### Error Analysis

Use the provided error analysis tools:

```bash
# Extract errors from results
./get_errors.sh benchmark_results/klue_dp_results_[timestamp].csv

# Test logging functionality
./test_logging.sh
```

## Scripts Overview

- `klue_dp-gemini2_5flash.py`: Main benchmark script
- `run`: Quick benchmark runner with logging
- `setup.sh`: Complete setup process
- `install_dependencies.sh`: Install Python dependencies
- `test_setup.py`: Verify setup and connectivity
- `get_errors.sh`: Extract error details from results
- `test_logging.sh`: Test logging functionality
- `verify_scripts.sh`: Verify all scripts are executable

## Performance Expectations

Typical performance ranges for Gemini 2.5 Flash on KLUE DP:

- **UAS**: 75-85% (depending on sentence complexity)
- **LAS**: 70-80% (depending on dependency label accuracy)
- **Processing Speed**: 0.5-2 seconds per sentence
- **Best Performance**: On simple sentences with clear dependency structures
- **Challenging Cases**: Complex sentences with multiple clauses, ambiguous dependencies

## Contributing

To contribute to this benchmark:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly with different sample sizes
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- KLUE dataset creators for providing the Korean language understanding benchmark
- Google Cloud for providing Vertex AI infrastructure
- The Korean NLP community for ongoing research and development 