# KLUE Machine Reading Comprehension Benchmark with Gemini 2.5 Flash on Vertex AI

This repository contains a benchmark script for evaluating Gemini 2.5 Flash on the Korean Language Understanding Evaluation (KLUE) Machine Reading Comprehension task using Google Cloud Vertex AI.

## Summary
In essence, run the following commands:
```bash
$ git clone https://github.com/aimldl/llm_benchmarks_asian_langs.git
$ cd klue_mrc
$ ./setup.sh full
$ ./run test
```


## Overview

The KLUE Machine Reading Comprehension (MRC) task involves answering questions based on given Korean text passages. The task includes:
- **Answerable Questions**: Questions that can be answered from the given context
- **Unanswerable Questions**: Questions that cannot be answered from the context (marked as "impossible")
- **Multiple Answer Formats**: Questions may have multiple valid answer formulations

## Features

- **Comprehensive Benchmarking**: Evaluates exact match, F1 score, and impossible question accuracy
- **Detailed Analysis**: Provides error analysis and per-question type performance breakdown
- **Flexible Configuration**: Supports various model parameters and sampling options
- **Result Export**: Saves results in JSON and CSV formats for further analysis
- **Progress Tracking**: Real-time progress bar and logging
- **Vertex AI Integration**: Uses Google Cloud Vertex AI for model inference
- **Impossible Question Handling**: Special handling for questions that cannot be answered from context

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
python klue_mrc-gemini2_5flash.py --project-id "your-project-id"
```

### Advanced Usage

```bash
# Test with limited samples (useful for quick testing)
python klue_mrc-gemini2_5flash.py --project-id "your-project-id" --max-samples 100

# Custom output directory
python klue_mrc-gemini2_5flash.py --project-id "your-project-id" --output-dir "my_results"

# Use different Vertex AI location
python klue_mrc-gemini2_5flash.py --project-id "your-project-id" --location "us-west1"

# Adjust model parameters
python klue_mrc-gemini2_5flash.py --project-id "your-project-id" --temperature 0.1 --max-tokens 2048

# Skip saving detailed predictions (saves disk space)
python klue_mrc-gemini2_5flash.py --project-id "your-project-id" --no-save-predictions
```

### Command Line Arguments

- `--project-id`: Google Cloud project ID (required if not set as environment variable)
- `--location`: Vertex AI location (default: "us-central1")
- `--max-samples`: Maximum number of samples to test (default: all test samples)
- `--output-dir`: Output directory for results (default: "benchmark_results")
- `--temperature`: Model temperature (default: 0.1)
- `--max-tokens`: Maximum output tokens (default: 2048)
- `--no-save-predictions`: Skip saving detailed prediction results

### Quick Start with Run Script

For convenience, a simple Bash script `run` is provided to quickly execute common benchmark scenarios:

```bash
# Make the script executable (if needed)
chmod +x run
```

### Run Script Usage

```bash
# Test with 10 samples (quick test)
./run test

# Run full benchmark (all samples)
./run full

# Run with custom number of samples
./run custom 50

# Show help
./run help
```

## Output Files

The benchmark generates several output files in the `benchmark_results` directory:

### Metrics File (`klue_mrc_metrics_[timestamp].json`)
Contains overall performance metrics:
```json
{
  "total_samples": 1000,
  "answerable_samples": 800,
  "impossible_samples": 200,
  "exact_match": 0.75,
  "f1_score": 0.82,
  "impossible_accuracy": 0.90,
  "total_time": 1200.5,
  "average_time_per_sample": 1.2,
  "samples_per_second": 0.83
}
```

### Detailed Results (`klue_mrc_results_[timestamp].json`)
Contains detailed results for each sample including:
- Sample ID and metadata
- Question and context
- Ground truth answers
- Model predictions
- Performance metrics per sample
- Error information (if any)

### CSV Results (`klue_mrc_results_[timestamp].csv`)
Tabular format for easy analysis in spreadsheet applications.

### Error Analysis (`klue_mrc_error_analysis_[timestamp].txt`)
Detailed analysis of failed predictions and errors.

## Logging

All benchmark runs are automatically logged to the `logs/` directory:

- **Full Logs** (`klue_mrc_[mode]_[timestamp].log`): Complete execution logs
- **Error Logs** (`klue_mrc_[mode]_[timestamp].err`): Error analysis and debugging information

Log files include:
- Command headers for easy identification
- Timestamps and working directory information
- Complete audit trail of benchmark execution
- Error analysis and debugging information

## Performance Metrics

The benchmark evaluates the following metrics:

### Primary Metrics
- **Exact Match**: Percentage of predictions that exactly match any ground truth answer
- **F1 Score**: Harmonic mean of precision and recall for answer prediction
- **Impossible Accuracy**: Accuracy on questions marked as "impossible"

### Secondary Metrics
- **Processing Time**: Total time and average time per sample
- **Throughput**: Samples processed per second
- **Per-Type Analysis**: Separate metrics for answerable vs impossible questions

## Dataset Information

The KLUE MRC dataset contains:
- **Training Set**: ~18,000 samples
- **Validation Set**: ~2,000 samples
- **Test Set**: ~2,000 samples (not used in this benchmark)

Each sample includes:
- **Title**: Article title
- **Context**: Text passage to read
- **Question**: Question to answer
- **Answers**: List of valid answer formulations
- **Is Impossible**: Boolean indicating if the question is unanswerable

## Troubleshooting

### Common Issues

1. **Authentication Errors**:
   ```bash
   # Ensure credentials are set up correctly
   gcloud auth application-default login
   export GOOGLE_CLOUD_PROJECT="your-project-id"
   ```

2. **API Quota Exceeded**:
   - Check your Vertex AI quota in Google Cloud Console
   - Consider using a smaller sample size for testing

3. **Memory Issues**:
   - Reduce `--max-tokens` parameter
   - Use smaller batch sizes

4. **Network Issues**:
   - Check your internet connection
   - Verify Vertex AI API is accessible from your location

### Getting Help

1. **Check Logs**: Review the log files in the `logs/` directory
2. **Test Setup**: Run `python test_setup.py` to verify your environment
3. **Error Analysis**: Use `./get_errors.sh` to extract error details from results

## Contributing

To contribute to this benchmark:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- KLUE dataset creators and maintainers
- Google Cloud Vertex AI team
- Hugging Face datasets library

## Related Work

- [KLUE Paper](https://arxiv.org/abs/2105.09680)
- [KLUE GitHub Repository](https://github.com/KLUE-benchmark/KLUE)
- [Google Cloud Vertex AI Documentation](https://cloud.google.com/vertex-ai) 