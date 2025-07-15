# KLUE Machine Reading Comprehension Benchmark with Gemini 2.5 Flash

Benchmark script for evaluating Gemini 2.5 Flash on the Korean Language Understanding Evaluation (KLUE) Machine Reading Comprehension task using Google Cloud Vertex AI.

## Quickstart

This guide assumes the repository has been cloned and the user has navigated into its directory:

```bash
git clone https://github.com/aimldl/llm_benchmarks_asian_langs.git
cd llm_benchmarks_asian_langs
```

### Setting Up the Virtual Environment
Two options are available for setting up the virtual environment:

#### Anaconda
Activate the pre-existing `klue` environment. If it doesn't exist, create it first.

```bash
(base) $ conda activate klue
(klue) $
```

#### `.venv`
For users preferring .venv, create and activate a new virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

### Running the Benchmarks
Open `run_klue_mrc.ipynb` to execute its cells, or run the following commands in a terminal.

```bash
cd klue_mrc
./setup.sh full          # Installs all necessary dependencies
./run test               # Runs 10 samples for a quick test
./run custom 50          # Runs 50 samples
./run full               # Runs the full benchmark (consider using `tmux` for background execution)
```

## Overview

The KLUE Machine Reading Comprehension (MRC) task evaluates the ability to answer questions based on Korean text passages, including:
- **Answerable Questions**: Questions with answers in the context
- **Unanswerable Questions**: Questions marked as "impossible" 
- **Multiple Answer Formats**: Various valid answer formulations

## Features

- **Comprehensive Evaluation**: Exact match, F1 score, and impossible question accuracy
- **Detailed Analysis**: Error analysis and per-question type breakdown
- **Flexible Configuration**: Customizable model parameters and sampling
- **Result Export**: JSON and CSV outputs for analysis
- **Progress Tracking**: Real-time progress and comprehensive logging
- **Vertex AI Integration**: Google Cloud Vertex AI for model inference

## Prerequisites

1. **Google Cloud Project** with Vertex AI API enabled
2. **Authentication** via Service Account Key, Application Default Credentials, or gcloud CLI

## Quick Setup

```bash
# Complete setup (install dependencies + test)
./setup.sh full

# Or step by step:
./setup.sh install    # Install dependencies only
./setup.sh test       # Test the setup
```

## Google Cloud Setup

1. **Create/Select Project**:
   ```bash
   gcloud projects create YOUR_PROJECT_ID
   gcloud services enable aiplatform.googleapis.com
   ```

2. **Set up Authentication** (choose one):

   **Service Account (Recommended)**:
   ```bash
   gcloud iam service-accounts create klue-benchmark --display-name="KLUE Benchmark"
   gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
       --member="serviceAccount:klue-benchmark@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
       --role="roles/aiplatform.user"
   gcloud iam service-accounts keys create key.json \
       --iam-account=klue-benchmark@YOUR_PROJECT_ID.iam.gserviceaccount.com
   export GOOGLE_APPLICATION_CREDENTIALS="path/to/key.json"
   ```

   **Application Default Credentials**:
   ```bash
   gcloud auth application-default login
   ```

3. **Set Project ID**:
   ```bash
   export GOOGLE_CLOUD_PROJECT="YOUR_PROJECT_ID"
   # For permanent setup, add to .bashrc:
   echo "export GOOGLE_CLOUD_PROJECT=\"$(gcloud config get-value project)\"" >> ~/.bashrc
   source ~/.bashrc
   ```

## Usage

### Quick Start with Run Script

```bash
# Test with 10 samples
./run test

# Run with custom number of samples
./run custom 50

# Run full benchmark
./run full

# Show help
./run help
```

### Direct Python Usage

```bash
# Basic usage
python klue_mrc-gemini2_5flash.py --project-id "your-project-id"

# Test with limited samples
python klue_mrc-gemini2_5flash.py --project-id "your-project-id" --max-samples 100

# Custom output directory
python klue_mrc-gemini2_5flash.py --project-id "your-project-id" --output-dir "my_results"

# Adjust model parameters
python klue_mrc-gemini2_5flash.py --project-id "your-project-id" --temperature 0.1 --max-tokens 2048
```

### Command Line Arguments

- `--project-id`: Google Cloud project ID (required if not set as environment variable)
- `--location`: Vertex AI location (default: "us-central1")
- `--max-samples`: Maximum number of samples to test (default: all test samples)
- `--output-dir`: Output directory for results (default: "benchmark_results")
- `--temperature`: Model temperature (default: 0.1)
- `--max-tokens`: Maximum output tokens (default: 2048)
- `--no-save-predictions`: Skip saving detailed prediction results

## Output Files

Generated in the `benchmark_results` directory:

### Metrics File (`klue_mrc_metrics_[timestamp].json`)
Overall performance metrics:
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

### Additional Files
- **Detailed Results** (`klue_mrc_results_[timestamp].json`): Per-sample results with predictions
- **CSV Results** (`klue_mrc_results_[timestamp].csv`): Tabular format for analysis
- **Error Analysis** (`klue_mrc_error_analysis_[timestamp].txt`): Failed predictions analysis

## Logging

Automatic logging to `logs/` directory:
- **Full Logs** (`klue_mrc_[mode]_[timestamp].log`): Complete execution logs
- **Error Logs** (`klue_mrc_[mode]_[timestamp].err`): Error analysis and debugging

## Performance Metrics

### Primary Metrics
- **Exact Match**: Percentage of predictions exactly matching ground truth
- **F1 Score**: Harmonic mean of precision and recall
- **Impossible Accuracy**: Accuracy on unanswerable questions

### Secondary Metrics
- **Processing Time**: Total and average time per sample
- **Throughput**: Samples processed per second
- **Per-Type Analysis**: Separate metrics for answerable vs impossible questions

## Dataset Information

KLUE MRC dataset:
- **Training Set**: ~18,000 samples
- **Validation Set**: ~2,000 samples
- **Test Set**: ~2,000 samples (not used in benchmark)

Each sample includes:
- **Title**: Article title
- **Context**: Text passage to read
- **Question**: Question to answer
- **Answers**: List of valid answer formulations
- **Is Impossible**: Boolean indicating if question is unanswerable

## Troubleshooting

### Common Issues

1. **Authentication Errors**:
   ```bash
   gcloud auth application-default login
   export GOOGLE_CLOUD_PROJECT="your-project-id"
   ```

2. **API Quota Exceeded**: Check Vertex AI quota in Google Cloud Console

3. **Memory Issues**: Reduce `--max-tokens` parameter

4. **Network Issues**: Verify Vertex AI API accessibility

### Getting Help

1. **Check Logs**: Review files in `logs/` directory
2. **Test Setup**: Run `python test_setup.py`
3. **Error Analysis**: Use `./get_errors.sh` to extract error details

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- KLUE dataset creators and maintainers
- Google Cloud Vertex AI team
- Hugging Face datasets library

## Related Work

- [KLUE Paper](https://arxiv.org/abs/2105.09680)
- [KLUE GitHub Repository](https://github.com/KLUE-benchmark/KLUE)
- [Google Cloud Vertex AI Documentation](https://cloud.google.com/vertex-ai) 