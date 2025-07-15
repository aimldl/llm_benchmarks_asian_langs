# KLUE Dialogue State Tracking Benchmark with Gemini 2.5 Flash

Benchmark script for evaluating Gemini 2.5 Flash on the Korean Language Understanding Evaluation (KLUE) Dialogue State Tracking task using Google Cloud Vertex AI.

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
Open `run_klue_dst.ipynb` to execute its cells, or run the following commands in a terminal.

```bash
cd klue_dst
./setup.sh full          # Installs all necessary dependencies
./run test               # Runs 10 samples for a quick test
./run custom 50          # Runs 50 samples
./run full               # Runs the full benchmark (consider using `tmux` for background execution)
```

## Overview

The KLUE Dialogue State Tracking (DST) task analyzes multi-turn Korean dialogues to predict:

- **Active Intent**: User's current intention (request, inform, book, confirm, etc.)
- **Requested Slots**: Information the user is asking for (location, time, price, etc.)
- **Slot Values**: Specific information provided by the user (e.g., "서울" for location)

This is crucial for task-oriented dialogue systems like virtual assistants and chatbots.

## Features

- **Comprehensive Evaluation**: Intent accuracy, slot F1 scores, and overall performance
- **Detailed Analysis**: Error analysis and per-domain breakdown
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
python klue_dst-gemini2_5flash.py --project-id "your-project-id"

# Test with limited samples
python klue_dst-gemini2_5flash.py --project-id "your-project-id" --max-samples 100

# Custom output directory
python klue_dst-gemini2_5flash.py --project-id "your-project-id" --output-dir "my_results"

# Adjust model parameters
python klue_dst-gemini2_5flash.py --project-id "your-project-id" --temperature 0.1 --max-tokens 2048
```

### Command Line Arguments

- `--project-id`: Google Cloud project ID (required if not set as environment variable)
- `--location`: Vertex AI location (default: "us-central1")
- `--max-samples`: Maximum number of samples to test (default: all validation samples)
- `--output-dir`: Output directory for results (default: "benchmark_results")
- `--temperature`: Model temperature (default: 0.1)
- `--max-tokens`: Maximum output tokens (default: 2048)
- `--no-save-predictions`: Skip saving detailed prediction results

## Output Files

Generated in the `benchmark_results` directory:

### Metrics File (`klue_dst_metrics_[timestamp].json`)
Overall performance metrics:
```json
{
  "total_samples": 1000,
  "intent_accuracy": 0.78,
  "requested_slots_f1": 0.72,
  "slot_values_f1": 0.68,
  "overall_f1": 0.70,
  "per_domain_f1": {
    "restaurant": 0.75,
    "hotel": 0.68,
    "movie": 0.72
  },
  "total_time": 2400.5,
  "average_time_per_sample": 2.4,
  "samples_per_second": 0.42
}
```

### Additional Files
- **Detailed Results** (`klue_dst_results_[timestamp].json`): Per-sample results with predictions
- **CSV Results** (`klue_dst_results_[timestamp].csv`): Tabular format for analysis
- **Error Analysis** (`klue_dst_error_analysis_[timestamp].txt`): DST errors analysis

## Logging

Automatic logging to `logs/` directory:
- **Full Logs** (`klue_dst_[mode]_[timestamp].log`): Complete execution logs
- **Error Logs** (`klue_dst_[mode]_[timestamp].err`): Error analysis and debugging

## Performance Metrics

### Primary Metrics
- **Intent Accuracy**: Percentage of correctly predicted intents
- **Requested Slots F1**: F1 score for requested slot prediction
- **Slot Values F1**: F1 score for slot value prediction
- **Overall F1**: Average of all F1 scores

### Secondary Metrics
- **Per-Domain Performance**: F1 scores broken down by dialogue domain
- **Processing Time**: Total and average time per sample
- **Success Rate**: Percentage of successful API calls

## Dataset Information

KLUE DST dataset:
- **Training Set**: ~8,000 samples
- **Validation Set**: ~1,000 samples
- **Test Set**: ~1,000 samples (not used in benchmark)

Each sample includes:
- **Dialogue Context**: Multi-turn conversation history
- **Active Intent**: User's current intention
- **Requested Slots**: Information being requested
- **Slot Values**: Specific values provided by user
- **Domains**: Restaurant, hotel, movie, music, etc.

## Available Scripts

### Core Scripts
- **`./run`** - Quick benchmark runner
- **`./setup.sh`** - Complete setup process
- **`./install_dependencies.sh`** - Install Python dependencies
- **`./get_errors.sh`** - Extract error details from results

### Usage Examples
```bash
# Complete setup and run
./setup.sh full
./run test

# Analyze errors from results
./get_errors.sh
```

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