# KLUE Relation Extraction Benchmark with Gemini 2.5 Flash

Benchmark script for evaluating Gemini 2.5 Flash on the Korean Language Understanding Evaluation (KLUE) Relation Extraction task using Google Cloud Vertex AI.

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
Open `run_klue_re.ipynb` to execute its cells, or run the following commands in a terminal.

```bash
cd klue_re
./setup.sh full          # Installs all necessary dependencies
./run test               # Runs 10 samples for a quick test
./run custom 50          # Runs 50 samples
./run full               # Runs the full benchmark (consider using `tmux` for background execution)
```

## Overview

The KLUE Relation Extraction (RE) task identifies relationships between two entities in Korean text. Given a sentence with marked subject and object entities, the model determines their relationship from predefined relation types.

### Relation Categories

**Organization Relations (org:)**:
- `org:top_members/employees`, `org:members`, `org:product`, `org:founded`
- `org:alternate_names`, `org:place_of_headquarters`, `org:website`
- `org:subsidiaries`, `org:parents`, `org:dissolved`

**Person Relations (per:)**:
- `per:title`, `per:employee_of`, `per:member_of`, `per:schools_attended`
- `per:works_for`, `per:countries_of_residence`, `per:cities_of_residence`
- `per:date_of_birth`, `per:place_of_birth`, `per:spouse`, `per:children`
- `per:parents`, `per:siblings`, `per:charges`, `per:age`

**Other**: `no_relation` - No relationship between entities

## Features

- **Comprehensive Evaluation**: Accuracy and per-relation type performance
- **Detailed Analysis**: Error analysis and per-relation breakdown
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
python klue_re-gemini2_5flash.py --project-id "your-project-id"

# Test with limited samples
python klue_re-gemini2_5flash.py --project-id "your-project-id" --max-samples 100

# Custom output directory
python klue_re-gemini2_5flash.py --project-id "your-project-id" --output-dir "my_results"

# Adjust model parameters
python klue_re-gemini2_5flash.py --project-id "your-project-id" --temperature 0.1 --max-tokens 2048
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

### Metrics File (`klue_re_metrics_[timestamp].json`)
Overall performance metrics:
```json
{
  "total_samples": 1000,
  "accuracy": 0.72,
  "per_relation_accuracy": {
    "org:top_members/employees": 0.78,
    "per:title": 0.75,
    "per:employee_of": 0.68,
    "no_relation": 0.85
  },
  "total_time": 1800.5,
  "average_time_per_sample": 1.8,
  "samples_per_second": 0.56
}
```

### Additional Files
- **Detailed Results** (`klue_re_results_[timestamp].json`): Per-sample results with predictions
- **CSV Results** (`klue_re_results_[timestamp].csv`): Tabular format for analysis
- **Error Analysis** (`klue_re_error_analysis_[timestamp].txt`): Relation extraction errors analysis

## Logging

Automatic logging to `logs/` directory:
- **Full Logs** (`klue_re_[mode]_[timestamp].log`): Complete execution logs
- **Error Logs** (`klue_re_[mode]_[timestamp].err`): Error analysis and debugging

## Performance Metrics

### Primary Metrics
- **Overall Accuracy**: Percentage of correctly classified relations
- **Per-Relation Accuracy**: Accuracy breakdown by relation type
- **Processing Time**: Total and average time per sample
- **Throughput**: Samples processed per second

### Error Analysis
- **Relation Extraction Errors**: Detailed analysis of misclassified relations
- **Entity Pair Confusion**: Shows which relation types are commonly confused
- **Error Patterns**: Identifies systematic extraction issues

## Dataset Information

KLUE RE dataset:
- **Training Set**: ~32,000 samples
- **Validation Set**: ~4,000 samples
- **Test Set**: ~4,000 samples (not used in benchmark)

Each sample includes:
- **Sentence**: Korean text containing entities
- **Subject Entity**: First entity with type and position
- **Object Entity**: Second entity with type and position
- **Relation**: Target relation type between entities

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