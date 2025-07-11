# KLUE DST (Dialogue State Tracking) Benchmark

This directory contains the implementation for benchmarking Gemini 2.5 Flash on the Korean Language Understanding Evaluation (KLUE) Dialogue State Tracking task using Google Cloud Vertex AI.

## Overview

Dialogue State Tracking (DST) is a crucial component of task-oriented dialogue systems. The task involves tracking the user's intent and the values of slots (parameters) throughout a conversation. This benchmark evaluates the model's ability to:

- Identify the user's current intent (e.g., request, inform, book, confirm)
- Extract requested slots (information the user is asking for)
- Track slot values (specific information provided by the user)

## Task Description

The KLUE DST task involves analyzing multi-turn Korean dialogues and predicting:
1. **Active Intent**: The user's current intention (e.g., request, inform, book, confirm, deny, affirm, search, recommend)
2. **Requested Slots**: Slots that the user is requesting information about (e.g., location, time, price, cuisine)
3. **Slot Values**: The actual values provided for each slot (e.g., "서울" for location, "한식" for cuisine)

## Directory Structure

```
klue_dst/
├── klue_dst-gemini2_5flash.py    # Main benchmark script
├── run                           # Benchmark runner script
├── setup.sh                      # Environment setup script
├── install_dependencies.sh       # Dependency installation script
├── test_setup.py                 # Environment testing script
├── get_errors.sh                 # Error analysis script
├── test_logging.sh               # Logging test script
├── verify_scripts.sh             # Script verification script
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── ABOUT_KLUE_DST.md            # Detailed task description
├── TROUBLESHOOTING.md            # Troubleshooting guide
├── VERTEX_AI_SETUP.md            # Vertex AI setup guide
├── logs/                         # Log files directory
├── benchmark_results/            # Benchmark results directory
├── result_analysis/              # Error analysis results
└── eval_dataset/                 # Evaluation dataset directory
```

## Quick Start

### 1. Prerequisites

- Python 3.8 or higher
- Google Cloud account with Vertex AI enabled
- Google Cloud SDK installed and configured
- Sufficient Vertex AI quota for Gemini 2.5 Flash

### 2. Setup

```bash
# Clone the repository and navigate to klue_dst directory
cd klue_dst

# Run the setup script
./setup.sh

# Or install dependencies manually
./install_dependencies.sh
```

### 3. Configure Google Cloud

```bash
# Set your Google Cloud project ID
export GOOGLE_CLOUD_PROJECT='your-project-id'

# Authenticate with Google Cloud
gcloud auth login
gcloud config set project $GOOGLE_CLOUD_PROJECT
```

### 4. Test the Setup

```bash
# Test the environment
python3 test_setup.py

# Test logging functionality
./test_logging.sh test
```

### 5. Run Benchmarks

```bash
# Run a small test (10 samples)
./run test

# Run with custom number of samples
./run custom 50

# Run the full benchmark (all validation samples)
./run full
```

## Usage

### Running Benchmarks

The `run` script provides three modes:

1. **Test Mode** (`./run test`): Runs with 10 samples for quick testing
2. **Custom Mode** (`./run custom N`): Runs with N samples
3. **Full Mode** (`./run full`): Runs with all validation samples

### Logging

All benchmark runs create detailed log files in the `logs/` directory:

- **Full logs**: `klue_dst_[mode]_[samples]samples_[timestamp].log`
- **Error logs**: `klue_dst_[mode]_[samples]samples_[timestamp].err`

The error logs contain:
- Error analysis with sample details
- ERROR level log messages
- Performance metrics for failed samples

### Error Analysis

Use the error analysis script to examine benchmark results:

```bash
# Analyze the latest log file
./get_errors.sh latest

# Analyze a specific log file
./get_errors.sh analyze logs/klue_dst_test_20241201_120000.log

# Analyze all log files
./get_errors.sh all
```

## Evaluation Metrics

The benchmark evaluates performance using multiple metrics:

### Primary Metrics
- **Intent Accuracy**: Percentage of correctly predicted intents
- **Requested Slots F1**: F1 score for requested slot prediction
- **Slot Values F1**: F1 score for slot value prediction
- **Overall F1**: Average of all F1 scores

### Secondary Metrics
- **Per-domain Performance**: F1 scores broken down by dialogue domain
- **Processing Time**: Total time and average time per sample
- **Success Rate**: Percentage of successful API calls

## Dataset

The benchmark uses the KLUE DST dataset from Hugging Face:
- **Dataset**: `klue` dataset with `dst` configuration
- **Split**: Validation set
- **Format**: Multi-turn dialogues with intent and slot annotations
- **Domains**: Restaurant, hotel, movie, music, etc.

## Model Configuration

The benchmark uses Gemini 2.5 Flash with the following settings:
- **Model**: `gemini-2.5-flash`
- **Temperature**: 0.1 (for consistent outputs)
- **Max Tokens**: 2048 (increased for DST task complexity)
- **Top-p**: 1.0
- **Top-k**: 1

## Prompt Engineering

The DST prompt includes:
- Detailed role definition as a dialogue state tracking expert
- Comprehensive explanation of DST components (intent, slots, values)
- Common intent and slot type definitions
- Step-by-step guidelines for analysis
- Structured output format requirements

## Results

Results are saved in multiple formats:

### JSON Files
- **Metrics**: `klue_dst_metrics_[timestamp].json`
- **Detailed Results**: `klue_dst_results_[timestamp].json`

### CSV Files
- **Results Table**: `klue_dst_results_[timestamp].csv`

### Error Analysis
- **Error Report**: `klue_dst_error_analysis_[timestamp].txt`

## Troubleshooting

### Common Issues

1. **Authentication Errors**
   ```bash
   # Ensure you're authenticated
   gcloud auth login
   gcloud auth application-default login
   ```

2. **Project ID Issues**
   ```bash
   # Set the correct project ID
   export GOOGLE_CLOUD_PROJECT='your-project-id'
   ```

3. **Quota Exceeded**
   - Check your Vertex AI quota in Google Cloud Console
   - Consider using a smaller sample size for testing

4. **Import Errors**
   ```bash
   # Reinstall dependencies
   ./install_dependencies.sh
   ```

### Getting Help

- Check the `TROUBLESHOOTING.md` file for detailed solutions
- Review log files in the `logs/` directory
- Use `./get_errors.sh latest` to analyze recent errors
- Run `python3 test_setup.py` to verify your environment

## Performance Expectations

Typical performance ranges for Gemini 2.5 Flash on KLUE DST:
- **Intent Accuracy**: 70-85%
- **Requested Slots F1**: 60-75%
- **Slot Values F1**: 55-70%
- **Overall F1**: 60-75%

Performance may vary based on:
- Dialogue complexity
- Domain diversity
- Slot value specificity
- Context length

## Contributing

To contribute to this benchmark:

1. Follow the existing code structure
2. Maintain consistent logging and error handling
3. Update documentation for any changes
4. Test thoroughly before submitting

## License

This project is licensed under the same terms as the parent repository.

## References

- [KLUE Dataset Paper](https://arxiv.org/abs/2105.09680)
- [Dialogue State Tracking Survey](https://arxiv.org/abs/2002.01389)
- [Google Cloud Vertex AI Documentation](https://cloud.google.com/vertex-ai)
- [Gemini 2.5 Flash Documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/2-5-flash) 