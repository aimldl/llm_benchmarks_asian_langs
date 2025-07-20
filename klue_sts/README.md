# KLUE Sentence Textual Similarity (STS) Benchmark

This directory contains the benchmark implementation for the KLUE Sentence Textual Similarity (STS) task using Gemini 2.5 Flash on Google Cloud Vertex AI.

## Quickstart

### 1. Setup Environment

```bash
# Install dependencies and verify setup
./setup.sh full

# Or install dependencies only
./setup.sh install

# Or run tests only
./setup.sh test
```

### 2. Run Benchmarks

```bash
# Test run with 10 samples
./run test

# Custom run with 50 samples
./run custom 50

# Full benchmark (entire dataset)
./run full
```

### 3. Monitor Progress

For long-running benchmarks, use tmux to keep them running:

```bash
# Start a new tmux session
tmux new -s klue_sts

# Run the benchmark
./run full

# Detach from session (Ctrl+B, then D)
# Reattach later
tmux attach -t klue_sts
```

## Task Description

The KLUE STS task evaluates a model's ability to measure semantic similarity between pairs of Korean sentences. Each sentence pair is scored on a scale from 0 to 5:

- **0**: Completely different meaning
- **1**: Mostly different meaning  
- **2**: Partially different meaning
- **3**: Similar meaning
- **4**: Very similar meaning
- **5**: Completely identical meaning

## Evaluation Metrics

### Primary Metrics
- **Pearson Correlation**: Linear correlation between predicted and true scores
- **Spearman Correlation**: Rank correlation between predicted and true scores

### Secondary Metrics
- **Mean Squared Error (MSE)**: Average squared prediction error
- **Mean Absolute Error (MAE)**: Average absolute prediction error

## Files and Directories

```
klue_sts/
├── klue_sts-gemini2_5flash.py    # Main benchmark script
├── setup.sh                      # Setup script
├── run                           # Benchmark runner script
├── requirements.txt              # Python dependencies
├── test_setup.py                 # Setup verification script
├── ABOUT_KLUE_STS.md            # Task description and metrics
├── README.md                     # This file
├── run_klue_sts.ipynb           # Jupyter notebook for interactive use
├── eval_dataset/                 # Evaluation dataset
│   ├── klue-sts-v1.1_dev.json
│   └── klue-sts-v1.1_dev_extracted.csv
├── benchmark_results/            # Benchmark outputs (created after running)
├── logs/                         # Log files (created after running)
└── result_analysis/              # Analysis scripts and results
```

## Configuration

### Environment Variables

Set your Google Cloud project ID:

```bash
export GOOGLE_CLOUD_PROJECT="your-project-id"
```

### Command Line Options

The benchmark script supports various options:

```bash
python klue_sts-gemini2_5flash.py \
    --project-id "your-project-id" \
    --max-samples 100 \
    --temperature 0.1 \
    --max-tokens 1024 \
    --output-dir "benchmark_results" \
    --save-interval 50
```

## Running the Benchmarks

### Test Run
Quick test with 10 samples to verify everything works:

```bash
./run test
```

### Custom Run
Run with a specific number of samples:

```bash
./run custom 50    # 50 samples
./run custom 100   # 100 samples
```

### Full Benchmark
Run on the entire dataset (may take several hours):

```bash
./run full
```

**Note**: For full benchmarks, we recommend using tmux to ensure the process continues even if your terminal disconnects.

## Output Files

After running a benchmark, you'll find:

### Metrics Files
- `klue_sts_metrics_YYYYMMDD_HHMMSS.json`: Final metrics
- `klue_sts_metrics_NNNNNN_YYYYMMDD_HHMMSS.json`: Intermediate metrics

### Results Files
- `klue_sts_results_YYYYMMDD_HHMMSS.json`: Detailed results (JSON)
- `klue_sts_results_YYYYMMDD_HHMMSS.csv`: Detailed results (CSV)

### Log Files
- `logs/klue_sts_*_YYYYMMDD_HHMMSS.log`: Full output logs
- `logs/klue_sts_*_YYYYMMDD_HHMMSS.err`: Error logs

### Error Analysis
- `klue_sts_error_analysis_YYYYMMDD_HHMMSS.txt`: Detailed error analysis

## Example Output

```
======================================================
KLUE Sentence Textual Similarity Benchmark Results
======================================================
Model: gemini-2.5-flash
Platform: Google Cloud Vertex AI
Project: your-project-id
Location: us-central1

Primary Metrics:
  Pearson Correlation: 0.8234
  Spearman Correlation: 0.8156
  Mean Squared Error (MSE): 0.4567
  Mean Absolute Error (MAE): 0.3456
  Valid Predictions: 1000/1000

Performance Metrics:
  Total Time: 1234.56 seconds
  Average Time per Sample: 1.23 seconds
  Samples per Second: 0.81
```

## Troubleshooting

### Common Issues

1. **Authentication Errors**
   ```bash
   gcloud auth application-default login
   ```

2. **Missing Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Project ID Not Set**
   ```bash
   export GOOGLE_CLOUD_PROJECT="your-project-id"
   ```

4. **Vertex AI API Not Enabled**
   Enable the Vertex AI API in your Google Cloud project.

### Getting Help

- Check the logs in the `logs/` directory for detailed error messages
- Review the error analysis files for specific prediction failures
- Ensure your Google Cloud project has sufficient quota for Vertex AI

## Performance Considerations

- **Rate Limiting**: The script includes built-in delays to respect API rate limits
- **Cost Management**: Monitor your Google Cloud usage and costs
- **Long-running Jobs**: Use tmux for full benchmarks to prevent disconnection issues
- **Memory Usage**: The script processes samples one at a time to minimize memory usage

## Contributing

To contribute to this benchmark:

1. Follow the existing code structure and style
2. Add appropriate error handling and logging
3. Update documentation for any new features
4. Test thoroughly before submitting changes

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## Acknowledgments

- KLUE benchmark dataset and task definitions
- Google Cloud Vertex AI for model hosting
- Gemini 2.5 Flash model for inference 