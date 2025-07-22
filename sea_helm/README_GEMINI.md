# SEA-HELM with Gemini 2.5 Flash on Vertex AI

This document describes how to run SEA-HELM (SouthEast Asian Holistic Evaluation of Language Models) benchmarks using Gemini 2.5 Flash on Google Cloud Vertex AI.

## Overview

SEA-HELM is a comprehensive evaluation suite for large language models across Southeast Asian languages. This integration allows you to evaluate Gemini 2.5 Flash on the full SEA-HELM benchmark suite using Google Cloud Vertex AI.

## Prerequisites

1. **Google Cloud Account**: You need a Google Cloud account with billing enabled
2. **Google Cloud Project**: A project with Vertex AI API enabled
3. **Authentication**: Proper authentication set up for Google Cloud
4. **Python Environment**: Python 3.8+ with required dependencies

## Quick Start

### 1. Setup Environment

Run the setup script to configure your environment:

```bash
cd sea_helm
bash setup_gemini.sh
```

This script will:
- Authenticate with Google Cloud
- Set up your project configuration
- Enable required APIs
- Install Python dependencies

### 2. Run Benchmark

Run the SEA-HELM benchmark with Gemini 2.5 Flash:

```bash
python3 seahelm_gemini2_5flash.py --project-id YOUR_PROJECT_ID
```

## Configuration

### Command Line Options

```bash
python3 seahelm_gemini2_5flash.py [OPTIONS]
```

#### Model Configuration
- `--model-name`: Gemini model name (default: gemini-2.5-flash)
- `--project-id`: Google Cloud project ID
- `--location`: Google Cloud location (default: us-central1)

#### Output Configuration
- `--output-dir`: Output directory for results (default: benchmark_results)
- `--tasks-configuration`: Tasks configuration to run (default: seahelm)

#### Generation Parameters
- `--max-tokens`: Maximum tokens to generate (default: 2048)
- `--temperature`: Sampling temperature (default: 0.1)
- `--top-p`: Top-p sampling parameter (default: 1.0)
- `--top-k`: Top-k sampling parameter (default: 1)
- `--sleep-interval`: Sleep interval between API calls (default: 0.04)

#### Model Type Flags
- `--base-model`: Treat as base model
- `--vision-model`: Treat as vision model
- `--reasoning-model`: Treat as reasoning model

#### Evaluation Parameters
- `--max-samples`: Maximum number of samples to evaluate
- `--limit`: Limit number of samples per task
- `--skip-task`: Tasks to skip
- `--no-batching`: Disable batching (default: True)
- `--no-cached-results`: Don't use cached results
- `--rerun-cached-results`: Rerun cached results

## Examples

### Basic Usage
```bash
# Run with default settings
python3 seahelm_gemini2_5flash.py --project-id my-project-123

# Run with custom parameters
python3 seahelm_gemini2_5flash.py \
    --project-id my-project-123 \
    --model-name gemini-2.5-flash \
    --max-tokens 4096 \
    --temperature 0.2 \
    --output-dir my_results
```

### Running Specific Tasks
```bash
# Run only specific task configurations
python3 seahelm_gemini2_5flash.py \
    --project-id my-project-123 \
    --tasks-configuration nlu \
    --limit 100
```

### Base Model Evaluation
```bash
# Evaluate as base model (disables certain tasks)
python3 seahelm_gemini2_5flash.py \
    --project-id my-project-123 \
    --base-model
```

## Output Structure

The benchmark generates the following output structure:

```
benchmark_results/
├── seahelm_gemini25flash_results_YYYYMMDD_HHMMSS.json  # Final results
├── {model_name}/
│   ├── config.json                                      # Configuration
│   ├── {task_name}_{lang}/
│   │   ├── inference_results.jsonl                      # Raw inference results
│   │   └── metrics.json                                 # Task metrics
│   └── ...
└── logs/
    └── seahelm_gemini25flash_YYYYMMDD_HHMMSS.log       # Log file
```

## Supported Models

- `gemini-2.5-flash`: Gemini 2.5 Flash (recommended for most use cases)
- `gemini-2.5-pro`: Gemini 2.5 Pro (for more complex reasoning tasks)

## Task Categories

SEA-HELM evaluates models across 5 core pillars:

1. **NLP Classics**: Traditional NLP tasks
2. **LLM-specifics**: Tasks designed for modern LLMs
3. **SEA Linguistics**: Southeast Asian language-specific tasks
4. **SEA Culture**: Cultural understanding tasks
5. **Safety**: Safety and alignment evaluation

## Troubleshooting

### Common Issues

1. **Authentication Error**
   ```bash
   # Re-authenticate with Google Cloud
   gcloud auth login
   gcloud auth application-default login
   ```

2. **Project Not Set**
   ```bash
   # Set your project
   gcloud config set project YOUR_PROJECT_ID
   export GOOGLE_CLOUD_PROJECT=YOUR_PROJECT_ID
   ```

3. **API Not Enabled**
   ```bash
   # Enable required APIs
   gcloud services enable aiplatform.googleapis.com
   gcloud services enable compute.googleapis.com
   gcloud services enable storage.googleapis.com
   ```

4. **Dependencies Missing**
   ```bash
   # Install dependencies
   pip3 install -r requirements.txt
   pip3 install google-genai>=0.3.0 google-cloud-aiplatform>=1.35.0
   ```

### Rate Limiting

If you encounter rate limiting:
- Increase `--sleep-interval` (e.g., 0.1 seconds)
- Reduce batch sizes
- Use smaller `--limit` values

### Cost Optimization

To optimize costs:
- Use `--limit` to test with fewer samples
- Use `--no-cached-results` to avoid re-running completed tasks
- Monitor usage in Google Cloud Console

## Integration with Existing SEA-HELM

This Gemini integration follows the same patterns as other SEA-HELM serving classes:

- **GeminiServing**: Implements the BaseServing interface
- **Compatible**: Works with existing SEA-HELM evaluation framework
- **Extensible**: Can be easily extended for other Gemini models

## Performance Considerations

- **Latency**: Vertex AI calls have network latency
- **Throughput**: Rate limits apply to API calls
- **Cost**: Pay per API call and token usage
- **Caching**: Results are cached to avoid re-computation

## Support

For issues related to:
- **SEA-HELM**: Check the main [SEA-HELM documentation](README.md)
- **Gemini API**: Check [Google AI documentation](https://ai.google.dev/)
- **Vertex AI**: Check [Vertex AI documentation](https://cloud.google.com/vertex-ai)

## License

This integration follows the same MIT license as the main SEA-HELM project. 