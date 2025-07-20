# LLM Benchmarks for Asian Languages 🌏

A comprehensive suite of benchmarks to evaluate Large Language Models (LLMs) on Korean language understanding tasks using Google Cloud Vertex AI and Gemini 2.5 Flash.

## Quickstart

```bash
# Clone and navigate
git clone https://github.com/aimldl/llm_benchmarks_asian_langs.git
cd llm_benchmarks_asian_langs

# Setup environment
conda create -n klue -y python=3 anaconda
conda activate klue

# Run any benchmark
cd klue_tc  # or klue_nli, klue_ner, klue_mrc, klue_dp, klue_re, klue_dst
./setup.sh full
./run test
```

## Available Tasks

| Task | Description | Dataset Size | Expected Performance | Complexity |
|------|-------------|--------------|---------------------|------------|
| **KLUE TC** | Topic Classification (7 news categories) | ~3K samples | 85-95% accuracy | Low |
| **KLUE NLI** | Natural Language Inference (entailment/contradiction/neutral) | ~3K samples | 75-85% accuracy | Medium |
| **KLUE NER** | Named Entity Recognition (person/location/organization) | ~1K samples | 70-85% F1 | Medium |
| **KLUE MRC** | Machine Reading Comprehension (Q&A from passages) | ~2K samples | 60-80% F1 | High |
| **KLUE DP** | Dependency Parsing (grammatical relationships) | ~1K samples | 75-90% UAS/LAS | Very High |
| **KLUE RE** | Relation Extraction (entity relationships) | ~1K samples | 65-80% accuracy | High |
| **KLUE DST** | Dialogue State Tracking (conversation state) | ~1K samples | 60-75% F1 | High |

## Project Structure

```
llm_benchmarks_asian_langs/
├── klue_tc/          # Topic Classification
├── klue_nli/         # Natural Language Inference  
├── klue_ner/         # Named Entity Recognition
├── klue_mrc/         # Machine Reading Comprehension
├── klue_dp/          # Dependency Parsing
├── klue_re/          # Relation Extraction
└── klue_dst/         # Dialogue State Tracking
```

Each task directory contains:
- **`README.md`**: Detailed task-specific documentation
- **`run`**: Benchmark execution script with test/custom/full modes
- **`setup.sh`**: Environment and dependency setup
- **`*.ipynb`**: Jupyter notebook for interactive execution
- **`benchmark_results/`**: Generated metrics and analysis
- **`logs/`**: Execution logs and error analysis

For more details, refer to [Directory and File Structure](files/directory_structure.md)

## Quick Setup

### 1. Environment Setup
```bash
# Option 1: Anaconda (recommended)
conda create -n klue -y python=3 anaconda
conda activate klue

# Option 2: Virtual environment
python -m venv .venv
source .venv/bin/activate
```

### 2. Google Cloud Configuration
```bash
# Set project and enable APIs
export GOOGLE_CLOUD_PROJECT="your-project-id"
gcloud services enable aiplatform.googleapis.com

# Authentication (choose one)
gcloud auth application-default login
# OR use service account key
export GOOGLE_APPLICATION_CREDENTIALS="path/to/key.json"
```

### 3. Run Benchmarks
```bash
# Navigate to any task
cd klue_tc    # or klue_nli, klue_ner, klue_mrc, klue_dp, klue_re, klue_dst

# Complete setup
./setup.sh full

# Execute benchmark
./run test        # 10 samples
./run custom 100  # 100 samples  
./run full        # All samples
```

## Execution Modes

All benchmarks support three execution modes:

- **`test`**: Quick validation with 10 samples
- **`custom N`**: Run with N samples (e.g., `./run custom 50`)
- **`full`**: Complete benchmark on all available samples

## Background Processing

For long-running benchmarks, use `tmux` to run processes in the background:

### Create and Start Session
```bash
tmux new -s klue
./run full
```

### Detach from Session
Press `Ctrl+b d` to detach while keeping the process running.

### Reattach to Session
```bash
# List sessions
tmux ls

# Reattach
tmux attach -t klue
```

## Performance Metrics

### Task-Specific Metrics

| Task | Primary Metrics | Secondary Metrics | Key Challenges |
|------|----------------|-------------------|----------------|
| **TC** | Accuracy, Per-category accuracy | Speed, Error analysis | Topic ambiguity |
| **NLI** | Accuracy, Per-class accuracy | Confusion matrix | Logical reasoning |
| **NER** | Precision, Recall, F1 | Per-entity F1 | Korean names/honorifics |
| **MRC** | Exact Match, F1, Impossible Accuracy | Per-type analysis | Unanswerable questions |
| **DP** | UAS, LAS | POS accuracy | Korean agglutination |
| **RE** | Accuracy, Per-relation accuracy | Error analysis | Complex sentence structures |
| **DST** | Intent Accuracy, Overall F1 | Requested Slots F1 | Multi-turn dialogue |

### Expected Performance Ranges

| Task | Metric | Expected Range | Best Performance |
|------|--------|----------------|------------------|
| TC | Accuracy | 85-95% | Simple news headlines |
| NLI | Accuracy | 75-85% | Simple premises |
| NER | F1-Score | 70-85% | Common entities |
| MRC | F1-Score | 60-80% | Answerable questions |
| DP | UAS/LAS | 75-90%/70-85% | Simple sentences |
| RE | Accuracy | 65-80% | Simple sentences |
| DST | F1-Score | 60-75% | Clear dialogue turns |

## Output Structure

Each benchmark generates:

- **Metrics**: Overall performance scores (accuracy, F1, etc.)
- **Detailed Results**: Per-sample predictions and analysis
- **Error Analysis**: Failed predictions and debugging info
- **Logs**: Complete execution logs for troubleshooting

Results are saved in `benchmark_results/` and `logs/` directories.

## Key Features

- **Standardized Structure**: Uniform scripts and directory layout across all tasks
- **Flexible Execution**: Multiple modes and customizable parameters
- **Comprehensive Logging**: Detailed logs and error analysis
- **Vertex AI Integration**: Production-grade Google Cloud platform
- **Task-Specific Evaluation**: Optimized prompts and metrics for each task
- **Background Processing**: tmux support for long-running benchmarks

## Prerequisites

- **Google Cloud Project** with Vertex AI API enabled
- **Authentication**: Service Account Key or Application Default Credentials
- **Python 3.8+**: For environment setup and execution

## Cost Considerations

⚠️ **Important**: Running full benchmarks incurs significant Google Cloud costs. Monitor usage and set budget alerts.

- **Vertex AI**: Production platform with high quotas
- **Google AI (Free)**: Not recommended - low rate limits cause failures

## Getting Started

1. **Choose a Task**: Navigate to any `klue_*` directory
2. **Read Task README**: Review task-specific documentation
3. **Setup Environment**: Run `./setup.sh full`
4. **Test Execution**: Run `./run test` for quick validation
5. **Full Benchmark**: Run `./run full` for complete evaluation

## Troubleshooting

- **Authentication**: Use `gcloud auth application-default login`
- **API Quotas**: Check Vertex AI quota in Google Cloud Console
- **Environment**: Recreate conda environment if needed
- **Logs**: Check `logs/` directory for detailed error information

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes and test thoroughly
4. Submit a pull request

## License

Apache License Version 2.0 - see LICENSE file for details.

## Acknowledgments

- KLUE dataset creators and maintainers
- Google Cloud Vertex AI team
- Hugging Face datasets library

## Related Work

- [KLUE Paper](https://arxiv.org/abs/2105.09680)
- [KLUE GitHub Repository](https://github.com/KLUE-benchmark/KLUE)
  - [KLUE dataset description](https://github.com/KLUE-benchmark/KLUE/wiki): [TC (YNAT)](https://github.com/KLUE-benchmark/KLUE/wiki/KLUE-TC-(YNAT)-dataset-description), [STS](https://github.com/KLUE-benchmark/KLUE/wiki/KLUE-STS-dataset-description), [NLI](https://github.com/KLUE-benchmark/KLUE/wiki/KLUE-NLI-dataset-description), [NER](https://github.com/KLUE-benchmark/KLUE/wiki/KLUE-NER-dataset-description), [RE](https://github.com/KLUE-benchmark/KLUE/wiki/KLUE-RE-dataset-description), [DP](https://github.com/KLUE-benchmark/KLUE/wiki/KLUE-DP-dataset-description), [MRC](https://github.com/KLUE-benchmark/KLUE/wiki/KLUE-MRC-dataset-descripton), [DST (WoS)](https://github.com/KLUE-benchmark/KLUE/wiki/KLUE-DST-(WoS)-dataset-description),
  - [KLUE-baseline GitHub Repository](https://github.com/KLUE-benchmark/KLUE-baseline)
- [KLUE HuggingFace Repository](https://huggingface.https://github.com/KLUE-benchmark/KLUE/wiki/KLUE-NER-dataset-descriptionco/klue)
  - [klue/Dataset card](https://huggingface.co/datasets/klue/klue)
- [Google Cloud Vertex AI Documentation](https://cloud.google.com/vertex-ai)
