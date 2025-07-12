# KLUE Named Entity Recognition (NER) Benchmark

This directory contains the benchmark implementation for the Korean Language Understanding Evaluation (KLUE) Named Entity Recognition task using Google Cloud Vertex AI with Gemini 2.5 Flash.

## Overview

The KLUE NER benchmark evaluates the performance of Gemini 2.5 Flash on identifying and classifying named entities in Korean text. The task involves recognizing six types of entities:

- **PS (Person)**: 인물 - People's names, nicknames, titles
- **LC (Location)**: 지명 - Places, regions, countries, cities, buildings
- **OG (Organization)**: 기관 - Companies, schools, government agencies, organizations
- **DT (Date)**: 날짜 - Years, months, days, weekdays, holidays
- **TI (Time)**: 시간 - Hours, minutes, seconds, time periods
- **QT (Quantity)**: 수량 - Numbers, units, amounts, ratios, counts

## Directory Structure

```
klue_ner/
├── klue_ner-gemini2_5flash.py    # Main benchmark script
├── run                           # Benchmark runner script
├── setup.sh                      # Complete setup script
├── install_dependencies.sh       # Dependencies installation
├── test_setup.py                 # Setup verification script
├── get_errors.sh                 # Error analysis script
├── test_logging.sh               # Logging test script
├── test_verbose_mode.sh          # Verbose mode test script
├── verify_scripts.sh             # Script verification
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── ABOUT_KLUE_NER.md             # KLUE NER task description
├── TROUBLESHOOTING.md            # Troubleshooting guide
├── VERTEX_AI_SETUP.md            # Vertex AI setup guide
├── logs/                         # Log files directory
├── benchmark_results/            # Benchmark results
├── result_analysis/              # Error analysis results
└── eval_dataset/                 # Evaluation dataset
```

## Quick Start

### 1. Prerequisites

- Python 3.8 or higher
- Google Cloud account with Vertex AI enabled
- Google Cloud CLI installed and configured

### 2. Setup

```bash
# Install dependencies
./setup.sh full

# Or step by step:
./install_dependencies.sh install
./test_setup.py
```

### 3. Configure Google Cloud

```bash
# Set your project ID
export GOOGLE_CLOUD_PROJECT='your-project-id'

# Authenticate and enable APIs
gcloud auth login
gcloud auth application-default login
gcloud services enable aiplatform.googleapis.com
```

### 4. Run Benchmark

```bash
# Quick test with 10 samples (clean mode - recommended)
./run test

# Custom number of samples (clean mode - recommended)
./run custom 100

# Full benchmark (all validation samples) (clean mode - recommended)
./run full

# Verbose mode for debugging (shows all logging details)
python klue_ner-gemini2_5flash.py --project-id "$GOOGLE_CLOUD_PROJECT" --max-samples 10 --verbose
```

## Detailed Usage

### Running the Benchmark

The benchmark can be run in three modes with two output levels:

#### Execution Modes:
1. **Test Mode** (`./run test`): Runs with 10 samples for quick testing
2. **Custom Mode** (`./run custom N`): Runs with N samples
3. **Full Mode** (`./run full`): Runs with all validation samples

#### Output Modes:
- **Default Mode (Clean)**: Minimal output with suppressed Google Cloud logging for better readability
- **Verbose Mode**: Full output with all logging details for debugging

**Recommended**: Use the default clean mode for normal operation. Use verbose mode only when debugging issues.

### Logging System

All benchmark runs are automatically logged:

- **Full logs**: `logs/klue_ner_[mode]_[samples]_[timestamp].log`
- **Error logs**: `logs/klue_ner_[mode]_[samples]_[timestamp].err`

Log files include:
- Command headers for easy identification
- Complete execution trace
- Error analysis and debugging information
- Performance metrics

### Error Analysis

Extract error information from benchmark results:

```bash
# Analyze latest result file
./get_errors.sh

# Analyze specific file
./get_errors.sh -f benchmark_results/klue_ner_results_20250101.csv

# Save to custom output file
./get_errors.sh -o my_errors.txt
```

### Testing and Verification

```bash
# Test logging functionality
./test_logging.sh test

# Test verbose mode functionality
./test_verbose_mode.sh

# Verify all scripts and setup
./verify_scripts.sh check

# Fix script permissions
./verify_scripts.sh fix
```

## Configuration

### Model Configuration

The benchmark uses the following default configuration:

- **Model**: `gemini-2.5-flash`
- **Max Tokens**: 2048 (increased for NER task)
- **Temperature**: 0.1
- **Safety Settings**: All disabled for maximum performance
- **Location**: `us-central1`

### Custom Configuration

You can modify the configuration in `klue_ner-gemini2_5flash.py`:

```python
config = BenchmarkConfig(
    max_samples=100,           # Number of samples to test
    max_tokens=2048,           # Maximum output tokens
    temperature=0.1,           # Model temperature
    save_interval=50,          # Save intermediate results every N samples
    project_id="your-project", # Google Cloud project ID
    location="us-central1",    # Vertex AI location
    verbose=False              # Set to True for verbose logging
)
```

## Output Files

### Benchmark Results

- **Metrics**: `benchmark_results/klue_ner_metrics_[timestamp].json`
- **Detailed Results**: `benchmark_results/klue_ner_results_[timestamp].json`
- **CSV Results**: `benchmark_results/klue_ner_results_[timestamp].csv`

**Note**: Intermediate results are also saved every 50 samples (configurable) with both JSON and CSV formats:
- `benchmark_results/klue_ner_results_000050_[timestamp].json`
- `benchmark_results/klue_ner_results_000050_[timestamp].csv`

### CSV Output Format

The CSV file contains the following columns:

| Column | Description |
|--------|-------------|
| id | Sample identifier |
| text | Input text |
| true_entities_count | Number of true entities |
| predicted_entities_count | Number of predicted entities |
| precision | Precision score |
| recall | Recall score |
| f1 | F1 score |
| correct_entities | Number of correctly identified entities |
| success | Whether prediction was successful |
| error | Error message (if any) |

### Error Analysis

Error analysis files contain:
- Sample-level error details
- Entity comparison information
- Error statistics and rates
- Debugging information

## Recent Improvements

### Progress Bar Enhancement
- **Reduced Update Frequency**: Progress bar now updates less frequently (approximately 25% of previous frequency) for better readability
- **Cleaner Output**: Improved formatting with better visual separation between progress updates
- **Better User Experience**: Less cluttered output while maintaining useful progress information

### CSV File Generation
- **Intermediate Results**: CSV files are now generated for both intermediate and final results
- **Easy Analysis**: CSV format provides quick access to summary statistics
- **Consistent Format**: Both JSON and CSV files are available for all result saves

### Verbose/Clean Mode System
- **Default Clean Mode**: Minimal output with suppressed Google Cloud logging for optimal readability
- **Verbose Mode**: Full logging details available for debugging and troubleshooting
- **Flexible Usage**: Users can choose the appropriate output level for their needs

## Performance Metrics

The benchmark calculates:

- **Precision**: Correct entities / Predicted entities
- **Recall**: Correct entities / True entities
- **F1 Score**: Harmonic mean of precision and recall
- **Per-entity Type Performance**: Accuracy for each entity type
- **Processing Time**: Total and per-sample processing times

## Dataset Information

The benchmark uses the KLUE NER dataset from Hugging Face:

- **Dataset**: `klue/ner`
- **Split**: `validation`
- **Format**: BIO tagging scheme
- **Entity Types**: 6 types (PS, LC, OG, DT, TI, QT)

### Entity Type Distribution

The dataset contains various entity types with different frequencies:
- Person (PS): Names, titles, nicknames
- Location (LC): Places, regions, buildings
- Organization (OG): Companies, institutions, agencies
- Date (DT): Temporal expressions
- Time (TI): Time expressions
- Quantity (QT): Numerical expressions

## Prompt Engineering

The benchmark uses a detailed prompt designed for Korean NER:

1. **Role Definition**: Clear definition of the AI's role as a NER specialist
2. **Entity Type Definitions**: Detailed descriptions of each entity type with examples
3. **Output Format**: Structured format `[entity_text:entity_type]`
4. **Context Guidelines**: Instructions for handling ambiguous cases
5. **Korean Language Considerations**: Specific guidance for Korean text processing

## Troubleshooting

### Common Issues

1. **Safety Filter Blocking**: The model may be blocked by safety filters
   - Solution: Safety settings are disabled by default
   - Check: Review error logs for safety-related messages

2. **API Rate Limits**: Too many requests to Vertex AI
   - Solution: Built-in rate limiting (0.04s between calls)
   - Check: Monitor API quotas in Google Cloud Console

3. **Authentication Issues**: Google Cloud authentication problems
   - Solution: Follow VERTEX_AI_SETUP.md guide
   - Check: Verify `GOOGLE_CLOUD_PROJECT` environment variable

4. **Dataset Access**: Issues loading KLUE dataset
   - Solution: Check internet connection and Hugging Face access
   - Check: Verify `datasets` library installation

### Debugging

1. **Check Logs**: Review `.log` and `.err` files in the `logs/` directory
2. **Run Tests**: Use `./test_setup.py` to verify setup
3. **Error Analysis**: Use `./get_errors.sh` to analyze specific errors
4. **Script Verification**: Use `./verify_scripts.sh` to check all components

## Advanced Usage

### Custom Entity Types

To modify entity types, edit the `ENTITY_TYPES` dictionary in the main script:

```python
ENTITY_TYPES = {
    "PS": "인물(Person)",
    "LC": "지명(Location)",
    "OG": "기관(Organization)",
    "DT": "날짜(Date)",
    "TI": "시간(Time)",
    "QT": "수량(Quantity)",
    # Add custom types here
}
```

### Custom Prompts

Modify the `create_prompt()` method to customize the prompt:

```python
def create_prompt(self, text: str) -> str:
    # Customize your prompt here
    prompt = f"""Your custom prompt for: {text}"""
    return prompt
```

### Batch Processing

For large datasets, consider:

1. **Intermediate Saves**: Results are saved every 50 samples by default
2. **Resume Capability**: Can restart from intermediate results
3. **Memory Management**: Process samples one at a time to manage memory

## Contributing

When contributing to this benchmark:

1. **Maintain Consistency**: Follow the same patterns as other KLUE benchmarks
2. **Update Documentation**: Keep README and other docs current
3. **Test Thoroughly**: Run tests before submitting changes
4. **Log Everything**: Ensure all operations are properly logged

## Related Documentation

- [ABOUT_KLUE_NER.md](ABOUT_KLUE_NER.md): Detailed KLUE NER task description
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md): Comprehensive troubleshooting guide
- [VERTEX_AI_SETUP.md](VERTEX_AI_SETUP.md): Google Cloud Vertex AI setup instructions

## License

This benchmark is part of the KLUE evaluation suite. Please refer to the main project license for usage terms. 