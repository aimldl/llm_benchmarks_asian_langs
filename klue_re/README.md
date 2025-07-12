# KLUE RE (Relation Extraction) Benchmark

This directory contains the benchmark implementation for the Korean Language Understanding Evaluation (KLUE) Relation Extraction (RE) task using Google Cloud Vertex AI and Gemini 2.5 Flash.

## Overview

The KLUE RE task involves identifying the relationship between two entities in Korean text. The model is given a sentence and two entities (subject and object) and must determine the relationship between them from a predefined set of relation types.

### Task Description

- **Input**: A sentence with two marked entities (subject and object)
- **Output**: The relationship type between the two entities
- **Evaluation**: Accuracy on the validation set
- **Dataset**: KLUE RE dataset from Hugging Face Hub

### Relation Types

The benchmark supports the following relation types:

#### Organization Relations (org:)
- `org:top_members/employees` - Organization's top members/employees
- `org:members` - Organization members
- `org:product` - Organization's products
- `org:founded` - Organization founding
- `org:alternate_names` - Organization alternate names
- `org:place_of_headquarters` - Organization headquarters location
- `org:number_of_employees/members` - Organization size
- `org:website` - Organization website
- `org:subsidiaries` - Organization subsidiaries
- `org:parents` - Parent organization
- `org:dissolved` - Organization dissolution

#### Person Relations (per:)
- `per:title` - Person's title
- `per:employee_of` - Person's employment
- `per:member_of` - Person's membership
- `per:schools_attended` - Person's education
- `per:works_for` - Person's workplace
- `per:countries_of_residence` - Person's residence country
- `per:stateorprovinces_of_residence` - Person's residence region
- `per:cities_of_residence` - Person's residence city
- `per:countries_of_birth` - Person's birth country
- `per:stateorprovinces_of_birth` - Person's birth region
- `per:cities_of_birth` - Person's birth city
- `per:date_of_birth` - Person's birth date
- `per:date_of_death` - Person's death date
- `per:place_of_birth` - Person's birth place
- `per:place_of_death` - Person's death place
- `per:cause_of_death` - Person's death cause
- `per:origin` - Person's origin
- `per:religion` - Person's religion
- `per:spouse` - Person's spouse
- `per:children` - Person's children
- `per:parents` - Person's parents
- `per:siblings` - Person's siblings
- `per:other_family` - Person's other family
- `per:charges` - Person's charges
- `per:alternate_names` - Person's alternate names
- `per:age` - Person's age

#### Other Relations
- `no_relation` - No relationship between entities

## Quick Start

### Prerequisites

1. **Google Cloud Project**: You need a Google Cloud project with Vertex AI enabled
2. **Authentication**: Set up authentication for Google Cloud
3. **Python 3.8+**: Ensure Python 3.8 or higher is installed
4. **Dependencies**: Install required Python packages

### Setup

1. **Clone and navigate to the directory**:
   ```bash
   cd klue_re
   ```

2. **Set your Google Cloud project ID**:
   ```bash
   export GOOGLE_CLOUD_PROJECT="your-project-id"
   ```

3. **Run the setup script**:
   ```bash
   ./setup.sh
   ```

4. **Verify the setup**:
   ```bash
   ./test_setup.py
   ```

### Running Benchmarks

#### Test Run (10 samples)
```bash
./run test
```

#### Full Benchmark (all validation samples)
```bash
./run full
```

#### Custom Benchmark (N samples)
```bash
./run custom 100
```

### Output Files

The benchmark generates several output files:

#### Log Files
- **Location**: `logs/` directory
- **Format**: `klue_re_[mode]_[samples]samples_[timestamp].log`
- **Content**: Complete execution log with timestamps and error analysis

#### Error Files
- **Location**: `logs/` directory
- **Format**: `klue_re_[mode]_[samples]samples_[timestamp].err`
- **Content**: Extracted error information and analysis

#### Results Files
- **Location**: `benchmark_results/` directory
- **Files**:
  - `klue_re_metrics_[timestamp].json` - Performance metrics
  - `klue_re_results_[timestamp].json` - Detailed results
  - `klue_re_results_[timestamp].csv` - Results in CSV format
  - `klue_re_error_analysis_[timestamp].txt` - Error analysis

#### Intermediate Results
- **Location**: `benchmark_results/` directory
- **Format**: `klue_re_[count]_[timestamp].json`
- **Purpose**: Save progress every 50 samples for long-running benchmarks

## Configuration

### Model Settings

The benchmark uses the following default settings for Gemini 2.5 Flash:

- **Model**: `gemini-2.5-flash`
- **Temperature**: 0.1 (low for consistent results)
- **Max Tokens**: 2048 (increased for RE task)
- **Top-p**: 1.0
- **Top-k**: 1
- **Safety Settings**: Minimal blocking

### Customization

You can modify the configuration in `klue_re-gemini2_5flash.py`:

```python
@dataclass
class BenchmarkConfig:
    model_name: str = "gemini-2.5-flash"
    temperature: float = 0.1
    max_tokens: int = 2048
    # ... other settings
```

### Command Line Options

The main script supports various command line options:

```bash
python klue_re-gemini2_5flash.py \
    --project-id "your-project-id" \
    --max-samples 100 \
    --temperature 0.1 \
    --max-tokens 2048 \
    --output-dir "custom_results"
```

## Prompt Engineering

The benchmark uses a detailed prompt designed specifically for Korean relation extraction:

### Prompt Structure
1. **Role Definition**: Expert relation extraction AI
2. **Task Description**: Clear explanation of the RE task
3. **Relation Type Definitions**: Comprehensive list with examples
4. **Analysis Guidelines**: Step-by-step instructions
5. **Output Format**: Clear specification for response format

### Key Features
- **Detailed Relation Definitions**: Each relation type is explained with examples
- **Context Awareness**: Instructions to consider full sentence context
- **Directional Relations**: Guidance on relation direction (A → B vs B → A)
- **Korean Language Support**: Optimized for Korean text understanding

## Performance Analysis

### Metrics

The benchmark calculates and reports:

- **Accuracy**: Overall correct predictions / total samples
- **Per-relation Performance**: Accuracy for each relation type
- **Processing Time**: Total time and time per sample
- **Error Analysis**: Detailed analysis of failed predictions

### Error Analysis

The system provides comprehensive error analysis:

1. **Sample-level Errors**: Individual failed predictions with context
2. **Pattern Analysis**: Common error patterns and their frequencies
3. **Relation-specific Issues**: Problems specific to certain relation types
4. **Suggestions**: Potential improvements based on error patterns

## Troubleshooting

### Common Issues

1. **Authentication Errors**
   - Ensure `GOOGLE_CLOUD_PROJECT` is set
   - Verify gcloud authentication: `gcloud auth login`
   - Check Vertex AI API is enabled

2. **Dataset Loading Issues**
   - Check internet connection
   - Verify Hugging Face Hub access
   - Try smaller sample sizes first

3. **Memory Issues**
   - Reduce batch size or max samples
   - Use intermediate saving more frequently
   - Monitor system resources

4. **API Rate Limits**
   - Increase sleep interval between calls
   - Use smaller sample sizes
   - Check Vertex AI quotas

### Getting Help

1. **Check Logs**: Review log files in `logs/` directory
2. **Error Analysis**: Use `./get_errors.sh` to extract error information
3. **Test Setup**: Run `./test_setup.py` to verify environment
4. **Documentation**: See `TROUBLESHOOTING.md` for detailed solutions

## Scripts Reference

### Core Scripts

- **`klue_re-gemini2_5flash.py`**: Main benchmark implementation
- **`run`**: Benchmark runner with logging
- **`setup.sh`**: Environment setup and verification
- **`test_setup.py`**: Comprehensive environment testing

### Utility Scripts

- **`get_errors.sh`**: Extract error information from logs
- **`test_logging.sh`**: Test logging functionality
- **`verify_scripts.sh`**: Verify all scripts and permissions
- **`install_dependencies.sh`**: Install Python dependencies

### Usage Examples

```bash
# Extract errors from all log files
./get_errors.sh -a

# Test logging with 5 samples
./test_logging.sh -s 5

# Verify all scripts
./verify_scripts.sh -f

# Install dependencies in virtual environment
./install_dependencies.sh -v
```

## Directory Structure

```
klue_re/
├── klue_re-gemini2_5flash.py    # Main benchmark script
├── run                          # Benchmark runner
├── setup.sh                     # Setup script
├── test_setup.py                # Setup testing
├── get_errors.sh                # Error extraction
├── test_logging.sh              # Logging test
├── verify_scripts.sh            # Script verification
├── install_dependencies.sh      # Dependency installation
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── ABOUT_KLUE_RE.md             # Task description
├── TROUBLESHOOTING.md           # Troubleshooting guide
├── VERTEX_AI_SETUP.md           # Vertex AI setup
├── logs/                        # Log files
├── benchmark_results/           # Results and metrics
├── result_analysis/             # Error analysis files
└── eval_dataset/                # Evaluation dataset
```

## Contributing

To contribute to this benchmark:

1. **Fork the repository**
2. **Create a feature branch**
3. **Make your changes**
4. **Test thoroughly**
5. **Submit a pull request**

### Development Guidelines

- **Consistency**: Follow the existing code style and structure
- **Documentation**: Update documentation for any changes
- **Testing**: Add tests for new features
- **Logging**: Ensure proper logging for debugging

## License

This project is licensed under the same license as the parent repository.

## Acknowledgments

- **KLUE Dataset**: Korean Language Understanding Evaluation dataset
- **Google Cloud**: Vertex AI platform and Gemini models
- **Hugging Face**: Dataset hosting and loading utilities

## Related Documentation

- [ABOUT_KLUE_RE.md](ABOUT_KLUE_RE.md) - Detailed task description
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Troubleshooting guide
- [VERTEX_AI_SETUP.md](VERTEX_AI_SETUP.md) - Vertex AI setup instructions 