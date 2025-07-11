# KLUE NER Troubleshooting Guide

This document provides solutions to common issues encountered when running the KLUE Named Entity Recognition benchmark.

## Common Issues and Solutions

### 1. Google Cloud Authentication Issues

**Problem**: Authentication errors when accessing Vertex AI
```
Error: Failed to initialize Vertex AI: 403 Forbidden
```

**Solutions**:
1. **Check Project ID**: Ensure `GOOGLE_CLOUD_PROJECT` is set correctly
   ```bash
   echo $GOOGLE_CLOUD_PROJECT
   export GOOGLE_CLOUD_PROJECT='your-project-id'
   ```

2. **Authenticate with gcloud**:
   ```bash
   gcloud auth login
   gcloud auth application-default login
   ```

3. **Enable Vertex AI API**:
   ```bash
   gcloud services enable aiplatform.googleapis.com
   ```

4. **Check IAM Permissions**: Ensure your account has the necessary roles:
   - Vertex AI User
   - Service Account User (if using service accounts)

### 2. Safety Filter Blocking

**Problem**: Model responses are blocked by safety filters
```
ERROR - Prediction failed: Cannot get the response text.
Response candidate content has no parts (and thus no text). The candidate is likely blocked by the safety filters.
```

**Solutions**:
1. **Safety settings are already disabled** in the benchmark by default
2. **Check for content issues**: Review the input text for potentially problematic content
3. **Try different samples**: Some content may still trigger filters despite settings
4. **Review error logs**: Check `.err` files for detailed error information

### 3. API Rate Limiting

**Problem**: Too many requests to Vertex AI API
```
Error: 429 Too Many Requests
```

**Solutions**:
1. **Built-in rate limiting**: The benchmark includes 0.04s delays between calls
2. **Reduce batch size**: Use smaller sample sizes for testing
3. **Check quotas**: Monitor your API quotas in Google Cloud Console
4. **Wait and retry**: API limits reset periodically

### 4. Dataset Loading Issues

**Problem**: Cannot load KLUE NER dataset
```
Error: Failed to load or process the dataset
```

**Solutions**:
1. **Check internet connection**: Dataset is downloaded from Hugging Face
2. **Verify datasets library**:
   ```bash
   pip install --upgrade datasets
   ```

3. **Clear cache** (if corrupted):
   ```bash
   rm -rf ~/.cache/huggingface/
   ```

4. **Check dataset availability**:
   ```python
   from datasets import load_dataset
   dataset = load_dataset('klue', 'ner', split='validation')
   ```

### 5. Memory Issues

**Problem**: Out of memory errors during processing
```
MemoryError: Unable to allocate memory
```

**Solutions**:
1. **Reduce batch size**: Process fewer samples at once
2. **Use smaller max_tokens**: Reduce the `max_tokens` parameter
3. **Clear memory**: Restart the process if memory usage is high
4. **Monitor memory usage**: Use system monitoring tools

### 6. Entity Recognition Accuracy Issues

**Problem**: Low F1 scores or poor entity recognition

**Solutions**:
1. **Review prompt engineering**: The prompt is designed for Korean NER
2. **Check entity boundaries**: Ensure proper tokenization
3. **Analyze error patterns**: Use `./get_errors.sh` to identify common errors
4. **Adjust model parameters**: Try different temperature or max_tokens settings

### 7. Logging Issues

**Problem**: Log files not created or incomplete

**Solutions**:
1. **Check directory permissions**: Ensure write access to `logs/` directory
2. **Verify script permissions**: Make sure `run` script is executable
   ```bash
   chmod +x run
   ```

3. **Check disk space**: Ensure sufficient disk space for log files
4. **Review log extraction**: Use `./test_logging.sh` to test logging functionality

### 8. Script Permission Issues

**Problem**: Scripts not executable
```
bash: ./run: Permission denied
```

**Solutions**:
1. **Fix permissions**:
   ```bash
   ./verify_scripts.sh fix
   ```

2. **Manual permission setting**:
   ```bash
   chmod +x run setup.sh install_dependencies.sh get_errors.sh test_logging.sh verify_scripts.sh
   ```

### 9. Python Import Errors

**Problem**: Missing Python packages
```
ModuleNotFoundError: No module named 'google.genai'
```

**Solutions**:
1. **Install dependencies**:
   ```bash
   ./install_dependencies.sh install
   ```

2. **Check installation**:
   ```bash
   ./install_dependencies.sh check
   ```

3. **Verify Python environment**: Ensure you're using the correct Python environment

### 10. Entity Type Mapping Issues

**Problem**: Incorrect entity type recognition

**Solutions**:
1. **Review entity definitions**: Check `ABOUT_KLUE_NER.md` for entity type definitions
2. **Verify prompt clarity**: The prompt includes detailed entity type descriptions
3. **Check output parsing**: Ensure the response parsing handles all entity types
4. **Analyze specific errors**: Use error analysis to identify problematic entity types

## Debugging Steps

### Step 1: Verify Setup
```bash
# Run comprehensive setup test
./test_setup.py

# Check all scripts and permissions
./verify_scripts.sh check
```

### Step 2: Test Logging
```bash
# Test logging functionality
./test_logging.sh test
```

### Step 3: Run Small Test
```bash
# Run with minimal samples
./run test
```

### Step 4: Analyze Errors
```bash
# Extract error information
./get_errors.sh

# Review log files
ls -la logs/
```

### Step 5: Check Configuration
```bash
# Verify environment variables
echo $GOOGLE_CLOUD_PROJECT

# Check Python packages
pip list | grep -E "(google|datasets|pandas|tqdm)"
```

## Performance Optimization

### 1. Reduce Processing Time
- Use smaller sample sizes for testing
- Adjust `sleep_interval_between_api_calls` (default: 0.04s)
- Consider parallel processing for large datasets

### 2. Improve Accuracy
- Review and refine the prompt engineering
- Analyze error patterns to identify common issues
- Consider entity type-specific optimizations

### 3. Memory Management
- Process samples one at a time
- Clear intermediate results periodically
- Monitor memory usage during long runs

## Error Analysis

### Common Error Patterns

1. **Entity Boundary Errors**: Incorrect entity start/end positions
2. **Entity Type Confusion**: Misclassification of entity types
3. **Missing Entities**: Entities not recognized at all
4. **False Positives**: Non-entities incorrectly identified
5. **Format Errors**: Incorrect output format from model

### Error Analysis Tools

1. **get_errors.sh**: Extracts error samples from results
2. **Log files**: Detailed execution traces in `.log` files
3. **Error logs**: Focused error information in `.err` files
4. **CSV analysis**: Manual analysis of result files

## Getting Help

### 1. Check Documentation
- [README.md](README.md): Main documentation
- [ABOUT_KLUE_NER.md](ABOUT_KLUE_NER.md): Task description
- [VERTEX_AI_SETUP.md](VERTEX_AI_SETUP.md): Google Cloud setup

### 2. Review Logs
- Check `logs/` directory for execution traces
- Review `.err` files for error details
- Analyze benchmark results for performance issues

### 3. Test Components
- Use `./test_setup.py` for setup verification
- Use `./test_logging.sh` for logging tests
- Use `./verify_scripts.sh` for script verification

### 4. Common Commands
```bash
# Quick setup verification
./test_setup.py

# Test logging system
./test_logging.sh test

# Analyze latest errors
./get_errors.sh

# Check all components
./verify_scripts.sh check
```

## Prevention

### Best Practices
1. **Always test with small samples first**
2. **Monitor API quotas and usage**
3. **Keep logs for debugging**
4. **Regularly update dependencies**
5. **Backup important results**

### Regular Maintenance
1. **Update Python packages**: `pip install --upgrade -r requirements.txt`
2. **Clear old logs**: Archive or remove old log files
3. **Check disk space**: Ensure sufficient storage
4. **Verify permissions**: Run `./verify_scripts.sh fix` periodically 