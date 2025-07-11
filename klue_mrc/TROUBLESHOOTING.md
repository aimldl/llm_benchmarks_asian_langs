# KLUE MRC Benchmark Troubleshooting Guide

This guide helps you resolve common issues when running the KLUE Machine Reading Comprehension benchmark.

## Common Issues and Solutions

### 1. Authentication and Project Setup

#### Issue: "Google Cloud project ID must be provided"
```
Error: Google Cloud project ID must be provided via the --project-id flag or by setting the GOOGLE_CLOUD_PROJECT environment variable.
```

**Solution:**
```bash
# Set the environment variable
export GOOGLE_CLOUD_PROJECT="your-project-id"

# Or provide it as a command line argument
python klue_mrc-gemini2_5flash.py --project-id "your-project-id"
```

#### Issue: "No credentials found"
```
✗ No credentials found
```

**Solution:**
```bash
# Set up application default credentials
gcloud auth application-default login

# Or use service account key
export GOOGLE_APPLICATION_CREDENTIALS="path/to/your/service-account-key.json"
```

#### Issue: "Vertex AI API not enabled"
```
Error: Vertex AI API is not enabled for this project
```

**Solution:**
```bash
# Enable the Vertex AI API
gcloud services enable aiplatform.googleapis.com
```

### 2. Dataset Loading Issues

#### Issue: "Failed to load KLUE MRC dataset"
```
✗ Failed to load KLUE MRC dataset: Connection error
```

**Solution:**
- Check your internet connection
- Verify you can access Hugging Face datasets
- Try downloading the dataset manually:
```bash
python -c "from datasets import load_dataset; load_dataset('klue', 'mrc', cache_dir='./cache')"
```

#### Issue: "Dataset schema has changed"
```
✗ A key was not found in the dataset item: 'answers'
```

**Solution:**
- Update the datasets library: `pip install --upgrade datasets`
- Check the current dataset schema:
```python
from datasets import load_dataset
dataset = load_dataset('klue', 'mrc')
print(dataset['validation'][0].keys())
```

### 3. Model and API Issues

#### Issue: "Cannot get the response text"
```
ERROR - Prediction failed: Cannot get the response text.
Response candidate content has no parts (and thus no text). The candidate is likely blocked by the safety filters.
```

**Solution:**
- The model response was blocked by safety filters
- This is usually temporary - try running again
- If persistent, check if your content violates safety guidelines
- Consider adjusting the safety settings in the code

#### Issue: "API quota exceeded"
```
Error: Quota exceeded for quota group 'default' and limit 'requests per minute'
```

**Solution:**
- Check your Vertex AI quota in Google Cloud Console
- Reduce the request rate by increasing `sleep_interval_between_api_calls`
- Use a smaller sample size for testing
- Consider requesting quota increase from Google Cloud

#### Issue: "Model not found"
```
Error: Model 'gemini-2.5-flash' not found
```

**Solution:**
- Verify the model name is correct
- Check if the model is available in your region
- Try using a different Vertex AI location:
```bash
python klue_mrc-gemini2_5flash.py --project-id "your-project-id" --location "us-west1"
```

### 4. Performance and Resource Issues

#### Issue: "Memory error"
```
MemoryError: Unable to allocate array
```

**Solution:**
- Reduce the batch size or max_samples
- Close other applications to free memory
- Use a machine with more RAM
- Process data in smaller chunks

#### Issue: "Timeout error"
```
TimeoutError: Request timed out
```

**Solution:**
- Increase timeout settings
- Check your network connection
- Try running during off-peak hours
- Use a different Vertex AI region

### 5. File and Directory Issues

#### Issue: "Permission denied"
```
PermissionError: [Errno 13] Permission denied: 'benchmark_results/'
```

**Solution:**
```bash
# Make sure you have write permissions
chmod 755 benchmark_results/
chmod 755 logs/

# Or run with appropriate permissions
sudo chown -R $USER:$USER .
```

#### Issue: "Directory not found"
```
FileNotFoundError: [Errno 2] No such file or directory: 'logs/'
```

**Solution:**
```bash
# Create required directories
mkdir -p logs benchmark_results result_analysis
```

### 6. Python and Dependency Issues

#### Issue: "Module not found"
```
ModuleNotFoundError: No module named 'google.genai'
```

**Solution:**
```bash
# Install dependencies
pip install -r requirements.txt

# Or install specific package
pip install google-genai
```

#### Issue: "Version conflict"
```
ImportError: cannot import name 'GenerateContentConfig' from 'google.genai.types'
```

**Solution:**
```bash
# Update to the latest version
pip install --upgrade google-genai

# Or install a specific version
pip install google-genai==0.3.0
```

### 7. Logging and Output Issues

#### Issue: "Log files not created"
```
No log files found in logs/ directory
```

**Solution:**
- Check if the `run` script is executable: `chmod +x run`
- Verify the logs directory exists: `mkdir -p logs`
- Check if the script has write permissions

#### Issue: "Error extraction not working"
```
No error analysis section found in the log.
```

**Solution:**
- This is normal if no errors occurred
- Check the full log file for actual content
- Verify the error extraction patterns match the output format

### 8. Benchmark-Specific Issues

#### Issue: "Low performance scores"
```
Exact Match: 0.15
F1 Score: 0.23
```

**Possible Causes and Solutions:**
- **Model Configuration**: Try adjusting temperature, max_tokens, or other parameters
- **Prompt Engineering**: The prompt might need optimization for Korean MRC
- **Data Quality**: Check if the dataset is loading correctly
- **Impossible Questions**: Ensure the model handles impossible questions properly

#### Issue: "Inconsistent results"
```
Results vary significantly between runs
```

**Solution:**
- Set a fixed random seed if applicable
- Use deterministic model parameters (temperature=0.0)
- Run multiple times and average results
- Check for API rate limiting or quota issues

## Debugging Steps

### 1. Test Environment Setup
```bash
# Run the test setup script
python test_setup.py

# Check if all components are working
./verify_scripts.sh
```

### 2. Test with Small Sample
```bash
# Run with just 5 samples to test
./run custom 5

# Check the output and logs
ls -la logs/
cat logs/klue_mrc_custom_5samples_*.log
```

### 3. Check Logs
```bash
# View the latest log file
ls -t logs/ | head -1 | xargs -I {} cat logs/{}

# Extract errors only
ls -t logs/ | head -1 | xargs -I {} cat logs/{} | grep -A 10 -B 2 "ERROR"
```

### 4. Verify Dataset
```python
# Test dataset loading
from datasets import load_dataset
dataset = load_dataset('klue', 'mrc')
print(f"Validation samples: {len(dataset['validation'])}")
print(f"Sample keys: {dataset['validation'][0].keys()}")
```

## Getting Help

If you're still experiencing issues:

1. **Check the logs**: Review the log files in the `logs/` directory
2. **Run test setup**: Execute `python test_setup.py` to verify your environment
3. **Check documentation**: Review the README.md and ABOUT_KLUE_MRC.md files
4. **Verify scripts**: Run `./verify_scripts.sh` to ensure all scripts are properly set up
5. **Search issues**: Look for similar issues in the repository or related forums

## Performance Optimization Tips

1. **Batch Processing**: Process multiple samples together when possible
2. **Caching**: Cache dataset and model responses to avoid redundant API calls
3. **Parallel Processing**: Use multiple processes for independent operations
4. **Resource Management**: Monitor memory and CPU usage during execution
5. **API Optimization**: Use appropriate rate limiting and retry strategies

## Common Configuration Adjustments

### For Better Performance
```bash
# Lower temperature for more consistent results
python klue_mrc-gemini2_5flash.py --temperature 0.0

# Increase max tokens for longer answers
python klue_mrc-gemini2_5flash.py --max-tokens 4096

# Use different location for better latency
python klue_mrc-gemini2_5flash.py --location "us-west1"
```

### For Faster Testing
```bash
# Use fewer samples for quick testing
python klue_mrc-gemini2_5flash.py --max-samples 10

# Skip saving detailed predictions
python klue_mrc-gemini2_5flash.py --no-save-predictions
```

### For Debugging
```bash
# Increase logging verbosity
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python -u klue_mrc-gemini2_5flash.py --max-samples 5
``` 