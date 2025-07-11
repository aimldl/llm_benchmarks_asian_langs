# KLUE DST Troubleshooting Guide

This guide provides solutions for common issues encountered when running the KLUE DST benchmark.

## Quick Diagnosis

Run these commands to diagnose your setup:

```bash
# Check environment
python3 test_setup.py

# Check script permissions
./verify_scripts.sh

# Test logging
./test_logging.sh test
```

## Common Issues and Solutions

### 1. Authentication and Authorization

#### Issue: "Failed to initialize Vertex AI"
**Symptoms:**
- Error: "Google Cloud project ID must be provided"
- Error: "Failed to initialize Vertex AI"

**Solutions:**
```bash
# Set your project ID
export GOOGLE_CLOUD_PROJECT='your-project-id'

# Authenticate with Google Cloud
gcloud auth login
gcloud auth application-default login

# Verify authentication
gcloud auth list
```

#### Issue: "Permission denied" or "Quota exceeded"
**Symptoms:**
- Error: "PERMISSION_DENIED"
- Error: "QUOTA_EXCEEDED"

**Solutions:**
1. Check your Vertex AI quota in Google Cloud Console
2. Ensure your account has the necessary permissions:
   - `aiplatform.endpoints.predict`
   - `aiplatform.models.predict`
3. Consider using a smaller sample size for testing

### 2. Python Environment Issues

#### Issue: "ModuleNotFoundError"
**Symptoms:**
- Error: "No module named 'google.genai'"
- Error: "No module named 'datasets'"

**Solutions:**
```bash
# Reinstall dependencies
./install_dependencies.sh

# Or install manually
pip3 install -r requirements.txt

# Verify installations
python3 -c "import google.genai; print('✓ google.genai OK')"
python3 -c "from datasets import load_dataset; print('✓ datasets OK')"
```

#### Issue: "Python version too old"
**Symptoms:**
- Error: "Python 3.8+ required"

**Solutions:**
```bash
# Check Python version
python3 --version

# Install Python 3.8+ if needed
# On Ubuntu/Debian:
sudo apt update
sudo apt install python3.8 python3.8-venv python3.8-pip

# On macOS:
brew install python@3.8
```

### 3. Dataset Loading Issues

#### Issue: "Failed to load dataset"
**Symptoms:**
- Error: "Dataset not found"
- Error: "Connection timeout"

**Solutions:**
```bash
# Check internet connection
ping huggingface.co

# Try with smaller dataset first
python3 -c "
from datasets import load_dataset
dataset = load_dataset('klue', 'dst', split='validation')
print(f'Dataset loaded: {len(dataset)} samples')
"
```

#### Issue: "Dataset features mismatch"
**Symptoms:**
- Error: "Key not found in dataset"

**Solutions:**
1. Check if the dataset schema has changed
2. Update the code to match the current dataset format
3. Verify dataset version compatibility

### 4. API and Network Issues

#### Issue: "API timeout"
**Symptoms:**
- Error: "Request timeout"
- Long response times

**Solutions:**
```bash
# Increase timeout in the script
# Edit klue_dst-gemini2_5flash.py and increase sleep_interval_between_api_calls

# Check network connectivity
curl -I https://us-central1-aiplatform.googleapis.com

# Try with smaller batch size
./run custom 5
```

#### Issue: "Rate limiting"
**Symptoms:**
- Error: "Too many requests"
- Frequent API failures

**Solutions:**
1. Increase the sleep interval between API calls
2. Use a smaller sample size
3. Check your Vertex AI quota limits
4. Consider running during off-peak hours

### 5. Script and Permission Issues

#### Issue: "Permission denied" for scripts
**Symptoms:**
- Error: "Permission denied"
- Scripts not executable

**Solutions:**
```bash
# Make scripts executable
chmod +x run setup.sh install_dependencies.sh test_setup.py get_errors.sh test_logging.sh verify_scripts.sh

# Verify permissions
ls -la *.sh run test_setup.py
```

#### Issue: "Script not found"
**Symptoms:**
- Error: "No such file or directory"

**Solutions:**
```bash
# Check if you're in the right directory
pwd
ls -la

# Ensure all files are present
./verify_scripts.sh
```

### 6. Memory and Resource Issues

#### Issue: "Out of memory"
**Symptoms:**
- Error: "MemoryError"
- System becomes unresponsive

**Solutions:**
1. Reduce the number of samples processed at once
2. Close other applications to free memory
3. Use a machine with more RAM
4. Process data in smaller batches

#### Issue: "Disk space full"
**Symptoms:**
- Error: "No space left on device"
- Cannot save results

**Solutions:**
```bash
# Check disk space
df -h

# Clean up old files
rm -rf logs/klue_dst_*_old_*
rm -rf benchmark_results/klue_dst_*_old_*

# Move results to external storage if needed
```

### 7. Logging and Output Issues

#### Issue: "Log files not created"
**Symptoms:**
- No files in logs/ directory
- Missing .log and .err files

**Solutions:**
```bash
# Check if logs directory exists
ls -la logs/

# Create logs directory if missing
mkdir -p logs

# Test logging functionality
./test_logging.sh test
```

#### Issue: "Error extraction not working"
**Symptoms:**
- .err files are empty or missing
- Error analysis not generated

**Solutions:**
```bash
# Test error extraction manually
./get_errors.sh latest

# Check log file format
head -20 logs/klue_dst_*.log
```

### 8. Performance Issues

#### Issue: "Very slow processing"
**Symptoms:**
- Takes hours to process small samples
- Low samples per second

**Solutions:**
1. Check your internet connection speed
2. Verify Vertex AI region settings
3. Consider using a different region
4. Check if other processes are using bandwidth

#### Issue: "Inconsistent results"
**Symptoms:**
- Different results on different runs
- High variance in metrics

**Solutions:**
1. Use consistent temperature settings (0.1)
2. Ensure deterministic processing
3. Run multiple times and average results
4. Check for API response variations

## Advanced Troubleshooting

### Debug Mode

Enable debug logging by modifying the Python script:

```python
# In klue_dst-gemini2_5flash.py, change:
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# To:
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
```

### Verbose Output

Run with verbose output to see more details:

```bash
# Add --verbose flag if supported
python3 klue_dst-gemini2_5flash.py --project-id $GOOGLE_CLOUD_PROJECT --max-samples 5 --verbose
```

### API Response Inspection

Add code to inspect API responses:

```python
# In the predict_single method, add:
print(f"Raw API response: {response}")
print(f"Response text: {response.text if response else 'None'}")
```

## Getting Help

### Before Asking for Help

1. **Check the logs**: Look at the latest log files in `logs/` directory
2. **Run diagnostics**: Execute `python3 test_setup.py` and `./verify_scripts.sh`
3. **Check environment**: Verify your Google Cloud setup and Python environment
4. **Search existing issues**: Look for similar problems in the documentation

### Information to Provide

When reporting issues, include:

1. **Error messages**: Complete error output
2. **Environment details**: Python version, OS, Google Cloud project
3. **Steps to reproduce**: Exact commands run
4. **Log files**: Relevant sections from log files
5. **Configuration**: Any custom settings or modifications

### Useful Commands for Debugging

```bash
# Check system information
python3 --version
pip3 list
gcloud version
gcloud auth list

# Check Google Cloud setup
gcloud config get-value project
gcloud auth application-default print-access-token

# Test API access
curl -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  https://us-central1-aiplatform.googleapis.com/v1/projects/$(gcloud config get-value project)/locations/us-central1/models

# Check disk and memory
df -h
free -h
```

## Prevention Tips

1. **Regular testing**: Run `./run test` regularly to catch issues early
2. **Environment isolation**: Use virtual environments to avoid conflicts
3. **Backup results**: Keep copies of important benchmark results
4. **Monitor quotas**: Check Google Cloud quotas before long runs
5. **Update dependencies**: Keep packages updated for security and compatibility 