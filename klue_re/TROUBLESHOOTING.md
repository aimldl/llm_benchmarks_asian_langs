# KLUE RE Troubleshooting Guide

This guide provides solutions to common issues encountered when running the KLUE RE benchmark.

## Quick Diagnosis

### Check Environment
```bash
# Verify Python version
python3 --version  # Should be 3.8+

# Check if GOOGLE_CLOUD_PROJECT is set
echo $GOOGLE_CLOUD_PROJECT

# Test basic imports
python3 -c "import google.genai, datasets, pandas, tqdm"
```

### Run Setup Test
```bash
./test_setup.py
```

## Common Issues and Solutions

### 1. Authentication and Project Issues

#### Error: "Google Cloud project ID must be provided"
**Symptoms:**
```
ValueError: Google Cloud project ID must be provided via the --project-id flag or by setting the GOOGLE_CLOUD_PROJECT environment variable.
```

**Solutions:**
1. **Set environment variable:**
   ```bash
   export GOOGLE_CLOUD_PROJECT="your-project-id"
   ```

2. **Use command line flag:**
   ```bash
   python klue_re-gemini2_5flash.py --project-id "your-project-id"
   ```

3. **Verify project exists:**
   ```bash
   gcloud projects list
   ```

#### Error: "Permission denied" or "Authentication failed"
**Symptoms:**
```
google.auth.exceptions.DefaultCredentialsError: Could not automatically determine credentials.
```

**Solutions:**
1. **Login to gcloud:**
   ```bash
   gcloud auth login
   ```

2. **Set application default credentials:**
   ```bash
   gcloud auth application-default login
   ```

3. **Verify authentication:**
   ```bash
   gcloud auth list
   ```

#### Error: "Vertex AI API not enabled"
**Symptoms:**
```
google.api_core.exceptions.PermissionDenied: 403 Vertex AI API has not been used in project
```

**Solutions:**
1. **Enable Vertex AI API:**
   ```bash
   gcloud services enable aiplatform.googleapis.com
   ```

2. **Enable required APIs:**
   ```bash
   gcloud services enable compute.googleapis.com
   gcloud services enable storage.googleapis.com
   ```

### 2. Dataset Loading Issues

#### Error: "Failed to load dataset"
**Symptoms:**
```
datasets.exceptions.DatasetNotFoundError: Dataset klue not found
```

**Solutions:**
1. **Check internet connection**
2. **Verify Hugging Face access:**
   ```bash
   python3 -c "from datasets import load_dataset; print('HF access OK')"
   ```

3. **Try with smaller dataset first:**
   ```bash
   ./run test  # Uses only 10 samples
   ```

#### Error: "Dataset schema changed"
**Symptoms:**
```
KeyError: 'guid' not found in dataset
```

**Solutions:**
1. **Update datasets library:**
   ```bash
   pip install --upgrade datasets
   ```

2. **Check dataset structure:**
   ```python
   from datasets import load_dataset
   dataset = load_dataset('klue', 're', split='validation')
   print(dataset[0].keys())
   ```

### 3. Model and API Issues

#### Error: "Model not found" or "Invalid model name"
**Symptoms:**
```
google.api_core.exceptions.NotFound: 404 Model not found
```

**Solutions:**
1. **Verify model name:**
   ```python
   # In klue_re-gemini2_5flash.py
   model_name: str = "gemini-2.5-flash"  # Correct name
   ```

2. **Check model availability:**
   ```bash
   gcloud ai models list --region=us-central1
   ```

3. **Try different region:**
   ```bash
   python klue_re-gemini2_5flash.py --location "us-east1"
   ```

#### Error: "Rate limit exceeded" or "Quota exceeded"
**Symptoms:**
```
google.api_core.exceptions.ResourceExhausted: 429 Quota exceeded
```

**Solutions:**
1. **Increase sleep interval:**
   ```python
   # In klue_re-gemini2_5flash.py
   sleep_interval_between_api_calls: float = 0.1  # Increase from 0.04
   ```

2. **Use smaller batch sizes:**
   ```bash
   ./run custom 50  # Use fewer samples
   ```

3. **Check quotas:**
   ```bash
   gcloud compute regions describe us-central1
   ```

#### Error: "Safety filter blocked"
**Symptoms:**
```
Response candidate content has no parts (and thus no text). The candidate is likely blocked by the safety filters.
```

**Solutions:**
1. **Adjust safety settings:**
   ```python
   # In klue_re-gemini2_5flash.py
   def configure_safety_settings(self, threshold=HarmBlockThreshold.BLOCK_NONE):
   ```

2. **Check prompt content for sensitive words**
3. **Use different temperature settings**

### 4. Memory and Performance Issues

#### Error: "Out of memory" or "MemoryError"
**Symptoms:**
```
MemoryError: Unable to allocate array
```

**Solutions:**
1. **Reduce batch size:**
   ```python
   # In klue_re-gemini2_5flash.py
   batch_size: int = 1  # Ensure this is 1
   ```

2. **Use smaller sample sizes:**
   ```bash
   ./run custom 100  # Instead of full dataset
   ```

3. **Increase save interval:**
   ```python
   save_interval: int = 25  # Save more frequently
   ```

#### Error: "Process killed" or "Killed"
**Symptoms:**
```
Killed
```

**Solutions:**
1. **Monitor memory usage:**
   ```bash
   htop  # or top
   ```

2. **Use swap space:**
   ```bash
   sudo swapon --show
   ```

3. **Restart with smaller workload**

### 5. File and Permission Issues

#### Error: "Permission denied" for scripts
**Symptoms:**
```
bash: ./run: Permission denied
```

**Solutions:**
1. **Make scripts executable:**
   ```bash
   chmod +x run setup.sh test_setup.py get_errors.sh test_logging.sh verify_scripts.sh install_dependencies.sh
   ```

2. **Use verify_scripts.sh:**
   ```bash
   ./verify_scripts.sh -f
   ```

#### Error: "Directory not found" or "File not found"
**Symptoms:**
```
FileNotFoundError: [Errno 2] No such file or directory
```

**Solutions:**
1. **Create required directories:**
   ```bash
   mkdir -p logs benchmark_results result_analysis eval_dataset
   ```

2. **Run setup script:**
   ```bash
   ./setup.sh
   ```

### 6. Python and Dependency Issues

#### Error: "Module not found" or "ImportError"
**Symptoms:**
```
ModuleNotFoundError: No module named 'google.genai'
```

**Solutions:**
1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Use virtual environment:**
   ```bash
   ./install_dependencies.sh -v
   ```

3. **Verify installation:**
   ```bash
   python3 -c "import google.genai, datasets, pandas, tqdm; print('All OK')"
   ```

#### Error: "Python version too old"
**Symptoms:**
```
SyntaxError: f-string expressions cannot contain backslashes
```

**Solutions:**
1. **Check Python version:**
   ```bash
   python3 --version  # Should be 3.8+
   ```

2. **Install newer Python:**
   ```bash
   # On Ubuntu/Debian
   sudo apt update
   sudo apt install python3.9
   ```

### 7. Logging and Output Issues

#### Error: "No log files generated"
**Symptoms:**
```
No log files found in logs/ directory
```

**Solutions:**
1. **Check if logs directory exists:**
   ```bash
   ls -la logs/
   ```

2. **Run with verbose output:**
   ```bash
   ./run test 2>&1 | tee debug.log
   ```

3. **Check script permissions:**
   ```bash
   ./verify_scripts.sh
   ```

#### Error: "Error extraction failed"
**Symptoms:**
```
Error extraction failed
```

**Solutions:**
1. **Check log file format:**
   ```bash
   head -20 logs/klue_re_*.log
   ```

2. **Run error extraction manually:**
   ```bash
   ./get_errors.sh -f logs/your_log_file.log
   ```

3. **Verify awk is available:**
   ```bash
   which awk
   ```

## Performance Optimization

### Speed Up Processing
1. **Reduce sleep interval (if no rate limits):**
   ```python
   sleep_interval_between_api_calls: float = 0.02  # Reduce from 0.04
   ```

2. **Increase batch size (if memory allows):**
   ```python
   batch_size: int = 5  # Increase from 1
   ```

3. **Use faster regions:**
   ```bash
   python klue_re-gemini2_5flash.py --location "us-central1"
   ```

### Reduce Costs
1. **Use smaller sample sizes for testing**
2. **Implement early stopping for poor performance**
3. **Cache results to avoid re-computation**

## Debugging Tips

### Enable Verbose Logging
```python
# In klue_re-gemini2_5flash.py
logging.basicConfig(level=logging.DEBUG)
```

### Test Individual Components
```bash
# Test dataset loading
python3 -c "from datasets import load_dataset; dataset = load_dataset('klue', 're', split='validation'); print(f'Loaded {len(dataset)} samples')"

# Test Vertex AI connection
python3 -c "import google.genai; client = google.genai.Client(vertexai=True, project='your-project'); print('Connection OK')"
```

### Check Intermediate Results
```bash
# Monitor progress
tail -f logs/klue_re_*.log

# Check intermediate saves
ls -la benchmark_results/klue_re_*.json
```

## Getting Help

### Collect Debug Information
```bash
# System information
uname -a
python3 --version
pip list | grep -E "(google|datasets|pandas|tqdm)"

# Environment variables
env | grep -E "(GOOGLE|PYTHON|PATH)"

# Test results
./test_setup.py > debug_setup.log 2>&1
```

### Common Debug Commands
```bash
# Check all scripts exist and are executable
./verify_scripts.sh

# Test logging system
./test_logging.sh -s 3

# Extract errors from logs
./get_errors.sh -a

# Check disk space
df -h

# Check memory usage
free -h
```

### Contact Information
- **GitHub Issues**: Create an issue in the repository
- **Documentation**: Check README.md and other .md files
- **Community**: Check KLUE benchmark community forums

## Prevention

### Best Practices
1. **Always run setup first:**
   ```bash
   ./setup.sh
   ./test_setup.py
   ```

2. **Use virtual environments:**
   ```bash
   ./install_dependencies.sh -v
   ```

3. **Start with small tests:**
   ```bash
   ./run test  # Before running full benchmark
   ```

4. **Monitor resources:**
   ```bash
   htop  # Monitor CPU and memory
   ```

5. **Regular backups:**
   ```bash
   # Backup important results
   cp -r benchmark_results/ backup_$(date +%Y%m%d)/
   ``` 