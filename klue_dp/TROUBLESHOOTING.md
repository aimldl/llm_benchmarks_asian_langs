# TROUBLESHOOTING

This document provides solutions to common issues encountered when running the KLUE Dependency Parsing benchmark.

## Common Issues and Solutions

### 1. Authentication and Project Setup

#### Issue: "Google Cloud project ID must be provided"
```
Error: GOOGLE_CLOUD_PROJECT environment variable is not set
Please set it with: export GOOGLE_CLOUD_PROJECT='your-project-id'
```

**Solution:**
```bash
# Set project ID
export GOOGLE_CLOUD_PROJECT="your-project-id"

# Or add to .bashrc for permanent setting
echo 'export GOOGLE_CLOUD_PROJECT="your-project-id"' >> ~/.bashrc
source ~/.bashrc
```

#### Issue: "No credentials found"
```
✗ No credentials found
```

**Solution:**
```bash
# Set up application default credentials
gcloud auth application-default login

# Or use service account
export GOOGLE_APPLICATION_CREDENTIALS="path/to/service-account-key.json"
```

#### Issue: "API not enabled"
```
Error: Vertex AI API is not enabled for this project
```

**Solution:**
```bash
# Enable Vertex AI API
gcloud services enable aiplatform.googleapis.com
```

### 2. Dataset Loading Issues

#### Issue: "Failed to load KLUE dataset"
```
✗ Failed to load KLUE dataset: Connection error
```

**Solution:**
- Check internet connection
- Verify Hugging Face Hub access
- Try loading a smaller subset first: `--max-samples 10`

#### Issue: "Dataset schema has changed"
```
❌ A key was not found in the dataset item: 'words'
```

**Solution:**
- Update the datasets library: `pip install --upgrade datasets`
- Check if the KLUE dataset structure has been updated

### 3. Model Response Issues

#### Issue: "Cannot get the response text"
```
ERROR - Prediction failed: Cannot get the response text.
Response candidate content has no parts (and thus no text).
```

**Causes and Solutions:**

1. **Safety Filter Blocking:**
   - The response was blocked by safety filters
   - **Solution:** Adjust safety settings in the code or use different prompts

2. **Token Limit Exceeded:**
   - Response exceeded max_tokens limit
   - **Solution:** Increase max_tokens: `--max-tokens 8192`

3. **Rate Limiting:**
   - Too many requests to Vertex AI
   - **Solution:** Increase sleep interval between calls

#### Issue: "MAX_TOKENS" finish reason
```
"finish_reason": "MAX_TOKENS"
```

**Solution:**
```bash
# Increase max tokens
python klue_dp-gemini2_5flash.py --max-tokens 8192 --project-id "your-project-id"
```

### 4. Performance Issues

#### Issue: Slow processing
```
Average time per sample: 5.0 seconds
```

**Solutions:**
- Reduce prompt complexity
- Use smaller max_tokens
- Check network latency to Vertex AI region
- Consider using a different Vertex AI location

#### Issue: Memory errors
```
MemoryError: Unable to allocate array
```

**Solutions:**
- Reduce batch size
- Use `--no-save-predictions` to save memory
- Process smaller subsets: `--max-samples 100`

### 5. Dependency Parsing Specific Issues

#### Issue: Low UAS/LAS scores
```
UAS: 0.4500 | LAS: 0.3200
```

**Possible Causes:**
1. **Complex Korean sentences** with multiple clauses
2. **Ambiguous dependencies** in the text
3. **Model limitations** with Korean syntax
4. **Prompt not detailed enough**

**Solutions:**
- Review error analysis in the output files
- Check per-POS performance breakdown
- Consider prompt engineering improvements
- Test with simpler sentences first

#### Issue: Incorrect POS tag interpretation
```
Error: Incorrect POS tag interpretation
```

**Solutions:**
- Verify POS tag definitions in the prompt
- Check if the model understands Korean POS tags
- Review the POS_TAGS dictionary in the code

#### Issue: Wrong dependency direction
```
Error: Wrong dependency direction
```

**Solutions:**
- Korean dependency parsing can be complex due to agglutination
- Check if the model understands Korean grammar structure
- Review dependency relation definitions in the prompt

### 6. Logging and Output Issues

#### Issue: Log files not created
```
Log files will be saved to the 'logs' directory
```

**Solution:**
```bash
# Check if logs directory exists
ls -la logs/

# Create logs directory if missing
mkdir -p logs
```

#### Issue: Error extraction not working
```
No error analysis section found in the log.
```

**Solution:**
- Check if the benchmark completed successfully
- Verify that errors occurred during execution
- Review the log file content manually

### 7. Script Execution Issues

#### Issue: "Permission denied"
```
bash: ./run: Permission denied
```

**Solution:**
```bash
# Make scripts executable
chmod +x run setup.sh install_dependencies.sh get_errors.sh test_logging.sh verify_scripts.sh
```

#### Issue: "Script not found"
```
Error: klue_dp-gemini2_5flash.py not found in current directory
```

**Solution:**
```bash
# Check if you're in the correct directory
pwd
ls -la *.py

# Navigate to the correct directory
cd /path/to/klue_dp
```

### 8. Environment Issues

#### Issue: Python package import errors
```
ModuleNotFoundError: No module named 'google.genai'
```

**Solution:**
```bash
# Install dependencies
pip install -r requirements.txt

# Or install manually
pip install google-genai datasets pandas tqdm google-cloud-aiplatform
```

#### Issue: Python version compatibility
```
SyntaxError: invalid syntax
```

**Solution:**
- Ensure Python 3.8+ is installed
- Check Python version: `python --version`
- Use virtual environment if needed

### 9. Network and Connectivity Issues

#### Issue: Connection timeout
```
ConnectionError: Connection timed out
```

**Solutions:**
- Check internet connection
- Verify firewall settings
- Try different Vertex AI regions
- Check if behind corporate proxy

#### Issue: SSL certificate errors
```
SSLError: Certificate verify failed
```

**Solution:**
```bash
# Update certificates
pip install --upgrade certifi

# Or temporarily disable SSL verification (not recommended for production)
export PYTHONHTTPSVERIFY=0
```

### 10. Resource Limitations

#### Issue: Quota exceeded
```
Quota exceeded for quota group 'default' and limit 'requests per minute'
```

**Solutions:**
- Reduce request frequency
- Increase sleep interval between calls
- Request quota increase from Google Cloud
- Use different project with higher quotas

#### Issue: Billing issues
```
Billing is not enabled for this project
```

**Solution:**
- Enable billing for your Google Cloud project
- Set up billing alerts
- Monitor usage in Google Cloud Console

## Debugging Tips

### 1. Enable Verbose Logging
```python
# In the Python script, change logging level
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
```

### 2. Test with Minimal Data
```bash
# Test with just 1 sample
python klue_dp-gemini2_5flash.py --max-samples 1 --project-id "your-project-id"
```

### 3. Check Intermediate Results
```bash
# Look for intermediate result files
ls -la benchmark_results/klue_dp_results_*.json
```

### 4. Analyze Error Patterns
```bash
# Extract errors from results
./get_errors.sh benchmark_results/klue_dp_results_[timestamp].csv
```

### 5. Test Logging Functionality
```bash
# Test logging without running the full benchmark
./test_logging.sh
```

## Getting Help

If you encounter issues not covered in this troubleshooting guide:

1. **Check the logs**: Review the `.log` and `.err` files in the `logs/` directory
2. **Review error analysis**: Check the error analysis files in `benchmark_results/`
3. **Test setup**: Run `./setup.sh test` to verify your environment
4. **Check documentation**: Review README.md and ABOUT_KLUE_DP.md
5. **Google Cloud issues**: Check Google Cloud documentation and support
6. **KLUE dataset issues**: Check the KLUE benchmark repository

## Performance Optimization

### For Better Performance:
1. **Use appropriate max_tokens**: 4096-8192 for DP task
2. **Optimize prompts**: Make them detailed but concise
3. **Use appropriate temperature**: 0.1 for consistent results
4. **Monitor rate limits**: Adjust sleep intervals if needed
5. **Use efficient regions**: Choose Vertex AI regions close to your location

### For Faster Testing:
1. **Use small sample sizes**: Start with 10-50 samples
2. **Skip detailed predictions**: Use `--no-save-predictions`
3. **Use test mode**: `./run test` for quick validation
4. **Monitor progress**: Check the progress bar and timing information 