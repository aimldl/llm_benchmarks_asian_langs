# Google Cloud Vertex AI Setup Guide for KLUE NER

This guide provides step-by-step instructions for setting up Google Cloud Vertex AI to run the KLUE Named Entity Recognition benchmark.

## Prerequisites

Before starting, ensure you have:

1. **Google Cloud Account**: A valid Google Cloud account with billing enabled
2. **Python 3.8+**: Python 3.8 or higher installed
3. **Google Cloud CLI**: gcloud command-line tool installed
4. **Internet Connection**: For downloading dependencies and datasets

## Step 1: Install Google Cloud CLI

### On Linux/macOS:
```bash
# Download and install gcloud CLI
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Or install via package manager (Ubuntu/Debian)
sudo apt-get install google-cloud-cli
```

### On Windows:
1. Download the installer from [Google Cloud CLI](https://cloud.google.com/sdk/docs/install)
2. Run the installer and follow the prompts
3. Restart your command prompt

### Verify Installation:
```bash
gcloud --version
```

## Step 2: Create a Google Cloud Project

### Option 1: Create New Project
```bash
# Create a new project
gcloud projects create [PROJECT_ID] --name="KLUE NER Benchmark"

# Set the project as default
gcloud config set project [PROJECT_ID]
```

### Option 2: Use Existing Project
```bash
# List existing projects
gcloud projects list

# Set an existing project as default
gcloud config set project [EXISTING_PROJECT_ID]
```

### Verify Project:
```bash
# Check current project
gcloud config get-value project
```

## Step 3: Enable Required APIs

Enable the necessary APIs for Vertex AI:

```bash
# Enable Vertex AI API
gcloud services enable aiplatform.googleapis.com

# Enable other required APIs
gcloud services enable compute.googleapis.com
gcloud services enable storage.googleapis.com
```

### Verify API Enablement:
```bash
# List enabled APIs
gcloud services list --enabled --filter="name:aiplatform"
```

## Step 4: Set Up Authentication

### Option 1: User Authentication (Recommended for Development)
```bash
# Login with your Google account
gcloud auth login

# Set up application default credentials
gcloud auth application-default login
```

### Option 2: Service Account (Recommended for Production)
```bash
# Create a service account
gcloud iam service-accounts create klue-ner-benchmark \
    --display-name="KLUE NER Benchmark Service Account"

# Grant necessary roles
gcloud projects add-iam-policy-binding [PROJECT_ID] \
    --member="serviceAccount:klue-ner-benchmark@[PROJECT_ID].iam.gserviceaccount.com" \
    --role="roles/aiplatform.user"

gcloud projects add-iam-policy-binding [PROJECT_ID] \
    --member="serviceAccount:klue-ner-benchmark@[PROJECT_ID].iam.gserviceaccount.com" \
    --role="roles/storage.objectViewer"

# Create and download key file
gcloud iam service-accounts keys create ~/klue-ner-key.json \
    --iam-account=klue-ner-benchmark@[PROJECT_ID].iam.gserviceaccount.com

# Set environment variable
export GOOGLE_APPLICATION_CREDENTIALS=~/klue-ner-key.json
```

## Step 5: Configure Environment Variables

Set the required environment variables:

```bash
# Set your project ID
export GOOGLE_CLOUD_PROJECT='[YOUR_PROJECT_ID]'

# Set Vertex AI location (recommended: us-central1)
export GOOGLE_CLOUD_LOCATION='us-central1'

# Verify environment variables
echo "Project ID: $GOOGLE_CLOUD_PROJECT"
echo "Location: $GOOGLE_CLOUD_LOCATION"
```

### Add to Shell Profile (Optional)
Add these to your `~/.bashrc`, `~/.zshrc`, or equivalent:

```bash
export GOOGLE_CLOUD_PROJECT='[YOUR_PROJECT_ID]'
export GOOGLE_CLOUD_LOCATION='us-central1'
```

## Step 6: Install Python Dependencies

Install the required Python packages:

```bash
# Install dependencies
./install_dependencies.sh install

# Or manually install
pip install -r requirements.txt
```

### Verify Installation:
```bash
# Check installed packages
./install_dependencies.sh check

# Or manually verify
python -c "import google.genai; print('google.genai installed')"
python -c "import datasets; print('datasets installed')"
python -c "import pandas; print('pandas installed')"
```

## Step 7: Test the Setup

Run the setup verification:

```bash
# Run comprehensive setup test
./test_setup.py

# Test logging functionality
./test_logging.sh test

# Verify all scripts
./verify_scripts.sh check
```

## Step 8: Run a Small Test

Test the complete setup with a small sample:

```bash
# Run test with 10 samples
./run test
```

This should:
1. Load the KLUE NER dataset
2. Initialize Vertex AI connection
3. Run entity recognition on 10 samples
4. Save results and logs
5. Display performance metrics

## Configuration Options

### Model Configuration

The benchmark uses these default settings:

```python
# In klue_ner-gemini2_5flash.py
config = BenchmarkConfig(
    model_name="gemini-2.5-flash",
    max_tokens=2048,           # Increased for NER task
    temperature=0.1,
    location="us-central1",
    sleep_interval_between_api_calls=0.04
)
```

### Custom Configuration

You can modify these settings:

```bash
# Run with custom parameters
python klue_ner-gemini2_5flash.py \
    --project-id [PROJECT_ID] \
    --max-samples 100 \
    --temperature 0.2 \
    --max-tokens 1024
```

## Billing and Quotas

### Check Billing Status
```bash
# Verify billing is enabled
gcloud billing accounts list

# Link billing account to project (if needed)
gcloud billing projects link [PROJECT_ID] --billing-account=[BILLING_ACCOUNT_ID]
```

### Monitor Usage
```bash
# Check Vertex AI quotas
gcloud compute regions describe us-central1 --format="value(quotas)"

# Monitor API usage in Google Cloud Console
# Go to: https://console.cloud.google.com/apis/credentials
```

### Cost Estimation

Typical costs for KLUE NER benchmark:
- **Small test (10 samples)**: ~$0.01-0.05
- **Medium test (100 samples)**: ~$0.10-0.50
- **Full benchmark (1000+ samples)**: ~$1.00-5.00

*Note: Costs vary based on input length and model usage*

## Troubleshooting

### Common Issues

1. **Authentication Errors**:
   ```bash
   # Re-authenticate
   gcloud auth login
   gcloud auth application-default login
   ```

2. **API Not Enabled**:
   ```bash
   # Enable Vertex AI API
   gcloud services enable aiplatform.googleapis.com
   ```

3. **Project Not Set**:
   ```bash
   # Set project
   gcloud config set project [PROJECT_ID]
   export GOOGLE_CLOUD_PROJECT=[PROJECT_ID]
   ```

4. **Permission Denied**:
   ```bash
   # Check IAM roles
   gcloud projects get-iam-policy [PROJECT_ID]
   ```

### Debugging Commands

```bash
# Check gcloud configuration
gcloud config list

# Test Vertex AI access
gcloud ai models list --region=us-central1

# Check project status
gcloud projects describe [PROJECT_ID]

# Verify APIs
gcloud services list --enabled --filter="name:aiplatform"
```

## Security Considerations

### Best Practices

1. **Use Service Accounts**: For production environments
2. **Limit Permissions**: Grant only necessary IAM roles
3. **Rotate Keys**: Regularly rotate service account keys
4. **Monitor Usage**: Set up billing alerts and usage monitoring
5. **Secure Credentials**: Never commit credentials to version control

### IAM Roles

Minimum required roles:
- `roles/aiplatform.user` - Use Vertex AI models
- `roles/storage.objectViewer` - Read datasets (if stored in GCS)

## Performance Optimization

### API Quotas

Vertex AI has rate limits:
- **Requests per minute**: Varies by project tier
- **Concurrent requests**: Limited per project
- **Token limits**: Per request and per minute

### Optimization Tips

1. **Rate Limiting**: Built-in 0.04s delay between requests
2. **Batch Processing**: Process samples sequentially
3. **Error Handling**: Automatic retry for transient errors
4. **Monitoring**: Track API usage and costs

## Next Steps

After successful setup:

1. **Run Full Benchmark**: `./run full`
2. **Analyze Results**: Check `benchmark_results/` directory
3. **Review Logs**: Examine `logs/` directory for execution details
4. **Error Analysis**: Use `./get_errors.sh` for detailed error information
5. **Performance Tuning**: Adjust parameters based on results

## Support Resources

### Documentation
- [Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs)
- [Google Generative AI](https://cloud.google.com/vertex-ai/generative-ai)
- [KLUE Benchmark](https://github.com/KLUE-benchmark/KLUE)

### Community
- [Google Cloud Community](https://cloud.google.com/community)
- [Stack Overflow](https://stackoverflow.com/questions/tagged/google-cloud-ai-platform)
- [GitHub Issues](https://github.com/KLUE-benchmark/KLUE/issues)

### Billing Support
- [Google Cloud Billing](https://cloud.google.com/billing/docs)
- [Cost Optimization](https://cloud.google.com/architecture/cost-optimization-on-google-cloud) 