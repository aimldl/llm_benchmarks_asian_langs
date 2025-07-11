# Vertex AI Setup Guide for KLUE DST

This guide provides detailed instructions for setting up Google Cloud Vertex AI to run the KLUE DST benchmark.

## Prerequisites

Before setting up Vertex AI, ensure you have:

1. **Google Cloud Account**: A valid Google Cloud account with billing enabled
2. **Google Cloud SDK**: The `gcloud` command-line tool installed
3. **Python Environment**: Python 3.8 or higher with pip
4. **Internet Connection**: Stable internet connection for API calls

## Step 1: Install Google Cloud SDK

### On Linux/macOS:
```bash
# Download and install the SDK
curl https://sdk.cloud.google.com | bash

# Restart your shell or run:
exec -l $SHELL

# Initialize gcloud
gcloud init
```

### On Windows:
1. Download the installer from: https://cloud.google.com/sdk/docs/install
2. Run the installer and follow the prompts
3. Open Command Prompt and run: `gcloud init`

### Verify Installation:
```bash
gcloud version
```

## Step 2: Create or Select a Google Cloud Project

### Create a New Project:
```bash
# Create a new project
gcloud projects create [PROJECT_ID] --name="KLUE DST Benchmark"

# Set the project as default
gcloud config set project [PROJECT_ID]
```

### Use Existing Project:
```bash
# List existing projects
gcloud projects list

# Set the project as default
gcloud config set project [PROJECT_ID]
```

### Set Environment Variable:
```bash
# Set the project ID as an environment variable
export GOOGLE_CLOUD_PROJECT=[PROJECT_ID]

# Add to your shell profile for persistence
echo 'export GOOGLE_CLOUD_PROJECT=[PROJECT_ID]' >> ~/.bashrc
source ~/.bashrc
```

## Step 3: Enable Required APIs

Enable the necessary Google Cloud APIs:

```bash
# Enable Vertex AI API
gcloud services enable aiplatform.googleapis.com

# Enable Cloud Storage API (if needed)
gcloud services enable storage.googleapis.com

# Verify enabled APIs
gcloud services list --enabled --filter="name:aiplatform.googleapis.com"
```

## Step 4: Set Up Authentication

### User Authentication:
```bash
# Authenticate with your Google account
gcloud auth login

# Set up application default credentials
gcloud auth application-default login
```

### Service Account (Recommended for Production):
```bash
# Create a service account
gcloud iam service-accounts create klue-dst-benchmark \
    --display-name="KLUE DST Benchmark Service Account"

# Grant necessary permissions
gcloud projects add-iam-policy-binding $GOOGLE_CLOUD_PROJECT \
    --member="serviceAccount:klue-dst-benchmark@$GOOGLE_CLOUD_PROJECT.iam.gserviceaccount.com" \
    --role="roles/aiplatform.user"

# Create and download key file
gcloud iam service-accounts keys create ~/klue-dst-key.json \
    --iam-account=klue-dst-benchmark@$GOOGLE_CLOUD_PROJECT.iam.gserviceaccount.com

# Set the key file as default
export GOOGLE_APPLICATION_CREDENTIALS=~/klue-dst-key.json
```

## Step 5: Configure Vertex AI Region

Set the Vertex AI region (us-central1 is recommended):

```bash
# Set the region
gcloud config set ai/region us-central1

# Or set as environment variable
export GOOGLE_CLOUD_REGION=us-central1
```

## Step 6: Verify Vertex AI Access

Test your Vertex AI setup:

```bash
# Test API access
curl -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  https://us-central1-aiplatform.googleapis.com/v1/projects/$GOOGLE_CLOUD_PROJECT/locations/us-central1/models

# Test with Python
python3 -c "
from google import genai
client = genai.Client(vertexai=True, project='$GOOGLE_CLOUD_PROJECT', location='us-central1')
print('✓ Vertex AI client initialized successfully')
"
```

## Step 7: Check Quotas and Limits

### View Current Quotas:
```bash
# Check Vertex AI quotas
gcloud compute regions describe us-central1 --project=$GOOGLE_CLOUD_PROJECT

# Check specific quotas
gcloud compute regions describe us-central1 --project=$GOOGLE_CLOUD_PROJECT \
  --format="value(quotas[metric=CPUS].limit,quotas[metric=CPUS].usage)"
```

### Request Quota Increase (if needed):
1. Go to Google Cloud Console: https://console.cloud.google.com
2. Navigate to IAM & Admin > Quotas
3. Search for "Vertex AI" or "AI Platform"
4. Select the quota you need to increase
5. Click "Edit Quotas" and submit a request

## Step 8: Install Python Dependencies

Install the required Python packages:

```bash
# Navigate to the klue_dst directory
cd klue_dst

# Install dependencies
./install_dependencies.sh

# Or install manually
pip3 install google-genai>=0.3.0
pip3 install datasets>=2.14.0
pip3 install pandas>=2.0.0
pip3 install tqdm>=4.65.0
pip3 install google-cloud-aiplatform>=1.35.0
pip3 install google-auth>=2.17.0
```

## Step 9: Test the Setup

Run the test script to verify everything is working:

```bash
# Test the environment
python3 test_setup.py

# Expected output should show:
# ✓ All tests passed! Environment is ready for KLUE DST benchmarking.
```

## Step 10: Run a Test Benchmark

Test the complete setup with a small benchmark:

```bash
# Run a test with 10 samples
./run test

# Check the results
ls -la logs/
ls -la benchmark_results/
```

## Configuration Options

### Environment Variables

Set these environment variables for customization:

```bash
# Required
export GOOGLE_CLOUD_PROJECT="your-project-id"

# Optional
export GOOGLE_CLOUD_REGION="us-central1"  # Default: us-central1
export GOOGLE_APPLICATION_CREDENTIALS="path/to/service-account-key.json"  # For service accounts
```

### Script Parameters

The benchmark script supports various parameters:

```bash
# Basic usage
python3 klue_dst-gemini2_5flash.py --project-id $GOOGLE_CLOUD_PROJECT

# With custom parameters
python3 klue_dst-gemini2_5flash.py \
  --project-id $GOOGLE_CLOUD_PROJECT \
  --location us-central1 \
  --max-samples 100 \
  --temperature 0.1 \
  --max-tokens 2048
```

## Troubleshooting

### Common Issues

1. **Authentication Errors**
   ```bash
   # Re-authenticate
   gcloud auth login
   gcloud auth application-default login
   
   # Check current account
   gcloud auth list
   ```

2. **Permission Errors**
   ```bash
   # Check project permissions
   gcloud projects get-iam-policy $GOOGLE_CLOUD_PROJECT
   
   # Ensure you have aiplatform.user role
   ```

3. **API Not Enabled**
   ```bash
   # Enable the API
   gcloud services enable aiplatform.googleapis.com
   
   # Wait a few minutes for propagation
   ```

4. **Quota Exceeded**
   ```bash
   # Check current usage
   gcloud compute regions describe us-central1 --project=$GOOGLE_CLOUD_PROJECT
   
   # Request quota increase in Google Cloud Console
   ```

### Debug Commands

```bash
# Test API access
curl -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  https://us-central1-aiplatform.googleapis.com/v1/projects/$GOOGLE_CLOUD_PROJECT/locations/us-central1/models

# Check project configuration
gcloud config list

# Verify authentication
gcloud auth print-access-token

# Test Vertex AI client
python3 -c "
from google import genai
try:
    client = genai.Client(vertexai=True, project='$GOOGLE_CLOUD_PROJECT', location='us-central1')
    print('✓ Vertex AI setup successful')
except Exception as e:
    print(f'✗ Error: {e}')
"
```

## Cost Considerations

### Pricing

Vertex AI pricing for Gemini 2.5 Flash:
- **Input tokens**: $0.000075 per 1K tokens
- **Output tokens**: $0.0003 per 1K tokens

### Cost Estimation

For the KLUE DST benchmark:
- **Average input length**: ~500 tokens per sample
- **Average output length**: ~100 tokens per sample
- **Cost per 1000 samples**: ~$0.04

### Cost Optimization

1. **Use smaller sample sizes for testing**
2. **Monitor usage in Google Cloud Console**
3. **Set up billing alerts**
4. **Use appropriate regions for lower latency**

## Security Best Practices

1. **Use Service Accounts**: For production use, use service accounts instead of user credentials
2. **Limit Permissions**: Grant only necessary permissions to service accounts
3. **Rotate Keys**: Regularly rotate service account keys
4. **Monitor Usage**: Set up monitoring and alerting for unusual usage patterns
5. **Secure Storage**: Store credentials securely and never commit them to version control

## Next Steps

After completing the setup:

1. **Run a test benchmark**: `./run test`
2. **Check results**: Review the generated log files and results
3. **Scale up**: Run larger benchmarks as needed
4. **Monitor costs**: Keep track of usage and costs in Google Cloud Console

## Support

If you encounter issues:

1. Check the troubleshooting section in this guide
2. Review the `TROUBLESHOOTING.md` file
3. Check Google Cloud documentation: https://cloud.google.com/vertex-ai/docs
4. Contact Google Cloud support if needed 