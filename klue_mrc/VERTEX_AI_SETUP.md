# Vertex AI Setup Guide for KLUE MRC Benchmark

This guide provides detailed instructions for setting up Google Cloud Vertex AI to run the KLUE Machine Reading Comprehension benchmark.

## Prerequisites

Before setting up Vertex AI, ensure you have:

1. **Google Cloud Account**: A valid Google Cloud account with billing enabled
2. **Google Cloud CLI**: The `gcloud` command-line tool installed
3. **Python Environment**: Python 3.8+ with pip installed
4. **Internet Connection**: Stable internet connection for API access

## Step 1: Install Google Cloud CLI

### On Linux/macOS:
```bash
# Download and install gcloud CLI
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
gcloud --version
```

## Step 2: Create or Select a Google Cloud Project

### Create a New Project:
```bash
# Create a new project (replace with your desired project ID)
gcloud projects create klue-mrc-benchmark --name="KLUE MRC Benchmark"

# Set the project as active
gcloud config set project klue-mrc-benchmark
```

### Use Existing Project:
```bash
# List your projects
gcloud projects list

# Set an existing project as active
gcloud config set project YOUR_EXISTING_PROJECT_ID
```

### Verify Project:
```bash
# Check current project
gcloud config get-value project

# Should output your project ID
```

## Step 3: Enable Required APIs

Enable the Vertex AI API and related services:

```bash
# Enable Vertex AI API
gcloud services enable aiplatform.googleapis.com

# Enable other required APIs
gcloud services enable compute.googleapis.com
gcloud services enable storage.googleapis.com

# Verify APIs are enabled
gcloud services list --enabled --filter="name:aiplatform.googleapis.com"
```

## Step 4: Set Up Authentication

Choose one of the following authentication methods:

### Method A: Application Default Credentials (Recommended for Development)

```bash
# Set up application default credentials
gcloud auth application-default login

# This will open a browser window for authentication
# Follow the prompts to complete the setup
```

### Method B: Service Account (Recommended for Production)

```bash
# Create a service account
gcloud iam service-accounts create klue-mrc-benchmark \
    --display-name="KLUE MRC Benchmark Service Account"

# Get your project ID
PROJECT_ID=$(gcloud config get-value project)

# Grant necessary permissions
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:klue-mrc-benchmark@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/aiplatform.user"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:klue-mrc-benchmark@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/storage.objectViewer"

# Create and download the service account key
gcloud iam service-accounts keys create klue-mrc-key.json \
    --iam-account=klue-mrc-benchmark@$PROJECT_ID.iam.gserviceaccount.com

# Set the environment variable
export GOOGLE_APPLICATION_CREDENTIALS="$(pwd)/klue-mrc-key.json"

# Verify the key file was created
ls -la klue-mrc-key.json
```

### Method C: User Account Authentication

```bash
# Authenticate with your Google account
gcloud auth login

# Set up application default credentials
gcloud auth application-default login
```

## Step 5: Set Environment Variables

Set the required environment variables:

```bash
# Get your project ID
PROJECT_ID=$(gcloud config get-value project)

# Set the project ID environment variable
export GOOGLE_CLOUD_PROJECT="$PROJECT_ID"

# Verify the environment variable is set
echo "GOOGLE_CLOUD_PROJECT: $GOOGLE_CLOUD_PROJECT"
```

### Make Environment Variables Permanent

Add the environment variables to your shell profile:

```bash
# For bash (add to ~/.bashrc)
echo "export GOOGLE_CLOUD_PROJECT=\"$PROJECT_ID\"" >> ~/.bashrc

# For zsh (add to ~/.zshrc)
echo "export GOOGLE_CLOUD_PROJECT=\"$PROJECT_ID\"" >> ~/.zshrc

# If using service account, also add:
echo "export GOOGLE_APPLICATION_CREDENTIALS=\"$(pwd)/klue-mrc-key.json\"" >> ~/.bashrc

# Reload the profile
source ~/.bashrc  # or source ~/.zshrc
```

## Step 6: Verify Vertex AI Setup

Test that Vertex AI is properly configured:

```bash
# Test Vertex AI access
python -c "
from google.cloud import aiplatform
import os

project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
if not project_id:
    print('❌ GOOGLE_CLOUD_PROJECT not set')
    exit(1)

try:
    aiplatform.init(project=project_id)
    print(f'✅ Vertex AI initialized successfully with project: {project_id}')
except Exception as e:
    print(f'❌ Vertex AI initialization failed: {e}')
"
```

## Step 7: Check Quotas and Limits

Verify your Vertex AI quotas:

```bash
# Check Vertex AI quotas
gcloud compute regions describe us-central1 --format="value(quotas)"

# Or check specific quotas
gcloud compute regions describe us-central1 --format="value(quotas[0].limit,quotas[0].usage)"
```

### Common Quota Limits:
- **Requests per minute**: 600 (default)
- **Requests per day**: 86,400 (default)
- **Concurrent requests**: 10 (default)

### Request Quota Increase (if needed):
1. Go to Google Cloud Console
2. Navigate to IAM & Admin > Quotas
3. Search for "Vertex AI"
4. Select the quota you want to increase
5. Click "Edit Quotas" and submit a request

## Step 8: Test the Complete Setup

Run the test setup script to verify everything is working:

```bash
# Navigate to the KLUE MRC directory
cd klue_mrc

# Run the test setup
python test_setup.py
```

Expected output:
```
============================================================
KLUE MRC Benchmark Setup Test (Vertex AI)
============================================================
Testing package imports...
✓ google.cloud.aiplatform
✓ vertexai
✓ datasets
✓ pandas
✓ tqdm
✓ huggingface_hub
✓ google.auth

✅ All packages imported successfully!

Testing environment variables...
✓ GOOGLE_CLOUD_PROJECT: your-project-id
⚠ GOOGLE_APPLICATION_CREDENTIALS: Not set (using default credentials)

Testing KLUE MRC dataset loading...
✓ KLUE mrc dataset for MRC loaded successfully
  - Train samples: 18000
  - Validation samples: 2000
  - Sample from validation set:
    - Title: [sample title]
    - Context: [sample context]...
    - Question: [sample question]
    - Answers: [sample answers]
    - Is Impossible: False

Testing Vertex AI authentication...
✓ Credentials found
  - Project: your-project-id
  - Credentials type: ApplicationDefaultCredentials
✓ Vertex AI initialization works

============================================================
Test Summary
============================================================
✅ All tests passed! Your setup is ready.
```

## Step 9: Run a Quick Test

Test the benchmark with a small sample:

```bash
# Run with 5 samples to test
./run custom 5

# Check the results
ls -la logs/
ls -la benchmark_results/
```

## Troubleshooting

### Common Issues:

#### 1. "Project not found"
```bash
# Verify your project exists
gcloud projects list

# Set the correct project
gcloud config set project YOUR_PROJECT_ID
```

#### 2. "API not enabled"
```bash
# Enable the required APIs
gcloud services enable aiplatform.googleapis.com
gcloud services enable compute.googleapis.com
```

#### 3. "Insufficient permissions"
```bash
# Check your current permissions
gcloud auth list

# Re-authenticate if needed
gcloud auth application-default login
```

#### 4. "Quota exceeded"
```bash
# Check your current usage
gcloud compute regions describe us-central1 --format="value(quotas)"

# Request quota increase if needed
# Go to: https://console.cloud.google.com/iam-admin/quotas
```

#### 5. "Billing not enabled"
```bash
# Enable billing for your project
# Go to: https://console.cloud.google.com/billing
```

## Cost Considerations

### Vertex AI Pricing (as of 2024):
- **Gemini 2.5 Flash**: ~$0.0005 per 1K input tokens, ~$0.0015 per 1K output tokens
- **API Requests**: Additional charges for API calls

### Estimated Costs for KLUE MRC:
- **Test run (10 samples)**: ~$0.01-0.05
- **Full validation set (2000 samples)**: ~$2-10
- **Full test set (2000 samples)**: ~$2-10

### Cost Optimization:
```bash
# Use smaller samples for testing
./run custom 10

# Monitor usage in Google Cloud Console
# Set up billing alerts
```

## Security Best Practices

### For Production Use:

1. **Use Service Accounts**: Instead of user credentials
2. **Limit Permissions**: Grant only necessary roles
3. **Rotate Keys**: Regularly rotate service account keys
4. **Monitor Usage**: Set up alerts for unusual activity
5. **Secure Storage**: Store credentials securely

### Example Secure Setup:
```bash
# Create a dedicated service account with minimal permissions
gcloud iam service-accounts create klue-mrc-limited \
    --display-name="KLUE MRC Limited Access"

# Grant only necessary permissions
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:klue-mrc-limited@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/aiplatform.user"

# Store key securely
gcloud iam service-accounts keys create /secure/path/key.json \
    --iam-account=klue-mrc-limited@$PROJECT_ID.iam.gserviceaccount.com

# Set restrictive permissions
chmod 600 /secure/path/key.json
```

## Next Steps

After completing the setup:

1. **Run the benchmark**: `./run test` or `./run full`
2. **Monitor performance**: Check logs and results
3. **Analyze results**: Review the generated metrics and analysis files
4. **Optimize**: Adjust parameters based on results
5. **Scale up**: Run larger benchmarks as needed

## Additional Resources

- [Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs)
- [Google Cloud CLI Documentation](https://cloud.google.com/sdk/docs)
- [Vertex AI Pricing](https://cloud.google.com/vertex-ai/pricing)
- [IAM Best Practices](https://cloud.google.com/iam/docs/best-practices)
- [Billing and Quotas](https://cloud.google.com/billing/docs) 