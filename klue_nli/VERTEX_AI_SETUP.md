# Vertex AI Setup Guide for KLUE NLI Benchmark

This guide provides detailed instructions for setting up Google Cloud Vertex AI to run the KLUE Natural Language Inference benchmark.

## Prerequisites

1. **Google Cloud Account**: You need a Google Cloud account with billing enabled
2. **Google Cloud CLI**: Install the Google Cloud CLI (gcloud) for command-line operations
3. **Python Environment**: Python 3.7+ with pip installed

## Step 1: Install Google Cloud CLI

### For Linux/macOS:
```bash
# Download and install gcloud CLI
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Or install via package manager (Ubuntu/Debian)
sudo apt-get install google-cloud-cli
```

### For Windows:
Download and install from: https://cloud.google.com/sdk/docs/install

### Verify Installation:
```bash
gcloud version
```

## Step 2: Create a Google Cloud Project

### Option A: Create a New Project
```bash
# Create a new project
gcloud projects create YOUR_PROJECT_ID --name="KLUE NLI Benchmark"

# Set the project as default
gcloud config set project YOUR_PROJECT_ID
```

### Option B: Use Existing Project
```bash
# List existing projects
gcloud projects list

# Set an existing project as default
gcloud config set project YOUR_EXISTING_PROJECT_ID
```

## Step 3: Enable Required APIs

Enable the Vertex AI API and other necessary services:

```bash
# Enable Vertex AI API
gcloud services enable aiplatform.googleapis.com

# Enable other useful APIs
gcloud services enable compute.googleapis.com
gcloud services enable storage.googleapis.com
```

## Step 4: Set Up Authentication

Choose one of the following authentication methods:

### Method A: Application Default Credentials (Recommended for Development)

```bash
# Authenticate with your Google account
gcloud auth login

# Set up application default credentials
gcloud auth application-default login

# Verify authentication
gcloud auth list
```

### Method B: Service Account (Recommended for Production)

```bash
# Create a service account
gcloud iam service-accounts create klue-nli-benchmark \
    --display-name="KLUE NLI Benchmark Service Account"

# Grant necessary permissions
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:klue-nli-benchmark@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/aiplatform.user"

# Create and download key
gcloud iam service-accounts keys create key.json \
    --iam-account=klue-nli-benchmark@YOUR_PROJECT_ID.iam.gserviceaccount.com

# Set environment variable
export GOOGLE_APPLICATION_CREDENTIALS="path/to/key.json"
```

### Method C: Workload Identity (For GKE)

If running on Google Kubernetes Engine:

```bash
# Enable Workload Identity
gcloud container clusters update YOUR_CLUSTER_NAME \
    --region=YOUR_REGION \
    --workload-pool=YOUR_PROJECT_ID.svc.id.goog

# Create Kubernetes service account
kubectl create serviceaccount klue-nli-benchmark

# Bind the service account to the Google service account
gcloud iam service-accounts add-iam-policy-binding \
    klue-nli-benchmark@YOUR_PROJECT_ID.iam.gserviceaccount.com \
    --role roles/iam.workloadIdentityUser \
    --member "serviceAccount:YOUR_PROJECT_ID.svc.id.goog[default/klue-nli-benchmark]"

# Annotate the Kubernetes service account
kubectl annotate serviceaccount klue-nli-benchmark \
    iam.gke.io/gcp-service-account=klue-nli-benchmark@YOUR_PROJECT_ID.iam.gserviceaccount.com
```

## Step 5: Set Environment Variables

Set the project ID as an environment variable:

```bash
# Set project ID
export GOOGLE_CLOUD_PROJECT="YOUR_PROJECT_ID"

# Verify it's set
echo $GOOGLE_CLOUD_PROJECT
```

### Permanent Setup (Optional)

Add to your shell profile for permanent setup:

```bash
# For bash
echo 'export GOOGLE_CLOUD_PROJECT="YOUR_PROJECT_ID"' >> ~/.bashrc
source ~/.bashrc

# For zsh
echo 'export GOOGLE_CLOUD_PROJECT="YOUR_PROJECT_ID"' >> ~/.zshrc
source ~/.zshrc
```

## Step 6: Verify Setup

Test your Vertex AI setup:

```bash
# Test Vertex AI initialization
python -c "
from google.cloud import aiplatform
aiplatform.init(project='$GOOGLE_CLOUD_PROJECT')
print('Vertex AI setup successful!')
"
```

## Step 7: Configure Billing

Ensure billing is enabled for your project:

```bash
# Check billing status
gcloud billing projects describe YOUR_PROJECT_ID

# If billing is not enabled, link a billing account
gcloud billing projects link YOUR_PROJECT_ID --billing-account=YOUR_BILLING_ACCOUNT_ID
```

## Step 8: Set Up Vertex AI Workbench (Optional)

For interactive development and testing:

```bash
# Create a Vertex AI Workbench instance
gcloud notebooks instances create klue-nli-notebook \
    --vm-image-project=deeplearning-platform-release \
    --vm-image-family=tf-latest-cpu \
    --machine-type=n1-standard-4 \
    --location=us-central1-a

# Connect to the instance
gcloud notebooks instances describe klue-nli-notebook --location=us-central1-a
```

## Step 9: Install Dependencies

Install the required Python packages:

```bash
# Install dependencies
pip install -r requirements.txt

# Or use the setup script
./setup.sh install
```

## Step 10: Test the Setup

Run the test script to verify everything is working:

```bash
# Test the complete setup
./setup.sh test

# Or run the test script directly
python test_setup.py
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Authentication Errors

**Error**: `google.auth.exceptions.DefaultCredentialsError`

**Solution**:
```bash
# Re-authenticate
gcloud auth application-default login

# Or set service account credentials
export GOOGLE_APPLICATION_CREDENTIALS="path/to/key.json"
```

#### 2. API Not Enabled

**Error**: `google.api_core.exceptions.PermissionDenied: 403`

**Solution**:
```bash
# Enable the API
gcloud services enable aiplatform.googleapis.com

# Wait a few minutes for the API to be fully enabled
```

#### 3. Project ID Not Set

**Error**: `GOOGLE_CLOUD_PROJECT environment variable is not set`

**Solution**:
```bash
# Set the project ID
export GOOGLE_CLOUD_PROJECT="YOUR_PROJECT_ID"

# Or provide it via command line
python klue_nli-gemini2_5flash.py --project-id "YOUR_PROJECT_ID"
```

#### 4. Billing Not Enabled

**Error**: `Billing has not been enabled for this project`

**Solution**:
```bash
# Enable billing
gcloud billing projects link YOUR_PROJECT_ID --billing-account=YOUR_BILLING_ACCOUNT_ID
```

#### 5. Quota Exceeded

**Error**: `Quota exceeded for quota group`

**Solution**:
- Request quota increase in Google Cloud Console
- Use a different region
- Reduce batch size or request frequency

### Getting Help

1. **Google Cloud Documentation**: https://cloud.google.com/vertex-ai/docs
2. **Vertex AI Troubleshooting**: https://cloud.google.com/vertex-ai/docs/troubleshooting
3. **Google Cloud Support**: https://cloud.google.com/support

## Cost Considerations

### Vertex AI Pricing

- **Gemini 2.5 Flash**: $0.075 per 1M input tokens, $0.30 per 1M output tokens
- **API Requests**: Additional charges may apply for high-volume usage

### Cost Optimization Tips

1. **Use Sampling**: Start with `--max-samples 100` for testing
2. **Monitor Usage**: Check billing dashboard regularly
3. **Set Budget Alerts**: Configure billing alerts in Google Cloud Console
4. **Use Efficient Prompts**: Optimize prompt length to reduce token usage

## Security Best Practices

1. **Use Service Accounts**: Prefer service accounts over user credentials for production
2. **Limit Permissions**: Grant only necessary IAM roles
3. **Rotate Keys**: Regularly rotate service account keys
4. **Monitor Access**: Use Cloud Audit Logs to monitor API access
5. **Secure Storage**: Store credentials securely, never commit to version control

## Next Steps

After completing the setup:

1. **Run a Small Test**: `./run test`
2. **Run Full Benchmark**: `./run full`
3. **Analyze Results**: Check the `benchmark_results/` directory
4. **Customize Configuration**: Modify parameters as needed

## Support

For issues specific to this benchmark:

1. Check the troubleshooting section above
2. Review the main README.md file
3. Check the test_setup.py output for specific error messages
4. Ensure all prerequisites are met 