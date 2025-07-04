# Vertex AI Setup Guide for KLUE TC Benchmark

This guide provides step-by-step instructions for setting up Google Cloud Vertex AI to run the KLUE Topic Classification benchmark.

## Prerequisites
1. **Google Cloud Account**: You need a Google Cloud account with billing enabled
2. **Google Cloud CLI**: Install the Google Cloud CLI for easier setup
3. **Python 3.8+**: Ensure you have Python 3.8 or higher installed

## Step 1: Install Google Cloud CLI

### On Linux/macOS:
```bash
# Download and install
curl https://sdk.cloud.google.com | bash

# Restart your shell or run:
exec -l $SHELL

# Verify the installation
Open a new terminal to take the installation effect and run:

$ gcloud --version
Google Cloud SDK 529.0.0
bq 2.1.19
bundled-python3-unix 3.12.9
core 2025.06.27
gcloud-crc32c 1.0.0
gsutil 5.35
$

# Initialize gcloud
gcloud init
```

### On Windows:
Download and install from: https://cloud.google.com/sdk/docs/install

## Step 2: Create or Select a Google Cloud Project
This step is necessary only when you haven't set up the default project when you initialized gcloud with the `gcloud init` command.

```bash
# List existing projects
gcloud projects list

# Create a new project (if needed)
gcloud projects create YOUR_PROJECT_ID --name="KLUE Benchmark Project"

# Set the project as default
gcloud config set project YOUR_PROJECT_ID
```

## Step 3: Enable Required APIs

```bash
# Enable Vertex AI API
gcloud services enable aiplatform.googleapis.com

# Verify the API is enabled
gcloud services list --enabled --filter="name:aiplatform.googleapis.com"
```
The output message will look simiar to:
```bash
$ gcloud services list --enabled --filter="name:aiplatform.googleapis.com"
NAME                       TITLE
aiplatform.googleapis.com  Vertex AI API
$
```

## Step 4: Set Up Authentication

Choose one of the following authentication methods:

### Method A: Application Default Credentials (Recommended for Development)

```bash
# Login with your Google account
gcloud auth login

# Set up application default credentials
gcloud auth application-default login

# Verify authentication
gcloud auth list
```

### Method B: Service Account (Recommended for Production)

```bash
# Create a service account
gcloud iam service-accounts create klue-benchmark \
    --display-name="KLUE Benchmark Service Account"

# Grant necessary permissions
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:klue-benchmark@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/aiplatform.user"

# Create and download key
gcloud iam service-accounts keys create key.json \
    --iam-account=klue-benchmark@YOUR_PROJECT_ID.iam.gserviceaccount.com

# Set environment variable
export GOOGLE_APPLICATION_CREDENTIALS="$(pwd)/key.json"
```

## Step 5: Set Environment Variables

`YOUR_PROJECT_ID` can be found by running:
```bash
gcloud config get-value project
```

You may want to hard-code the output of the `gcloud config get-value project` command in the following line:

```bash
# Set your project ID
export GOOGLE_CLOUD_PROJECT="YOUR_PROJECT_ID"

# Verify the environment variable is set
echo $GOOGLE_CLOUD_PROJECT
```

## Step 6: Install Python Dependencies

```bash
# Install required packages
pip install -r requirements.txt

# Or use the installation script
chmod +x install_dependencies.sh
./install_dependencies.sh
```

## Step 7: Test the Setup

```bash
# Run the test script
python test_setup.py
```

You should see output similar to:
```
============================================================
KLUE TC Benchmark Setup Test (Vertex AI)
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

Testing KLUE dataset loading...
✓ KLUE TC dataset loaded successfully
  - Train samples: 10000
  - Validation samples: 1000
  - Test samples: 1000

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

## Step 8: Run the Benchmark

After successful setup, run:

```bash
# a small test with limited samples first
python klue_tc-gemini2_5flash.py --project-id $GOOGLE_CLOUD_PROJECT --max-samples 10

# the full benchmark
python klue_tc-gemini2_5flash.py --project-id $GOOGLE_CLOUD_PROJECT
# or
python klue_tc-gemini2_5flash.py --project-id $GOOGLE_CLOUD_PROJECT --max-samples 1000
```

## Step 9: Analyze results in the `benchmark_results/` directory

Analyze results after running the full benchmark successfully.

Additionally, you may monitor costs and usage in Google Cloud Console

## Troubleshooting

### Common Issues and Solutions

#### 1. Authentication Errors

**Error**: `google.auth.exceptions.DefaultCredentialsError: Could not automatically determine credentials.`

**Solution**:
```bash
# Re-authenticate
gcloud auth application-default login
```

#### 2. Project Not Found

**Error**: `google.api_core.exceptions.NotFound: 404 Project not found`

**Solution**:
```bash
# Check your project ID
gcloud config get-value project

# Set the correct project
gcloud config set project YOUR_PROJECT_ID
export GOOGLE_CLOUD_PROJECT="YOUR_PROJECT_ID"
```

#### 3. API Not Enabled

**Error**: `google.api_core.exceptions.PermissionDenied: 403 Vertex AI API has not been used in project`

**Solution**:
```bash
# Enable the API
gcloud services enable aiplatform.googleapis.com
```

#### 4. Quota Exceeded

**Error**: `google.api_core.exceptions.ResourceExhausted: 429 Quota exceeded`

**Solution**:
- Check your Vertex AI quotas in the Google Cloud Console
- Request quota increase if needed
- Add delays between requests in the script

#### 5. Billing Not Enabled

**Error**: `google.api_core.exceptions.PermissionDenied: 403 Billing has not been enabled for this project`

**Solution**:
- Enable billing for your Google Cloud project
- Link a billing account to your project

### Debugging Commands

```bash
# Check authentication status
gcloud auth list

# Check current project
gcloud config get-value project

# Check enabled APIs
gcloud services list --enabled

# Test Vertex AI access
python -c "
from google.cloud import aiplatform
aiplatform.init(project='YOUR_PROJECT_ID')
print('Vertex AI initialized successfully')
"
```

## Cost Considerations

- **Vertex AI Pricing**: Charges per request and token usage
- **Estimated Cost**: $5-15 USD for full KLUE TC test set (This is a rough estimate which can be inaccurate.)
- **Cost Control**: Use `--max-samples` to limit testing scope
- **Monitoring**: Check costs in Google Cloud Console

## Security Best Practices

1. **Service Accounts**: Use service accounts for production workloads
2. **Least Privilege**: Grant only necessary permissions
3. **Key Rotation**: Regularly rotate service account keys
4. **Environment Variables**: Use environment variables for sensitive data
5. **Audit Logging**: Enable audit logs for Vertex AI usage

## Support

- [Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs)
- [Google Cloud Support](https://cloud.google.com/support)
- [Vertex AI Pricing](https://cloud.google.com/vertex-ai/pricing) 