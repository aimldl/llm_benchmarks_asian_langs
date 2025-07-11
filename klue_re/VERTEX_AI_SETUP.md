# Vertex AI Setup Guide for KLUE RE

This guide provides detailed instructions for setting up Google Cloud Vertex AI to run the KLUE RE benchmark.

## Prerequisites

### 1. Google Cloud Account
- **Google Cloud Account**: You need a Google Cloud account with billing enabled
- **Billing**: Vertex AI requires billing to be enabled
- **Quotas**: Ensure you have sufficient quotas for AI Platform

### 2. Local Environment
- **Python 3.8+**: Required for the benchmark scripts
- **gcloud CLI**: Google Cloud command-line interface
- **Internet Connection**: Required for API calls and dataset loading

## Step-by-Step Setup

### Step 1: Install Google Cloud CLI

#### On Linux/macOS:
```bash
# Download and install gcloud CLI
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Verify installation
gcloud --version
```

#### On Windows:
1. Download the installer from: https://cloud.google.com/sdk/docs/install
2. Run the installer and follow the prompts
3. Open Command Prompt and verify: `gcloud --version`

### Step 2: Authenticate with Google Cloud

```bash
# Login to your Google Cloud account
gcloud auth login

# Set up application default credentials
gcloud auth application-default login

# Verify authentication
gcloud auth list
```

### Step 3: Create or Select a Project

#### Option A: Use Existing Project
```bash
# List your projects
gcloud projects list

# Set the project
gcloud config set project YOUR_PROJECT_ID

# Verify the project is set
gcloud config get-value project
```

#### Option B: Create New Project
```bash
# Create a new project
gcloud projects create YOUR_PROJECT_ID --name="KLUE RE Benchmark"

# Set the project
gcloud config set project YOUR_PROJECT_ID

# Enable billing (you'll need to do this in the web console)
echo "Please enable billing for your project in the Google Cloud Console"
```

### Step 4: Enable Required APIs

```bash
# Enable Vertex AI API
gcloud services enable aiplatform.googleapis.com

# Enable Compute Engine API (required for Vertex AI)
gcloud services enable compute.googleapis.com

# Enable Cloud Storage API (for model artifacts)
gcloud services enable storage.googleapis.com

# Verify APIs are enabled
gcloud services list --enabled | grep -E "(aiplatform|compute|storage)"
```

### Step 5: Set Up Billing

1. **Go to Google Cloud Console**: https://console.cloud.google.com
2. **Navigate to Billing**: Menu → Billing
3. **Link Project to Billing Account**: Select your project and link it to a billing account
4. **Verify**: Ensure billing is enabled for your project

### Step 6: Configure IAM Permissions

#### Option A: Use Default Service Account (Recommended for testing)
```bash
# Get your project number
PROJECT_NUMBER=$(gcloud projects describe $(gcloud config get-value project) --format="value(projectNumber)")

# Grant Vertex AI User role to your account
gcloud projects add-iam-policy-binding $(gcloud config get-value project) \
    --member="user:$(gcloud config get-value account)" \
    --role="roles/aiplatform.user"

# Grant Storage Object Viewer role (for model access)
gcloud projects add-iam-policy-binding $(gcloud config get-value project) \
    --member="user:$(gcloud config get-value account)" \
    --role="roles/storage.objectViewer"
```

#### Option B: Create Service Account (Recommended for production)
```bash
# Create service account
gcloud iam service-accounts create klue-re-benchmark \
    --description="Service account for KLUE RE benchmark" \
    --display-name="KLUE RE Benchmark"

# Grant necessary roles
gcloud projects add-iam-policy-binding $(gcloud config get-value project) \
    --member="serviceAccount:klue-re-benchmark@$(gcloud config get-value project).iam.gserviceaccount.com" \
    --role="roles/aiplatform.user"

gcloud projects add-iam-policy-binding $(gcloud config get-value project) \
    --member="serviceAccount:klue-re-benchmark@$(gcloud config get-value project).iam.gserviceaccount.com" \
    --role="roles/storage.objectViewer"

# Create and download key (optional, for programmatic access)
gcloud iam service-accounts keys create ~/klue-re-key.json \
    --iam-account=klue-re-benchmark@$(gcloud config get-value project).iam.gserviceaccount.com

# Set environment variable for service account key
export GOOGLE_APPLICATION_CREDENTIALS=~/klue-re-key.json
```

### Step 7: Verify Vertex AI Access

```bash
# Test Vertex AI access
gcloud ai models list --region=us-central1

# Test specific model availability
gcloud ai models describe gemini-2.5-flash --region=us-central1
```

### Step 8: Set Environment Variables

```bash
# Set your project ID
export GOOGLE_CLOUD_PROJECT=$(gcloud config get-value project)

# Set default region
export GOOGLE_CLOUD_REGION=us-central1

# Verify environment variables
echo "Project: $GOOGLE_CLOUD_PROJECT"
echo "Region: $GOOGLE_CLOUD_REGION"
```

### Step 9: Install Python Dependencies

```bash
# Navigate to the KLUE RE directory
cd klue_re

# Install dependencies
./install_dependencies.sh -v

# Or install manually
pip install google-genai>=0.3.0 datasets>=2.14.0 pandas>=1.5.0 tqdm>=4.64.0 google-cloud-aiplatform>=1.35.0
```

### Step 10: Test the Setup

```bash
# Run the setup test
./test_setup.py

# Test with a small sample
./run test
```

## Configuration Options

### Model Configuration

The benchmark uses Gemini 2.5 Flash by default. You can modify the configuration in `klue_re-gemini2_5flash.py`:

```python
@dataclass
class BenchmarkConfig:
    model_name: str = "gemini-2.5-flash"  # Model to use
    location: str = "us-central1"         # Vertex AI region
    temperature: float = 0.1              # Model temperature
    max_tokens: int = 2048                # Maximum output tokens
    # ... other settings
```

### Available Models

| Model Name | Description | Region Support |
|------------|-------------|----------------|
| `gemini-2.5-flash` | Fast, cost-effective model | All regions |
| `gemini-2.5-pro` | More capable but slower | All regions |
| `gemini-1.5-flash` | Previous generation | All regions |
| `gemini-1.5-pro` | Previous generation | All regions |

### Available Regions

| Region | Description | Latency |
|--------|-------------|---------|
| `us-central1` | Iowa (default) | Low |
| `us-east1` | South Carolina | Low |
| `us-west1` | Oregon | Medium |
| `europe-west1` | Belgium | Medium |
| `asia-northeast1` | Tokyo | High |

## Cost Management

### Pricing Information

- **Gemini 2.5 Flash**: ~$0.075 per 1M input tokens, ~$0.30 per 1M output tokens
- **Gemini 2.5 Pro**: ~$3.50 per 1M input tokens, ~$10.50 per 1M output tokens
- **API Calls**: Additional charges for API requests

### Cost Optimization

1. **Use Gemini 2.5 Flash**: More cost-effective for most tasks
2. **Limit Sample Size**: Start with small tests before full benchmarks
3. **Monitor Usage**: Check billing dashboard regularly
4. **Set Budget Alerts**: Configure billing alerts in Google Cloud Console

### Budget Setup

```bash
# Set up billing alerts (in Google Cloud Console)
echo "Go to: https://console.cloud.google.com/billing"
echo "Select your billing account"
echo "Click 'Budgets & alerts'"
echo "Create a new budget with alerts"
```

## Troubleshooting

### Common Issues

#### 1. "API not enabled" Error
```bash
# Enable the API
gcloud services enable aiplatform.googleapis.com

# Wait a few minutes for propagation
sleep 60

# Verify
gcloud services list --enabled | grep aiplatform
```

#### 2. "Permission denied" Error
```bash
# Check your permissions
gcloud projects get-iam-policy $(gcloud config get-value project) \
    --flatten="bindings[].members" \
    --format="table(bindings.role)" \
    --filter="bindings.members:$(gcloud config get-value account)"
```

#### 3. "Quota exceeded" Error
```bash
# Check your quotas
gcloud compute regions describe us-central1

# Request quota increase (if needed)
echo "Go to: https://console.cloud.google.com/iam-admin/quotas"
```

#### 4. "Model not found" Error
```bash
# List available models
gcloud ai models list --region=us-central1

# Check specific model
gcloud ai models describe gemini-2.5-flash --region=us-central1
```

### Debug Commands

```bash
# Check authentication
gcloud auth list

# Check project configuration
gcloud config list

# Test API access
gcloud ai models list --region=us-central1 --limit=5

# Check billing status
gcloud billing accounts list

# Verify service account (if using)
gcloud iam service-accounts list
```

## Security Best Practices

### 1. Use Service Accounts for Production
```bash
# Create dedicated service account
gcloud iam service-accounts create klue-re-production \
    --description="Production service account for KLUE RE"

# Grant minimal required permissions
gcloud projects add-iam-policy-binding $(gcloud config get-value project) \
    --member="serviceAccount:klue-re-production@$(gcloud config get-value project).iam.gserviceaccount.com" \
    --role="roles/aiplatform.user"
```

### 2. Enable Audit Logging
```bash
# Enable audit logging for Vertex AI
gcloud services enable cloudaudit.googleapis.com

# View audit logs
gcloud logging read "resource.type=aiplatform.googleapis.com/Model" --limit=10
```

### 3. Network Security
```bash
# Restrict API access to specific IPs (if needed)
gcloud compute firewall-rules create allow-vertex-ai \
    --direction=INGRESS \
    --priority=1000 \
    --network=default \
    --action=ALLOW \
    --rules=tcp:443 \
    --source-ranges=YOUR_IP_RANGE
```

## Monitoring and Logging

### Enable Monitoring
```bash
# Enable monitoring APIs
gcloud services enable monitoring.googleapis.com
gcloud services enable logging.googleapis.com

# Create monitoring dashboard (optional)
echo "Go to: https://console.cloud.google.com/monitoring"
```

### View Logs
```bash
# View Vertex AI logs
gcloud logging read "resource.type=aiplatform.googleapis.com/Model" --limit=20

# View specific model logs
gcloud logging read "resource.labels.model_name=gemini-2.5-flash" --limit=10
```

## Next Steps

After completing the setup:

1. **Run the benchmark**: `./run test`
2. **Monitor costs**: Check billing dashboard
3. **Review logs**: Check for any issues
4. **Scale up**: Run full benchmark when ready

## Support Resources

- **Google Cloud Documentation**: https://cloud.google.com/vertex-ai/docs
- **Vertex AI Pricing**: https://cloud.google.com/vertex-ai/pricing
- **gcloud CLI Reference**: https://cloud.google.com/sdk/gcloud/reference
- **IAM Best Practices**: https://cloud.google.com/iam/docs/best-practices

## Quick Reference

### Essential Commands
```bash
# Set project
gcloud config set project YOUR_PROJECT_ID

# Enable APIs
gcloud services enable aiplatform.googleapis.com

# Set environment
export GOOGLE_CLOUD_PROJECT=$(gcloud config get-value project)

# Test setup
./test_setup.py

# Run benchmark
./run test
```

### Environment Variables
```bash
export GOOGLE_CLOUD_PROJECT="your-project-id"
export GOOGLE_CLOUD_REGION="us-central1"
export GOOGLE_APPLICATION_CREDENTIALS="path/to/service-account-key.json"  # Optional
``` 