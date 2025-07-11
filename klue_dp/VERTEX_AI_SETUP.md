# Vertex AI Setup Guide for KLUE DP Benchmark

This guide provides step-by-step instructions for setting up Google Cloud Vertex AI to run the KLUE Dependency Parsing benchmark.

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
The output message will look similar to:
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
KLUE DP Benchmark Setup Test (Vertex AI)
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
✓ KLUE DP dataset loaded successfully
  - Train samples: 10000
  - Validation samples: 1000

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
# Make the run script executable (if needed)
chmod +x run

# Run a small test with 10 samples (recommended first)
./run test

# Run the full benchmark (all validation samples)
./run full

# Run with custom number of samples
./run custom 100
```

**Note**: The `./run` script automatically handles logging and saves all output to the `logs/` directory with command headers for easy identification.

### Alternative: Direct Python Usage

```bash
# Test with limited samples
python klue_dp-gemini2_5flash.py --project-id "your-project-id" --max-samples 10

# Run full benchmark
python klue_dp-gemini2_5flash.py --project-id "your-project-id"

# Custom configuration
python klue_dp-gemini2_5flash.py \
    --project-id "your-project-id" \
    --max-samples 100 \
    --temperature 0.1 \
    --max-tokens 4096 \
    --location "us-central1"
```

## Step 9: Monitor and Analyze Results

### Check Log Files
```bash
# View recent log files
ls -la logs/

# Check the latest log
tail -f logs/klue_dp_test_10samples_*.log

# View error log
cat logs/klue_dp_test_10samples_*.err
```

### Analyze Results
```bash
# Check benchmark results
ls -la benchmark_results/

# Extract errors from results
./get_errors.sh benchmark_results/klue_dp_results_*.csv

# Test logging functionality
./test_logging.sh
```

## Troubleshooting

### Common Issues

1. **"Project ID not set"**
   ```bash
   # Set project ID
   export GOOGLE_CLOUD_PROJECT="your-project-id"
   ```

2. **"API not enabled"**
   ```bash
   # Enable Vertex AI API
   gcloud services enable aiplatform.googleapis.com
   ```

3. **"No credentials found"**
   ```bash
   # Set up authentication
   gcloud auth application-default login
   ```

4. **"Permission denied"**
   ```bash
   # Make scripts executable
   chmod +x run setup.sh install_dependencies.sh
   ```

### Performance Tips

1. **Start Small**: Begin with 10-50 samples to test your setup
2. **Monitor Quotas**: Check your Vertex AI quotas in Google Cloud Console
3. **Use Efficient Regions**: Choose Vertex AI regions close to your location
4. **Optimize Parameters**: Adjust max_tokens and temperature for your needs

### Cost Management

1. **Monitor Usage**: Check billing in Google Cloud Console
2. **Set Alerts**: Configure billing alerts to avoid unexpected charges
3. **Use Test Mode**: Use `./run test` for quick validation without full costs
4. **Clean Up**: Remove unnecessary resources after testing

## Next Steps

After successful setup:

1. **Run Initial Test**: `./run test` to verify everything works
2. **Review Results**: Check the generated log files and results
3. **Analyze Performance**: Review UAS/LAS scores and error patterns
4. **Optimize**: Adjust prompts and parameters based on results
5. **Scale Up**: Run larger benchmarks as needed

## Support

If you encounter issues:

1. Check the `TROUBLESHOOTING.md` file
2. Review log files in the `logs/` directory
3. Run `./setup.sh test` to verify your environment
4. Check Google Cloud documentation and support
5. Review the KLUE benchmark repository for dataset issues

## Security Best Practices

1. **Use Service Accounts**: For production, use service accounts instead of user credentials
2. **Limit Permissions**: Grant only necessary permissions to service accounts
3. **Secure Keys**: Store service account keys securely and rotate them regularly
4. **Monitor Access**: Regularly review who has access to your project
5. **Enable Audit Logs**: Enable Cloud Audit Logs to track API usage 