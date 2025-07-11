# LLM Benchmarks for Asian Languages 🌏

This repository provides a suite of benchmarks to evaluate Large Language Models (LLMs) on a variety of tasks for Asian languages.

## Key Features

The benchmark framework is designed with a focus on consistency, reproducibility, and ease of use.

* **Standardized Structure:** All tasks follow a uniform directory and script structure, ensuring a consistent user experience across the entire suite.
* **Robust Implementation:** Features include comprehensive logging for detailed analysis, consistent error handling, and automated verification scripts.
* **Flexible Execution:** Each task provides standardized script functionality with multiple operational modes, such as `test`, `custom`, and `full`, to accommodate different testing needs.
* **Tailored Evaluation:** Benchmarks include task-specific prompt engineering and evaluation metrics to ensure relevant and accurate performance assessment.
* **Comprehensive Documentation:** Each task directory contains detailed documentation, including setup instructions and troubleshooting guides.

---

## 🚀 Getting Started

### Prerequisites

Ensure you have [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed. For detailed instructions, refer to [INSTALL-CONDA.md](INSTALL-CONDA.md).

### Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/aimldl/llm_benchmarks_asian_langs.git
    cd llm_benchmarks_asian_langs
    ```
3.  **Create and activate the conda environment**
   Create the 'klue' environment with Python 3
    ```bash
    (base) $ conda create -n klue -y python=3 anaconda
    ```

    # Activate the environment
    ```bash
    (base) $ conda activate klue
    (klue) $ 
    ```
    To deactivate the environment when you're done, run `conda deactivate`.

---

## ⚙️ Recommended: Configure Google Cloud for Vertex AI

To run these benchmarks with the Gemini models, you must configure access to a Google API. While there are two options for accessing Gemini, using Vertex AI is strongly recommended for this project.

* **Google AI (Free Tier):** This API is intended for development and has very low rate limits. The benchmark scripts make thousands of API calls and will quickly exceed the free quota, causing the tasks to fail before completion.
* **Google Cloud Vertex AI (Paid Service):** This is the production-grade platform with high quotas suitable for intensive tasks. To ensure the benchmarks can run to completion, you must set up and use the Vertex AI API.

> **⚠️ Cost Warning**
>
> Be aware that running these benchmark tasks will execute a large volume of API calls to Gemini. This **will incur significant costs** on your Google Cloud billing account. Please monitor your usage and set up budget alerts before proceeding.

### Setup Instructions

Follow these steps to configure your local environment to use a Google Cloud project with the Vertex AI API enabled.

1.  **Initialize the gcloud SDK**

    This command walks you through linking a Google Cloud project and configuring defaults.
    ```bash
    gcloud init
    ```

2.  **Authenticate your account**

    This command opens a browser window to grant the SDK access to your user account.
    ```bash
    gcloud auth login
    ```
---

## ▶️ Running the KLUE Benchmarks

All KLUE task implementations share a consistent structure and provide robust logging, detailed documentation, and task-specific prompt engineering.

You have two options for running the benchmarks.

### Option 1: Use the Jupyter Notebook

This is the recommended method for a quick start.

1.  **Launch Jupyter Lab**
    ```bash
    (klue) $ jupyter lab
    ```
Once launched, open the `.ipynb` notebook for the target benchmark. 

For example, run `run_klue_tc.ipynb` for the KLUE Topic Classification (TC) task. Open and run the cells in `run_klue.ipynb`. This notebook automates the setup and execution steps for the benchmark tasks.

### Option 2: Use the Command Line

This method allows for running tasks independently from their respective directories.

1.  **Navigate to a task directory.**

For example, for Topic Classification:
    ```bash
    (klue) $ cd klue_tc/
    ```
3.  **Run the setup script.** 

The `full` argument installs all required packages.
    ```bash
    (klue) $ ./setup.sh full
    ```
4.  **Execute the benchmark.** 

The `run` script can be executed with different modes (`test`, `custom`, `full`).
    ```bash
    (klue) $ ./run test        #  10 samples
    (klue) $ ./run custom 100  # N samples, N=100
    (klue) $ ./run full        # The number of samples varies from task to task
    ```
    Follow the `README.md` in each task sub-directory for more details.
