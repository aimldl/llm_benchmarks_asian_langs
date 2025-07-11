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

1.  **Clone the repository:**
* [https://github.com/aimldl/llm_benchmarks_asian_langs.git](https://github.com/aimldl/llm_benchmarks_asian_langs.git)
    ```bash
    git clone https://github.com/aimldl/llm_benchmarks_asian_langs.git
    cd llm_benchmarks_asian_langs
    ```
3.  **Create and activate the conda environment:**
    ```bash
    # Create the 'klue' environment with Python 3
    conda create -n klue python=3 anaconda

    # Activate the environment
    conda activate klue
    ```
    To deactivate the environment when you're done, run `conda deactivate`.

---

## ⚙️ Configuration for Google Cloud (Optional)

This step is only required if you plan to use Google's Vertex AI Gemini APIs.

1.  **Initialize the gcloud SDK:**
    ```bash
    gcloud init
    ```
2.  **Authenticate your account:**
    ```bash
    gcloud auth login
    ```

---

## ▶️ Running the KLUE Benchmarks

All KLUE task implementations share a consistent structure and provide robust logging, detailed documentation, and task-specific prompt engineering.

You have two options for running the benchmarks.

### Option 1: Use the Jupyter Notebook

This is the recommended method for a quick start.

1.  **Launch Jupyter Lab:**
    ```bash
    (klue) $ jupyter lab
    ```
2.  Open and run the cells in `run_klue.ipynb`. This notebook automates the setup and execution steps for the benchmark tasks.

### Option 2: Use the Command Line

This method allows for running tasks independently from their respective directories.

1.  **Navigate to a task directory.** For example, for Topic Classification:
    ```bash
    (klue) $ cd klue_tc/
    ```
2.  **Run the setup script.** The `full` argument installs all required packages.
    ```bash
    (klue) $ ./setup.sh full
    ```
3.  **Execute the benchmark.** The `run` script can be executed with different modes (`test`, `custom`, `full`).
    ```bash
    # Run in 'test' mode for a quick check
    (klue) $ ./run test
    ```
    Follow the `README.md` in each task sub-directory for more details.
