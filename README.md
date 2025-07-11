# LLM Benchmarks for Asian Languages 🌏

This repository provides a suite of LLM benchmarks on various tasks for Asian languages.

---

## 🚀 Getting Started

### Prerequisites

Ensure you have [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed. For detailed instructions, refer to [INSTALL-CONDA.md](INSTALL-CONDA.md).

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/aimldl/llm_benchmarks_asian_langs.git](https://github.com/aimldl/llm_benchmarks_asian_langs.git)
    cd llm_benchmarks_asian_langs
    ```
2.  **Create and activate the conda environment:**
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

# LLM Benchmarks for Asian Languages

## Install Anaconda
Install Anaconda. For details, see [INSTALL-CONDA.md](INSTALL-CONDA.md).

## Project Directory
`cd` into the project directory. 
- For example, I've git cloned this repository under the `~/github/aimldl` directory.
- My project directory is `~/github/aimldl/llm_benchmarks_asian_langs`.

```bash
(base) ~$ cd ~/github/aimldl/llm_benchmarks_asian_langs
(base) ~/github/aimldl/llm_benchmarks_asian_langs$
```

## Launch Jupyter Lab
```bash
(base) $ jupyter lab
```
To cancel Jupyter Lab, press `Ctrl+C` twice.

## Initialize the `gcloud` SDK and authenticate into your account
This step is necessary to use Vertex AI Gemini API.

### Install the `gcloud` SDK
```bash
$ gcloud init
```

### Authenticate into your account
```bash
$ gcloud auth login
```

## Run the KLUE benchmarks

All KLUE task directories provide:
* Consistent directory and script structure
* Robust logging and error analysis
* Detailed documentation and troubleshooting
* Task-specific prompt engineering and evaluation

Implementation Consistency
* All tasks maintain identical directory structure
* Consistent logging and error handling across all tasks
* Standardized script functionality (test/custom/full modes)
* Comprehensive documentation for each task
* Automated testing and verification scripts

## Create a new conda environment `klue`
```bash
(base) $ conda create -n klue python=3 anaconda
```

## Activate the klue environment
```bash
(base) $ conda activate klue
(klue) $
```
To deactivate an active environment, run:
```bash
(klue) $ conda deactivate
```

### Two options to run tasks for KLUE.
- Option 1. Use a notebook.
- Option 2. Use a terminal and follow the instructions in each sub-directory

<img src="images/jupyter_lab-llm_benchmarks_asian_langs.png">

For example,
```bash
(klue) aimldl@tkim-glinux:~$ cd ~/github/aimldl/llm_benchmarks_asian_langs
(klue) aimldl@tkim-glinux:~/github/aimldl/llm_benchmarks_asian_langs$ jupyter lab
```

For example, 
- Option 1. Open [run_klue.ipynb](run_klue.ipynb) and run the notebook.
- Option 2. Open a terminal, `cd` into `klue_tc`, and follow instructions at `README.md`.

Note: The sub-directory `klue_tc` is designed to run independetly. The notebook [run_klue.ipynb](run_klue.ipynb) was created laster ro run the same commands, but it's assumed that the following cells are executed in the `klue` environment. Running the commands in the following cell 

```bash
cd klue_tc
./setup.sh full
./run test
```

is identical to

```bash
(klue) $ cd klue_tc
(klue) $ ./setup.sh full
(klue) $ ./run test
```
