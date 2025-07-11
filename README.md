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
