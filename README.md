# LLM Benchmarks for Asian Languages

## Install Anaconda
Install Anaconda. For details, see [INSTALL-CONDA.md](INSTALL-CONDA.md).

## Create a new conda environment `klue`
```bash
conda create -n klue python=3 anaconda
```

## Activate the `klue` environment
```bash
(base) $ conda activate klue
(klue) $
```

For example,
```bash
(base) aimldl@tkim-glinux:~$ conda activate klue
(klue) aimldl@tkim-glinux:~$
```

To deactivate an active environment, run:                
```bash                                                                                                            conda deactivate 
```

## Initialize the `gcloud` SDK and authenticate into your account
Install the `gcloud` SDK
```bash
$ gcloud init
```

And authenticate into your account
```bash
$ gcloud auth login
```

## Launch Jupyter Lab
`cd` into the project directory. For example, `~/github/aimldl/llm_benchmarks_asian_langs`. And launche Jupyter Lab.

```bash
(klue) $ cd ~/github/aimldl/llm_benchmarks_asian_langs
(klue) $ jupyter lab
```
<img src="images/jupyter_lab-llm_benchmarks_asian_langs.png">

For example,
```bash
(klue) aimldl@tkim-glinux:~$ cd ~/github/aimldl/llm_benchmarks_asian_langs
(klue) aimldl@tkim-glinux:~/github/aimldl/llm_benchmarks_asian_langs$ jupyter lab
```

To cancel Jupyter Lab, press Ctrl+C twice.

## Run the KLUE benchmarks

### Two options to run tasks for KLUE.
- Option 1. Use a notebook.
- Option 2. Use a terminal and follow the instructions in each sub-directory

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
