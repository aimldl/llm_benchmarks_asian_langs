# litellm-batch_completion.md

[https://github.com/aisingapore/SEA-HELM](https://github.com/aisingapore/SEA-HELM)

## Overview

SEA-HELM (Southeast Asian Holistic Evaluation of Language Models) is an open-source evaluation framework for language models, particularly focusing on their performance in Southeast Asian contexts. The full codebase for SEA-HELM is available on GitHub, specifically within the aisingapore/SEA-HELM repository.

This repository contains the necessary files and documentation for understanding, running, and contributing to the SEA-HELM evaluation. It includes details on score calculations, how tasks are grouped into competencies (NLU, NLG, NLR, Instruction-Following, Multi-Turn, Cultural, Safety), and guidelines for adding new tasks or metrics. The repository also provides information on reproducing leaderboards and serving models for evaluation.

## Model Serving Implementations

|  | File & Directory Structure | Compute | Memo |
| :---- | :---- | :---- | :---- |
| Serving | serving/ |  |  |
| vLLM | ├── vllm\_serving.py | GPU | It works OOTB |
| LiteLLM | ├── litellm\_serving.py | CPU | It doesn't work OOTB |
| ollama |  | CPU |  |

```shell
serving/
  ...
├── litellm_serving.py
  ...
└── vllm_serving.py
```

## Experiments

Active evaluation of both local and API-based language models using SEA-HELM was conducted.

| Mode | Model | Config, Results, Log |
| :---- | :---- | :---- |
| local | Llama-3-2-3B-Instruct | output\_local\_test/ └── Llama-3-2-3B-Instruct     ├── inference     ├── Llama-3-2-3B-Instruct\_run\_config\_\*.yaml     ├── Llama-3-2-3B-Instruct\_seahelm\_results\_\*.json     └── logfile.log |
| remote | gemini-2.5-pro | output\_gemini/ └── gemini-2.5-pro    ├──gemini-2.5-pro\_run\_config\_\*.yaml    ├──gemini-2.5-pro\_seahelm\_results\_\*.json    ├── inference    └── logfile.log |
| remote | gemini-2.5-flash | (Not tested or implemented yet) |
| remote | gemini-2.5-flash-lite | (Not tested or implemented yet) |

### mode=local

* Local models are used to generate the code with LiteLLM and ensure it works.  
* A lightweight local model provides a faster turnaround time than a remote heavy model.  
* Q: Why meta-llama/Llama-3-2-3B-Instruct?  
  A: **meta-llama/Llama-3-2-3B-Instruct** is the model used by the default **run\_evaluation.sh** script.

### mode=remote (API-based)

* Use **vertex\_ai/gemini-2.5-flash** instead of **vertex\_ai/gemini-2.5-pro**  
  * to reduce the turnaround time for a single experiment  
* Using a lighter model reduces the total generation time of an LLM  
  * TTFT (Time to First Token) and TPOT (Time Per Output Token).  
* Reduce the number of benchmark tasks (run\_config\_\*.yaml)

In **run\_evaluation.sh,** 

```shell
#!/bin/bash

# Add a list of models (either local path or HuggingFace model id) to be evaluated
MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
OUTPUT="results"
  ...
```

This script includes repeated if statements which were removed in my scripts.

In **run\_local\_test,**

```shell
#!/bin/bash
  ...
MODEL_NAME="meta-llama/Llama-3-2-3B-Instruct"  # Prefix "Meta-" was removed
OUTPUT_DIR="output_local_test"
```

### Running experiments

#### run\_local\_test

```shell
$ ./run_local_test
```

```shell
#!/bin/bash
# run-local_test
#   runs a local test with a specific model configuration.

set -euo pipefail

# Configure
MODEL="meta-llama/Llama-3-2-3B-Instruct"
OUTPUT_DIR="output_local_test"
MODEL_TYPE="litellm"

# Ensure the main script exists and is executable.
if [ ! -x "./run_evaluation.sh" ]; then
    echo "Error: 'run_evaluation.sh' was not found or is not executable." >&2
    echo "Ensure it exists in the current directory and has execute permissions." >&2
    echo "  $ chmod +x run_evaluation.sh" >&2
    exit 1
fi

# Run
./run_evaluation.sh "$MODEL" "$OUTPUT_DIR" "$MODEL_TYPE"
```

#### run\_gemini

```shell
$ ./run_gemini
```

```shell
#!/bin/bash
# run-gemini
#   runs an evaluation with a Gemini model configuration.

set -euo pipefail

# Configure
MODEL="vertex_ai/gemini-2.5-pro"
OUTPUT_DIR="output_gemini"
MODEL_TYPE="litellm"

# Ensure the main script exists and is executable.
if [ ! -x "./run_evaluation.sh" ]; then
    echo "Error: 'run_evaluation.sh' was not found or is not executable." >&2
    echo "Ensure it exists in the current directory and has execute permissions." >&2
    echo "  $ chmod +x run_evaluation.sh" >&2
    exit 1
fi

# Run
./run_evaluation.sh "$MODEL" "$OUTPUT_DIR" "$MODEL_TYPE"
```

#### run\_evaluation.sh

* The original run\_evaluation.sh was rewritten.   
* The original script was renamed and backed up as run\_evaluation-original.sh. 

```shell
#!/bin/bash
# run_evaluation.sh
# A reusable base script to run seahelm evaluations.

set -euo pipefail

# --- Main Function ---
main() {
    # Set variables with defaults that can be overridden by command-line arguments.
    local model="${1:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
    local output_base_dir="${2:-results}"
    local model_type="${3:-vllm}"
    
    # --- Configuration Variables (Hardcoded) ---
    # If needed, take in input argument with the syntax 
    # "${variable:-default_value}"
    local python_script="seahelm_evaluation.py"
    local tasks="seahelm"
    local is_base_model=false
    local rerun_cached_results=false

    # --- Prepare Command-Line Arguments ---
    # Conditionally set arguments based on the boolean flags.
    local base_model_arg=""
    if [ "$is_base_model" = true ]; then
        base_model_arg="--base_model"
    fi

    local rerun_results_arg=""
    if [ "$rerun_cached_results" = true ]; then
        rerun_results_arg="--rerun_cached_cached_results"
    fi

    # --- Create Output Directory ---
    # The output directory name is derived from the model name.
    local output_dir="${output_base_dir}/$(basename "${model}")"
    mkdir -p "${output_dir}"
    echo "Output directory created: ${output_dir}"

    # --- Check for Dependencies ---
    if [ ! -f "$python_script" ]; then
        echo "Error: Python script '$python_script' not found!" >&2
        exit 1
    fi

    # --- Run the Evaluation ---
    # Set environment variables for the evaluation process.
    export LITELLM_LOG="ERROR"

    # Use an array to build the command for clarity and correct quoting.
    local seahelm_eval_args=(
        "python" "$python_script"
        "--tasks" "$tasks"
        "--output_dir" "$output_dir"
        "--model_name" "$model"
        "--model_type" "$model_type"
        "$base_model_arg"
        "$rerun_results_arg"
    )

    # Conditionally add --model_args for vllm
    # LiteLLM and OLLama don't need this argument
    if [[ "$model_type" == "vllm" ]]; then
        seahelm_eval_args+=("--model_args" "dtype=bfloat16,enable_prefix_caching=True,tensor_parallel_size=1")
    fi

    echo "Running evaluation command:"
    printf "%s " "${seahelm_eval_args[@]}"
    echo ""

    # Execute the final command.
    "${seahelm_eval_args[@]}"
}

# Run the main function with all provided arguments.
main "$@"
```

## Parameters

### The \--base\_model flag

**A base model** is the raw, pre-instruction-tuned version of a language model. The \--base\_model flag is a way of telling the evaluation script to handle this type of model correctly, rather than treating it like a chatbot. In other words, it is an LLM that has not been fine-tuned for conversational or instruction-following tasks. It is a foundational model trained on a vast amount of data to predict the next token in a sequence.

| [https://github.com/aisingapore/SEA-HELM](https://github.com/aisingapore/SEA-HELM) \> [Important notes about running SEA-HELM](https://github.com/aisingapore/SEA-HELM?tab=readme-ov-file#important-notes-about-running-sea-helm) 1\. Please ensure that the \--base\_model flag is included if the model to be evaluated is a base model. This will load a default base model chat template (see chat\_templates\\base\_model.jinja). The \--base\_model will also disable the MT-Bench tasks. |
| :---- |

The `seahelm_evaluation.py` script treats a model as a base model if you pass the `--base_model` flag. This flag then triggers two key actions within the script to handle the base model's unique characteristics:

1. **It applies a default chat template** (`chat_templates/base_model.jinja`). This is necessary because base models do not have their own built-in conversation format. The template provides a basic structure to format user prompts for the model.  
2. **It disables MT-Bench tasks.** MT-Bench is designed to evaluate a model's ability to engage in multi-turn conversations and follow complex instructions, which are skills typically developed through fine-tuning. Since a base model lacks this training, evaluating it on MT-Bench would yield poor or irrelevant results, so the task is skipped to avoid misleading scores.

### The **\--rerun\_cached\_results** flag

| [https://github.com/aisingapore/SEA-HELM](https://github.com/aisingapore/SEA-HELM) \> [Important notes about running SEA-HELM](https://github.com/aisingapore/SEA-HELM?tab=readme-ov-file#important-notes-about-running-sea-helm) 2\. All inference results are cached in the results folder to allow for resumption of incomplete runs. If there is a need to rerun the inference, please either delete the corresponding model folder in the results folder or set the flag \--rerun\_cached\_results |
| :---- |

### The \-**\-model\_args** parameter

* This parameter is used for vLLM.   
* LiteLLM and Ollama don't use the same arguments as vLLM.  
* These parameters become irrelevant or unsupported   
  * when a CPU-focused backend like LiteLLM or Ollama is used.

#### \--model\_args is for vllm

The `--model_args` argument is specific to the **vLLM backend**. It's used to pass arguments directly to the vLLM engine, which is designed for high-performance inference on GPUs. Arguments like `tensor_parallel_size` and `dtype=bfloat16` are directly tied to GPU-accelerated operations.

* `tensor_parallel_size=1`: This parameter controls how a model is split across multiple GPUs. A value of 1 means it's running on a single GPU. It is not a relevant parameter for a CPU-only setup.  
* `dtype=bfloat16`: Bfloat16 is a data type optimized for GPU performance. While some CPUs can handle it, it's not a standard setting for CPU-based inference.

| [https://github.com/aisingapore/SEA-HELM](https://github.com/aisingapore/SEA-HELM) \> Tip  \--model\_args takes any kwargs specified in [https://docs.vllm.ai/en/latest/serving/engine\_args.html](https://docs.vllm.ai/en/latest/serving/engine_args.html) and allows for control of how vLLM serves the model |
| :---- |

| [https://github.com/aisingapore/SEA-HELM](https://github.com/aisingapore/SEA-HELM) \> [Important notes about running SEA-HELM](https://github.com/aisingapore/SEA-HELM?tab=readme-ov-file#important-notes-about-running-sea-helm)  3\. LLM-as-a-Judge in SEA-MT-Bench The OpenAI calls are currently performed using the OpenAI's Batch API to save cost. This might result in a small increase in the wait times depending on the load of the OpenAI servers. If present as a task to run, MT-Bench is run as the first task so as to allow for the judgements to be done while the rest of the evaluations are being done. |
| :---- |

#### How LiteLLM and Ollama Handle CPUs

**LiteLLM** is a library that provides a unified, OpenAI-compatible API for over 100 different LLMs. It acts as a wrapper, simplifying calls to various backends, including local ones like Ollama and other CPU-only models.

* **CPU Support:** LiteLLM is designed to work with various local models, many of which can run on a CPU. It does not require specific `--model_args` for CPU operation because it leverages the underlying backend's (e.g., Ollama's) ability to use a CPU.  
* **No `--model_args` is needed:**

**Ollama** is a powerful tool for running open-source models locally. It includes a server and a command-line interface that can run models on either a GPU or a CPU.

* **CPU Support:** By default, Ollama will use the CPU if a GPU is not detected or if it's configured to do so. It will automatically manage the necessary CPU resources and memory. You can explicitly tell Ollama to use the CPU by setting the `num_gpu` parameter to 0 in its configuration, though this is usually not required.  
* **No `--model_args` is needed:** When you use LiteLLM to call an Ollama model (`MODEL_TYPE="litellm"`), your script doesn't pass any `--model_args` to Ollama directly. LiteLLM handles the communication and passes the model name to the Ollama server, which then runs it on the available hardware (CPU or GPU).

## Files in the seahelm\_tasks directory

Each subdirectory represents a benchmark task and contains its own 

* configuration (config.yaml),   
* Python scripts for task logic, and   
* data files (data/).

### e.g. instruction\_following

* metric\_file: seahelm\_tasks/**instruction\_following**/ifeval/**if\_eval.py**  
* filepath: seahelm\_tasks/**instruction\_following**/ifeval/**data/id\_sea\_ifeval.jsonl**

#### seahelm\_tasks/**instruction\_following**/ifeval/**config.yaml**

The instruction\_following task is specified

```shell
if-eval:
  metadata:
    version: 1
  name: "if-eval"
  competency: "instruction-following"
  metric_file: "seahelm_tasks/instruction_following/ifeval/if_eval.py"
  metric_class: "IFEvalMetric"
  metric: "overall_lang_normalized_acc"
  temperature: 0
```

Config for each supported language are stored in the .../ifeval/data/**\[language\]**\_sea\_ifeval**.jsonl**

```shell
  languages:
    en:
      filepath: "seahelm_tasks/instruction_following/ifeval/data/en_sea_ifeval.jsonl"
      max_tokens: 1024
      prompt_template:
        template: "{text}"
    id:
      filepath: "seahelm_tasks/instruction_following/ifeval/data/id_sea_ifeval.jsonl"
      max_tokens: 1024
      prompt_template:
        template: "{text}"
    vi:
      filepath: "seahelm_tasks/instruction_following/ifeval/data/vi_sea_ifeval.jsonl"
      max_tokens: 1024
      prompt_template:
        template: "{text}"
    th:
      filepath: "seahelm_tasks/instruction_following/ifeval/data/th_sea_ifeval.jsonl"
      max_tokens: 1024
      prompt_template:
        template: "{text}"
    tl:
      filepath: "seahelm_tasks/instruction_following/ifeval/data/tl_sea_ifeval.jsonl"
      max_tokens: 1024
      prompt_template:
        template: "{text}"
    jv:
      filepath: "seahelm_tasks/instruction_following/ifeval/data/jv_sea_ifeval.jsonl"
      max_tokens: 1024
      prompt_template:
        template: "{text}"
    su:
      filepath: "seahelm_tasks/instruction_following/ifeval/data/su_sea_ifeval.jsonl"
      max_tokens: 1024
      prompt_template:
        template: "{text}"
```

Note configuration for 7 languages exists.

And 7 files named **\[language\]**\_sea\_ifeval**.jsonl** exist.

```shell
$ tree seahelm_tasks/instruction_following/ifeval/data
seahelm_tasks/instruction_following/ifeval/data
├── en_sea_ifeval.jsonl
├── id_sea_ifeval.jsonl
├── jv_sea_ifeval.jsonl
├── su_sea_ifeval.jsonl
├── th_sea_ifeval.jsonl
├── tl_sea_ifeval.jsonl
└── vi_sea_ifeval.jsonl

1 directory, 7 files
$
```

### run\_config\_\*.yaml

This yaml file starts with:

```shell
tasks:
```

and the replica of "seahelm\_tasks/instruction\_following/ifeval/**config.yaml**" with indentation. 

```shell

  if-eval:
    metadata:
      version: 1
    name: if-eval
    competency: instruction-following
    metric_file: seahelm_tasks/instruction_following/ifeval/if_eval.py
    metric_class: IFEvalMetric
    metric: overall_lang_normalized_acc
    temperature: 0
```

while the number of languages is reduced to 4 by Gemini-CLI automatically.

```shell
   languages:
      id:
        filepath: seahelm_tasks/instruction_following/ifeval/data/id_sea_ifeval.jsonl
        max_tokens: 1024
        prompt_template:
          template: '{text}'
      vi:
        filepath: seahelm_tasks/instruction_following/ifeval/data/vi_sea_ifeval.jsonl
        max_tokens: 1024
        prompt_template:
          template: '{text}'
      th:
        filepath: seahelm_tasks/instruction_following/ifeval/data/th_sea_ifeval.jsonl
        max_tokens: 1024
        prompt_template:
          template: '{text}'
      tl:
        filepath: seahelm_tasks/instruction_following/ifeval/data/tl_sea_ifeval.jsonl
        max_tokens: 1024
        prompt_template:
          template: '{text}'
```

The **mt-bench** task is specified and the rest of tasks follow:

```shell
 mt-bench:
    metadata:
      ...
```

## \[model\]\_seahelm\_results\_\*.json

This json file has a list of

* benchmark task  
* performance metric  
* error

```json
{
  "id": {
    "nlu": {
      "sentiment": {
        "normalized_accuracy": 0,
        "error": "Failed to run evaluation for task"
      },
        ...
 "th": {
    "nlu": {
      "sentiment": {
        "normalized_accuracy": 0,
        "error": "Failed to run evaluation for task"
      },
      "qa": {
        "normalized_f1": 0,
        "error": "Failed to run evaluation for task"
      }
    }
  }
}
```

After running an experiment, all the results had an error.

```shell
"Failed to run evaluation for task"
```

## logfile.log: Analyzing the cause of the error

### Local model: meta-llama/Llama-3-2-3B-Instruct

SEA-HELM/output\_local\_test/Llama-3-2-3B-Instruct/**logfile.log**

```shell
2025-08-20 14:55:37 | INFO     | seahelm_evaluation
Loading model meta-llama/Llama-3-2-3B-Instruct using vLLMs...
```

```shell
2025-08-20 15:29:25 | INFO     | seahelm_evaluation
Model type is set to None. Please ensure that the model inferences are in the correct folder and format.
```

Recall that run\_config\_\*.yaml has a reduced number of languages (4 languages) while config.yaml has 7 languages in the list. It's because 

```shell
2025-08-20 15:29:25 | WARNING  | seahelm_evaluation
No valid OpenAI models found. Skipping task: mt-bench
```

As a result, the task skipped some languages.

```shell
2025-08-20 15:29:27 | INFO     | seahelm_evaluation
  ...
Task in skip task list: ['mt-bench']. Skipping task 'mt-bench' for lang 'id'.
Task in skip task list: ['mt-bench']. Skipping task 'mt-bench' for lang 'vi'.
Task in skip task list: ['mt-bench']. Skipping task 'mt-bench' for lang 'th'.
Task in skip task list: ['mt-bench']. Skipping task 'mt-bench' for lang 'tl'.
```

The start of a benchmark task evaluation is below.

```shell
---------- Inference | Lang: ID | Task: SENTIMENT ----------
Testing Competency: NLU
Drawing and preparing instances from seahelm_tasks/nlu/sentiment_analysis/data/id_nusax.jsonl
Performing inference for task 'SENTIMENT' with 0 examples
```

Note 0 examples are considered.

```shell
2025-08-20 15:45:51 | INFO     | utils
LiteLLM completion() model= meta-llama/Llama-3-2-3B-Instruct; provider = ollama
LiteLLM completion() model= meta-llama/Llama-3-2-3B-Instruct; provider = ollama
  ...
LiteLLM completion() model= meta-llama/Llama-3-2-3B-Instruct; provider = ollama
```

### Remote model: vertex\_ai/gemini-2.5-pro

SEA-HELM/output\_local\_test/gemini-2.5-pro/**logfile.log**

```shell
2025-08-20 15:53:33 | INFO     | seahelm_evaluation
Loading model vertex_ai/gemini-2.5-pro using OLLAMA...
```

```shell
2025-08-20 15:53:34 | WARNING  | seahelm_evaluation
No valid OpenAI models found. Skipping task: mt-bench
```

```shell
2025-08-20 15:53:35 | INFO     | seahelm_evaluation
Task in skip task list: ['mt-bench']. Skipping task 'mt-bench' for lang 'id'.
  ...
Task in skip task list: ['mt-bench']. Skipping task 'mt-bench' for lang 'tl'.
```

```shell
2025-08-20 15:53:35 | INFO     | seahelm_evaluation
---------- Inference | Lang: ID | Task: SENTIMENT ----------
Testing Competency: NLU
Drawing and preparing instances from seahelm_tasks/nlu/sentiment_analysis/data/id_nusax.jsonl
Performing inference for task 'SENTIMENT' with 0 examples
```

```shell
2025-08-20 15:53:37 | INFO     | utils
LiteLLM completion() model= vertex_ai/gemini-2.5-pro; provider = ollama
LiteLLM completion() model= vertex_ai/gemini-2.5-pro; provider = ollama
  ...
LiteLLM completion() model= vertex_ai/gemini-2.5-pro; provider = ollama
```

#### A longer log file

The following output is reformatted for a better readability.

```shell
2025-08-20 15:53:33 | INFO | seahelm_evaluation
Loading model vertex_ai/gemini-2.5-pro using OLLAMA...
---------- Preparation of output folder ----------
Preparing output folder ...
Folder: output_vertex_ai_gemini_2_5_pro/gemini-2.5-pro/inference
Completed preparation of output folder!

2025-08-20 15:53:33 | INFO | seahelm_evaluation
<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><
Evaluating vertex_ai/gemini-2.5-pro as instruction-tuned model...
<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><

2025-08-20 15:53:34 | WARNING | seahelm_evaluation
No valid OpenAI models found. Skipping task: mt-bench

2025-08-20 15:53:34 | INFO | seahelm_evaluation
---------- Configuration saving ----------
Saving run config to output folder...
Filepath: output_vertex_ai_gemini_2_5_pro/gemini-2.5-pro/gemini-2.5-pro_run_config_2025-08-20T15:53:34.067068.yaml

2025-08-20 15:53:35 | INFO | seahelm_evaluation
Config file saved!
Task in skip task list: ['mt-bench']. Skipping task 'mt-bench' for lang 'id'.
Task in skip task list: ['mt-bench']. Skipping task 'mt-bench' for lang 'vi'.
Task in skip task list: ['mt-bench']. Skipping task 'mt-bench' for lang 'th'.
Task in skip task list: ['mt-bench']. Skipping task 'mt-bench' for lang 'tl'.
---------- Inference | Lang: ID | Task: SENTIMENT ----------
Testing Competency: NLU
Drawing and preparing instances from seahelm_tasks/nlu/sentiment_analysis/data/id_nusax.jsonl
Performing inference for task 'SENTIMENT' with 0 examples

2025-08-20 15:53:37 | INFO | utils
LiteLLM completion() model= vertex_ai/gemini-2.5-pro; provider = ollama
LiteLLM completion() model= vertex_ai/gemini-2.5-pro; provider = ollama
  ...
LiteLLM completion() model= vertex_ai/gemini-2.5-pro; provider = ollama
  ...
```

## Fixing the errors

### No valid OpenAI models found. Skipping task: mt-bench

| [https://github.com/aisingapore/SEA-HELM](https://github.com/aisingapore/SEA-HELM) \> Evaluating Models using SEA-HELM \>  [Setup environment for SEA-HELM](https://github.com/aisingapore/SEA-HELM?tab=readme-ov-file#setup-environment-for-sea-helm) 2\. Ensure that the appropriate HF\_TOKEN has been set in the environment (with access to gated models). Note: Running LLM-as-a-Judge for MT-bench SEA-HELM currently uses gpt-4-1106-preview as the Judge LLM. As such, there is a need to access the OpenAI servers. Please ensure that OPENAI\_API\_KEY environment variable is set. `export OPENAI_API_KEY=...`  |
| :---- |

Double-check the API key.

```shell
(.venv) user@tkim-main-workstation:~/benchmarking-sea-helm-with-gemini/SEA-HELM$
```

```shell
$ echo $OPENAI_API_KEY
sk-proj-aChZZYuoFtHL-f8KOf-Dt0KX_zpd3Yce00mBoFsM9UhrA5HLHQ93CUbET4ogo4jevGuoRAvZNET3BlbkFJRuduZze7hdAYo2GSjtQw8jdNHJPLzd5ewm3t1-qMhpZokuBX5FdiKpjc6AI8>
$
```

The trailing \> was the problem. Replaced it to the full API key.

```shell
$ echo $OPENAI_API_KEY
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
$
```

### Performing inference for task 'SENTIMENT' with 0 examples

```shell

```

```shell

```

```shell

```

```shell

```

```shell

```

```shell

```

```shell

```

```shell

```

```shell

```

```shell

```

```shell

```

```shell

```