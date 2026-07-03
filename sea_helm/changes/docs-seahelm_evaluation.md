# Architecture and Code Analysis of `seahelm_evaluation.py`

The `seahelm_evaluation.py` script is designed to evaluate Language Models (LLMs) against a set of predefined tasks. It follows a modular architecture, separating concerns into different components.

## Architecture Overview

*   **`SeaHelmEvaluation` Class:** This is the core class that orchestrates the entire evaluation process. It handles configuration loading, logging, inference, and evaluation.
*   **Task Configuration:** Tasks are defined in YAML files (e.g., `seahelm_tasks/task_config.yaml`, `seahelm_tasks/**/config.yaml`). These configurations specify task details, languages, metrics, and data file paths.
*   **LLM Serving:** The script integrates with different LLM serving mechanisms (vLLM, OpenAI, LiteLLM) through dedicated classes (`VLLMServing`, `OpenAIServing`, `LiteLLMServing`) located in the `serving` directory. This allows for flexible model integration.
*   **Dataset Handling:** It uses the `datasets` library to load and process task-specific datasets.
*   **Metric Evaluation:** Each task has a corresponding metric class (e.g., in `seahelm_tasks/nlg/translation/translation.py`) that defines how to evaluate the model's responses for that specific task. These are dynamically loaded.
*   **Logging and Output:** The script sets up comprehensive logging and saves evaluation results (inference data and aggregated metrics) to a specified output directory.
*   **Multiprocessing:** For certain tasks like `mt-bench`, it leverages multiprocessing to speed up evaluation.

## Code Analysis

### `SeaHelmEvaluation` Class

*   **`__init__`:**
    *   Initializes with an LLM object, task configuration, output directory, model details, and various flags (e.g., `is_base_model`, `is_vision_model`, `is_reasoning_model`).
    *   Loads task configurations from YAML files using `OmegaConf`.
    *   Sets up logging specific to the model and output directory.
    *   Determines the number of in-context examples based on whether it's a base or instruction-tuned model.
    *   Handles task skipping based on model type or user-provided list.
    *   Captures environment information and model arguments for reproducibility.
    *   Filters tasks based on the provided `tasks_configuration`.
*   **`load_config_from_folders`:**
    *   Recursively searches for `config.yaml` files within the `seahelm_tasks` folder.
    *   Aggregates configurations from all found YAML files into a single `OmegaConf` object.
    *   Builds a `task_list_by_lang` dictionary to easily access tasks per language.
    *   Handles task aliases for aggregated tasks.
*   **`setup_seahelm_logging`:** Creates the necessary output directories for inference results.
*   **`save_config`:** Saves the merged configuration (including run environment and arguments) to a YAML file in the output directory.
*   **`_check_if_task_should_run`:** A helper function to determine if a given task for a specific language should be run, considering skip lists and task existence.
*   **`get_generation_kwargs`:** Constructs generation arguments (e.g., temperature, max_tokens, stop tokens) based on task configuration and model type (base vs. instruction-tuned).
*   **`generate_formatted_conversation`:**
    *   Prepares the conversation history for the LLM, including few-shot examples if `num_examples > 0`.
    *   Supports both single-turn and multi-turn few-shot examples.
    *   Handles cases where example files are missing.
*   **`update_conversation`:** Appends a new turn to the conversation, handling vision model content formatting.
*   **`get_prompt_formatter`:** Returns a function that formats prompts for each row in the dataset, incorporating conversation history and few-shot examples. This is used with `dataset.map`.
*   **`get_update_function`:** A generic function to update a column in a dataset, used for adding responses, errors, or tokenized prompts.
*   **`update_reasoning_generation_kwargs`:** Adjusts generation parameters (e.g., `max_tokens`, `temperature`) specifically for reasoning models.
*   **`run_single_task_inference`:**
    *   Performs inference for a single task and language.
    *   Loads the dataset, applies prompt formatting, and generates responses using the provided LLM object.
    *   Handles caching of inference results.
    *   Parses LLM outputs (responses, errors, tokenized prompts).
    *   Updates the dataset with generated responses and errors.
    *   Saves inference results to a file (CSV or JSONL).
    *   Includes error handling for inference failures.
*   **`get_inference_filepath`:** Generates the file path for saving/loading inference results.
*   **`write_out_inference_results`:** Writes the inference DataFrame to a specified file type (CSV or JSONL).
*   **`read_inference_results`:** Reads cached inference results from a file.
*   **`get_metric_class`:** Dynamically imports and returns the appropriate metric class for a given task based on the configuration.
*   **`run_single_task_evaluation`:**
    *   Evaluates the model's responses for a single task and language using the dynamically loaded metric class.
    *   Calculates metrics, error counts, and inference time.
    *   Updates the overall `metrics` dictionary.
    *   Includes error handling for evaluation failures.
*   **`write_metric_to_file`:** Writes the aggregated metrics to a JSON file.
*   **`run_evaluation`:**
    *   The main entry point for running the entire evaluation.
    *   Iterates through defined tasks and languages, performing inference and evaluation.
    *   Handles special cases like `mt-bench` (which uses multiprocessing for evaluation) and translation tasks (which might require specific model handling).
    *   Aggregates final metrics using `aggregate_metrics`.

### Main Execution Block (`if __name__ == "__main__":`)

*   **Argument Parsing:** Uses `argparse` to define command-line arguments for tasks, output directory, model type, model name, model arguments, and various flags (e.g., `is_base_model`, `rerun_cached_results`).
*   **Logger Setup:** Initializes the root logger.
*   **LLM Initialization:** Based on `model_type`, it initializes the appropriate LLM serving class (`LiteLLMServing`, `OpenAIServing`, or `VLLMServing`).
*   **`SeaHelmEvaluation` Instantiation:** Creates an instance of `SeaHelmEvaluation` with the parsed arguments.
*   **Evaluation Execution:** Calls `seahelm_eval.run_evaluation()` to start the evaluation process.

## Key Dependencies

*   `argparse`: For command-line argument parsing.
*   `glob`: For finding configuration files.
*   `importlib`: For dynamic module and class loading (metrics).
*   `json`: For handling JSON output.
*   `logging`: For logging messages.
*   `os`: For file system operations.
*   `datetime`: For timestamping output files.
*   `functools.partial`: For creating partial functions for multiprocessing.
*   `multiprocessing.Pool`: For parallel execution of `mt-bench` evaluation.
*   `datasets`: For efficient dataset loading and manipulation.
*   `litellm`: For LiteLLM integration.
*   `pandas`: For data manipulation (DataFrames).
*   `omegaconf`: For loading and merging YAML configurations.
*   `torch.utils.collect_env.get_pretty_env_info`: For collecting environment information.
*   `base_logger`: Custom logging setup.
*   `constants`: Defines various constants like skip tasks and stop tokens.
*   `serving`: Directory containing LLM serving classes.
*   `seahelm_tasks.aggregate_metrics`: For aggregating evaluation metrics.
*   `utils`: Contains utility functions like `get_error_count`, `get_git_commit_hash`, `simple_parse_args_string`.

## Overall Flow

1.  User runs `seahelm_evaluation.py` with command-line arguments.
2.  The script initializes logging and the appropriate LLM serving client.
3.  `SeaHelmEvaluation` class is instantiated, loading configurations and setting up the evaluation environment.
4.  `run_evaluation` method is called, which iterates through defined tasks and languages.
5.  For each task, `run_single_task_inference` is called to generate model responses.
    *   This involves loading datasets, formatting prompts, and calling the LLM.
    *   Inference results are saved.
6.  After inference, `run_single_task_evaluation` is called to evaluate the generated responses using task-specific metrics.
    *   Evaluation results are saved.
7.  Special handling for `mt-bench` (multiprocessing) and translation tasks.
8.  Finally, all metrics are aggregated and saved to a final JSON file.

This script provides a robust and extensible framework for evaluating LLMs across various tasks and configurations.
