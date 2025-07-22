# New Task Guide

## Folder setup
To create a new task, we will need to create a new folder. For example,
```
mkdir seahelm_tasks/<competency>/<task_name>
```

Each folder should minimally contain the following folders/files:
```text
<task_name>
├──data             # Folder containing the data for the task
│   └── ...
├──config.yaml      # YAML containing the task configuration 
├──readme.md        # Text containing the task descriptions and list of changes made
└──task_metric.py   # Python file containing the metrics used for the task
```
## Creating a new `config.yaml`

<details>
<summary>Example config file structure</summary>

```yaml
<task_name>:
  metadata:
    version: ...
  name: ...
  competency: ...
  aggregation_group: ...
  metric_file: ...
  metric_class: ...
  metric: ...
  <additional kwargs>: ...
  temperature: ...
  languages:
    <lang>:
      filepath: ...
      example_filepath: ...
      max_tokens: ...
      prompt_template:
        template: ...
        fewshot_example: ...
        fewshot_label: ...
```
</details>

### Basic task descriptions
The description of the tasks should be defined as follows.
```yaml
<task_name>:
  metadata:
    version: ... # version number of the task
  name: ... # name of the task
  competency: ... # competency which the task should be under
  aggregation_group: ... # <optional> Set multiple tasks to the same aggregation group for the scores to be averaged together e.g. translation
```

> [!Note]  
> **Versioning of config/data**  
> To encourage transparency and reproducibility of the results, any changes to the config, data or metric calculation should be accompanied by an increase in the version number.
>
> Please also indicate in the `readme.md` any changes that were made.

> [!tip]  
> **Multiple tasks in a single config file**  
> The config file allows for multiple tasks to be defined by specifying more than one `<task_name>` and filling in the various keys. For an example of this, check out the translation task

### Metric class
The metrics can be specified as follows. The filepath to the metric should be relative to the base of the git repo i.e. `seahelm_tasks\<competency>\<task_name>`.
```yaml
  metric_file: ... # filepath to the metric python file
  metric_class: ... # name of the metric class
  metric: ... # metric used for the calculation of the aggregated scores
  <additional kwargs>: ... # Additional kwargs that can be accessed by `metric_class`
```
> [!tip]  
> **Additional parameters requiried for the metric class**  
> Additional parameters can be defined in the config file and accessed by `metric_class`.
> Examples of such parameters include: `null_label` for multiple choice tasks or `judge_models` for LLM-as-a-judge based tasks.

### Languages
For each language in each task, please fill in the following parts. Filepaths are relative to the base of the git repo i.e. `seahelm_tasks\<competency>\<task_name>\<data>`. 

```yaml
  languages:
    <lang>: # Language. Current code only support the two letter isocodes (ID, TA, TH, VI, TL, ...)
      filepath: ... # filepath the test data
      example_filepath: ... # filepath containing the fewshot examples
      max_tokens: ... # max number of tokens allowed for each generation
      prompt_template:
        template: ... # prompt template
        fewshot_example: ... # template used for the fewshot examples
        fewshot_label: ... # template used when few shot examples are presented as a multi-turn conversation
```

* `template` should contain the actual prompt template used to format the data. The template should also contain the text `{fewshot_example}`
* `fewshot_example` should contain the template to format the fewshot examples. Please ensure that the example begins with a double newline character (`\n\n`).
* `fewshot_label` should contain the answer tag `<tag>: <label>`. Label should be the actual text that the model is expected to output.

> [!tip]  
> **Using the YAML block syntax for more readable prompt templates**  
> Please use the following syntax to use the block style
> ```
> template: |-
>   text here
>   next line of text
> ```

<details>
<summary>Example `prompt_template` configuration (Indonesian sentiment task)</summary>

````yaml
      prompt_template:
        template: |-
          Apa sentimen dari kalimat berikut ini? Gunakan salah satu dari pilihan di bawah ini: Positif, Negatif, atau Netral.

          Jawablah hanya dengan menggunakan format berikut ini:
          Jawaban: $OPTION
          Ganti $OPTION dengan jawaban yang telah dipilih.{fewshot_examples}

          Kalimat:
          ```
          {text}
          ```
        fewshot_example: |2-


          Kalimat:
          ```
          {text}
          ```
          Jawaban: {label}
        fewshot_label: 'Jawaban: {label}'
````
</details>

### Generation params

The following generation parameters can be set individually for each task. Note that `max_tokens` should be set individually for each language.

```yaml
  temperature: ...
  languages:
    <lang>:
      max_tokens: ...
```

> [!note]  
> **Temperature and max token count for reasoning models**  
> The temperature for reasoning models are overwritten by the temperature specified in [seahelm_tasks/reasoning_generation_config.yaml](../seahelm_tasks/reasoning_generation_config.yaml) (default is 0.6). This was the default value specified for the DeepSeek R1 models.
>
> The max token count is also increased by the amount specified in `seahelm_tasks/reasoning_generation_config.yaml` (default is 20000). The max number of tokens is thus set to `20000 + max_tokens` where `max_tokens` is specified in the task config file.

## Writing the metric class
For each task, please create a new metric class and inherit from the base metric class `SeaHelmMetric`.

### Overview for the evaluation of responses
The metrics are calculated using the following steps in the `evaluate_responses()` function:
1. (Either) Drop error responses using `drop_error_responses()` or replace error responses using `replace_error_responses()`
2. Postprocess of responses `postprocess_responses()`
    - Default post processing steps are to extract out the answer using regex
    - Strip out "$" signs at the start and end of the text
    - Strip out any leading and trailing white spaces
3. Calculate the counts of unique responses
4. Calculate the metric using `calculate_metrics()`

### Calculation of metrics
Please ensure that `calculate_metrics()` is defined in the new metric class. Output expected from the function is a dictionary containing the various metrics and the inference pandas dataframe.

```python
def calculate_metrics(self) -> tuple[dict, pd.DataFrame]:
    predictions = self.inference_df[self.postprocessed_response_column] # get the processed responses
    references = self.inference_df[self.label_column] # get the references from the label_column

    # perform metric calculations here
    metric_a = ...

    # perform score normalization of metrics
    # min is 0 for generative task and 1/n_options for multiple choice tasks
    normalized_metric_a = 100 * self.normalize_score(metric_a, min, max)

    # calculate number of null cases (for multiple choice tasks)
    null_count = ...

    metrics = {
      "normalized_metric_a": normalized_metric_a,
      "null_count": null_count
    }
    return metrics, self.inference_df
```
