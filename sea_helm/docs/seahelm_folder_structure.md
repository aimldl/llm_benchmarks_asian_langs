# SEA-HELM Folder structure
```
.
├── README.md
├── chat_templates          # Folder with test datasets for SEA-HELM
│   ├── base_model.jinja    # base model chat template
│   └── llama_template_wo_sys_prompt.jinja
├── rouge_score                 # Folder containing XLSum's ROUGE metric implementation
│   └── ...
├── seahelm_tasks               # Folder with test datasets for SEA-HELM. Only a few notable example folders are expanded.
│   ├── cultural
│   │   └── kalahi
│   │       ├──data             # Folder containing the data for Kalahi
│   │       │   └── ...
│   │       ├──config.yaml      # YAML containing the task configuration 
│   │       └──kalahi.py        # Python file containing the metrics used in Kalahi
│   ├── instruction_following
│   │   └── ifeval
│   │       ├──data
│   │       │   └── ...
│   │       ├──config.yaml
│   │       ├──if_eval.py
│   │       └──instruction_checkers.py    # Python file containing the instruction checkers for each constraint
│   ├── lindsea
│   │   └── ...
│   ├── multi_turn
│   │   └── mt_bench
│   │       ├──data
│   │       │   └── ...
│   │       ├──config.yaml
│   │       ├──mt_bench_prompts.py      # Python file containing the LLM-as-a-Judge prompts
│   │       └──mt_bench.py
│   ├── nlg
│   │   └── ...
│   ├── nlr
│   │   └── ...
│   ├── nlu
│   │   └── ...
│   └──...                      # New competencies/tasks to be added
├── serving                     # Folder containing the wrappers to interface with the model serving frameworks
│   ├── base_serving.py         # Base serving class
│   ├── litellm_serving.py      # LiteLLM wrapper
│   ├── openai_serving.py       # openai_serving wrapper
│   └── vllm_serving.py         # vLLM wrapper
├── base_logger.py              # Logging codes
├── constants.py                # Constants needed for SEA-HELM evaluation metrics
├── requirements.txt
├── run_evaluation.sh           # Script for running evaluation
├── seahelm_evaluation.py       # Main SEA-HELM evaluation script
└── utils.py                    # Utility functions for SEA-HELM

```