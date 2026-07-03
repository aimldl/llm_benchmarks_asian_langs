
# Plan
This .md file contains the plan to interact with Gemini-CLI to achieve my goal.

## Purpose
The ultimate goal is to evaluate the performance of Vertex AI Gemini with the SEA-HELM benchmark.
Thee versions of Gemini will be used:
- Gemini 2.5 Pro,
- Gemini 2.5 Flash,
- Gemini 2.5 Flash-lite.

To do the performance evaluation, I will run "run-vertex".
  $ ./run-vertex 

Therefore it is important to set up "run-vertex" correctly.

## Your mission



### Step 1. Study the content in serving_models.md
The following part enclosed by "----------" is the content of a Markdown file "SEA-HELM/docs/serving_models.md".

----------
# Model Serving
Inferencing in SEA-HELM is supported through the use of the vLLM and LiteLLM inference frameworks.
The following model types are accepted: `vllm`, `litellm`, `openai`, `none`

## vLLM
The `VLLMServing` class serves the model using the offline inference method found in vLLM. This allows for any model that is supported by vLLM to be served. Additionally, vLLM engine arguments can be configured using the `--model_args` cli argument. For the full list of engine args, please see the vLLM documentation on [Engine Args](https://docs.vllm.ai/en/latest/serving/engine_args.html#engine-args)

## LiteLLM
The `LiteLLMServing` class interfaces with the liteLLM package to provide support for closed source API servers such as OpenAI, Claude and Vertex. 

> [!Important]  
> **Specifying the model provider**  
> Please ensure that the model provider is specified using the `api_provider` in `--model_args`:  
> * Example (OpenAI): `api_provider=openai`
> * Example (Anthropic): `api_provider=anthropic`

It also supports the use of vLLM OpenAI-Compatible Server that is started using `vllm serve`. Please ensure that the correct `api_provider`, `base_url` and `api_key` are passed as one of the model_args. For example:
```
--model_args api_provider=openai,base_url=http://localhost:8000/v1,api_key=token-abc123
```

> [!Tip]  
> **Tokenization of prompts**  
> The evaluation framework will make an additional call to tokenize the prompts so as to gather statistics on the given prompt. If there are no tokenization end points available, please set the flag `--skip_tokenize_prompts`.

> [!Tip]  
> **Setting SSL verify to `False`**  
> To set SSL verify to false, please pass the key `ssl_verify=False` as one of the `--model_args`

## OpenAI (Batching API)
Support for the OpenAI Batch API is also provided. This provided a cost saving at the expense of potentially having to wait for longer if the OpenAI server are busy. To run this set:
```bash
--model_type openai
--model_args "api_provider=openai"
```

## None
Setting `model_type` to `none` is a special case to allow for the recalculation of evaluation metrics without any new inference being made. As such, no model will be loaded for vLLM and no API calls will be made. Please ensure that all the results are cached in the inference folder before running this.
----------

### Step 2. Help me modify the Bash script "run-vertex"

After this step, I will execute the Bash script "run-vertex".
I want to modify the following part in "run-vertex"

```bash
# Examples of MODEL_ARGS
#  OpenAI: --model_args "api_provider=openai,base_url=http://localhost:8000/v1,api_key=token-abc123"
#  Ollama: --model_args "api_provider=ollama,base_url=http://localhost:11434"

API_PROVIDER="vertex"  # TODO: Ensure this is correct 
PORT_NUMBER=""         # TODO: Find the correct port number for Vertex AI Gemini API. "http:///localpost:$PORT_NUMBER" 
API_KEY=`echo $GOOGLE_API_KEY`
MODEL_ARGS="api_provider=$API_PROVIDER,base_url=http://localhost:$PORT_NUMBER,api_key=$API_KEY"
```

See each line and help me find the right value for the variable.
Specifically, API_PROVIDER and PORT_NUMBER are my target.

To find the correct answer, study and review the codebase in the SEA-HELM directory.
And then refer to the above "Examples of MODEL_ARGS" both for OpenAI and Ollama.
Google-search the relevant parts and let me know some possible answers.

### Step 3. Modify the values for API_PROVIDER and PORT_NUMBER in the "run-vertex" file.
Based on the study, review and your Google-search results, interact with me modify the values for the variables.

### Step 4. Guide me through the commands to correctly run experiments to evaluate Vertex AI Gemini 2.5 Pro.
Do let me know if there are some prerequisites to run the experiments correctly.
For example, " $ ollama serve" should be executed in another Terminal.
And other crucial information should be provided.

### Step 5. Execute "run-vertex"
Run:
```bash
$ ./run-vertex
```

### Step 6. Debug errors and warnings until the successful execution of the performance evaluation
Running
```bash
$ ./run-vertex
```
may incur errors and warnings until the SEA-HELM performance evaluation is conducted successfully.

Help me debug the errors and warnings.


