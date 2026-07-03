# test-litellm_vertex_ai_and_openai.py
# https://docs.litellm.ai/docs/completion/batching#send-1-completion-call-to-many-models-return-all-responses
#
#  $ gcloud auth application-default login
#  (Log in for Vertex AI)
#  $ python test-litellm_vertex_ai_and_openai.py

import litellm
from litellm import batch_completion_models_all_responses

# Set environment variables 'OPENAI_API_KEY'
# For Vertex AI, run:
#   $ gcloud auth application-default login

responses = batch_completion_models_all_responses(
    models=["openai/gpt-3.5-turbo", "vertex_ai/gemini-2.5-flash"],
    messages=[{"role": "user", "content": "Hello, SEA-HELM?"}]
)
for response in responses:
    # Now you can access each ModelResponse object
    print("-" * 20)
    print(f"Model: {response.model}")
    print(f"Content: {response.choices[0].message.content}")