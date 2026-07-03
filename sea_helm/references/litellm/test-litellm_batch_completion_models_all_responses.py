# test-litellm_batch_completion_models_all_responses.py
# https://docs.litellm.ai/docs/completion/batching#send-1-completion-call-to-many-models-return-all-responses
#
#  $ gcloud auth application-default login
#  (Log in for Vertex AI)
#  $ python test-litellm_batch_completion_models_all_responses.py

import litellm
import os
from litellm import batch_completion_models_all_responses

#os.environ['OPENAI_API_KEY'] = ""
#os.environ['ANTHROPIC_API_KEY'] = ""
#  Caution: Don't use GOOGLE_API_KEY. Instead, run:
#    $ gcloud auth application-default login
#    (Log in for Vertex AI)

responses = batch_completion_models_all_responses(
    models=["openai/gpt-3.5-turbo", "anthropic/claude-3-haiku-20240307", "vertex_ai/gemini-2.5-flash"],
    messages=[{"role": "user", "content": "Hey, how's it going"}]
)
print(responses)

for response in responses:
    # Now you can access each ModelResponse object
    print("-" * 20)
    print(f"Model: {response.model}")
    print(f"Content: {response.choices[0].message.content}")
