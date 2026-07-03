import litellm
import os
from litellm import batch_completion_models

#os.environ['OPENAI_API_KEY'] = ""
#os.environ['ANTHROPIC_API_KEY'] = ""
#os.environ['GOOGLE_API_KEY='] = ""

responses = batch_completion_models(
    models=["gpt-3.5-turbo", "claude-instant-1.2", "gemini-2.5-flash"], 
    messages=[{"role": "user", "content": "Hey, how's it going"}]
)
#print(responses)
print(responses.choices[0].message.content)