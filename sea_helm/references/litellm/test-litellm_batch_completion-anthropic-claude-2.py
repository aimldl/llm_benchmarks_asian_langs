import litellm
from litellm import completion
from litellm import batch_completion
#import os
#
# Sets an environment variable ANTHROPIC_API_KEY to a value.
#os.environ['ANTHROPIC_API_KEY'] = ""
#
# In my case, ANTHROPIC_API_KEY has already been saved in ~/.bashrc

# https://docs.litellm.ai/docs/providers/anthropic#usage
messages = [{"role": "user", "content": "Hey! how's it going?"}]
response = completion(model="claude-opus-4-20250514", messages=messages)
print(response)

# https://docs.litellm.ai/docs/completion/batching#example-code
# responses = batch_completion(
#     model="claude-2",
#     messages = [
#         [
#             {
#                 "role": "user",
#                 "content": "good morning? "
#             }
#         ],
#         [
#             {
#                 "role": "user",
#                 "content": "what's the time? "
#             }
#         ]
#     ]
# )
# print(responses)