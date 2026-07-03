# test-anthropic.py
# Run
#   $ pip install anthropic
# before running:
#   $ python test-anthropic.py
#   [TextBlock(citations=None, text="Hello! It's nice to meet you. How are you doing today?", type='text')]
#   $

import anthropic
import os

client = anthropic.Anthropic(
    # defaults to os.environ.get("ANTHROPIC_API_KEY")
    #api_key="my_api_key",
    # Save the my_api_key in an environment variable "ANTHROPIC_API_KEY"
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
)
message = client.messages.create(
    model="claude-opus-4-1-20250805",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Hello, Claude"}
    ]
)
print(message.content)
# Expected output:
#[TextBlock(citations=None, text="Hello! It's nice to meet you. How are you doing today?", type='text')]
