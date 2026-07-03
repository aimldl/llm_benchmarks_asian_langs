# test-any_llm_api_via_litellm.py
#
# See the official manual at https://openai.github.io/openai-agents-python/models/litellm/
#
# Step 1. test.anthropic.py
#   $ pip install anthropic
#   $ python test-anthropic.py
#   [TextBlock(citations=None, text="Hello! It's nice to meet you. How are you doing today?", type='text')]
#   $
#
# Step 2. test-any_llm_api_via_litellm.py
#   First, install
#   $ pip install "openai-agents[litellm]"
#
#   Anthropic
#   The model name in the official manual or "claude-3-5-sonnet-20240620" fails. 
#   $ ANTHROPIC_MODEL="anthropic/claude-3-5-sonnet-20240620"
#   $ ANTHROPIC_API_KEY=`echo $ANTHROPIC_API_KEY`
#   $ 
#   $ python test-any_llm_api_via_litellm.py --model $ANTHROPIC_MODEL --api-key $ANTHROPIC_API_KEY
#     ...
#   litellm.exceptions.NotFoundError: litellm.NotFoundError: AnthropicException - {"type":"error","error":{"type":"not_found_error","message":"model: claude-3-5-sonnet-20240620"},"request_id":"req_011CSJQJzfR9MYB3hu5MGsFK"}
#   $

#   "claude-3-haiku-20240307" works fine.
#   $ ANTHROPIC_MODEL="anthropic/claude-3-haiku-20240307"
#   $ ANTHROPIC_API_KEY=`echo $ANTHROPIC_API_KEY`
#   $ 
#   $ python test-any_llm_api_via_litellm.py --model $ANTHROPIC_MODEL --api-key $ANTHROPIC_API_KEY
#   Clouds drift overhead,
#   Chilly winds whisper through streets,
#   Tokyo's forecast, cold.
#   $ 

#   OpenAI > "openai/gpt-4.1"
#   $ OPENAI_MODEL="openai/gpt-4.1"
#   $ OPENAI_API_KEY=`echo $OPENAI_API_KEY`
#   $ 
#   $ python test-any_llm_api_via_litellm.py --model $OPENAI_MODEL --api-key $OPENAI_API_KEY
#   
#   Give Feedback / Get Help: https://github.com/BerriAI/litellm/issues/new
#   LiteLLM.Info: If you need to debug this error, use `litellm._turn_on_debug()'.
#   
#   Traceback (most recent call last):
#     File "/home/user/.pyenv/versions/3.11.9/lib/python3.11/site-packages/litellm/llms/openai/openai.py", line 812, in acompletion
#       headers, response = await self.make_openai_chat_completion_request(
#                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

#   
#     File "/home/user/.pyenv/versions/3.11.9/lib/python3.11/site-packages/litellm/litellm_core_utils/exception_mapping_utils.py", line 329, in exception_type
#       raise RateLimitError(
#   litellm.exceptions.RateLimitError: litellm.RateLimitError: RateLimitError: OpenAIException - You exceeded your current quota, please check your plan and billing details. For more information on this error, read the docs: https://platform.openai.com/docs/guides/error-codes/api-errors.
#
#   That error means your API request was rejected because you’ve run out of credits (quota) or your account doesn’t have an active paid plan set up. Here’s how you can fix it:
#   1. Check your billing status at https://platform.openai.com/account/billing
#   2. Upgrade or add more credits
#   3. Verify your usage at https://platform.openai.com/usage
#      Chat Completions > 2 requests 145 input tokens
#   4. Wait for quota reset (if applicable)
#
#   $ OPENAI_MODEL="openai/gpt-4.1"
#   $ OPENAI_API_KEY=`echo $OPENAI_API_KEY`
#   $
#   $ python test-any_llm_api_via_litellm.py --model $OPENAI_MODEL --api-key $OPENAI_API_KEY
#   [debug] getting weather for Tokyo
#   Tokyo skies so bright,
#   Golden sun upon the streets,
#   Warmth in every light.
#   $

#python test-any_llm_api_via_litellm.py --model "openai/gpt-4.1" --api-key [OPENAI_API_KEY]

#   Vertex AI
#   https://docs.litellm.ai/docs/providers/vertex
#   $ VERTEX_AI_MODEL="vertex_ai/gemini-1.5-pro"
#   $ VERTEX_AI_MODEL="vertex_ai/gemini-2.5-pro"
#   $ VERTEX_API_KEY=`echo $GOOGLE_API_KEY`
#   $
#   $ python test-any_llm_api_via_litellm.py --model $VERTEX_AI_MODEL --api-key $VERTEX_API_KEY



from __future__ import annotations

import asyncio

from agents import Agent, Runner, function_tool, set_tracing_disabled
from agents.extensions.models.litellm_model import LitellmModel

@function_tool
def get_weather(city: str):
    print(f"[debug] getting weather for {city}")
    return f"The weather in {city} is sunny."

async def main(model: str, api_key: str):
    agent = Agent(
        name="Assistant",
        instructions="You only respond in haikus.",
        model=LitellmModel(model=model, api_key=api_key),
        tools=[get_weather],
    )

    result = await Runner.run(agent, "What's the weather in Tokyo?")
    print(result.final_output)

if __name__ == "__main__":
    # First try to get model/api key from args
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=False)
    parser.add_argument("--api-key", type=str, required=False)
    args = parser.parse_args()

    model = args.model
    if not model:
        model = input("Enter a model name for Litellm: ")

    api_key = args.api_key
    if not api_key:
        api_key = input("Enter an API key for Litellm: ")

    asyncio.run(main(model, api_key))