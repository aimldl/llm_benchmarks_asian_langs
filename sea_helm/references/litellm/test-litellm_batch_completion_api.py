


from litellm import batch_completion

# Define a list of message lists for batch processing
messages_for_batch = [
    [{"role": "user", "content": "What is the capital of France?"}],
    [{"role": "user", "content": "Tell me a fun fact about pandas."}],
    [{"role": "user", "content": "Explain the concept of quantum entanglement."}]
]

# Call batch_completion
responses = batch_completion(
    model="gpt-3.5-turbo",  # Or any other supported model
    messages=messages_for_batch
)

# Process the responses
for response in responses:
    print(response.choices[0].message.content)