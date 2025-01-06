import tiktoken
import json

def count_tokens(text, model="gpt-3.5-turbo"):
    """
    Counts the number of tokens in the given text for the specified model.

    :param text: The input text to tokenize.
    :param model: The OpenAI model for which tokens are counted.
    :return: The number of tokens in the text.
    """
    # Get the encoding for the specified model
    encoding = tiktoken.encoding_for_model(model)
    
    # Encode the text and count the number of tokens
    num_tokens = len(encoding.encode(text))
    return num_tokens

if __name__ == "__main__":
    # Load the JSON data file
    json_file = "transcripts.json"  # Update with your JSON file path
    with open(json_file, "r") as f:
        data = json.load(f)

    # Loop through each transcript and count tokens
    model = "gpt-3.5-turbo"
    total_tokens = 0
    for index, entry in enumerate(data):
        content = entry.get("content", "")
        num_tokens = count_tokens(content, model=model)
        total_tokens += num_tokens
        print(f"Transcript {index + 1}: {num_tokens} tokens")

    print(f"Total tokens for all transcripts: {total_tokens}")