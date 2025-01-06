import openai
from dotenv import load_dotenv
import os
import json

# Load environment variables from .env
load_dotenv()

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Example transcripts
transcripts = [
    """Today I had a great meeting with Sarah about the new project launch. 
       We discussed timelines, responsibilities, and next steps. Super productive!""",
    """I spent the morning organizing my desk. It was so messy! 
       Afterward, I called Mom to catch up—she’s doing well.""",
    """Feeling a bit tired today but managed to go for a run. 
       Cleared my mind and made me feel better."""
]

# Function to extract meaningful information
def extract_meaningful_info(transcript):
    refined_prompt = (
        "Extract meaningful details from the following text in a friendly tone. "
        "Focus on these categories: "
        "1) Event details, 2) People involved, 3) Emotions, 4) Outcomes or actions, "
        "5) Memorable moments. Return the information in JSON format."
    )
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an assistant that extracts key details from text and formats them as JSON."},
            {"role": "user", "content": f"{refined_prompt}\n\nText:\n{transcript}"}
        ],
        max_tokens=150
    )
    return response["choices"][0]["message"]["content"].strip()

# Extract and store meaningful information in unstructured JSON
extracted_data = []
for idx, transcript in enumerate(transcripts, start=1):
    result = extract_meaningful_info(transcript)
    extracted_data.append(json.loads(result))  # Parse JSON response

# Save extracted data to a file
with open("extracted_data.json", "w") as f:
    json.dump(extracted_data, f, indent=4)

# Display extracted data
print(json.dumps(extracted_data, indent=4))