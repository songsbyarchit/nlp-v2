import openai
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Confirm API setup
print("OpenAI API initialized successfully.")