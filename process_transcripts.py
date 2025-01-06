import openai
from dotenv import load_dotenv
import os
import json
from sqlalchemy import create_engine, Column, Integer, Text, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime, timezone

# SQLAlchemy setup
Base = declarative_base()
engine = create_engine("sqlite:///transcripts.db")
Session = sessionmaker(bind=engine)
session = Session()

# Define Transcript model
class Transcript(Base):
    __tablename__ = "transcripts"
    id = Column(Integer, primary_key=True)
    content = Column(Text, nullable=False)
    extracted_data = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

# Create the database table
Base.metadata.create_all(engine)

print("Database and table set up successfully.")

# Load environment variables from .env
load_dotenv()

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Function to load transcripts from a JSON file
def load_transcripts(file_path="transcripts.json"):
    with open(file_path, "r") as f:
        return json.load(f)

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

# Load transcripts from file
transcripts_data = load_transcripts()

# Extract and store meaningful information in the database
for entry in transcripts_data:
    transcript = entry["content"]
    result = extract_meaningful_info(transcript)
    
    # Strip and parse JSON response
    try:
        parsed_data = json.loads(result.strip())  # Ensure clean parsing
        # Save transcript and extracted data to database
        new_entry = Transcript(
            content=transcript,
            extracted_data=json.dumps(parsed_data),  # Store as JSON string
            timestamp=datetime.now(timezone.utc)
        )
        session.add(new_entry)
        print(f"Transcript {entry['id']} processed and added to the database.")
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON for Transcript {entry['id']}: {e}")
        continue

# Commit all changes to the database
session.commit()
print("All transcripts have been saved to the database.")