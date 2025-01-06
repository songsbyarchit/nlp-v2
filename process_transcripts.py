import openai
from dotenv import load_dotenv
import os
import json
from sqlalchemy import create_engine, Column, Integer, Text, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime, timezone
from flask import Flask, render_template
import threading
import logging
from generate_timestamps import generate_sample_timestamps

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# SQLAlchemy setup
Base = declarative_base()
engine = create_engine("sqlite:///transcripts.db")
Session = sessionmaker(bind=engine)
session = Session()

timestamps = generate_sample_timestamps()  # Get the list of timestamps

# Define Transcript model
class Transcript(Base):
    __tablename__ = "transcripts"
    id = Column(Integer, primary_key=True)
    content = Column(Text, nullable=False)
    extracted_data = Column(Text, nullable=False)
    timestamp = Column(DateTime, nullable=False)  # Add this line for the timestamp column

# Create the database table
Base.metadata.create_all(engine)

print("Database and table set up successfully.")

# Load environment variables from .env
load_dotenv()

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)

# Function to load transcripts from a JSON file
def load_transcripts(file_path="transcripts.json"):
    with open(file_path, "r") as f:
        return json.load(f)

@app.route("/")
def index():
    # Query all rows from the database
    transcripts = session.query(Transcript).all()
    
    # Pass the data to the template
    return render_template("index.html", transcripts=transcripts)

def extract_meaningful_info(transcript):
    refined_prompt = (
        "You are an assistant that extracts key details from a given text. "
        "Please format the extracted information into a clean, strictly valid JSON format. "
        "Ensure the JSON is complete, with no extra commas, improper quotation marks, or missing values. "
        "Here are the categories to focus on: "
        "1) Event details (date, time, subject), "
        "2) People involved (names, roles), "
        "3) Emotions (e.g., happy, sad, excited), "
        "4) Outcomes or actions (e.g., tasks completed, next steps), "
        "5) Memorable moments (significant or notable events). "
        "Do not include any extra text outside of the JSON structure or I'll never use this LLM API ever again."
    )
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an assistant that extracts key details from text and formats them as JSON."},
            {"role": "user", "content": f"{refined_prompt}\n\nText:\n{transcript}"}
        ],
        max_tokens=200
    )
    
    # Log the response for debugging purposes
    logging.debug(f"OpenAI response for Transcript: {response['choices'][0]['message']['content']}")
    
    return response["choices"][0]["message"]["content"].strip()

def safe_parse_json(text):
    """Attempt to parse the result and fix common issues."""
    try:
        # Attempt parsing
        return json.loads(text)
    except json.JSONDecodeError as e:
        # Add fallback: try cleaning common issues like fancy quotes, extra commas
        text = text.replace('“', '"').replace('”', '"')  # Replace fancy quotes
        text = text.strip().rstrip(",")  # Remove any trailing commas
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse JSON after cleaning: {e}")
            return None

# Load transcripts from file
transcripts_data = load_transcripts()

# Extract and store meaningful information in the database
for entry_index, entry in enumerate(transcripts_data):  # Use enumerate to get entry_index
    transcript = entry["content"]
    
    # Check if transcript content already exists in the database
    existing_entry = session.query(Transcript).filter_by(content=transcript).first()
    if existing_entry:
        logging.info(f"Transcript {entry['id']} already exists in the database. Skipping processing.")
        continue

    logging.info(f"Processing Transcript {entry['id']}...")  # Debugging line

    # Process and add new transcript
    result = extract_meaningful_info(transcript)

    # Call the safe_parse_json function to handle parsing automatically
    parsed_data = safe_parse_json(result.strip())

    # Check if parsing was successful before proceeding
    if parsed_data:
        # Use the entry_index to pick the correct timestamp
        timestamp_obj = datetime.strptime(timestamps[entry_index], "%Y-%m-%d %H:%M:%S")

        new_entry = Transcript(
            content=transcript,
            extracted_data=json.dumps(parsed_data),  # Store as JSON string
            timestamp=timestamp_obj  # Use the datetime object instead of the string
        )
        session.add(new_entry)
        logging.info(f"Transcript {entry['id']} processed and added to the database successfully.")
    else:
        logging.error(f"Error parsing JSON for Transcript {entry['id']}: Failed to clean and parse JSON.")
        continue

# Commit all changes to the database
session.commit()
logging.info("All transcripts have been saved to the database.")

if __name__ == "__main__":
    app.run(debug=True)