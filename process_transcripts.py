import openai
from dotenv import load_dotenv
import os
import json
from sqlalchemy import create_engine, Column, Integer, Text, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime, timezone
from flask import Flask, render_template, request
import threading
import logging
import re
from sqlalchemy import func
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

def extract_date_from_prompt(user_prompt):
    logging.debug(f"User Prompt: {user_prompt}")  # Log user prompt

    date_found = None
    current_date = datetime.now()
    current_year = current_date.year

    # If it's before December, the latest December was last year
    if current_date.month < 12:
        latest_december_year = current_year - 1
    else:
        latest_december_year = current_year

    date_match = re.search(r"(december (\d{1,2})(st|nd|rd|th)?(?:, (\d{4}))?)", user_prompt, re.IGNORECASE)
    
    if date_match:
        day = date_match.group(2)
        year = date_match.group(4) if date_match.group(4) else latest_december_year
        date_found = f"december {day}, {year}"
        logging.debug(f"Extracted Date: {date_found}")
    elif re.search(r"december", user_prompt, re.IGNORECASE):
        date_found = f"december {latest_december_year}"
        logging.debug(f"Extracted Month Only: {date_found}")

    if not date_found:
        logging.debug("No date extracted from user prompt.")
    
    return date_found

def parse_query_for_dates(user_prompt):
    # Basic regex to match date-related queries, e.g., "December 3rd", "second week of December 2024"
    # This can be expanded further for more complex queries
    date_match = re.search(r"(December \d{1,2}(st|nd|rd|th)?(, \d{4})?)", user_prompt, re.IGNORECASE)
    if date_match:
        # Parse the date to datetime object
        date_str = date_match.group(1)
        try:
            # Handle specific dates like "December 3rd"
            date_obj = datetime.strptime(date_str, "%B %d, %Y") if ',' in date_str else datetime.strptime(date_str, "%B %d")
            return date_obj
        except ValueError:
            # Handle errors in date parsing if format doesn't match
            logging.error(f"Date parsing failed for query: {user_prompt}")
            return None
    else:
        # Return None if no date is found in the query
        return None

def process_query(user_prompt):
    extracted_date = extract_date_from_prompt(user_prompt)
    logging.debug(f"Extracted date for query: {extracted_date}")

    if extracted_date:
        results = []

        # Handle the case where user only provides a month-year (like "December 2024")
        if len(extracted_date.split()) == 2:  # month and year format, e.g. "December 2024"
            # Extract month and year, convert month name to number (e.g., "December" -> "12")
            month_str, year_str = extracted_date.split()
            month_number = datetime.strptime(month_str, "%B").month  # Convert month name to month number
            extracted_month_year = f"{month_number:02d}-{year_str}"
            logging.debug(f"Formatted month-year for query: {extracted_month_year}")

            # Query the database with the formatted month-year
            for entry in session.query(Transcript).filter(func.strftime('%m-%Y', Transcript.timestamp) == extracted_month_year).all():
                logging.debug(f"Checking Transcript ID: {entry.id}, Timestamp: {entry.timestamp}, Content: {entry.content}")
                result = extract_meaningful_info(entry.content, user_prompt)
                results.append(result)

        else:  # This means we have a full date like "December 3, 2024"
            # Try to convert it into a valid date (YYYY-MM-DD)
            try:
                extracted_date_obj = datetime.strptime(extracted_date, "%B %d, %Y")  # "December 3, 2024"
                extracted_date_formatted = extracted_date_obj.strftime("%Y-%m-%d")  # "2024-12-03"
                logging.debug(f"Formatted date for query: {extracted_date_formatted}")

                # Query the database with the formatted date
                for entry in session.query(Transcript).filter(func.strftime('%Y-%m-%d', Transcript.timestamp) == extracted_date_formatted).all():
                    logging.debug(f"Checking Transcript ID: {entry.id}, Timestamp: {entry.timestamp}, Content: {entry.content}")
                    result = extract_meaningful_info(entry.content, user_prompt)
                    results.append(result)

            except ValueError as e:
                logging.error(f"Date format conversion error: {e}")
                return "Invalid date format. Please try again with a valid date."

        if results:
            return " ".join(" ".join(results).split()[:100])  # Limit to 100 words for brevity
        else:
            logging.debug("No results found for the extracted date.")
            return f"No events found for {extracted_date}."
    else:
        logging.debug("No date was extracted from the user prompt.")
        return "Sorry, I could not understand your query."

def extract_meaningful_info(transcript, user_prompt):
    logging.debug(f"Processing transcript: {transcript[:50]}...")  # Log first 50 characters of transcript
    logging.debug(f"User Prompt: {user_prompt}")  # Log user prompt

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
        "Your task is to extract the relevant information from the transcript, "
        "but also match the date range or event details based on the following query: "
        f"User's query: {user_prompt}. "
        "Do not include any extra text outside of the JSON structure or I will discard your response."
    )

    logging.debug(f"Refined Prompt for AI: {refined_prompt}")  # Log the exact prompt sent to AI

    # Call OpenAI's API with the enhanced prompt
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an assistant that extracts key details from text and formats them as JSON."},
            {"role": "user", "content": f"{refined_prompt}\n\nText:\n{transcript}"}
        ],
        max_tokens=200
    )

    # Log the API response for debugging
    logging.debug(f"OpenAI Response: {response['choices'][0]['message']['content']}")
    
    # Return the cleaned result
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
    
    user_prompt = "Default query for processing all transcripts"  # Or leave it as empty ""
    # Process and add new transcript
    result = extract_meaningful_info(transcript, user_prompt)

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

@app.route("/query", methods=["GET"])
def query_form():
    return render_template("query_form.html")

@app.route("/query", methods=["POST"])
def query():
    # Get the user's prompt from the form
    user_prompt = request.form["query"]
    
    # Log the user prompt
    logging.debug(f"User Prompt: {user_prompt}")
    
    # Process the query with the user prompt
    response = process_query(user_prompt)  # Pass user_prompt here
    
    # Render the query results page with the response
    return render_template("query_results.html", response=response, prompt=user_prompt)

if __name__ == "__main__":
    app.run(debug=True)