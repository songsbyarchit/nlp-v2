import openai
import numpy as np
from scipy.spatial.distance import cosine
from dotenv import load_dotenv
import os
import json
from sqlalchemy import create_engine, Column, Integer, Text, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime, timezone, timedelta
from flask import Flask, render_template, request
import threading
import logging
import pprint  # For pretty-printing data
import re
from sqlalchemy import func
from generate_timestamps import generate_sample_timestamps

# Function to generate an embedding for a given text using OpenAI API
def generate_embedding(text):
    response = openai.Embedding.create(
        model="text-embedding-ada-002",  # Choose the correct model
        input=text
    )
    return response['data'][0]['embedding']

transcript_embeddings = {}  # This will store embeddings for your transcripts

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
    """
    Use OpenAI to dynamically extract and normalize date ranges from a user query.
    Handles typos, informal formats, quarters, and half-years.
    """
    logging.debug(f"User Prompt: {user_prompt}")  # Log user prompt

    # Refined prompt to query OpenAI
    refined_prompt = (
        "You are an assistant that extracts specific dates or date ranges from queries. "
        "Handle flexible formats such as '9Dec', 'dec 9th', 'Q1', 'H1', or typos like '9 decenbr'. "
        "Output in one of the following formats:\n"
        "1. 'YYYY-MM-DD' for a single date.\n"
        "2. 'YYYY-MM-DD to YYYY-MM-DD' for a range.\n\n"
        f"Query: {user_prompt}"
    )

    # Call OpenAI API to extract date or range
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You extract date ranges from text queries."},
                {"role": "user", "content": refined_prompt}
            ],
            max_tokens=50,  # Keep response concise
            temperature=0  # Ensure deterministic output
        )
        extracted_date = response['choices'][0]['message']['content'].strip()
        logging.debug(f"Extracted Date/Range: {extracted_date}")
        return extracted_date
    except Exception as e:
        logging.error(f"Error extracting date from prompt: {e}")
        return None

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

def generate_story(extracted_data):
    """
    This function takes the extracted event details and turns it into a full, human-readable story.
    It uses the AI to summarize the data into a coherent paragraph or story.
    """
    # Format the extracted event data for better story context
    prompt = (
        f"Please summarize the following event details into a coherent narrative so I can review what I did back then. Your response should be limited to 100 words:\n\n{extracted_data}\n\n"
        "Ensure that the summary is human-readable and flows like a natural narrative. Don't make the contents overly flowery. Keep it mostly matter of fact and don't extrapolate data which isn't given. But do include and make use of ALL data you do have. It can be upto 100 words but it doesn't have to be."
    )
    
    # Call OpenAI's API to generate a story
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an assistant that summarizes event details into a coherent story."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500  # Limit the AI response to ~100 words (tokens)
    )

    # Extract and return the AI-generated story
    story = response['choices'][0]['message']['content'].strip()
    return story

def process_query(user_prompt):
    # Extract the date or date range from the user prompt
    extracted_date = extract_date_from_prompt(user_prompt)
    logging.debug(f"Extracted date for query: {extracted_date}")

    # If no date range or date is extracted, you might want to search using embeddings
    if not extracted_date:
        logging.debug("No specific date range found. Proceeding with semantic search.")
        
        # Use semantic search to find the best matching transcripts based on query
        top_transcripts = find_best_matching_transcript(user_prompt)

        # Create a response for the user based on top matching transcripts
        response = ""
        for transcript_id, similarity in top_transcripts:
            # Retrieve the transcript content by ID
            transcript_entry = session.query(Transcript).get(transcript_id)
            response += f"\n{transcript_entry.content} (Similarity: {similarity:.2f})"
        
        if response:
            return f"Top matching transcripts based on your query: {response}"
        else:
            return "No matching transcripts found for your query."

    # Process date-based search if a date or range is extracted
    results = []
    
    # Check if the extracted date represents a larger period (like H1 or Q1)
    if "H1" in extracted_date or "Q1" in extracted_date:
        logging.debug("Large time period detected (e.g., H1, Q1). Using semantic search.")
        
        # Use the `find_best_matching_transcript` function to get matching transcripts based on embeddings
        top_transcripts = find_best_matching_transcript(user_prompt)

        # Collect results from the top matching transcripts
        for transcript_id, similarity in top_transcripts:
            # Retrieve the transcript content by ID
            transcript_entry = session.query(Transcript).get(transcript_id)
            results.append(f"Transcript ID: {transcript_id}, Similarity: {similarity:.2f}, Content: {transcript_entry.content}")
    
    elif "to" in extracted_date:  # Date range like "2024-01-01 to 2024-06-30"
        try:
            start_date_str, end_date_str = extracted_date.split(" to ")
            start_date = datetime.strptime(start_date_str.strip(), "%Y-%m-%d")
            end_date = datetime.strptime(end_date_str.strip(), "%Y-%m-%d")
            logging.debug(f"Start Date: {start_date}, End Date: {end_date}")

            # Query for transcripts in the date range
            for entry in session.query(Transcript).filter(Transcript.timestamp >= start_date, Transcript.timestamp <= end_date).all():
                logging.debug(f"Checking Transcript ID: {entry.id}, Timestamp: {entry.timestamp}, Content: {entry.content}")
                result = extract_meaningful_info(entry.content, user_prompt)
                results.append(result)

        except ValueError as e:
            logging.error(f"Error parsing date range: {e}")
            return "Invalid date range format. Please try again."

    else:  # Handle single date formats like "2024-01-01"
        try:
            extracted_date_obj = datetime.strptime(extracted_date, "%Y-%m-%d")  # "2024-01-01"
            extracted_date_formatted = extracted_date_obj.strftime("%Y-%m-%d")  # "2024-01-01"
            logging.debug(f"Formatted date for query: {extracted_date_formatted}")

            for entry in session.query(Transcript).filter(func.strftime('%Y-%m-%d', Transcript.timestamp) == extracted_date_formatted).all():
                logging.debug(f"Checking Transcript ID: {entry.id}, Timestamp: {entry.timestamp}, Content: {entry.content}")
                result = extract_meaningful_info(entry.content, user_prompt)
                results.append(result)

        except ValueError as e:
            logging.error(f"Date format conversion error: {e}")
            return "Invalid date format. Please try again with a valid date."

    # After collecting results, we need to summarize them in human-readable text.
    if results:
        # Combine the results and limit to 100 words
        combined_results = " ".join(results)

        # Generate a human-readable story (pass results to AI)
        story = generate_story(combined_results)

        return f"Story Summary: {story}"
    else:
        logging.debug("No results found for the extracted date.")
        return f"No events found for {extracted_date}."

def extract_meaningful_info(transcript, user_prompt):
    # Assuming the AI response is a well-formed JSON string.
    logging.debug(f"Processing transcript: {transcript[:50]}...")  # Log first 50 characters of transcript

    refined_prompt = (
        "You are an assistant that extracts key details from a given text and formats them into a clean, strictly valid JSON format. "
        "Please focus on capturing as much important information as possible and organize it under these categories: "
        "1) Event details: Include purpose, specific activities, timing, topics discussed, and any relevant descriptions. "
        "2) People involved: Include names, roles, relationships, and their status or contributions. "
        "3) Emotions: Capture both specific emotions (e.g., happy, frustrated) and transitions (e.g., initial and post-event feelings). "
        "4) Outcomes or actions: Highlight changes, decisions, completed tasks, or next steps taken. "
        "5) Memorable moments: Include significant highlights or anecdotes. "
        "6) Timing: Specify when the event occurred (e.g., morning, afternoon, evening, exact time)."
        "7) Relationships: Describe the relationship between people involved (e.g., family, professional, friend). Include any notable dynamics or history."
        "8) Multi-step emotions: Track how emotions changed before, during, and after the event (e.g., 'Feeling tired' → 'Feeling better')."
        "9) Outcomes linked to emotions: Highlight how specific actions or decisions were influenced by emotions (e.g., 'cleared mind' after a walk)."
        "Make sure all critical information is captured and nothing important is omitted. "
        "Here’s an example of the output format:\n\n"
        '{\n'
        '  "Event details": {\n'
        '    "Meeting purpose": "Discuss the new project launch",\n'
        '    "Topics discussed": ["Timelines", "Responsibilities", "Next steps"]\n'
        '  },\n'
        '  "People involved": ["Sarah"],\n'
        '  "Emotions": ["Positive", "Productive"],\n'
        '  "Outcomes or actions": "Discussion on timelines, responsibilities, and next steps",\n'
        '  "Memorable moments": "Great meeting"\n'
        '}\n\n'
        "Text to analyze:\n"
        f"{transcript}"
    )

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an assistant that extracts key details from text and formats them as JSON."},
            {"role": "user", "content": f"{refined_prompt}\n\nText:\n{transcript}"}
        ],
        max_tokens=4096
    )
    
    # Parse the response (assuming it's valid JSON)
    response_text = response['choices'][0]['message']['content']
    try:
        response_json = json.loads(response_text)
        # Extract the relevant fields from the JSON
        summary = json.dumps(response_json, indent=2)
        return summary
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding response JSON: {e}")
        return "Error processing the response."

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

for entry_index, entry in enumerate(transcripts_data):  # Use enumerate to get entry_index
    transcript = entry["content"]
    
    # Check if transcript content already exists in the database
    existing_entry = session.query(Transcript).filter_by(content=transcript).first()
    if existing_entry:
        logging.info(f"Transcript {entry['id']} already exists in the database. Skipping processing.")
        print("\n")  # Add a blank line for readability
        continue

    logging.info(f"Processing Transcript {entry['id']}...\n")  # Debugging line with a blank line after it
    
    user_prompt = "Default query for processing all transcripts"  # Or leave it as empty ""
    # Process and add new transcript
    result = extract_meaningful_info(transcript, user_prompt)

    # Call the safe_parse_json function to handle parsing automatically
    parsed_data = safe_parse_json(result.strip())

    # Generate an embedding for the transcript content and store it
    embedding = generate_embedding(transcript)  # Generate embedding for the transcript
    transcript_embeddings[entry.id] = embedding  # Store embedding by transcript ID

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

        # Log what was added in a pretty format
        print("\n--- New Transcript Added to Database ---")
        pprint.pprint({
            "Content": transcript,
            "Extracted Data": parsed_data,
            "Timestamp": timestamp_obj.strftime("%Y-%m-%d %H:%M:%S"),
            "Embedding": embedding  # Optionally log the embedding
        }, indent=2)  # Pretty print with indentation
        print("\n")  # Add another blank line after the log

        logging.info(f"Transcript {entry['id']} processed and added to the database successfully.\n")
    else:
        logging.error(f"Error parsing JSON for Transcript {entry['id']}: Failed to clean and parse JSON.")
        print("\n")  # Add a blank line for readability
        continue

# Commit all changes to the database
session.commit()
logging.info("\nAll transcripts have been saved to the database.\n")  # Add blank lines for final message

def find_best_matching_transcript(user_query):
    query_embedding = generate_embedding(user_query)  # Get embedding for the query
    
    # Find the best matching transcript by calculating cosine similarity
    similarities = []
    for transcript_id, embedding in transcript_embeddings.items():
        similarity = 1 - cosine(query_embedding, embedding)
        similarities.append((transcript_id, similarity))

    # Sort by similarity, highest first
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Return the top 3 most similar transcripts (adjust as needed)
    top_transcripts = similarities[:3]
    return top_transcripts

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