import random
from datetime import datetime, timedelta

def generate_sample_timestamps():
    # Set the start date (December 1, 2024)
    start_date = datetime(2024, 12, 1)

    # Generate 27 sample timestamps, one for each day from Dec 1 to Dec 27
    sample_timestamps = []

    for day in range(27):
        # Get a random time for each day (from 00:00 to 23:59)
        random_hour = random.randint(0, 23)
        random_minute = random.randint(0, 59)
        random_second = random.randint(0, 59)

        # Create a timestamp for the current day with the random time
        timestamp = start_date + timedelta(days=day, hours=random_hour, minutes=random_minute, seconds=random_second)

        # Append the formatted timestamp to the list
        sample_timestamps.append(timestamp.strftime("%Y-%m-%d %H:%M:%S"))

    return sample_timestamps