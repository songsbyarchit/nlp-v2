import random
from datetime import datetime, timedelta

def generate_sample_timestamps():
    # Set the start date to January 1, 2024
    start_date = datetime(2024, 1, 1)

    # Generate a timestamp for every day of 2024
    sample_timestamps = []

    for day in range(366):  # 2024 is a leap year, so 366 days
        # Randomize time of day (hours, minutes, seconds)
        random_hour = random.randint(0, 23)
        random_minute = random.randint(0, 59)
        random_second = random.randint(0, 59)

        # Create a timestamp for the current day with the random time
        timestamp = start_date + timedelta(days=day, hours=random_hour, minutes=random_minute, seconds=random_second)

        # Append the formatted timestamp to the list
        sample_timestamps.append(timestamp.strftime("%Y-%m-%d %H:%M:%S"))

    return sample_timestamps