# src/data_fetcher.py
import sys
import os

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import API_KEY, BASE_URL  # Only import once

import requests
import pandas as pd

def get_matches(competition="PL", season=2024):
    """
    Fetches matches from the given competition and season.
    Saves the data to data/processed_data.csv
    """
    headers = {"X-Auth-Token": API_KEY}
    url = f"{BASE_URL}competitions/{competition}/matches?season={season}"
    
    response = requests.get(url, headers=headers)
    print("Status Code:", response.status_code)  # Debug

    if response.status_code != 200:
        print("Error fetching data:", response.text)
        return
    
    data = response.json()
    print("Total matches found:", len(data['matches']))  # Debug
    
    matches = []
    for m in data['matches']:
        matches.append({
            'date': m['utcDate'],
            'home_team': m['homeTeam']['name'],
            'away_team': m['awayTeam']['name'],
            'home_score': m['score']['fullTime']['home'],
            'away_score': m['score']['fullTime']['away'],
            'status': m['status']
        })
    
    df = pd.DataFrame(matches)
    df.to_csv("data/processed_data.csv", index=False)
    print("✅ Data saved to data/processed_data.csv")
    return df

# Run script directly
if __name__ == "__main__":
    get_matches()
