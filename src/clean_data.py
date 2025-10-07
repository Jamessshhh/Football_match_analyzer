# src/clean_data.py
import pandas as pd

# ---------------------------
# Load processed data
# ---------------------------
df = pd.read_csv("data/processed_data.csv")

# Convert date to datetime and sort
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

# ---------------------------
# Compute match result
# ---------------------------
def match_result(row):
    if row['home_score'] > row['away_score']:
        return "Home Win"
    elif row['home_score'] < row['away_score']:
        return "Away Win"
    else:
        return "Draw"

df['result'] = df.apply(match_result, axis=1)

# ---------------------------
# Function to calculate points
# ---------------------------
def points(result, team_type):
    if result == "Home Win" and team_type == "home":
        return 3
    elif result == "Away Win" and team_type == "away":
        return 3
    elif result == "Draw":
        return 1
    else:
        return 0

# ---------------------------
# Initialize recent form columns
# ---------------------------
df['home_team_recent_form'] = 0.0
df['away_team_recent_form'] = 0.0

# ---------------------------
# Initialize Head-to-Head columns
# ---------------------------
df['head_to_head_home_win'] = 0
df['head_to_head_away_win'] = 0
df['head_to_head_draw'] = 0

# ---------------------------
# Calculate Recent Form and Head-to-Head
# ---------------------------
teams = pd.concat([df['home_team'], df['away_team']]).unique()

for team in teams:
    team_matches = df[(df['home_team'] == team) | (df['away_team'] == team)].sort_values('date').reset_index()
    recent_points = []

    for idx, row in team_matches.iterrows():
        # -------------------
        # Recent Form (last 5 matches)
        # -------------------
        team_type = "home" if row['home_team'] == team else "away"
        pts = points(row['result'], team_type)
        recent_points.append(pts)
        last5 = recent_points[-5:]
        avg_last5 = sum(last5) / len(last5)
        
        if team_type == "home":
            df.loc[row['index'], 'home_team_recent_form'] = avg_last5
        else:
            df.loc[row['index'], 'away_team_recent_form'] = avg_last5

        # -------------------
        # Head-to-Head (historical matches between home and away teams)
        # -------------------
        # Only consider matches before current match
        past_matches = df[(df['date'] < row['date']) & 
                          (((df['home_team']==row['home_team']) & (df['away_team']==row['away_team'])) |
                           ((df['home_team']==row['away_team']) & (df['away_team']==row['home_team'])))]
        
        h2h_home_win = len(past_matches[past_matches['result']=='Home Win'])
        h2h_away_win = len(past_matches[past_matches['result']=='Away Win'])
        h2h_draw = len(past_matches[past_matches['result']=='Draw'])
        
        df.loc[row['index'], 'head_to_head_home_win'] = h2h_home_win
        df.loc[row['index'], 'head_to_head_away_win'] = h2h_away_win
        df.loc[row['index'], 'head_to_head_draw'] = h2h_draw

# ---------------------------
# Save cleaned data
# ---------------------------
df.to_csv("data/cleaned_data.csv", index=False)
print("✅ Cleaned data with Recent Form and Head-to-Head stats saved to data/cleaned_data.csv")
