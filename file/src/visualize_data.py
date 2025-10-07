import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned data
df = pd.read_csv("data/cleaned_data.csv")

# Determine match outcome from home team perspective
def match_result(row):
    if row['home_score'] > row['away_score']:
        return "Home Win"
    elif row['home_score'] < row['away_score']:
        return "Away Win"
    else:
        return "Draw"

df['result'] = df.apply(match_result, axis=1)

# Plot result distribution
sns.set(style="whitegrid")
plt.figure(figsize=(7,5))
sns.countplot(data=df, x='result', palette='coolwarm')
plt.title("Match Result Distribution")
plt.xlabel("Result Type")
plt.ylabel("Number of Matches")
plt.tight_layout()
plt.show()



# ---- Goals per Month ----
plt.figure(figsize=(8,5))
sns.barplot(data=df, x='month', y='total_goals', estimator=sum, palette='crest')
plt.title("Total Goals Scored per Month")
plt.xlabel("Month")
plt.ylabel("Total Goals")
plt.tight_layout()
plt.show()



# ---- Average Goals per Weekday ----
plt.figure(figsize=(8,5))
sns.barplot(data=df, x='weekday', y='total_goals', estimator='mean', palette='mako')
plt.title("Average Goals per Match by Weekday")
plt.xlabel("Weekday")
plt.ylabel("Average Goals")
plt.tight_layout()
plt.show()



# ---- Top 10 Highest Scoring Teams ----
# Combine goals scored by each team (home + away)
home_goals = df.groupby("home_team")["home_score"].sum()
away_goals = df.groupby("away_team")["away_score"].sum()

total_goals_team = (home_goals + away_goals).sort_values(ascending=False).head(10)

# Convert to DataFrame for plotting
top_teams = total_goals_team.reset_index()
top_teams.columns = ["team", "total_goals"]

plt.figure(figsize=(10,6))
sns.barplot(data=top_teams, x="total_goals", y="team", palette="viridis")
plt.title("Top 10 Highest Scoring Teams of the Season")
plt.xlabel("Total Goals Scored")
plt.ylabel("Team")
plt.tight_layout()
plt.show()
