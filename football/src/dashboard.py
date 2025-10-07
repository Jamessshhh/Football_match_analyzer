# src/dashboard.py

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for deployment
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# -----------------------------
# Load cleaned data
# -----------------------------
df = pd.read_csv("data/cleaned_data.csv")

# Ensure 'date' is datetime
df['date'] = pd.to_datetime(df['date'])

# -----------------------------
# Compute match result if missing
# -----------------------------
def match_result(row):
    if row['home_score'] > row['away_score']:
        return "Home Win"
    elif row['home_score'] < row['away_score']:
        return "Away Win"
    else:
        return "Draw"

if 'result' not in df.columns:
    df['result'] = df.apply(match_result, axis=1)

# -----------------------------
# Encode categorical variables
# -----------------------------
le_home = LabelEncoder()
le_away = LabelEncoder()
le_weekday = LabelEncoder()

df['home_team_enc'] = le_home.fit_transform(df['home_team'])
df['away_team_enc'] = le_away.fit_transform(df['away_team'])
df['weekday_enc'] = le_weekday.fit_transform(df['date'].dt.day_name())

# -----------------------------
# Features and target
# -----------------------------
features = [
    'home_team_enc', 'away_team_enc', 'month', 'weekday_enc',
    'home_score', 'away_score', 'total_goals',
    'home_team_recent_form', 'away_team_recent_form',
    'head_to_head_home_win', 'head_to_head_away_win', 'head_to_head_draw'
]

# Compute month & total goals
df['month'] = df['date'].dt.month
df['total_goals'] = df['home_score'] + df['away_score']

X = df[features]
y = df['result']

# -----------------------------
# Train model
# -----------------------------
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X, y)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("⚽ Football Match Predictor")

# Inputs
home_team = st.selectbox("Select Home Team", df['home_team'].unique())
away_team = st.selectbox("Select Away Team", df['away_team'].unique())
month = st.slider("Match Month", 1, 12, 5)
weekday = st.selectbox("Match Weekday", df['date'].dt.day_name().unique())
home_score = st.number_input("Average Home Goals", min_value=0, max_value=10, value=1)
away_score = st.number_input("Average Away Goals", min_value=0, max_value=10, value=1)
total_goals = home_score + away_score

# Recent form
home_recent = df[df['home_team']==home_team]['home_team_recent_form'].iloc[-1]
away_recent = df[df['away_team']==away_team]['away_team_recent_form'].iloc[-1]

# Head-to-head
h2h = df[(df['home_team']==home_team) & (df['away_team']==away_team)]
if not h2h.empty:
    h2h_home_win = h2h['head_to_head_home_win'].iloc[-1]
    h2h_away_win = h2h['head_to_head_away_win'].iloc[-1]
    h2h_draw = h2h['head_to_head_draw'].iloc[-1]
else:
    h2h_home_win = h2h_away_win = h2h_draw = 0.33  # default equal probability

# Predict button
if st.button("Predict Result"):
    input_df = pd.DataFrame({
        'home_team_enc':[le_home.transform([home_team])[0]],
        'away_team_enc':[le_away.transform([away_team])[0]],
        'month':[month],
        'weekday_enc':[le_weekday.transform([weekday])[0]],
        'home_score':[home_score],
        'away_score':[away_score],
        'total_goals':[total_goals],
        'home_team_recent_form':[home_recent],
        'away_team_recent_form':[away_recent],
        'head_to_head_home_win':[h2h_home_win],
        'head_to_head_away_win':[h2h_away_win],
        'head_to_head_draw':[h2h_draw]
    })

    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    classes = model.classes_

    st.success(f"Predicted Match Result: {prediction}")

    # Show probabilities
    prob_df = pd.DataFrame({'Result': classes, 'Probability': probabilities*100})
    st.subheader("Predicted Probabilities (%)")
    st.bar_chart(prob_df.set_index('Result'))

    # Show last 5 matches trends
    st.subheader("Last 5 Matches Points Trend")
    home_last5 = df[df['home_team']==home_team].tail(5)['home_team_recent_form']
    away_last5 = df[df['away_team']==away_team].tail(5)['away_team_recent_form']

    fig, ax = plt.subplots()
    ax.plot(range(1,6), home_last5, marker='o', label=f'{home_team} Recent Form')
    ax.plot(range(1,6), away_last5, marker='o', label=f'{away_team} Recent Form')
    ax.set_xlabel("Last 5 Matches")
    ax.set_ylabel("Average Points")
    ax.set_xticks(range(1,6))
    ax.legend()
    st.pyplot(fig)
