# src/dashboard.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# ---------------------------
# Load cleaned data
# ---------------------------
df = pd.read_csv("data/cleaned_data.csv")
df['date'] = pd.to_datetime(df['date'])

# ---------------------------
# Encode categorical variables
# ---------------------------
le_home = LabelEncoder()
le_away = LabelEncoder()
le_weekday = LabelEncoder()

df['home_team_enc'] = le_home.fit_transform(df['home_team'])
df['away_team_enc'] = le_away.fit_transform(df['away_team'])
df['weekday_enc'] = le_weekday.fit_transform(df['date'].dt.day_name())

# ---------------------------
# Features for prediction
# ---------------------------
features = [
    'home_team_enc', 'away_team_enc', 'home_score', 'away_score', 'total_goals',
    'home_team_recent_form', 'away_team_recent_form',
    'head_to_head_home_win', 'head_to_head_away_win', 'head_to_head_draw'
]

if 'total_goals' not in df.columns:
    df['total_goals'] = df['home_score'] + df['away_score']

X = df[features]
y = df['result']

# ---------------------------
# Train Random Forest Model
# ---------------------------
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X, y)

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("⚽ Football Match Predictor")

# Dynamic Month & Season Selection
st.subheader("Match Date Selection")
month = st.slider("Match Month", 1, 12, 5)
year = st.selectbox("Season Year", sorted(df['date'].dt.year.unique()))

# User Inputs
home_team = st.selectbox("Select Home Team", df['home_team'].unique())
away_team = st.selectbox("Select Away Team", df['away_team'].unique())
home_score = st.number_input("Average Home Goals", min_value=0, max_value=10, value=1)
away_score = st.number_input("Average Away Goals", min_value=0, max_value=10, value=1)
total_goals = home_score + away_score

# ---------------------------
# Predict Button
# ---------------------------
if st.button("Predict Result"):
    # Recent Form
    home_recent = df[df['home_team']==home_team]['home_team_recent_form'].iloc[-1]
    away_recent = df[df['away_team']==away_team]['away_team_recent_form'].iloc[-1]

    # Head-to-Head stats
    past_matches = df[
        ((df['home_team']==home_team) & (df['away_team']==away_team)) |
        ((df['home_team']==away_team) & (df['away_team']==home_team))
    ]
    h2h_home_win = len(past_matches[past_matches['result']=='Home Win'])
    h2h_away_win = len(past_matches[past_matches['result']=='Away Win'])
    h2h_draw = len(past_matches[past_matches['result']=='Draw'])

    # Prepare input for model
    input_df = pd.DataFrame({
        'home_team_enc':[le_home.transform([home_team])[0]],
        'away_team_enc':[le_away.transform([away_team])[0]],
        'home_score':[home_score],
        'away_score':[away_score],
        'total_goals':[total_goals],
        'home_team_recent_form':[home_recent],
        'away_team_recent_form':[away_recent],
        'head_to_head_home_win':[h2h_home_win],
        'head_to_head_away_win':[h2h_away_win],
        'head_to_head_draw':[h2h_draw]
    })

    # Predict result and probabilities
    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]

    # ---------------------------
    # Display Result
    # ---------------------------
    st.success(f"Predicted Match Result: {prediction}")
    st.info(f"Home Win Probability: {probabilities[0]*100:.2f}%")
    st.info(f"Draw Probability: {probabilities[1]*100:.2f}%")
    st.info(f"Away Win Probability: {probabilities[2]*100:.2f}%")

    # ---------------------------
    # Visual Probabilities
    # ---------------------------
    st.subheader("Match Outcome Probabilities")
    prob_df = pd.DataFrame({
        'Outcome': ['Home Win', 'Draw', 'Away Win'],
        'Probability': [probabilities[0]*100, probabilities[1]*100, probabilities[2]*100]
    })

    fig, ax = plt.subplots()
    ax.barh(prob_df['Outcome'], prob_df['Probability'], color=['green', 'blue', 'red'])
    ax.set_xlabel("Probability (%)")
    ax.set_xlim(0, 100)
    st.pyplot(fig)

    # ---------------------------
    # Last 5 Matches Points Trend
    # ---------------------------
    st.subheader("Last 5 Matches Points Trend")

    def last_5_points(team):
        team_matches = df[(df['home_team']==team) | (df['away_team']==team)].sort_values('date', ascending=False).head(5)
        points_list = []
        for _, row in team_matches.iterrows():
            if row['home_team']==team:
                pts = 3 if row['result']=='Home Win' else 1 if row['result']=='Draw' else 0
            else:
                pts = 3 if row['result']=='Away Win' else 1 if row['result']=='Draw' else 0
            points_list.append(pts)
        points_list.reverse()  # oldest to newest
        return points_list

    home_points = last_5_points(home_team)
    away_points = last_5_points(away_team)

    fig, ax = plt.subplots()
    ax.plot(range(1,6), home_points, marker='o', label=home_team, color='green')
    ax.plot(range(1,6), away_points, marker='o', label=away_team, color='red')
    ax.set_xticks(range(1,6))
    ax.set_xlabel("Last 5 Matches")
    ax.set_ylabel("Points")
    ax.set_title("Team Points Trend")
    ax.legend()
    st.pyplot(fig)
