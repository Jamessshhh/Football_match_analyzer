import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load cleaned data
df = pd.read_csv("data/cleaned_data.csv")

# ---- Step 1: Create match result ----
def match_result(row):
    if row['home_score'] > row['away_score']:
        return "Home Win"
    elif row['home_score'] < row['away_score']:
        return "Away Win"
    else:
        return "Draw"

df['result'] = df.apply(match_result, axis=1)

# ---- Step 2: Encode categorical variables ----
le_home = LabelEncoder()
le_away = LabelEncoder()
df['home_team_enc'] = le_home.fit_transform(df['home_team'])
df['away_team_enc'] = le_away.fit_transform(df['away_team'])
df['weekday_enc'] = LabelEncoder().fit_transform(df['weekday'])

# ---- Step 3: Feature engineering ----
# Use team IDs + match features
features = ['home_team_enc', 'away_team_enc', 'month', 'weekday_enc', 'home_score', 'away_score', 'total_goals']
X = df[features]
y = df['result']

# ---- Step 4: Split data ----
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ---- Step 5: Train Random Forest ----
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# ---- Step 6: Predict & evaluate ----
y_pred = model.predict(X_test)

print("=== Classification Report ===")
print(classification_report(y_test, y_pred))

print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))



features = [
    'home_team_enc', 'away_team_enc', 'month', 'weekday_enc',
    'home_score', 'away_score', 'total_goals',
    'home_team_recent_form', 'away_team_recent_form'
]

X = df[features]
y = df['result']

# Train model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X, y)
