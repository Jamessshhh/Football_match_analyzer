import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load cleaned data
df = pd.read_csv("data/cleaned_data.csv")

# ---- Create match result ----
def match_result(row):
    if row['home_score'] > row['away_score']:
        return "Home Win"
    elif row['home_score'] < row['away_score']:
        return "Away Win"
    else:
        return "Draw"

df['result'] = df.apply(match_result, axis=1)

# ---- Encode categorical variables ----
le_home = LabelEncoder()
le_away = LabelEncoder()
df['home_team_enc'] = le_home.fit_transform(df['home_team'])
df['away_team_enc'] = le_away.fit_transform(df['away_team'])
df['weekday_enc'] = LabelEncoder().fit_transform(df['weekday'])

# ---- Features and target ----
features = ['home_team_enc', 'away_team_enc', 'month', 'weekday_enc', 'home_score', 'away_score', 'total_goals']
X = df[features]
y = df['result']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# ---- Confusion Matrix Heatmap ----
cm = confusion_matrix(y_test, y_pred, labels=["Away Win", "Draw", "Home Win"])
cm_df = pd.DataFrame(cm, index=["Away Win", "Draw", "Home Win"], columns=["Away Win", "Draw", "Home Win"])

plt.figure(figsize=(6,5))
sns.heatmap(cm_df, annot=True, fmt="d", cmap="YlGnBu")
plt.title("Predicted vs Actual Match Results")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.show()

# ---- Optional: Bar chart of predicted results ----
plt.figure(figsize=(6,4))
sns.countplot(x=y_pred, palette="coolwarm")
plt.title("Predicted Match Outcome Distribution")
plt.xlabel("Predicted Result")
plt.ylabel("Number of Matches")
plt.tight_layout()
plt.show()
