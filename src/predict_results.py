import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Load cleaned data
df = pd.read_csv("data/cleaned_data.csv")

# Create match result from home team perspective
def match_result(row):
    if row['home_score'] > row['away_score']:
        return "Home Win"
    elif row['home_score'] < row['away_score']:
        return "Away Win"
    else:
        return "Draw"

df['result'] = df.apply(match_result, axis=1)

# Encode categorical variables
le_home = LabelEncoder()
le_away = LabelEncoder()
df['home_team_enc'] = le_home.fit_transform(df['home_team'])
df['away_team_enc'] = le_away.fit_transform(df['away_team'])

# Features and target
X = df[['home_team_enc', 'away_team_enc']]
y = df['result']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(multi_class='multinomial', max_iter=500)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("=== Classification Report ===")
print(classification_report(y_test, y_pred))

print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))
