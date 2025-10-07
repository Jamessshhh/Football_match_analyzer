# src/explore_data.py

import pandas as pd

# Load the processed data
df = pd.read_csv("data/processed_data.csv")

print("✅ Data loaded successfully!\n")

# 1️⃣ Basic Info
print("=== Data Overview ===")
print(df.info(), "\n")

# 2️⃣ Summary statistics (numerical columns)
print("=== Summary Statistics ===")
print(df.describe(), "\n")

# 3️⃣ Check for missing values
print("=== Missing Values ===")
print(df.isnull().sum(), "\n")

# 4️⃣ Check unique values in key columns
important_columns = ["homeTeam", "awayTeam", "status", "utcDate"]
for col in important_columns:
    if col in df.columns:
        print(f"Unique values in '{col}': {df[col].nunique()}")
