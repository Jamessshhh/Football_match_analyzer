# src/load_data.py

import pandas as pd

# Path to your saved CSV file
csv_path = "data/processed_data.csv"

# Load the CSV into a DataFrame
df = pd.read_csv(csv_path)

# Display some basic info
print("✅ Data loaded successfully!\n")
print("Shape of DataFrame:", df.shape)
print("\nFirst 5 rows:\n", df.head())
