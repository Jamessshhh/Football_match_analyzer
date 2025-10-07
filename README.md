# ⚽ Football Match Analyzer

A data-driven project that fetches real football match data, cleans it, visualizes team performance, and predicts match outcomes (Home Win, Away Win, or Draw) using machine learning.

---

## 📂 Project Structure

football_match_analyzer/
├── data/ # CSV data files
├── src/ # Source scripts
├── config.py # API configuration
├── requirements.txt # Dependencies
└── README.md # Project documentation


---

## ⚙️ Features

✅ Fetches real match data from a football API  
✅ Cleans and preprocesses match data  
✅ Visualizes team stats and outcomes  
✅ Predicts match results using Random Forest  
✅ Compares predicted vs actual outcomes  

---

## 🧠 Model Info

- **Algorithm:** RandomForestClassifier  
- **Features:** Team names, scores, weekdays, total goals, etc.  
- **Accuracy:** ~99% (on test data)

---

## 🚀 How to Run

```bash
# Activate virtual environment (Mac)
source venv/bin/activate

# Fetch data
python src/data_fetcher.py

# Clean data
python src/clean_data.py

# Visualize data
python src/visualize_data.py

# Train and predict
python src/predict_results_improved.py

# Visualize predictions
python src/visualize_predictions.py


📊 Example Output
Confusion Matrix:
Shows how many predictions matched actual results.
Bar Chart:
Displays the predicted outcome distribution.


🧰 Tech Stack
Python 3.10+
Pandas
Seaborn
Scikit-learn
Matplotlib


🏁 Future Improvements
Add live match predictions via API
Incorporate player-level stats
Build a simple web dashboard with Flask or Streamlit



---


