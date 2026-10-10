
# UPI Expense Analyzer

## Project Overview

UPI Expense Analyzer is a Python-based data science project developed to analyze digital payment transactions, identify spending patterns, detect unusual transactions, and predict future expenses.

The project uses Streamlit for the interactive dashboard and machine learning algorithms for financial analysis.

## Technologies Used

- Python
- Pandas and NumPy
- Streamlit
- Plotly
- Scikit-learn
- Pytest

## Key Features

1. Transaction Data Preprocessing
2. Exploratory Data Analysis (EDA)
3. Category-wise Expense Analysis
4. Monthly Spending Trends
5. K-Means Transaction Clustering
6. Isolation Forest Anomaly Detection
7. Expense Prediction using Linear Regression and Random Forest
8. Smart Spending Insights and Illustrative Budget Recommendations
9. Interactive Financial Dashboard
10. CSV Report Download

## Machine Learning Algorithms

### K-Means Clustering
Groups transactions using transaction amount, hour, and day of the week. StandardScaler is used for feature scaling and PCA is used for 2D visualization.

### Isolation Forest
Identifies unusual transactions based on amount, transaction timing, merchant frequency, and category frequency.

### Expense Prediction
Uses Linear Regression and Random Forest Regression to predict next-month spending.

Models are evaluated using MAE, RMSE, and R² score.

## Project Structure

```text
upi_expense_analyzer/
├── app.py
├── data/
├── pages/
├── src/
├── test/
│   └── test_ml.py
├── requirements.txt
└── README.md
```

## Installation

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run the Application

```bash
python -m streamlit run app.py
```

Open http://localhost:8501

## Automated Testing

Run:

```bash
python -m pytest test/test_ml.py -v
```

The three automated tests verify:

- K-Means clustering
- Isolation Forest anomaly detection
- Expense prediction

## Dataset

The project uses a synthetic UPI transaction dataset for educational analysis.

## Limitations

- Synthetic transactions may not represent actual user spending.
- Forecast accuracy depends on the available historical data.
- Anomaly detection results are not proof of fraudulent activity.
- Budget suggestions are illustrative and not personalized financial advice.

## Conclusion

The UPI Expense Analyzer demonstrates how exploratory data analysis, visualization, and basic machine learning algorithms can be combined to study digital transaction patterns and generate financial insights.
