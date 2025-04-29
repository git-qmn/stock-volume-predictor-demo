# Stock Earnings Volume Prediction

## Overview
This project predicts the volume of stock traded immediately following a company's earnings announcement. We aim to help investors and traders anticipate activity after earnings by analyzing historical financial ratios and past trading behavior. The model uses machine learning to capture key drivers behind volume surges following corporate events.

## Course Information
- **Course:** BA870 - Financial Analytics, Master of Science in Business Analytics, Boston University
- **Team Members:** Quan Nguyen, Michael Webber, Jean Alvergnas

## Dataset
We collected and processed stock and financial data from:
- **Compustat** and **CRSP** via Wharton Research Data Services (WRDS)
- Historical financial ratios (e.g., EV/EBITDA, Return on Assets, Debt-to-Equity)
- Trading volume from January 2010 to December 2024

Final datasets include:
- `final_training_set.csv`
- `final_testing_set.csv`
- `completed_tickers.txt` (list of tech stocks)

Preprocessing steps:
- Handling missing values
- Standardizing features
- Aligning financial ratios to the correct earnings dates

## Installation
1. Clone the repository:
```bash
git clone https://github.com/YOUR_GITHUB/stock-earnings-volume-prediction.git
cd stock-earnings-volume-prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:
```bash
streamlit run app.py
```

## Usage
- Navigate through five main sections:
  - App Overview
  - Dataset Overview
  - Volume Prediction
  - Feature Importance
  - Top Stocks by Volume
- Select a stock to view predicted trading volume after earnings.
- Explore historical stock price and trading volume trends.
- Analyze which financial metrics are most influential in volume prediction.

## Methodology
- **Model:** Random Forest Regressor
- **Inputs:** Financial ratios available immediately after earnings releases
- **Target:** Volume traded on the day following earnings

**Why Random Forest?**
- Captures complex, non-linear interactions
- Robust against overfitting
- Provides feature importance for interpretability

**Performance Metrics:**
- R-squared: 82%
- Adjusted R-squared: 81.5%
- Mean Absolute Error (MAE): ~3.2 million shares

## Challenges
- Financial ratios may not fully capture market sentiment, news, or analyst activity.
- Heavy focus on U.S. technology stocks may limit generalizability across sectors.
- Random events (e.g., major news or macroeconomic shocks) can lead to volume anomalies not predicted by financial data.
- Variability in earnings reporting dates introduces some noise.

## Contribution
- **Quan Nguyen** 
- **Michael Webber** 
- **Jean Alvergnas** 

## Contact
For questions or feedback, please reach out at: **qmn@bu.edu**

