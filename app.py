import streamlit as st
st.set_page_config(page_title="Earnings Volume Predictor", layout="wide")

import pandas as pd
import numpy as np
import joblib
import yfinance as yf
import matplotlib.pyplot as plt

# Upload the test data
testing_data = pd.read_csv("final_testing_set.csv")

# Load full pipeline (model + scaler)
@st.cache_resource
def load_pipeline():
    pipeline = joblib.load("model/random_forest_pipeline.pkl")
    selected_features = pipeline.feature_names_in_
    return pipeline, selected_features

pipeline, selected_features = load_pipeline()

# Get financial ratios from yfinance
def get_financial_ratios(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info

    try:
        ratios = {
            'EV/EBITDA': info.get('enterpriseValue', 0) / info.get('marketCap', 1) if info.get('marketCap') else 0,
            'P/E Ratio': info.get('trailingPE', 0),
            'P/S Ratio': info.get('priceToSalesTrailing12Months', 0),
            'Net Margin': info.get('netMargins', 0),
            'EBITDA Margin': info.get('operatingMargins', 0),
            'Return on Assets': info.get('returnOnAssets', 0),
            'Return on Equity': info.get('returnOnEquity', 0),
            'Debt-to-Equity': info.get('debtToEquity', 0),
            'Interest Coverage': info.get('ebitdaMargins', 0),
            'Quick Ratio': info.get('quickRatio', 0),
            'Current Ratio': info.get('currentRatio', 0),
            'Asset Turnover': info.get('returnOnAssets', 0),
            'Price-to-Book': info.get('priceToBook', 0)
        }
        return pd.DataFrame([ratios])
    except Exception as e:
        return None

# Create tab structure
tabs = st.tabs([
    "Company Snapshot",
    "Volume Prediction After Earnings",
    "Model Insight"
])

# Load tickers from file
with open("completed_tickers.txt", "r") as f:
    tickers = [line.strip() for line in f if line.strip()]

# Sidebar for ticker selection
with st.sidebar:
    ticker = st.selectbox("Select a stock ticker:", tickers)
    st.caption("Note: Model trained mainly on tech stocks. Predictions for other sectors may have varied accuracy.")  
    st.caption("Model trained on data up to October 2023. Predictions may not reflect current market conditions.")

# Page 1: Company Snapshot
with tabs[0]:
    st.header("Company Snapshot")

    if ticker:
        stock = yf.Ticker(ticker)
        info = stock.info

        st.subheader(info.get('longName', ticker))

        market_cap = info.get('marketCap', None)
        if isinstance(market_cap, (int, float)):
            market_cap_display = f"{market_cap:,}"
        else:
            market_cap_display = "N/A"

        st.markdown((
            f"**Sector:** {info.get('sector', 'N/A')}  \n"
            f"**Industry:** {info.get('industry', 'N/A')}  \n"
            f"**Market Cap:** {market_cap_display}  \n"
            f"**Trailing P/E:** {info.get('trailingPE', 'N/A')}"
        ))

        st.subheader("Recent Volume and Price")
        hist = stock.history(period="1mo")
        if not hist.empty:
            st.line_chart(hist[['Volume', 'Close']])

# Page 2: Volume Prediction After Earnings
with tabs[1]:
    st.header("Volume Prediction After Earnings")
    # Filter the test set for the selected ticker and get the most recent record
    ticker_data = testing_data[testing_data['Ticker'] == ticker].sort_values('date', ascending=False).head(1)

    if not ticker_data.empty:
        # Prepare input features (exclude label and non-features)
        input_df = ticker_data[selected_features]
        
        # Make the prediction
        prediction = pipeline.predict(input_df)[0]

        # Extract the actual volume
        actual_volume = ticker_data['predicted_volume'].values[0]
        
        # Compute difference and percent difference
        volume_diff = actual_volume - prediction
        percent_diff = (volume_diff / actual_volume) * 100 if actual_volume != 0 else 0

        # Display results
        st.subheader(f"{ticker} - Volume Prediction Summary")
        st.write(f"**Prediction Date:** {ticker_data['date'].values[0]}")
        st.write(f"**Actual Volume:** {int(actual_volume):,}")
        st.write(f"**Predicted Volume:** {int(prediction):,}")
        st.write(f"**Difference:** {int(volume_diff):,} shares")
        st.write(f"**Percent Difference:** {percent_diff:.2f}%")

        # Category messages
        st.divider()
        if prediction >= 50_000_000:
            st.info("High expected trading activity following earnings announcement.")
        elif prediction >= 10_000_000:
            st.info("Moderate trading volume expected post-earnings.")
        else:
            st.info("Low trading volume expected following earnings release.")
    else:
        st.warning("No recent data available for this ticker.")

# Page 3: Model Insight
with tabs[2]:
    st.header("Model Insight")

    model = pipeline.named_steps['model'] if 'model' in pipeline.named_steps else pipeline.named_steps['randomforestregressor']
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    sorted_features = np.array(selected_features)[sorted_idx]
    sorted_importance = importances[sorted_idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(sorted_features[::-1], sorted_importance[::-1])
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importances")
    st.pyplot(fig)
