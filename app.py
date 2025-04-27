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

# Load tickers from file
with open("completed_tickers.txt", "r") as f:
    tickers = [line.strip() for line in f if line.strip()]

# Sidebar navigation
page = st.sidebar.radio("Navigate", [
    "Overview",
    "Volume Prediction",
    "Feature Importance"
])

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
    except Exception:
        return None

# Page 1: Overview
if page == "Overview":
    st.title("Volume Prediction After Financial Releases")

    st.subheader("Team Members")
    st.markdown("""
    - Quan Nguyen  
    - Michael Webber  
    - Jean Alvergnas  
    """)

    st.subheader("🎯 App Purpose")
    st.write("""
    This Streamlit app predicts the volume of stock traded on the day following a financial release.
    It leverages past trading behavior and key financial ratios to anticipate activity after earnings announcements.
    """)

    st.subheader("💡 Why is This Valuable?")
    st.markdown("""
    - **Signal Strength of Market Reaction** ➔ Big volume spikes show how strongly investors react to earnings.
    - **Help Large Investors Manage Liquidity** ➔ Easier to buy/sell large amounts.
    - **Improve Short-Term Trading Strategies** ➔ Volume surges enable breakouts and momentum trades.
    - **Better Risk Management** ➔ High volume usually signals higher volatility.
    - **Power Event-Driven Strategies** ➔ Expected volume helps funds assess opportunities.
    - **Predictable Output** ➔ Volume tends to spike around earnings, mergers, and major news.
    """)

    st.subheader("Model Used: Random Forest")
    st.write("""
    - The model was trained using historical data with key financial ratios.
    - Model performance was evaluated using **Mean Absolute Error (MAE)** and **R-squared (R²)**.
    """)

# Page 2: Volume Prediction
elif page == "Volume Prediction":
    st.title("Volume Prediction After Earnings Release")

    ticker = st.selectbox("Select a stock ticker for prediction:", tickers)
    st.session_state['selected_ticker'] = ticker

    tab1, tab2 = st.tabs(["Company Snapshot", "Volume Prediction Summary"])

    with tab1:
        st.header("Company Snapshot")

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

    with tab2:
        st.header("Volume Prediction Summary")

        ticker_data = testing_data[testing_data['Ticker'] == ticker].sort_values('date', ascending=False).head(1)

        if not ticker_data.empty:
            input_df = ticker_data[selected_features]
            input_df = input_df.fillna(0)

            prediction = pipeline.predict(input_df)[0]
            actual_volume = ticker_data['predicted_volume'].values[0]

            volume_diff = actual_volume - prediction
            percent_diff = (volume_diff / actual_volume) * 100 if actual_volume != 0 else 0

            st.subheader(f"{ticker} - Volume Prediction Summary")
            st.write(f"**Prediction Date:** {ticker_data['date'].values[0]}")
            st.write(f"**Actual Volume:** {int(actual_volume):,}")
            st.write(f"**Predicted Volume:** {int(prediction):,}")
            st.write(f"**Difference:** {int(volume_diff):,} shares")
            st.write(f"**Percent Difference:** {percent_diff:.2f}%")

            st.divider()
            if prediction >= 50_000_000:
                st.info("High expected trading activity following earnings announcement.")
            elif prediction >= 10_000_000:
                st.info("Moderate trading volume expected post-earnings.")
            else:
                st.info("Low trading volume expected following earnings release.")
        else:
            st.warning("No recent data available for this ticker.")

# Page 3: Feature Importance
elif page == "Feature Importance":
    st.title("Model Feature Importance")

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
