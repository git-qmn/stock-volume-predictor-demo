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

if page == "Overview":
    st.title("Volume Prediction After Financial Releases")

    st.subheader("Team Members")
    st.markdown("""
    - Quan Nguyen  
    - Michael Webber  
    - Jean Alvergnas  
    """)

    st.divider()

    st.subheader("App Purpose")
    st.write("""
    This Streamlit app predicts the volume of stock traded on the day following a financial release.
    It leverages past trading behavior and key financial ratios to anticipate activity after earnings announcements.
    """)

    st.divider()

    st.subheader("Why is This Valuable?")
    st.markdown("""
    - **Signal Strength of Market Reaction** ➔ Big volume spikes show how strongly investors react to earnings announcements.
    - **Help Large Investors Manage Liquidity** ➔ High volume days make it easier for funds to buy/sell large amounts without moving the stock price.
    - **Improve Short-Term Trading Strategies** ➔ Traders use volume surges for breakouts, reversals, and momentum trading opportunities.
    - **Better Risk Management** ➔ High expected volume helps manage volatility risks in trading portfolios.
    - **Power Event-Driven Strategies** ➔ Funds trading around earnings rely on expected volume to assess trade sizing.
    - **Predictable Output** ➔ Volume tends to spike predictably around earnings, mergers, and major corporate events.
    """)

    st.divider()

    st.subheader("Model Used: Random Forest")
    st.write("""
    - **Random Forest** was selected for its ability to handle non-linear relationships and provide robust predictions.
    - Ensemble methods like Random Forest help avoid overfitting compared to single models.
    - Feature importance from the model gives useful insights into drivers of post-earnings volume.
    - Model performance was evaluated using:
        - **Mean Absolute Error (MAE)**
        - **R-squared (R²)** to measure variance explained.
    """)

    st.divider()

    st.subheader("Dataset Description")
    st.write("""
    - Data includes company financial fundamentals (e.g., P/E Ratio, Return on Assets) and stock trading volumes.
    - Each record links a company's financial release date to the next available trading day's actual volume.
    - Financial data was collected from public earnings reports and market databases.
    - Volume data was sourced from U.S. stock exchange feeds.
    - Dataset focuses primarily on major U.S.-listed companies from the technology, semiconductor, cloud computing, and cybersecurity sectors.
    - The dataset covers earnings announcements up to **June 2024**.
    """)

    st.divider()

    st.subheader("Model Inputs")
    st.write("""
    - The model uses financial ratios that would have been publicly available immediately after the earnings release.
    - No future data or forward-looking indicators are used — ensuring real-time applicability.
    - Inputs include ratios like EV/EBITDA, Net Margin, Return on Equity, Debt-to-Equity, and Asset Turnover.
    """)

    st.divider()

    st.subheader("Feature Engineering Highlights")
    st.write("""
    - Selected key financial ratios that showed the strongest historical relationship with post-earnings volume movements.
    - Standardized and cleaned features to ensure consistency across different companies and industries.
    - Focused on variables that are timely, widely reported, and reliable.
    """)

    st.divider()

    st.subheader("Modeling Approach")
    st.write("""
    - Trained a Random Forest Regressor on historical financial and volume data to predict next-day trading volume after earnings releases.
    - Hyperparameters were tuned using cross-validation to avoid overfitting.
    - Model generalizes best to U.S. technology, semiconductor, cloud, and cybersecurity sectors based on training data composition.
    """)

    st.divider()

    st.subheader("Limitations and Future Work")
    st.write("""
    - Current model generalizes mainly to the U.S. technology and growth stock sectors; expansion to other sectors like financial services, healthcare, and industrials would enhance model generalizability.
    - Incorporating forward-looking indicators (e.g., analyst EPS revisions, options activity) could improve predictive power.
    - Future iterations could predict not only volume spikes but also price movements around earnings events.
    - Expansion to international markets (e.g., Europe, Asia) could test robustness across different trading environments.
    """)

    st.divider()

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
