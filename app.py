import streamlit as st
st.set_page_config(page_title="Earnings Volume Predictor", layout="wide")

import pandas as pd
import numpy as np
import joblib
import yfinance as yf
import plotly.graph_objs as go

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

    tab1, tab2 = st.tabs(["Company Snapshot", "Volume Prediction Summary"])

    # --- Company Snapshot ---
    with tab1:
        st.header("Company Snapshot")

        stock = yf.Ticker(ticker)
        info = stock.info

        # --- Basic Company Info ---
        st.subheader(f"{info.get('longName', ticker)}")
        
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"**Sector:** {info.get('sector', 'N/A')}")
            st.markdown(f"**Industry:** {info.get('industry', 'N/A')}")
            st.markdown(f"**Market Cap:** {info.get('marketCap', 'N/A'):,}")
            st.markdown(f"**Trailing P/E:** {info.get('trailingPE', 'N/A')}")
            st.markdown(f"**Forward P/E:** {info.get('forwardPE', 'N/A')}")

        with col2:
            st.markdown(f"**Price/Sales:** {info.get('priceToSalesTrailing12Months', 'N/A')}")
            st.markdown(f"**Price/Book:** {info.get('priceToBook', 'N/A')}")
            st.markdown(f"**Beta:** {info.get('beta', 'N/A')}")
            st.markdown(f"**EPS (TTM):** {info.get('trailingEps', 'N/A')}")
            st.markdown(f"**1Y Target Est:** {info.get('targetMeanPrice', 'N/A')}")

        st.divider()

        # --- Time Range Selection ---
        st.subheader("Recent Stock Price")
        time_range = st.selectbox(
            "Select time range:", 
            options=["1D", "5D", "1M", "6M", "YTD", "1Y", "5Y", "Max"],
            index=2
        )

        # --- Map user selection to yfinance interval ---
        time_mapping = {
            "1D": ("1d", "5m"),
            "5D": ("5d", "15m"),
            "1M": ("1mo", "60m"),
            "6M": ("6mo", "1d"),
            "YTD": ("ytd", "1d"),
            "1Y": ("1y", "1d"),
            "5Y": ("5y", "1wk"),
            "Max": ("max", "1mo")
        }

        period, interval = time_mapping.get(time_range, ("1mo", "1d"))

        hist = stock.history(period=period, interval=interval)

        if not hist.empty:
            # --- Close Price Chart ---
            fig_price = go.Figure()
            fig_price.add_trace(go.Scatter(x=hist.index, y=hist['Close'], mode='lines', name='Close Price', line=dict(color='royalblue')))
            fig_price.update_layout(
                title="Recent Stock Price",
                xaxis_title="Date",
                yaxis_title="Close Price ($)",
                height=400,
                margin=dict(l=20, r=20, t=50, b=20)
            )
            st.plotly_chart(fig_price, use_container_width=True)

            # --- Volume Chart ---
            st.subheader("Recent Volume Traded")
            fig_volume = go.Figure()
            fig_volume.add_trace(go.Bar(x=hist.index, y=hist['Volume'], name='Volume', marker_color='lightblue'))
            fig_volume.update_layout(
                title="Volume Over Time",
                xaxis_title="Date",
                yaxis_title="Volume",
                height=300,
                margin=dict(l=20, r=20, t=50, b=20)
            )
            st.plotly_chart(fig_volume, use_container_width=True)

        st.divider()
    
# Stock Details
    st.subheader("Stock Details")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"**Previous Close:** {info.get('previousClose', 'N/A')}")
        st.markdown(f"**Open:** {info.get('open', 'N/A')}")
        st.markdown(f"**Bid:** {info.get('bid', 'N/A')}")
        st.markdown(f"**Ask:** {info.get('ask', 'N/A')}")
    
    with col2:
        st.markdown(f"**Day's Range:** {info.get('dayLow', 'N/A')} - {info.get('dayHigh', 'N/A')}")
        st.markdown(f"**52 Week Range:** {info.get('fiftyTwoWeekLow', 'N/A')} - {info.get('fiftyTwoWeekHigh', 'N/A')}")
        st.markdown(f"**Volume:** {info.get('volume', 'N/A'):,}")
        st.markdown(f"**Avg. Volume:** {info.get('averageVolume', 'N/A'):,}")
    
    with col3:
        st.markdown(f"**Market Cap (intraday):** {info.get('marketCap', 'N/A'):,}")
        st.markdown(f"**Beta (5Y Monthly):** {info.get('beta', 'N/A')}")
        st.markdown(f"**P/E Ratio (TTM):** {info.get('trailingPE', 'N/A')}")
        st.markdown(f"**EPS (TTM):** {info.get('trailingEps', 'N/A')}")
    
    with col4:
        st.markdown(f"**Earnings Date:** {info.get('earningsTimestamp', 'N/A')}")
        st.markdown(f"**Forward Dividend & Yield:** {info.get('dividendRate', 'N/A')} ({info.get('dividendYield', 'N/A')})")
        st.markdown(f"**Ex-Dividend Date:** {info.get('exDividendDate', 'N/A')}")
        st.markdown(f"**1Y Target Est:** {info.get('targetMeanPrice', 'N/A')}")

    # Volume Prediction Summary
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
