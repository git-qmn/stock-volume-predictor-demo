import streamlit as st
st.set_page_config(page_title="Earnings Volume Predictor", layout="wide")

import pandas as pd
import numpy as np
import joblib
import yfinance as yf
import matplotlib.pyplot as plt

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

    if ticker:
        fin_df = get_financial_ratios(ticker)
        if fin_df is not None:
            try:
                if 'Current Volume' not in fin_df.columns:
                    fin_df['Current Volume'] = 0

                input_df = fin_df[selected_features]

                # Fill any missing features with 0
                input_df = input_df.fillna(0)

                prediction = pipeline.predict(input_df)[0]

                stock = yf.Ticker(ticker)
                info = stock.info

                st.subheader(f"{info.get('longName', ticker)} ({ticker})")

                market_cap = info.get('marketCap', None)
                if isinstance(market_cap, (int, float)):
                    market_cap_display = f"{market_cap:,}"
                else:
                    market_cap_display = "N/A"

                st.write(f"**Sector:** {info.get('sector', 'N/A')}")
                st.write(f"**Industry:** {info.get('industry', 'N/A')}")
                st.write(f"**Market Cap:** {market_cap_display}")
                st.write(f"**Trailing P/E:** {info.get('trailingPE', 'N/A')}")

                st.divider()

                st.success(
                    f"Predicted trading volume on the first market day after earnings release: "
                    f"{int(prediction):,} shares"
                )

                if prediction >= 50_000_000:
                    st.info("High expected trading activity following earnings announcement.")
                elif prediction >= 10_000_000:
                    st.info("Moderate trading volume expected post-earnings.")
                else:
                    st.info("Low trading volume expected following earnings release.")

                st.divider()

                st.subheader("Recent Volume Trends")
                hist = stock.history(period="1mo")
                if not hist.empty:
                    st.line_chart(hist['Volume'])
                else:
                    st.write("No historical volume data available.")

            except Exception as e:
                st.error(f"Prediction failed: {e}")
        else:
            st.warning("Could not fetch financial fundamentals for this ticker.")

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
