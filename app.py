
import streamlit as st
st.set_page_config(page_title="Earnings Volume Predictor", layout="wide")

import pandas as pd
import numpy as np
import joblib
import yfinance as yf
import matplotlib.pyplot as plt

# ===== Load Model Artifacts =====
@st.cache_resource
def load_artifacts():
    model = joblib.load("model/volume_model.pkl")
    scaler = joblib.load("model/scaler.pkl")
    selected_features = joblib.load("model/selected_features.pkl")
    return model, scaler, selected_features

model, scaler, selected_features = load_artifacts()

# ===== Get Financial Ratios from yfinance =====
def get_financial_ratios(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info

    try:
        ratios = {
            'evm': info.get('enterpriseValue', 0) / info.get('marketCap', 1),
            'pe_exi': info.get('trailingPE', 0),
            'ps': info.get('priceToSalesTrailing12Months', 0),
            'npm': info.get('netMargins', 0),
            'opmbd': info.get('operatingMargins', 0),
            'roa': info.get('returnOnAssets', 0),
            'roe': info.get('returnOnEquity', 0),
            'de_ratio': info.get('debtToEquity', 0),
            'intcov_ratio': info.get('ebitdaMargins', 0),
            'quick_ratio': info.get('quickRatio', 0),
            'curr_ratio': info.get('currentRatio', 0),
            'at_turn': info.get('returnOnAssets', 0),  # Proxy reuse
            'ptb': info.get('priceToBook', 0)
        }
        return pd.DataFrame([ratios])
    except:
        return None

# ===== Streamlit Tabs =====
tabs = st.tabs([
    "📊 Company Snapshot",
    "🔮 Volume Prediction After Earnings",
    "🧠 Model Insight"
])

# Ticker input shared across pages
with st.sidebar:
    ticker = st.text_input("Enter a stock ticker (e.g., AAPL)", value="ORCL").upper()

# ===== Page 1: Company Snapshot =====
with tabs[0]:
    st.header("📊 Company Snapshot")

    if ticker:
        stock = yf.Ticker(ticker)
        info = stock.info

        st.subheader(info.get('longName', ticker))
        st.markdown(f"**Sector:** {info.get('sector', 'N/A')}  
"
                    f"**Industry:** {info.get('industry', 'N/A')}  
"
                    f"**Market Cap:** {info.get('marketCap', 'N/A'):,}  
"
                    f"**Trailing P/E:** {info.get('trailingPE', 'N/A')}")

        st.subheader("Recent Volume & Price")
        hist = stock.history(period="1mo")
        if not hist.empty:
            st.line_chart(hist[['Volume', 'Close']])

# ===== Page 2: Volume Prediction =====
with tabs[1]:
    st.header("🔮 Volume Prediction After Earnings")

    if ticker:
        fin_df = get_financial_ratios(ticker)
        if fin_df is not None:
            try:
                input_df = fin_df[selected_features]
                input_scaled = scaler.transform(input_df)
                prediction = model.predict(input_scaled)[0]
                st.success(f"📈 Predicted trading volume on the first market day after earnings release: **{int(prediction):,} shares**")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
        else:
            st.warning("Could not fetch fundamentals for this ticker.")

# ===== Page 3: Model Insight =====
with tabs[2]:
    st.header("🧠 Model Insight")

    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    sorted_features = np.array(selected_features)[sorted_idx]
    sorted_importance = importances[sorted_idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(sorted_features[::-1], sorted_importance[::-1])
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importances (Top → Bottom)")
    st.pyplot(fig)
