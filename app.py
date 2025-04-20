
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import yfinance as yf
import matplotlib.pyplot as plt

# ========== Load Model Artifacts ==========
@st.cache_resource
def load_artifacts():
    model = joblib.load("model/volume_model.pkl")
    scaler = joblib.load("model/scaler.pkl")
    selected_features = joblib.load("model/selected_features.pkl")
    return model, scaler, selected_features

model, scaler, selected_features = load_artifacts()

# ========== Extract Financials from yfinance ==========
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
            'at_turn': info.get('returnOnAssets', 0),  # reused proxy
            'ptb': info.get('priceToBook', 0)
        }
        return pd.DataFrame([ratios])
    except:
        return None

# ========== Page Setup ==========
st.set_page_config(page_title="Stock Volume Prediction", layout="wide")
tabs = st.tabs(["🔮 Predict Volume", "📊 Financial Details", "🧠 Feature Insights"])

# ========== PAGE 1: Predict Volume ==========
with tabs[0]:
    st.header("🔮 Predict Tomorrow's Trading Volume")
    ticker = st.text_input("Enter Stock Ticker (e.g., AAPL)", value="ORCL").upper()

    if st.button("Predict"):
        fin_df = get_financial_ratios(ticker)
        if fin_df is not None:
            try:
                input_df = fin_df[selected_features]
                input_scaled = scaler.transform(input_df)
                prediction = model.predict(input_scaled)[0]
                st.success(f"📈 Predicted Volume for Tomorrow: **{int(prediction):,} shares**")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
        else:
            st.warning("Couldn't fetch financials for this ticker.")

# ========== PAGE 2: Ticker Info ==========
with tabs[1]:
    st.header("📊 Financial Snapshot")
    if ticker:
        stock = yf.Ticker(ticker)
        info = stock.info

        st.subheader(f"{info.get('longName', ticker)}")
        st.markdown(f"**Sector:** {info.get('sector', 'N/A')}  \n"
                    f"**Market Cap:** {info.get('marketCap', 'N/A'):,}  \n"
                    f"**Trailing P/E:** {info.get('trailingPE', 'N/A')}")

        st.subheader("Recent Volume Trend")
        hist = stock.history(period="1mo")
        if not hist.empty:
            st.line_chart(hist['Volume'])

# ========== PAGE 3: Feature Importance ==========
with tabs[2]:
    st.header("🧠 Feature Contribution to Prediction")
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    sorted_features = np.array(selected_features)[sorted_idx]
    sorted_importance = importances[sorted_idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(sorted_features[::-1], sorted_importance[::-1])
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importances (Top to Bottom)")
    st.pyplot(fig)
