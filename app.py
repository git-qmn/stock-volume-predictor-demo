import streamlit as st
import pandas as pd
import numpy as np
import joblib
import yfinance as yf
import plotly.graph_objs as go
import matplotlib.pyplot as plt

# --- Set page configuration ---
st.set_page_config(page_title="Earnings Volume Predictor", layout="wide")

# --- Initialize Session State for Time Range ---
if "time_range" not in st.session_state:
    st.session_state["time_range"] = "1M"

# --- Upload the test data ---
testing_data = pd.read_csv("final_testing_set.csv")

# --- Load full pipeline (model + features) ---
@st.cache_resource
def load_pipeline():
    pipeline = joblib.load("model/random_forest_pipeline.pkl")
    selected_features = pipeline.feature_names_in_
    return pipeline, selected_features

pipeline, selected_features = load_pipeline()

# --- Load allowed tickers ---
with open("completed_tickers.txt", "r") as f:
    tickers = [line.strip() for line in f if line.strip()]

# --- Sidebar Navigation ---
page = st.sidebar.radio("Navigate", ["Overview", "Volume Prediction", "Feature Importance"])

# --- Helper: Time mapping for charts ---
time_options = ["1D", "5D", "1M", "6M", "YTD", "1Y", "5Y", "Max"]
time_mapping = {
    "1D": ("1d", "5m"),
    "5D": ("5d", "15m"),
    "1M": ("1mo", "1d"),
    "6M": ("6mo", "1d"),
    "YTD": ("ytd", "1d"),
    "1Y": ("1y", "1d"),
    "5Y": ("5y", "1wk"),
    "Max": ("max", "1mo")
}

# --- Page 1: Overview ---
if page == "Overview":
    st.title("Volume Prediction After Financial Releases")

    st.subheader("Team Members")
    st.markdown("- Quan Nguyen  \n- Michael Webber  \n- Jean Alvergnas")

    st.divider()

    st.subheader("App Purpose")
    st.write("""
    This Streamlit app predicts the trading volume of stocks the day after earnings announcements.
    It uses historical trading behavior and key financial ratios to anticipate market activity.
    """)

    st.divider()

    st.subheader("Why is This Valuable?")
    st.markdown("""
    - **Signal Strength of Market Reaction**
    - **Help Large Investors Manage Liquidity**
    - **Improve Short-Term Trading Strategies**
    - **Better Risk Management**
    - **Power Event-Driven Strategies**
    """)

    st.divider()

    st.subheader("Model Used: Random Forest")
    st.write("""
    Random Forest was selected for its robustness and ability to model non-linear relationships.
    Performance was evaluated using MAE (Mean Absolute Error) and R² (variance explained).
    """)

    st.divider()

    st.subheader("Dataset Description")
    st.write("""
    Dataset covers U.S. tech and growth stocks. Features include fundamental ratios (P/E, ROE, etc.)
    and trading volumes around earnings dates up to June 2024.
    """)

    st.divider()

# --- Page 2: Volume Prediction ---
elif page == "Volume Prediction":
    st.title("Volume Prediction After Earnings Release")

    ticker = st.selectbox("Select a stock ticker:", tickers)

    # Unified time range selection
    st.session_state["time_range"] = st.selectbox(
        "Select time range for charts:",
        options=time_options,
        index=time_options.index(st.session_state["time_range"]),
        key="time_range_selector"
    )

    # Pull stock info and history once
    stock = yf.Ticker(ticker)
    info = stock.info
    period, interval = time_mapping.get(st.session_state["time_range"], ("1mo", "1d"))
    hist = stock.history(period=period, interval=interval)

    tab1, tab2 = st.tabs(["Company Snapshot", "Volume Prediction Summary"])

    # --- Tab 1: Company Snapshot ---
    with tab1:
        st.header(f"{info.get('longName', ticker)} Overview")

        # Company Business Description
        st.subheader("Company Overview")
        st.write(info.get('longBusinessSummary', 'No description available.'))
        if info.get('website'):
            st.markdown(f"[Visit Website]({info.get('website')})")

        st.divider()

        # Basic Company Stats
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Valuation Measures")
            st.markdown(f"**Market Cap:** {info.get('marketCap', 'N/A'):,}")
            st.markdown(f"**Enterprise Value:** {info.get('enterpriseValue', 'N/A'):,}")
            st.markdown(f"**Trailing P/E:** {info.get('trailingPE', 'N/A')}")
            st.markdown(f"**Forward P/E:** {info.get('forwardPE', 'N/A')}")
            st.markdown(f"**PEG Ratio:** {info.get('pegRatio', 'N/A')}")
            st.markdown(f"**Price/Sales:** {info.get('priceToSalesTrailing12Months', 'N/A')}")
            st.markdown(f"**Price/Book:** {info.get('priceToBook', 'N/A')}")

        with col2:
            st.markdown("### Financial Highlights")
            st.markdown(f"**Profit Margin:** {info.get('profitMargins', 'N/A')}")
            st.markdown(f"**Return on Assets:** {info.get('returnOnAssets', 'N/A')}")
            st.markdown(f"**Return on Equity:** {info.get('returnOnEquity', 'N/A')}")
            st.markdown(f"**Revenue (ttm):** {info.get('totalRevenue', 'N/A'):,}")
            st.markdown(f"**Net Income (ttm):** {info.get('netIncomeToCommon', 'N/A'):,}")

        st.divider()

        # Stock Price Chart
        st.subheader("Recent Stock Price")
        if not hist.empty:
            fig_price = go.Figure()
            fig_price.add_trace(go.Scatter(x=hist.index, y=hist['Close'], mode='lines', line=dict(color='blue')))
            fig_price.update_layout(title="Recent Stock Price", height=400)
            st.plotly_chart(fig_price, use_container_width=True)
        else:
            st.warning("No stock price data available.")

    # --- Tab 2: Volume Prediction Summary ---
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

            # Prediction Details
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Prediction Date:** {ticker_data['date'].values[0]}")
                st.markdown(f"**Predicted Volume:** {int(prediction):,}")
                st.markdown(f"**Percent Difference:** {percent_diff:.2f}%")
            with col2:
                st.markdown(f"**Actual Volume:** {int(actual_volume):,}")
                st.markdown(f"**Difference:** {int(volume_diff):,} shares")

            st.divider()

            # Recent Volume Chart (synced with dropdown)
            st.subheader("Recent Volume Traded")
            if not hist.empty:
                fig_vol = go.Figure()
                fig_vol.add_trace(go.Bar(x=hist.index, y=hist['Volume'], marker_color='lightblue'))
                fig_vol.update_layout(title="Recent Trading Volume", height=400)
                st.plotly_chart(fig_vol, use_container_width=True)
            else:
                st.warning("No volume data available.")

            st.divider()

            # Actual vs Predicted Comparison Chart
            st.subheader("Actual vs Predicted Volume")
            fig_comp = go.Figure()
            fig_comp.add_trace(go.Bar(x=["Actual"], y=[actual_volume], marker_color='green'))
            fig_comp.add_trace(go.Bar(x=["Predicted"], y=[prediction], marker_color='blue'))
            fig_comp.update_layout(barmode='group', height=400)
            st.plotly_chart(fig_comp, use_container_width=True)

            st.divider()

            # Business Interpretation
            st.subheader("Business Interpretation")
            if prediction >= 50_000_000:
                st.success("High expected trading activity following earnings announcement.")
            elif prediction >= 10_000_000:
                st.info("Moderate trading volume expected.")
            else:
                st.warning("Low expected trading activity post-earnings.")

        else:
            st.warning("No recent prediction data available for this ticker.")

# --- Page 3: Feature Importance ---
elif page == "Feature Importance":
    st.title("Model Feature Importance")

    model = pipeline.named_steps['model'] if 'model' in pipeline.named_steps else pipeline.named_steps['randomforestregressor']
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    sorted_features = np.array(selected_features)[sorted_idx]
    sorted_importance = importances[sorted_idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(sorted_features[::-1], sorted_importance[::-1])
    ax.set_xlabel("Feature Importance")
    ax.set_title("Top Features Driving Prediction")
    st.pyplot(fig)
