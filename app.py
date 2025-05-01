import streamlit as st
st.set_page_config(page_title="Earnings Volume Predictor", layout="wide")

import pandas as pd
import numpy as np
import joblib
import yfinance as yf
import plotly.graph_objs as go
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

st.markdown("""
    <style>
    [data-testid="stSidebar"] h3 {
        font-size: 42px !important;
    }
    </style>
""", unsafe_allow_html=True)

st.sidebar.title("App Navigation")

page = st.sidebar.selectbox(
    label="",
    options=[
        "App Overview",
        "Data Description and Feature Importance",
        "Volume Prediction",
        "Top Stocks by Volume"
    ],
    label_visibility="collapsed"
)


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
if page == "App Overview":
    st.title("Volume Prediction After Financial Releases")

    st.subheader("Team Members")
    st.markdown("""
    - **Quan Nguyen**  
    - **Michael Webber**  
    - **Jean Alvergnas**  
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

    st.subheader("Model Inputs")
    st.write("""
    - The model uses financial ratios that would have been publicly available immediately after the earnings release.
    - No future data or forward-looking indicators are used — ensuring real-time applicability.
    - Inputs include ratios like EV/EBITDA, Net Margin, Return on Equity, Debt-to-Equity, and Asset Turnover. Please refer to the Data Description tab to get a comprehensive list of the ratios used
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

# Page 2: Data Description and Feature Importance
elif page == "Data Description and Feature Importance":
    st.title("Data Description")

    data_dict = {
        "tic": "Ticker symbol",
        "calendar_qtr_x": "Calendar cuarter that the financials were released",
        "public_date": "Date the earnings reports were made public",
        "evm": "Enterprise Value Multiple",
        "pe_exi": "Price to Earnings ratio",
        "ps": "Price to Sales",
        "npm": "Net Profit Margin",
        "opmbd": "Operating Margin before Depreciation",
        "roa": "Return on Assets",
        "roe": "Return on Equity",
        "de_ratio": "Debt to Equity Ratio",
        "intcov_ratio": "Interest Coverage Ratio",
        "quick_ratio": "Quick Ratio",
        "curr_ratio": "Current Ratio",
        "at_turn": "Asset Turnover",
        "ptb": "Price to Book",
        "date": "Date",
        "VOL": "Volume Traded",
        "predicted_volume": "Predicted Volume Traded"
    }

    st.subheader("Overview")
    
    st.markdown("""
    **Three Final Datasets**: final_training_data.csv, final_testing_data.csv, completed_tickers.txt

    All of the initial datasets were pulled from the Wharton Research Data Services (WRDS). In order to get the financial ratios that we were interested in we used Compustat. In addition to these
    ratios, we also needed the stock prices which we had to pull from CRSP. The list of technology companies that we trained and tested were consistent across both aspects and 
    can be found in the completed_tickers.txt file. 
    
    All of the data is from **January 1st, 2010** to **December 31st, 2024**. The training data stops on **June 30th, 2024** and the remaining period is designated to the final testing set. Since WRDS does
    not have any information for the current year, we had to use the end of 2024 to locate the most recent and available earning announcements to allow our model to generate a prediction. Outside
    of the time period, the training and testing set contain the same exact columns which you can find below.
    """)

    st.divider()

    st.subheader("Data Dictionary")
    st.write("""Please scroll to see every variable.""")
    data_dict_df = pd.DataFrame(list(data_dict.items()), columns=["Column", "Description"])
    st.dataframe(data_dict_df)
    
    st.divider()

    st.subheader("Feature Importance")

    # --- Calculate feature importance
    model = pipeline.named_steps['model'] if 'model' in pipeline.named_steps else pipeline.named_steps['randomforestregressor']
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    sorted_features = np.array(selected_features)[sorted_idx]
    sorted_importance = importances[sorted_idx]

    # --- Plotly Bar Chart (sorted descending)
    fig = go.Figure(go.Bar(
        x=sorted_importance[::-1],
        y=sorted_features[::-1],
        orientation='h',
        marker=dict(color='royalblue')
    ))

    fig.update_layout(
        title="Top Feature Importances",
        xaxis_title="Importance",
        yaxis_title="Feature",
        height=600,
        template="plotly_white",
        margin=dict(l=40, r=20, t=50, b=40)
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption("""
    **Interpretation:**  
    - "Current Volume" is the strongest driver of post-earnings trading activity.
    - Operational metrics like Return on Assets, EV/EBITDA, and EBITDA Margin also contribute meaningfully.
    - Financial health indicators (e.g., Debt-to-Equity, Interest Coverage) play a secondary role.
    """)

    st.divider()
    
    st.subheader("Key Takeaways")
    st.write("""
    - Stocks with recent strong trading activity tend to exhibit larger post-earnings volume spikes.
    - Companies with higher profitability and operational efficiency metrics show more predictable volume patterns.
    - Financial fundamentals enhance prediction beyond technical trading patterns alone.
    """)

# Page 3: Volume Prediction
elif page == "Volume Prediction":
    st.title("Volume Prediction After Earnings Release")

    ticker = st.selectbox("Select a stock ticker:", tickers)

    tab1, tab2 = st.tabs(["Company Snapshot", "Volume Prediction Summary"])

    # --- Company Snapshot ---
    with tab1:
        st.header(f"{ticker} - Company Snapshot")

        stock = yf.Ticker(ticker)
        try:
            info = stock.info
        except Exception:
            info = {} 
        
        # --- Company Overview Section ---
        st.subheader(f"{info.get('longName', ticker)} Overview")
        
        company_summary = info.get('longBusinessSummary', None)
        if company_summary:
            st.write(company_summary)
        else:
            st.write("No company overview available.")
        
        company_website = info.get('website', None)
        if company_website:
            st.markdown(f"[Visit Website]({company_website})")
        
        st.divider()
        # --- Company Basic Info (Valuation Measures + Financial Highlights) ---
        st.subheader("Company Overview")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Valuation Measures")
            st.markdown(f"**Market Cap:** {info.get('marketCap', 'N/A'):,}" if info.get('marketCap') else "**Market Cap:** N/A")
            st.markdown(f"**Enterprise Value:** {info.get('enterpriseValue', 'N/A'):,}" if info.get('enterpriseValue') else "**Enterprise Value:** N/A")
            st.markdown(f"**Trailing P/E:** {info.get('trailingPE', 'N/A')}")
            st.markdown(f"**Forward P/E:** {info.get('forwardPE', 'N/A')}")
            st.markdown(f"**PEG Ratio:** {info.get('pegRatio', 'N/A')}")
            st.markdown(f"**Price/Sales:** {info.get('priceToSalesTrailing12Months', 'N/A')}")
            st.markdown(f"**Price/Book:** {info.get('priceToBook', 'N/A')}")
            st.markdown(f"**Enterprise Value/Revenue:** {info.get('enterpriseToRevenue', 'N/A')}")
            st.markdown(f"**Enterprise Value/EBITDA:** {info.get('enterpriseToEbitda', 'N/A')}")

        with col2:
            st.markdown("### Financial Highlights")
            st.markdown(f"**Profit Margin:** {info.get('profitMargins', 'N/A')}")
            st.markdown(f"**Return on Assets (ttm):** {info.get('returnOnAssets', 'N/A')}")
            st.markdown(f"**Return on Equity (ttm):** {info.get('returnOnEquity', 'N/A')}")
            st.markdown(f"**Revenue (ttm):** {info.get('totalRevenue', 'N/A'):,}" if info.get('totalRevenue') else "**Revenue (ttm):** N/A")
            st.markdown(f"**Net Income (ttm):** {info.get('netIncomeToCommon', 'N/A'):,}" if info.get('netIncomeToCommon') else "**Net Income (ttm):** N/A")
            st.markdown(f"**Diluted EPS (ttm):** {info.get('trailingEps', 'N/A')}")
            st.markdown(f"**Total Cash (mrq):** {info.get('totalCash', 'N/A'):,}" if info.get('totalCash') else "**Total Cash (mrq):** N/A")
            st.markdown(f"**Total Debt/Equity (mrq):** {info.get('debtToEquity', 'N/A')}")
            st.markdown(f"**Levered Free Cash Flow:** {info.get('leveredFreeCashflow', 'N/A'):,}" if info.get('leveredFreeCashflow') else "**Levered Free Cash Flow:** N/A")

        st.divider()

        # --- Recent Stock Price Chart ---
        st.subheader("Recent Stock Price")
        
        time_range = st.selectbox(
            "Select time range for stock price:", 
            options=["1D", "5D", "1M", "6M", "YTD", "1Y", "5Y", "Max"], 
            index=2
        )
        
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
        period, interval = time_mapping.get(time_range, ("1mo", "1d"))
        
        hist = stock.history(period=period, interval=interval)
        
        if not hist.empty:
            fig_price = go.Figure()
            fig_price.add_trace(go.Scatter(
                x=hist.index, 
                y=hist['Close'], 
                mode='lines', 
                name='Close Price', 
                line=dict(color='green')
            ))
            fig_price.update_layout(
                title="Recent Stock Price",
                xaxis_title="Date",
                yaxis_title="Close Price ($)",
                height=400,
                margin=dict(l=20, r=20, t=50, b=20),
                xaxis_rangeslider_visible=False
            )
            st.plotly_chart(fig_price, use_container_width=True)
        else:
            st.warning("No stock price data available for this time range.")
        
        st.divider()

        
    # --- Stock Details (fixed to match exactly 4-column layout you want) ---
        st.subheader("Stock Details")
        
        detail_col1, detail_col2, detail_col3, detail_col4 = st.columns(4)
        
        with detail_col1:
            st.markdown(f"**Previous Close:** {info.get('previousClose', 'N/A')}")
            st.markdown(f"**Open:** {info.get('open', 'N/A')}")
            st.markdown(f"**Bid:** {info.get('bid', 'N/A')}")
            st.markdown(f"**Ask:** {info.get('ask', 'N/A')}")
        
        with detail_col2:
            st.markdown(f"**Day's Range:** {info.get('dayLow', 'N/A')} - {info.get('dayHigh', 'N/A')}")
            st.markdown(f"**52 Week Range:** {info.get('fiftyTwoWeekLow', 'N/A')} - {info.get('fiftyTwoWeekHigh', 'N/A')}")
            st.markdown(f"**Volume:** {info.get('volume', 'N/A'):,}" if info.get('volume') else "**Volume:** N/A")
            st.markdown(f"**Avg Volume:** {info.get('averageVolume', 'N/A'):,}" if info.get('averageVolume') else "**Avg Volume:** N/A")
        
        with detail_col3:
            st.markdown(f"**Market Cap (intraday):** {info.get('marketCap', 'N/A'):,}" if info.get('marketCap') else "**Market Cap (intraday):** N/A")
            st.markdown(f"**Beta (5Y Monthly):** {info.get('beta', 'N/A')}")
            st.markdown(f"**PE Ratio (TTM):** {info.get('trailingPE', 'N/A')}")
            st.markdown(f"**EPS (TTM):** {info.get('trailingEps', 'N/A')}")
        
        with detail_col4:
            st.markdown(f"**Earnings Date:** {info.get('earningsDate', ['N/A'])[0] if isinstance(info.get('earningsDate'), list) else info.get('earningsDate', 'N/A')}")
            st.markdown(f"**Forward Dividend & Yield:** {info.get('dividendRate', 'N/A')} ({info.get('dividendYield', 'N/A')})")
            st.markdown(f"**Ex-Dividend Date:** {info.get('exDividendDate', 'N/A')}")
            st.markdown(f"**1Y Target Est:** {info.get('targetMeanPrice', 'N/A')}")

        st.divider()

# --- Volume Prediction Summary ---
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
    
            # --- Main Prediction Numbers ---
            st.subheader(f"{ticker} - Volume Prediction Details")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Earnings Release Date:** {ticker_data['date'].values[0]}")
                st.markdown(f"**Predicted Volume:** {int(prediction):,}")
                st.markdown(f"**Percent Difference:** {percent_diff:.2f}%")
            
            with col2:
                st.markdown(f"**Actual Volume:** {int(actual_volume):,}")
                st.markdown(f"**Difference:** {int(volume_diff):,} shares")
            
                avg_volume = info.get('averageVolume', None)
                if avg_volume:
                    avg_vol_change = (prediction - avg_volume) / avg_volume * 100
                    st.markdown(f"**Volume Change vs Avg.:** {avg_vol_change:.2f}%")
                else:
                    st.markdown(f"**Volume Change vs Avg.:** N/A")

            st.divider()
            # --- Recent Volume Chart ---
            st.subheader("Recent Trading Volume")
            
            volume_time_range = st.selectbox(
                "Select time range for trading volume:", 
                options=["1D", "5D", "1M", "6M", "YTD", "1Y", "5Y", "Max"], 
                index=2
            )
            
            volume_time_mapping = {
                "1D": ("1d", "5m"),
                "5D": ("5d", "15m"),
                "1M": ("1mo", "1d"),
                "6M": ("6mo", "1d"),
                "YTD": ("ytd", "1d"),
                "1Y": ("1y", "1d"),
                "5Y": ("5y", "1wk"),
                "Max": ("max", "1mo")
            }
            volume_period, volume_interval = volume_time_mapping.get(volume_time_range, ("1mo", "1d"))
            
            hist_volume = stock.history(period=volume_period, interval=volume_interval)
            
            if not hist_volume.empty:
                fig_vol = go.Figure()
                fig_vol.add_trace(go.Bar(
                    x=hist_volume.index,
                    y=hist_volume['Volume'],
                    marker_color='lightblue',
                    name='Trading Volume'
                ))
                fig_vol.update_layout(
                    title="Recent Trading Volume",
                    xaxis_title="Date",
                    yaxis_title="Volume",
                    height=400,
                    margin=dict(l=20, r=20, t=50, b=20),
                    xaxis_rangeslider_visible=False
                )
                st.plotly_chart(fig_vol, use_container_width=True)
            else:
                st.warning("No volume data available for this time range.")
            
            st.divider()

            # --- Model Performance Metrics ---
            st.subheader("Model Performance")
            
            # Real evaluation results
            r2_score = 0.8206095  # R-squared
            adjusted_r2 = 0.815333  # Adjusted R-Squared
            mae_score = 3228716.84  # Mean Absolute Error
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("R-squared (Model Predictive Power)", f"{r2_score * 100:.2f}%")

            with col2:
                st.metric("Adjusted R-squared (Model Predictive Power)", f"{adjusted_r2 * 100:.2f}%")
            
            with col3:
                st.metric("Mean Absolute Error (MAE)", f"{mae_score:,.0f} shares")
            
            st.caption("""
            **Interpretation:**  
            The model explains approximately 82% of the variation in post-earnings trading volumes, meaning it captures most of the key factors influencing volume changes.
            Additionally, the Adjusted $R^{2}$ shows that the variation covered by the features in the model does not directly correlate to having a lot of features present.
            On average, the model's predictions differ from the actual trading volume by about 3.2 million shares.  
            While not perfect, the model provides a strong, reliable signal for anticipating major shifts in trading activity.
            """)

            st.divider()
    
            # --- Volume Compared to Stock's Average Volume ---
            st.subheader("Volume vs. Stock Average Volume")
            
            avg_volume = info.get('averageVolume', None)
            
            if avg_volume:
                volume_change = (prediction - avg_volume) / avg_volume * 100
                st.markdown(f"**Predicted Volume Change vs Average:** {volume_change:.2f}%")
            
                st.caption("""
                **Interpretation:**  
                A volume change of {:.2f}% compared to the stock’s typical daily trading activity suggests a relatively mild shift in investor interest.  
                Small changes like this often indicate normal market fluctuations rather than major news or sentiment changes.  
                Traders may not expect significant liquidity improvements or unusual price volatility based on this prediction.
                """.format(volume_change))
            else:
                st.write("Average volume data not available.")
            
            st.divider()

            # --- Business Interpretation ---
            st.subheader("Business Interpretation")
            
            if prediction >= 50_000_000:
                interpretation_text = (
                    "High trading volume is expected following this earnings announcement, indicating a significant investor reaction.\n\n"
                    "Such spikes often reflect strong sentiment shifts or major reassessments of company fundamentals.\n\n"
                    "Traders and portfolio managers should be prepared for increased liquidity and potential price volatility."
                )
            elif prediction >= 10_000_000:
                interpretation_text = (
                    "Moderate trading volume is anticipated after earnings, suggesting a measured investor response.\n\n"
                    "While not as dramatic as major spikes, moderate volume increases can still offer trading opportunities, especially for event-driven or momentum strategies.\n\n"
                    "Liquidity is expected to improve compared to normal trading days."
                )
            else:
                interpretation_text = (
                    "Low trading volume is forecasted following the earnings release, implying limited investor reaction.\n\n"
                    "This could suggest that the results met expectations or that market participants are waiting for more information.\n\n"
                    "Trading conditions may remain stable, with minimal volatility expected."
                )
            
            # Display nicely
            st.write(interpretation_text)
            
            st.divider()

    
        else:
            st.warning("No recent prediction data available for this ticker.")
    

# Page 5: Top Stocks
elif page == "Top Stocks by Volume":
    st.title("Top 5 Traded Stocks in the Past 3 Months")
    st.markdown("Displays the daily volume traded over the past 90 days for 5 selected major stocks.")
    tickers = ['AAPL', 'MSFT', 'TSLA', 'NVDA', 'GOOGL']
    end_date = pd.to_datetime("today")
    start_date = end_date - pd.Timedelta(days=90)
    fig = go.Figure()
    volume_frames = []

    """ for ticker in tickers:
        df = yf.download(ticker, start=start_date, end=end_date, interval='1d', progress=False)
        if not df.empty:
            df['Volume_Millions'] = df['Volume'] / 1_000_000
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['Volume_Millions'],
                mode='lines',
                name=ticker
            ))
            # Save for table (no extra ticker column needed)
            df = df[['Volume']].copy()
            df.columns = [ticker]  # Rename 'Volume' column to ticker name
            volume_frames.append(df)

    # --- Display Line Chart
    fig.update_layout(
        title="Volume Traded (in Millions) Over the Past 3 Months",
        xaxis_title="Date",
        yaxis_title="Volume (Millions)",
        template="plotly_white",
        height=500,
        margin=dict(l=20, r=20, t=50, b=20),
        legend_title="Stock Ticker",
        xaxis_rangeslider_visible=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # --- Clean Recent Volume Table
    st.subheader("Recent Volume Data (Last 5 Days)")

    if volume_frames:
        combined_df = pd.concat(volume_frames, axis=1)
        combined_df.index = pd.to_datetime(combined_df.index).date
        combined_df = combined_df.sort_index(ascending=False)
        combined_df = combined_df.applymap(lambda x: f"{int(x):,}")
        
        st.dataframe(combined_df.head(5))
    else:
        st.warning("No volume data available.")
    """








