# 📈 Stock Volume Predictor

A 3-page Streamlit web app that predicts tomorrow's stock trading volume using financial ratios from Yahoo Finance and a pre-trained XGBoost model.

---

## 🚀 Features

- 🔮 **Prediction Page**: Enter a stock ticker to get a real-time volume forecast for tomorrow.
- 📊 **Financial Details**: View recent price trends and key stats like P/E ratio, market cap, and sector.
- 🧠 **Feature Insights**: Explore feature importance from the trained XGBoost model.

---

## 🛠 Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the App

```bash
streamlit run app.py
```

Make sure you have the following files:
- `model/volume_model.pkl`
- `model/scaler.pkl`
- `model/selected_features.pkl`

---

## 🌐 Live Deployment Options

- [Streamlit Cloud](https://streamlit.io/cloud)
- [Hugging Face Spaces](https://huggingface.co/spaces)
- [Render](https://render.com)

---

## 📬 Contact

Built by Quan Minh Nguyen