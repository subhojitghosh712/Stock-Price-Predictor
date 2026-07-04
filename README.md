# 📈 Stock Price Prediction Dashboard

A comprehensive Multi-Horizon Machine Learning System for stock market analysis and prediction. This project provides both a FastAPI-based modern web application and a Streamlit dashboard to forecast stock prices using advanced ensemble machine learning techniques (Random Forest and XGBoost) along with extensive technical analysis charts.

## 🚀 Features

- **Multi-Horizon Predictions**: Predicts stock price movements over multiple timeframes: 1 Day, 5 Days, 1 Month (21 days), and 6 Months (126 days).
- **Dual ML Approach**: Uses both Classification (for predicting market direction - Bullish/Bearish) and Regression (for predicting expected price returns) simultaneously.
- **Advanced Feature Engineering**: Generates around 30 technical features on-the-fly, including moving averages, volatility, momentum, and daily ranges.
- **Interactive Technical Charts**: Visualizes stock data with interactive Plotly charts, including:
  - Candlestick charts with Bollinger Bands
  - Relative Strength Index (RSI)
  - Moving Average Convergence Divergence (MACD)
- **Two UI Options**:
  - A responsive web dashboard using HTML/CSS/JS served by a blazing-fast FastAPI backend.
  - A Streamlit dashboard for quick Python-native deployments.
- **Data Export**: Allows users to download the raw engineered dataset for further local analysis.

---

## 🧠 Machine Learning Models & Architecture

The application fetches historical data directly from Yahoo Finance (`yfinance`) and computes a rich set of features. For each prediction horizon, it dynamically trains machine learning models on the historical data of the requested stock.

### Models Used
1. **Random Forest**: 
   - `RandomForestClassifier`: Predicts the probability of the price going up (Confidence).
   - `RandomForestRegressor`: Predicts the expected return percentage.
2. **XGBoost** (Optional but recommended):
   - If installed, the system will also train `XGBClassifier` and `XGBRegressor` models.
   - The final prediction is an ensemble (average) of both Random Forest and XGBoost outputs, yielding more robust predictions.

### Engineered Features
The models are trained on the following derived features:
- **Price Returns**: 1-day, 5-day, and 20-day percentage returns.
- **Moving Averages (MA)**: 10, 20, 50, 100, and 200-day MAs, along with their ratios compared to the current closing price.
- **Volatility**: Rolling standard deviations of returns over 10, 20, and 50 days.
- **Volume Metrics**: 20-day volume moving average and volume ratio.
- **Price Action**: Daily range, open-close differences, higher highs, and higher lows.
- **Momentum**: Price differences over 5, 10, and 20-day windows.

---

## 📁 Project Structure

```text
.
├── app.py           # FastAPI backend server (API & static file routing)
├── main.py          # Streamlit version of the dashboard
├── basic.py         # Standalone script for Nifty 50 backtesting & evaluation
├── static/          # Frontend assets for the FastAPI app
│   ├── index.html   # Main web interface
│   ├── app.js       # Frontend logic and API integration
│   └── style.css    # Modern UI styling (glassmorphism, gradients)
└── README.md        # Project documentation
```

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### 1. Clone or Download the Repository
Navigate to the project folder in your terminal:
```bash
cd "Stock Price Prediction"
```

### 2. Create a Virtual Environment (Recommended)
```bash
python -m venv .venv
# On Windows
.venv\Scripts\activate
# On Mac/Linux
source .venv/bin/activate
```

### 3. Install Dependencies
Install the required Python packages:
```bash
pip install fastapi uvicorn yfinance pandas numpy scikit-learn plotly streamlit
```
*Optional but recommended for better predictions:*
```bash
pip install xgboost
```

---

## 💻 Usage

You have two choices for running the application:

### Option A: FastAPI + Modern Web Frontend (Recommended)
This runs a fast backend API and serves the beautifully designed HTML/JS/CSS frontend.
```bash
uvicorn app:app --reload
```
Once running, open your browser and go to: `http://localhost:8000`

### Option B: Streamlit Dashboard
This runs the pure Python Streamlit app.
```bash
streamlit run main.py
```
A browser window should automatically open pointing to `http://localhost:8501`.

### Standalone Backtesting
If you want to run a quick backtest on the Nifty 50 index to see raw model accuracy:
```bash
python basic.py
```

---

## ⚠️ Disclaimer

**This software is for educational and informational purposes only.** It should not be considered financial advice. Stock markets are inherently volatile and unpredictable. Machine learning models are based on historical data and do not guarantee future results. Always do your own research or consult with a qualified financial advisor before making any investment decisions. The creators of this project hold no liability for any financial losses incurred while using this tool.
