import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import warnings

warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, mean_absolute_error

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="wide"
)

st.markdown("""
<style>
    /* Animated Gradient Background */
    .stApp {
        background: linear-gradient(-45deg, #0f172a, #1e293b, #0f172a, #334155);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
        color: #f8fafc;
    }
    
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Glassmorphism for metrics */
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1), 0 1px 3px rgba(0, 0, 0, 0.08);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        animation: fadeUp 0.6s ease-out;
    }
    
    [data-testid="stMetric"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 15px rgba(0, 0, 0, 0.2), 0 4px 6px rgba(0, 0, 0, 0.1);
        border-color: rgba(255, 255, 255, 0.2);
    }
    
    /* Glassmorphism for DataFrames/Tables */
    [data-testid="stDataFrame"] > div {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
    }
    
    /* Smooth Inputs */
    .stTextInput > div > div > input {
        background-color: rgba(255, 255, 255, 0.05) !important;
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 8px !important;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.3) !important;
    }
    
    /* Buttons styling */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
        box-shadow: 0 4px 6px rgba(59, 130, 246, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(59, 130, 246, 0.4);
        background: linear-gradient(135deg, #60a5fa 0%, #3b82f6 100%);
        border: none;
        color: white;
    }
    
    /* Headers animation */
    h1, h2, h3 {
        animation: fadeDown 0.8s ease-out;
    }
    
    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes fadeDown {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: rgba(255, 255, 255, 0.02);
        padding: 5px 10px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px !important;
        color: #cbd5e1 !important;
        background-color: transparent !important;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: rgba(255, 255, 255, 0.1) !important;
        color: white !important;
        border-bottom: 2px solid #3b82f6 !important;
        backdrop-filter: blur(10px);
    }
</style>
""", unsafe_allow_html=True)

st.title("📈 Stock Price Prediction Dashboard")
st.markdown("Multi-Horizon Machine Learning System (Classification + Regression)")

ticker = st.text_input("Ticker Symbol (e.g., AAPL, TATASTEEL.NS)", "AAPL").upper()
period = "max"

@st.cache_data
def load_data(ticker, period):
    df = yf.download(ticker, period=period, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

df = load_data(ticker, period)

df = df.dropna(subset=['Close'])

if len(df) < 300:
    st.error(f"Not enough historical data found for {ticker}. Check if the ticker symbol is correct.")
    st.stop()

current_price = df['Close'].iloc[-1]

df["Return_1"] = df["Close"].pct_change(1)
df["Return_5"] = df["Close"].pct_change(5)
df["Return_20"] = df["Close"].pct_change(20)

df["MA10"] = df["Close"].rolling(10).mean()
df["MA20"] = df["Close"].rolling(20).mean()
df["MA50"] = df["Close"].rolling(50).mean()
df["MA100"] = df["Close"].rolling(100).mean()
df["MA200"] = df["Close"].rolling(200).mean()

df["MA10_Ratio"] = df["Close"] / df["MA10"]
df["MA20_Ratio"] = df["Close"] / df["MA20"]
df["MA50_Ratio"] = df["Close"] / df["MA50"]
df["MA200_Ratio"] = df["Close"] / df["MA200"]

df["Volume_MA20"] = df["Volume"].rolling(20).mean()
df["Volume_Ratio"] = df["Volume"] / df["Volume_MA20"]

df["Volatility_10"] = df["Return_1"].rolling(10).std()
df["Volatility_20"] = df["Return_1"].rolling(20).std()
df["Volatility_50"] = df["Return_1"].rolling(50).std()

df["Daily_Range"] = df["High"] - df["Low"]
df["Range_Percent"] = (df["High"] - df["Low"]) / df["Close"]
df["Open_Close_Diff"] = df["Close"] - df["Open"]

df["Momentum_5"] = df["Close"] - df["Close"].shift(5)
df["Momentum_10"] = df["Close"] - df["Close"].shift(10)
df["Momentum_20"] = df["Close"] - df["Close"].shift(20)

df["Higher_High"] = (df["High"] > df["High"].shift(1)).astype(int)
df["Higher_Low"] = (df["Low"] > df["Low"].shift(1)).astype(int)

features = [
    "Open", "High", "Low", "Close", "Volume",
    "Return_1", "Return_5", "Return_20",
    "MA10", "MA20", "MA50", "MA100", "MA200",
    "MA10_Ratio", "MA20_Ratio", "MA50_Ratio", "MA200_Ratio",
    "Volume_Ratio", "Volatility_10", "Volatility_20", "Volatility_50",
    "Daily_Range", "Range_Percent", "Open_Close_Diff",
    "Momentum_5", "Momentum_10", "Momentum_20",
    "Higher_High", "Higher_Low"
]

df_features = df[features].copy()

df_features.replace([np.inf, -np.inf], np.nan, inplace=True)
df_features.ffill(inplace=True)
df_features.bfill(inplace=True) 

latest_features = df_features.iloc[-1:]

horizons = {
    "1 Day": 1,
    "5 Days": 5,
    "1 Month (21d)": 21,
    "6 Months (126d)": 126
}

st.write("---")
st.header(f"Asset: {ticker}")
st.subheader(f"Latest Market Price: ₹{current_price:.2f}" if ".NS" in ticker or ".BO" in ticker else f"Latest Market Price: ${current_price:.2f}")
st.write("---")

cols = st.columns(4)
predictions_data = []

for i, (label, days) in enumerate(horizons.items()):
    
    target_return = (df["Close"].shift(-days) - df["Close"]) / df["Close"]
    
    target_return.replace([np.inf, -np.inf], np.nan, inplace=True)
    target_class = (target_return > 0).astype(int)
    
    valid_idx = target_return.dropna().index
    X_train = df_features.loc[valid_idx].dropna()
    
    y_reg_train = target_return.loc[X_train.index]
    y_class_train = target_class.loc[X_train.index]
    
    rf_class = RandomForestClassifier(n_estimators=500, max_depth=8, random_state=42, n_jobs=-1)
    rf_reg = RandomForestRegressor(n_estimators=500, max_depth=8, random_state=42, n_jobs=-1)
    
    rf_class.fit(X_train, y_class_train)
    rf_reg.fit(X_train, y_reg_train)
    
    prob_up = rf_class.predict_proba(latest_features)[0][1]
    expected_return = rf_reg.predict(latest_features)[0]
    
    if XGB_AVAILABLE:
        xgb_class = XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=4, random_state=42, eval_metric="logloss")
        xgb_reg = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=4, random_state=42)
        
        xgb_class.fit(X_train, y_class_train)
        xgb_reg.fit(X_train, y_reg_train)
        
        prob_up = (prob_up + xgb_class.predict_proba(latest_features)[0][1]) / 2
        expected_return = (expected_return + xgb_reg.predict(latest_features)[0]) / 2

    expected_price = current_price * (1 + expected_return)
    price_delta = expected_price - current_price
    direction = "📈 BULLISH" if expected_return > 0 else "📉 BEARISH"
    
    with cols[i]:
        st.markdown(f"### {label}")
        st.metric(
            "Expected Price", 
            f"{expected_price:.2f}", 
            f"{price_delta:.2f} ({(expected_return*100):.2f}%)"
        )
        st.write(f"**Signal:** {direction}")
        st.write(f"**Confidence:** {prob_up*100:.1f}%")
        
    predictions_data.append({
        "Horizon": label,
        "Expected Price": expected_price,
        "Return %": expected_return * 100,
        "Signal": direction,
        "Confidence %": prob_up * 100
    })

st.write("---")

df_charts = df.copy()

delta = df_charts["Close"].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()
rs = avg_gain / avg_loss
df_charts["RSI"] = 100 - (100 / (1 + rs))

ema12 = df_charts["Close"].ewm(span=12).mean()
ema26 = df_charts["Close"].ewm(span=26).mean()
df_charts["MACD"] = ema12 - ema26
df_charts["MACD_SIGNAL"] = df_charts["MACD"].ewm(span=9).mean()
df_charts["MACD_HIST"] = df_charts["MACD"] - df_charts["MACD_SIGNAL"]

df_charts["BB_MID"] = df_charts["Close"].rolling(20).mean()
bb_std = df_charts["Close"].rolling(20).std()
df_charts["BB_UPPER"] = df_charts["BB_MID"] + 2 * bb_std
df_charts["BB_LOWER"] = df_charts["BB_MID"] - 2 * bb_std

df_charts = df_charts.dropna()

tab1, tab2 = st.tabs(["📊 Technical Charts", "📦 Data & Export"])

with tab1:
    st.subheader("Candlestick + Bollinger Bands (Last 200 Days)")
    
    plot_df = df_charts.tail(200)
    
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=plot_df.index, open=plot_df["Open"], high=plot_df["High"], low=plot_df["Low"], close=plot_df["Close"], name="Price"))
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["BB_UPPER"], name="BB Upper", line=dict(color='rgba(255, 255, 255, 0.3)')))
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["BB_LOWER"], name="BB Lower", fill="tonexty", fillcolor='rgba(255, 255, 255, 0.1)', line=dict(color='rgba(255, 255, 255, 0.3)')))
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("RSI (Relative Strength Index)")
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=plot_df.index, y=plot_df["RSI"], name="RSI", line=dict(color='purple')))
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
    fig_rsi.update_layout(height=300)
    st.plotly_chart(fig_rsi, use_container_width=True)

    st.subheader("MACD (Moving Average Convergence Divergence)")
    fig_macd = go.Figure()
    fig_macd.add_trace(go.Scatter(x=plot_df.index, y=plot_df["MACD"], name="MACD", line=dict(color='blue')))
    fig_macd.add_trace(go.Scatter(x=plot_df.index, y=plot_df["MACD_SIGNAL"], name="Signal", line=dict(color='orange')))
    fig_macd.add_trace(go.Bar(x=plot_df.index, y=plot_df["MACD_HIST"], name="Histogram", marker_color='gray'))
    fig_macd.update_layout(height=300)
    st.plotly_chart(fig_macd, use_container_width=True)

with tab2:
    st.subheader("Prediction Summary Table")
    summary_df = pd.DataFrame(predictions_data)
    st.dataframe(summary_df)
    
    st.subheader("Download Raw Dataset")
    csv = df_features.to_csv().encode("utf-8")
    st.download_button(
        label="Download Historical Feature Data CSV",
        data=csv,
        file_name=f"{ticker}_historical_features.csv",
        mime="text/csv"
    )