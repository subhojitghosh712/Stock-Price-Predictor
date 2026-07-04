import yfinance as yf
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
import warnings

warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
try:
    from xgboost import XGBClassifier, XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_data(ticker, period="max"):
    df = yf.download(ticker, period=period, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna(subset=['Close'])
    return df

def get_features(df):
    df = df.copy()
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
    
    return df, df_features

@app.get("/api/predict/{ticker}")
def predict(ticker: str):
    ticker = ticker.upper()
    df = load_data(ticker)
    if len(df) < 300:
        raise HTTPException(status_code=400, detail=f"Not enough historical data found for {ticker}.")

    current_price = float(df['Close'].iloc[-1])
    
    df, df_features = get_features(df)
    latest_features = df_features.iloc[-1:]

    horizons = {
        "1 Day": 1,
        "5 Days": 5,
        "1 Month (21d)": 21,
        "6 Months (126d)": 126
    }

    predictions_data = []

    for label, days in horizons.items():
        target_return = (df["Close"].shift(-days) - df["Close"]) / df["Close"]
        target_return.replace([np.inf, -np.inf], np.nan, inplace=True)
        target_class = (target_return > 0).astype(int)
        
        valid_idx = target_return.dropna().index
        X_train = df_features.loc[valid_idx].dropna()
        
        y_reg_train = target_return.loc[X_train.index]
        y_class_train = target_class.loc[X_train.index]
        
        rf_class = RandomForestClassifier(n_estimators=1000, max_depth=8, random_state=42, n_jobs=-1)
        rf_reg = RandomForestRegressor(n_estimators=1000, max_depth=8, random_state=42, n_jobs=-1)
        
        rf_class.fit(X_train, y_class_train)
        rf_reg.fit(X_train, y_reg_train)
        
        prob_up = float(rf_class.predict_proba(latest_features)[0][1])
        expected_return = float(rf_reg.predict(latest_features)[0])
        
        if XGB_AVAILABLE:
            xgb_class = XGBClassifier(n_estimators=1000, learning_rate=0.05, max_depth=4, random_state=42, eval_metric="logloss")
            xgb_reg = XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=4, random_state=42)
            
            xgb_class.fit(X_train, y_class_train)
            xgb_reg.fit(X_train, y_reg_train)
            
            prob_up = (prob_up + float(xgb_class.predict_proba(latest_features)[0][1])) / 2
            expected_return = (expected_return + float(xgb_reg.predict(latest_features)[0])) / 2

        expected_price = current_price * (1 + expected_return)
        price_delta = expected_price - current_price
        direction = "📈 BULLISH" if expected_return > 0 else "📉 BEARISH"
        
        predictions_data.append({
            "horizon": label,
            "expectedPrice": round(expected_price, 2),
            "priceDelta": round(price_delta, 2),
            "returnPct": round(expected_return * 100, 2),
            "signal": direction,
            "confidence": round(prob_up * 100, 1)
        })

    return {
        "ticker": ticker,
        "currentPrice": current_price,
        "predictions": predictions_data
    }

@app.get("/api/chart/{ticker}")
def chart_data(ticker: str):
    ticker = ticker.upper()
    df = load_data(ticker)
    if len(df) < 200:
         raise HTTPException(status_code=400, detail="Not enough data for charts.")

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
    plot_df = df_charts.tail(200)

    dates = plot_df.index.strftime('%Y-%m-%d').tolist()

    return {
        "dates": dates,
        "open": plot_df["Open"].tolist(),
        "high": plot_df["High"].tolist(),
        "low": plot_df["Low"].tolist(),
        "close": plot_df["Close"].tolist(),
        "bb_upper": plot_df["BB_UPPER"].tolist(),
        "bb_lower": plot_df["BB_LOWER"].tolist(),
        "rsi": plot_df["RSI"].tolist(),
        "macd": plot_df["MACD"].tolist(),
        "macd_signal": plot_df["MACD_SIGNAL"].tolist(),
        "macd_hist": plot_df["MACD_HIST"].tolist()
    }

@app.get("/api/download/{ticker}")
def download_data(ticker: str):
    ticker = ticker.upper()
    df = load_data(ticker)
    if len(df) == 0:
        raise HTTPException(status_code=400, detail="No data.")
    _, df_features = get_features(df)
    csv = df_features.to_csv()
    return Response(content=csv, media_type="text/csv", headers={"Content-Disposition": f"attachment; filename={ticker}_historical_features.csv"})

app.mount("/", StaticFiles(directory="static", html=True), name="static")
