import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score 

nifty50 = yf.Ticker("^NSEI")
nifty50 = nifty50.history(period="max")

del nifty50["Dividends"]
del nifty50["Stock Splits"]

nifty50["Tomorrow"] = nifty50["Close"].shift(-1)
nifty50["Target"] = (nifty50["Tomorrow"] > nifty50["Close"]).astype(int)

model = RandomForestClassifier(n_estimators=2000, min_samples_split=50, random_state=1)

train = nifty50.iloc[:-100]
test = nifty50.iloc[-100:]

predictors = ["Close", "Volume", "Open", "High", "Low"]

print(nifty50)

def predict(train, test, predictors, model):
    model.fit(train[predictors],train["Target"])
    preds = model.predict_proba(test[predictors])[:,1]
    preds[preds >= .6] = 1
    preds[preds < .6] = 0
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined

def backtest(data, model, predictors, start=2500, step=250):
    all_predictions = []
    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i + step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)
    return pd.concat(all_predictions)

horizons = [2, 5, 60, 250, 1000]
new_predictors = []

for horizon in horizons:
    rolling_averages = nifty50.rolling(horizon).mean()

    ratio_column = f"Close_Ratio_{horizon}"
    nifty50[ratio_column] = nifty50["Close"] / rolling_averages["Close"]

    trend_column = f"Trend_{horizon}"
    nifty50[trend_column] = nifty50.shift(1).rolling(horizon).sum()["Target"]

    new_predictors += [ratio_column, trend_column]

predictions = backtest(nifty50, model, predictors)
print("Accuracy - ", precision_score(predictions["Target"], predictions["Predictions"])*100,"%")

