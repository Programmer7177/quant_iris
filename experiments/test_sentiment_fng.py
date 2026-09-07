"""
BENCHMARK: INTEGRATING FEAR & GREED SENTIMENT INDEX (2018 - 2026)
Tests if adding Sentiment (Crowd Psychology / Contrarian Indicator):
  - fng_value (0 = Extreme Fear, 100 = Extreme Greed)
  - fng_change_7d
  - fng_dist_ema30 (Sentiment stretch)
improves Short (7d, 14d) and Mid-Term (30d) Directional Accuracy.
"""
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge

BTC_PATH = r"C:\Mirza Personal\crypto quant\data\btc_coinbase_10y.csv"
FNG_PATH = r"C:\Mirza Personal\crypto quant\data\fear_greed_full.csv"

def rsi(series, n):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = (-delta.clip(upper=0)).rolling(n).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def main():
    btc = pd.read_csv(BTC_PATH, parse_dates=["open_time"]).sort_values("open_time").reset_index(drop=True)
    btc["date"] = btc["open_time"].dt.strftime("%Y-%m-%d")
    
    fng = pd.read_csv(FNG_PATH)
    df = pd.merge(btc, fng, on="date", how="inner").sort_values("date").reset_index(drop=True)
    
    c = df["close"]
    h = df["high"]
    l = df["low"]
    
    # Features
    df["rsi_14"] = rsi(c, 14)
    df["rsi_30"] = rsi(c, 30)
    df["rsi_90"] = rsi(c, 90)
    hl_ratio = np.log(h / (l + 1e-9))
    df["vol_parkinson_14"] = np.sqrt((1.0 / (4.0 * np.log(2.0))) * (hl_ratio**2).rolling(14).mean()) * np.sqrt(365)
    df["dist_ema50"] = (c - c.ewm(span=50).mean()) / c.ewm(span=50).mean()
    
    # Sentiment features
    df["fng_level"] = df["fng_value"]
    df["fng_dist_ema30"] = (df["fng_value"] - df["fng_value"].ewm(span=30).mean())
    df["fng_change_7d"] = df["fng_value"] - df["fng_value"].shift(7)
    
    # Targets
    for horizon in [7, 14, 30]:
        df[f"fwd_ret_{horizon}"] = np.log(c.shift(-horizon) / c)
        
    df = df.dropna().reset_index(drop=True)
    n = len(df)
    split_idx = int(n * 0.65) # 65% train, 35% test out-of-sample (~1000 days test)
    
    train = df.iloc[:split_idx]
    
    print("=== BENCHMARK CROWD SENTIMENT: FEAR & GREED (2018 - 2026) ===")
    print(f"Total Sampel: {n} hari | Test Out-of-Sample: {n - split_idx} hari ({df['date'].iloc[split_idx]} s.d. {df['date'].iloc[-1]})\n")
    
    # Information Coefficients
    print("--- KORELASI STATISTIK (IC) FEAR & GREED VS RETURN KE DEPAN ---")
    for f in ["fng_level", "fng_dist_ema30", "fng_change_7d"]:
        ic7 = df[f].corr(df["fwd_ret_7"])
        ic14 = df[f].corr(df["fwd_ret_14"])
        ic30 = df[f].corr(df["fwd_ret_30"])
        print(f"  {f:<18}: IC (7h) = {ic7:+.4f} | IC (14h) = {ic14:+.4f} | IC (30h) = {ic30:+.4f}")
    print("-" * 75 + "\n")
    
    base_feats = ["rsi_14", "rsi_30", "rsi_90", "vol_parkinson_14", "dist_ema50"]
    sentiment_feats = base_feats + ["fng_level", "fng_dist_ema30", "fng_change_7d"]
    
    for h in [7, 14, 30]:
        test = df.iloc[split_idx:-h]
        y_tr = train[f"fwd_ret_{h}"].values
        y_te = test[f"fwd_ret_{h}"].values
        actual_dir = np.sign(y_te)
        
        # Model Baseline (No Sentiment)
        m_base = Ridge(alpha=100.0).fit(train[base_feats].values, y_tr)
        pred_base = m_base.predict(test[base_feats].values)
        acc_base = np.mean(np.sign(pred_base) == actual_dir)
        mape_base = np.mean(np.abs(test["close"] * np.exp(y_te) - test["close"] * np.exp(pred_base)) / (test["close"] * np.exp(y_te)))
        
        # Model + Sentiment
        m_sent = Ridge(alpha=100.0).fit(train[sentiment_feats].values, y_tr)
        pred_sent = m_sent.predict(test[sentiment_feats].values)
        acc_sent = np.mean(np.sign(pred_sent) == actual_dir)
        mape_sent = np.mean(np.abs(test["close"] * np.exp(y_te) - test["close"] * np.exp(pred_sent)) / (test["close"] * np.exp(y_te)))
        
        print(f"[HORIZON {h:2d} HARI]")
        print(f"  Tanpa Sentimen : Akurasi Arah = {acc_base:.2%} | Error MAPE = {mape_base:.2%}")
        print(f"  + Fear & Greed : Akurasi Arah = {acc_sent:.2%} | Error MAPE = {mape_sent:.2%}")
        delta = acc_sent - acc_base
        print(f"  --> Dampak Sentimen: Akurasi {delta:+.2%} | MAPE {mape_sent - mape_base:+.2%}\n")

if __name__ == "__main__":
    main()
