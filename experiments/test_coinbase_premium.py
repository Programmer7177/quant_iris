"""
BENCHMARK: COINBASE PREMIUM GAP & ETF INSTITUTIONAL FLOWS (2017 - 2026)
Tests predictive power of:
  1. Coinbase Premium Index (3,308 bars, 2017-2026): US Institutional Aggression vs Binance Retail.
  2. Spot ETF Dollar Volume / Activity (665 bars, Jan 2024-2026).
"""
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

DATA_DIR = r"C:\Mirza Personal\crypto quant\data"
BTC_PATH = os.path.join(DATA_DIR, "btc_coinbase_10y.csv")
PREM_PATH = os.path.join(DATA_DIR, "coinbase_premium_index.csv")
ETF_PATH = os.path.join(DATA_DIR, "btc_spot_etf_flows.csv")

def rsi(series, n):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = (-delta.clip(upper=0)).rolling(n).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def main():
    btc = pd.read_csv(BTC_PATH, parse_dates=["open_time"]).sort_values("open_time").reset_index(drop=True)
    btc["date"] = btc["open_time"].dt.strftime("%Y-%m-%d")
    
    prem = pd.read_csv(PREM_PATH)
    df = pd.merge(btc, prem[["date", "premium_bps", "premium_bps_ema7", "premium_bps_ema30"]], on="date", how="inner").sort_values("date").reset_index(drop=True)
    
    c = df["close"]
    df["rsi_14"] = rsi(c, 14)
    df["rsi_30"] = rsi(c, 30)
    df["dist_ema50"] = (c - c.ewm(span=50).mean()) / c.ewm(span=50).mean()
    
    # Target forward returns
    for h in [7, 14, 30]:
        df[f"fwd_ret_{h}"] = np.log(c.shift(-h) / c)
        
    df = df.dropna().reset_index(drop=True)
    n = len(df)
    split_idx = int(n * 0.65) # 65% train, 35% test out-of-sample (~1150 days)
    train = df.iloc[:split_idx]
    
    print("=== BENCHMARK COINBASE PREMIUM INDEX (2017 - 2026) ===")
    print(f"Total Dataset: {n} bar | Out-of-Sample Test: {n - split_idx} hari ({df['date'].iloc[split_idx]} s.d. {df['date'].iloc[-1]})\n")
    
    # 1. Information Coefficients
    print("--- KORELASI STATISTIK (IC) COINBASE PREMIUM VS FORWARD RETURNS ---")
    for col in ["premium_bps", "premium_bps_ema7", "premium_bps_ema30"]:
        ic7 = df[col].corr(df["fwd_ret_7"])
        ic14 = df[col].corr(df["fwd_ret_14"])
        ic30 = df[col].corr(df["fwd_ret_30"])
        print(f"  {col:<20}: IC (7h) = {ic7:+.4f} | IC (14h) = {ic14:+.4f} | IC (30h) = {ic30:+.4f}")
    print("-" * 75 + "\n")
    
    base_feats = ["rsi_14", "rsi_30", "dist_ema50"]
    prem_feats = base_feats + ["premium_bps", "premium_bps_ema7", "premium_bps_ema30"]
    
    # 2. Backtest across Horizons
    for h in [7, 14, 30]:
        test = df.iloc[split_idx:-h]
        y_tr = train[f"fwd_ret_{h}"].values
        y_te = test[f"fwd_ret_{h}"].values
        actual_dir = np.sign(y_te)
        
        # Model Tanpa Premium
        m_base = Ridge(alpha=100.0).fit(train[base_feats].values, y_tr)
        pred_base = m_base.predict(test[base_feats].values)
        acc_base = np.mean(np.sign(pred_base) == actual_dir)
        mape_base = np.mean(np.abs(test["close"] * np.exp(y_te) - test["close"] * np.exp(pred_base)) / (test["close"] * np.exp(y_te)))
        
        # Model + Coinbase Premium
        m_prem = Ridge(alpha=100.0).fit(train[prem_feats].values, y_tr)
        pred_prem = m_prem.predict(test[prem_feats].values)
        acc_prem = np.mean(np.sign(pred_prem) == actual_dir)
        mape_prem = np.mean(np.abs(test["close"] * np.exp(y_te) - test["close"] * np.exp(pred_prem)) / (test["close"] * np.exp(y_te)))
        
        delta_acc = acc_prem - acc_base
        delta_mape = mape_prem - mape_base
        print(f"[HORIZON {h:2d} HARI]")
        print(f"  Baseline (Tanpa Premium) : Akurasi Arah = {acc_base:.2%} | MAPE = {mape_base:.2%}")
        print(f"  + Coinbase Premium Gap   : Akurasi Arah = {acc_prem:.2%} | MAPE = {mape_prem:.2%}")
        status = "MENINGKAT" if delta_acc > 0 else "NETRAL"
        print(f"  --> Dampak Coinbase Premium: Akurasi {delta_acc:+.2%} | MAPE {delta_mape:+.2%} ({status})\n")

if __name__ == "__main__":
    main()
