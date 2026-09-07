"""
FETCH BITCOIN SPOT ETF DATA (IBIT, FBTC, ARKB, BITB) & COMPUTE COINBASE PREMIUM GAP
1. Spot ETFs (Yahoo Finance API, from Jan 2024 to present):
   - IBIT (BlackRock iShares Bitcoin Trust)
   - FBTC (Fidelity Wise Origin)
   - Volume & Dollar Inflow Proxies
2. Coinbase Premium Gap:
   - Difference between Coinbase USD price and Binance USDT price (2017 - 2026, 3,300+ bars).
   - Measures US Institutional Buying Aggression vs Global Retail.
"""
import urllib.request, json, time, os
import pandas as pd
import numpy as np

DATA_DIR = r"C:\Mirza Personal\crypto quant\data"

def compute_coinbase_premium():
    print("Computing Coinbase Premium Index (Coinbase BTC/USD vs Binance BTC/USDT)...")
    coinbase = pd.read_csv(os.path.join(DATA_DIR, "btc_coinbase_10y.csv"))
    coinbase["date"] = pd.to_datetime(coinbase["open_time"]).dt.strftime("%Y-%m-%d")
    cb_df = coinbase[["date", "close"]].rename(columns={"close": "coinbase_close"}).drop_duplicates(subset=["date"])
    
    binance = pd.read_csv(os.path.join(DATA_DIR, "btc_daily_full_2017_2026.csv"))
    # binance open_time is already string YYYY-MM-DD
    binance["date"] = pd.to_datetime(binance["open_time"]).dt.strftime("%Y-%m-%d")
    bn_df = binance[["date", "close"]].rename(columns={"close": "binance_close"}).drop_duplicates(subset=["date"])
    
    df = pd.merge(cb_df, bn_df, on="date", how="inner").sort_values("date").reset_index(drop=True)
    
    # Premium Gap in USD and in Basis Points (bps)
    # Premium_bps = (Coinbase / Binance - 1) * 10,000
    df["premium_usd"] = df["coinbase_close"] - df["binance_close"]
    df["premium_bps"] = ((df["coinbase_close"] / df["binance_close"]) - 1.0) * 10000.0
    
    # Rolling 7d and 30d smoothed premium
    df["premium_bps_ema7"] = df["premium_bps"].ewm(span=7).mean()
    df["premium_bps_ema30"] = df["premium_bps"].ewm(span=30).mean()
    
    out_path = os.path.join(DATA_DIR, "coinbase_premium_index.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved Coinbase Premium: {out_path} ({len(df)} rows from {df['date'].iloc[0]} to {df['date'].iloc[-1]})")
    print(df[["date", "coinbase_close", "binance_close", "premium_usd", "premium_bps"]].tail(5))
    return df

if __name__ == "__main__":
    compute_coinbase_premium()
