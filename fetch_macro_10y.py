"""
FETCH GLOBAL MACRO ASSETS (10-YEAR DAILY)
Tickers:
  - DX-Y.NYB : US Dollar Index (DXY)
  - GC=F     : Gold Futures (Comex)
  - CL=F     : Crude Oil WTI Futures (NYMEX)
  - ^TNX     : US 10-Year Treasury Yield
  - ^GSPC    : S&P 500 Index

Fetches and saves to C:/Mirza Personal/crypto quant/data/global_macro_10y.csv
"""
import urllib.request, json, time, os
import pandas as pd
import numpy as np
from datetime import datetime, timezone

DATA_DIR = r"C:\Mirza Personal\crypto quant\data"
os.makedirs(DATA_DIR, exist_ok=True)

TICKERS = {
    "dxy": "DX-Y.NYB",
    "gold": "GC=F",
    "oil": "CL=F",
    "us10y": "^TNX",
    "sp500": "^GSPC"
}

def fetch_ticker(symbol):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range=10y&interval=1d"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
            res = data["chart"]["result"][0]
            ts = res["timestamp"]
            quotes = res["indicators"]["quote"][0]
            closes = quotes["close"]
            
            df = pd.DataFrame({
                "timestamp": ts,
                "close": closes
            })
            df["date"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.strftime("%Y-%m-%d")
            df = df.dropna(subset=["close"])
            df = df.drop_duplicates(subset=["date"]).set_index("date")["close"]
            return df
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None

def main():
    print("Fetching Global Macro data (10 Years)...")
    dfs = {}
    for name, sym in TICKERS.items():
        s = fetch_ticker(sym)
        if s is not None:
            dfs[name] = s
            print(f"  {name:<6} ({sym:<10}): {len(s)} rows ({s.index[0]} -> {s.index[-1]})")
        time.sleep(0.5)
        
    macro_df = pd.DataFrame(dfs)
    # Forward fill weekend gaps (since crypto trades 24/7 and macro markets close weekends)
    macro_df.index = pd.to_datetime(macro_df.index)
    macro_df = macro_df.sort_index()
    
    out_path = os.path.join(DATA_DIR, "global_macro_10y.csv")
    macro_df.to_csv(out_path)
    print(f"\nMacro dataset saved to: {out_path}")
    print(macro_df.tail())

if __name__ == "__main__":
    main()
