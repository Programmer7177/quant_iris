"""
FETCH ADVANCED MACRO RISK, INFLATION, CHINA, AND CREDIT LIQUIDITY
Tickers:
  1. CNY=X  : USD/CNY Exchange Rate (Chinese PBOC Liquidity & Yuan Devaluation)
  2. FXI    : China Large-Cap ETF (Direct Chinese Economic & Stimulus Proxy)
  3. TIP    : US Treasury Inflation-Protected Securities (Direct Inflation Expectation)
  4. TLT    : 20+ Year US Treasury Bond ETF (Long-Term US Debt Market Stress & Curve Inversion)
  5. HYG    : US High Yield Corporate Bond ETF (Credit Risk Appetite & Financial System Stress)

Saves to: C:/Mirza Personal/crypto quant/data/advanced_macro_10y.csv
"""
import urllib.request, json, time, os
import pandas as pd
import numpy as np

DATA_DIR = r"C:\Mirza Personal\crypto quant\data"
os.makedirs(DATA_DIR, exist_ok=True)

ADV_TICKERS = {
    "usdcny": "CNY=X",
    "china_fxi": "FXI",
    "infl_tip": "TIP",
    "us_debt_tlt": "TLT",
    "credit_hyg": "HYG"
}

def fetch_yfinance_rest(symbol):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range=10y&interval=1d"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
            res = data["chart"]["result"][0]
            ts = res["timestamp"]
            closes = res["indicators"]["quote"][0]["close"]
            df = pd.DataFrame({"timestamp": ts, "close": closes})
            df["date"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.strftime("%Y-%m-%d")
            df = df.dropna(subset=["close"]).drop_duplicates(subset=["date"]).set_index("date")["close"]
            return df
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None

def main():
    print("Fetching Advanced Macro: China, Inflation Expectations, Debt, Credit Risk...")
    dfs = {}
    for name, sym in ADV_TICKERS.items():
        s = fetch_yfinance_rest(sym)
        if s is not None:
            dfs[name] = s
            print(f"  {name:<14} ({sym:<6}): {len(s)} rows ({s.index[0]} -> {s.index[-1]})")
        time.sleep(0.4)
        
    adv_df = pd.DataFrame(dfs)
    adv_df.index = pd.to_datetime(adv_df.index)
    adv_df = adv_df.sort_index().ffill().bfill()
    
    out_path = os.path.join(DATA_DIR, "advanced_macro_10y.csv")
    adv_df.to_csv(out_path)
    print(f"\nSaved to: {out_path}")
    print(adv_df.tail(4))

if __name__ == "__main__":
    main()
