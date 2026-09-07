"""
FETCH DEEP MACRO LIQUIDITY & DEBT INDICATORS (FRED & STOOQ)
Series:
1. WALCL       : Fed Total Assets (Balance Sheet in Millions USD)
2. WTREGEN     : Treasury General Account (TGA Balance in Millions USD)
3. RRPONTSYD   : Overnight Reverse Repo (ON RRP in Billions USD)
   --> Fed Net Liquidity = WALCL - WTREGEN - (RRPONTSYD * 1000)
4. T10YIE      : 10-Year Breakeven Inflation Rate (Market Expected Inflation)
5. GFDEGDQ188S : US Federal Debt to GDP Ratio (Quarterly)
6. CNY=X       : USD/CNY (Chinese Yuan Exchange Rate - PBOC Liquidity proxy)
"""
import io, urllib.request, os, time, json
import pandas as pd
import numpy as np

DATA_DIR = r"C:\Mirza Personal\crypto quant\data"
os.makedirs(DATA_DIR, exist_ok=True)

FRED_SERIES = {
    "fed_assets": "WALCL",       # Fed Balance Sheet
    "tga_balance": "WTREGEN",     # Treasury General Account
    "on_rrp": "RRPONTSYD",       # Reverse Repo
    "breakeven_infl": "T10YIE",  # 10Y Breakeven Inflation
    "debt_to_gdp": "GFDEGDQ188S" # US Debt to GDP
}

def fetch_fred(series_id):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            content = resp.read().decode("utf-8")
        df = pd.read_csv(io.StringIO(content))
        df.columns = ["date", series_id.lower()]
        df["date"] = pd.to_datetime(df["date"])
        df[series_id.lower()] = pd.to_numeric(df[series_id.lower()], errors="coerce")
        df = df.dropna().set_index("date")
        return df[series_id.lower()]
    except Exception as e:
        print(f"Error fetching FRED {series_id}: {e}")
        return None

def fetch_cny():
    url = "https://query1.finance.yahoo.com/v8/finance/chart/CNY=X?range=10y&interval=1d"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
            res = data["chart"]["result"][0]
            ts = res["timestamp"]
            closes = res["indicators"]["quote"][0]["close"]
            df = pd.DataFrame({"timestamp": ts, "usdcny": closes})
            df["date"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.strftime("%Y-%m-%d")
            df["date"] = pd.to_datetime(df["date"])
            df = df.dropna(subset=["usdcny"]).drop_duplicates(subset=["date"]).set_index("date")["usdcny"]
            return df
    except Exception as e:
        print(f"Error fetching CNY: {e}")
        return None

def main():
    print("Fetching FRED Deep Macro & China Data...")
    dfs = {}
    for name, s_id in FRED_SERIES.items():
        s = fetch_fred(s_id)
        if s is not None:
            dfs[name] = s
            print(f"  {name:<16} ({s_id}): {len(s)} rows ({s.index[0].date()} -> {s.index[-1].date()})")
        time.sleep(0.3)
        
    cny_s = fetch_cny()
    if cny_s is not None:
        dfs["usdcny"] = cny_s
        print(f"  {'usdcny':<16} (CNY=X): {len(cny_s)} rows")
        
    macro_deep = pd.DataFrame(dfs)
    macro_deep = macro_deep.sort_index()
    
    # Filter 2015 onwards to match BTC 10y
    macro_deep = macro_deep[macro_deep.index >= pd.Timestamp("2015-01-01")]
    
    # Daily calendar reindex & forward fill
    idx = pd.date_range(start=macro_deep.index[0], end=macro_deep.index[-1], freq="D")
    macro_daily = macro_deep.reindex(idx).ffill().bfill()
    macro_daily.index.name = "date"
    
    # Calculate Fed Net Liquidity in Billions USD:
    # Fed Assets (WALCL in Millions) / 1000 - TGA (WTREGEN in Millions) / 1000 - RRP (RRPONTSYD in Billions)
    walcl_b = macro_daily["fed_assets"] / 1000.0
    tga_b = macro_daily["tga_balance"] / 1000.0
    rrp_b = macro_daily["on_rrp"]
    macro_daily["fed_net_liquidity_b"] = walcl_b - tga_b - rrp_b
    
    out_path = os.path.join(DATA_DIR, "deep_macro_liquidity_10y.csv")
    macro_daily.to_csv(out_path)
    print(f"\nDeep Macro dataset saved: {out_path}")
    print(macro_daily.tail(5))

if __name__ == "__main__":
    main()
