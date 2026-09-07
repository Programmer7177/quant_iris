"""
PULL FULL HISTORICAL FEAR & GREED INDEX (2018 - 2026)
From alternative.me API (Keyless, 3,100+ daily observations)
Saves to: C:/Mirza Personal/crypto quant/data/fear_greed_full.csv
"""
import urllib.request, json, os
import pandas as pd
from datetime import datetime, timezone

DATA_DIR = r"C:\Mirza Personal\crypto quant\data"
os.makedirs(DATA_DIR, exist_ok=True)

def fetch_fear_greed():
    url = "https://api.alternative.me/fng/?limit=0"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=25) as resp:
            data = json.loads(resp.read().decode())["data"]
            
        rows = []
        for d in data:
            ts = int(d["timestamp"])
            dt = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
            val = float(d["value"])
            cls = d["value_classification"]
            rows.append({"date": dt, "fng_value": val, "fng_class": cls})
            
        df = pd.DataFrame(rows).drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
        out_path = os.path.join(DATA_DIR, "fear_greed_full.csv")
        df.to_csv(out_path, index=False)
        print(f"DONE: Fetched {len(df)} rows ({df['date'].iloc[0]} -> {df['date'].iloc[-1]})")
        print(f"Saved: {out_path}")
        print(df.tail(4))
    except Exception as e:
        print(f"Error fetching Fear & Greed: {e}")

if __name__ == "__main__":
    fetch_fear_greed()
