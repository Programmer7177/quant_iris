"""
Fetch 10 Years of Daily BTC-USD from Coinbase Exchange API.
Replicates iris-terminal/src/lib/sources/coinbase.ts getDailyHistory().
Keyless, no geo-block, ~10 req/s limit.
"""
import urllib.request, json, os, time
from datetime import datetime, timezone

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

BASE = "https://api.exchange.coinbase.com"
PAGE_DAYS = 300
BATCH = 4
DELAY = 0.08  # 80ms between batches
DAY_S = 86400
YEARS = 10
MIN_CANDLES = 3500

def fetch_page(end_ts):
    """Fetch one page of daily candles (max 300)."""
    start_ts = end_ts - PAGE_DAYS * DAY_S
    url = f"{BASE}/products/BTC-USD/candles?granularity={DAY_S}&start={start_ts}&end={end_ts}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        print(f"  WARN: page failed ({e})")
        return None

def main():
    now = int(time.time())
    pages = (YEARS * 365) // PAGE_DAYS + 1  # ~13 pages
    by_ts = {}
    
    print(f"Fetching {YEARS}Y daily BTC-USD from Coinbase ({pages} pages, batch={BATCH})...")
    
    for base in range(0, pages, BATCH):
        batch_end = base + min(BATCH, pages - base)
        # Fetch batch concurrently-ish (sequential with small delay)
        batch_results = []
        for i in range(base, batch_end):
            end_ts = now - i * PAGE_DAYS * DAY_S
            result = fetch_page(end_ts)
            if result is not None:
                batch_results.append(result)
        
        saw_data = False
        for rows in batch_results:
            if not isinstance(rows, list) or len(rows) == 0:
                continue
            saw_data = True
            for r in rows:
                # Coinbase format: [time, low, high, open, close, volume]
                ts, low, high, o, close, vol = r[0], r[1], r[2], r[3], r[4], r[5]
                if not all(isinstance(x, (int, float)) for x in [ts, close]):
                    continue
                if close <= 0:
                    continue
                by_ts[ts] = {
                    "ts": ts, "low": low, "high": high,
                    "open": o, "close": close, "volume": vol
                }
        
        pct = min(100, int((batch_end / pages) * 100))
        print(f"  [{pct:3d}%] page {batch_end}/{pages} — {len(by_ts)} candles so far")
        
        if not saw_data:
            print("  Empty batch — reached earliest listing date")
            break
        
        if batch_end < pages:
            time.sleep(DELAY)
    
    # Sort oldest-first
    candles = sorted(by_ts.values(), key=lambda c: c["ts"])
    
    if len(candles) < MIN_CANDLES:
        print(f"ERROR: only {len(candles)} candles, need {MIN_CANDLES}")
        return
    
    # Save CSV
    csv_path = os.path.join(DATA_DIR, "btc_coinbase_10y.csv")
    with open(csv_path, "w") as f:
        f.write("open_time,open,high,low,close,volume\n")
        for c in candles:
            dt = datetime.fromtimestamp(c["ts"], tz=timezone.utc).strftime("%Y-%m-%d")
            f.write(f"{dt},{c['open']},{c['high']},{c['low']},{c['close']},{c['volume']}\n")
    
    # Summary
    first = candles[0]
    last = candles[-1]
    first_dt = datetime.fromtimestamp(first["ts"], tz=timezone.utc).strftime("%Y-%m-%d")
    last_dt = datetime.fromtimestamp(last["ts"], tz=timezone.utc).strftime("%Y-%m-%d")
    
    print(f"\n{'='*50}")
    print(f"DONE: {len(candles)} daily candles")
    print(f"Range: {first_dt} → {last_dt}")
    print(f"BTC: ${first['close']:,.0f} → ${last['close']:,.0f}")
    print(f"Saved: {csv_path}")

if __name__ == "__main__":
    main()
