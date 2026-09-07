"""
FASE 1: DATA PIPELINE
Fetch spot & perpetual data from Binance REST API:
- OHLCV (1d, 1h, 15m)
- Funding rate history
- Open interest
- Order book depth (L2 snapshot)
Save raw data to CSV/JSON.
"""

import json
import os
import time
import urllib.request
import pandas as pd
import numpy as np

DATA_DIR = r"C:\Users\mirza\quant-crypto\data"
os.makedirs(DATA_DIR, exist_ok=True)

def fetch_klines(symbol="BTCUSDT", interval="1d", limit=1000, is_futures=False):
    """Fetch historical candlestick data from Binance."""
    base_url = "https://fapi.binance.com" if is_futures else "https://api.binance.com"
    endpoint = "/fapi/v1/klines" if is_futures else "/api/v3/klines"
    url = f"{base_url}{endpoint}?symbol={symbol}&interval={interval}&limit={limit}"
    
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode())
        
    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ]
    df = pd.DataFrame(data, columns=cols)
    for c in ["open", "high", "low", "close", "volume", "quote_asset_volume"]:
        df[c] = df[c].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
    df.set_index("open_time", inplace=True)
    return df

def fetch_funding_history(symbol="BTCUSDT", limit=1000):
    """Fetch historical funding rates for perpetual futures."""
    url = f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={symbol}&limit={limit}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode())
    df = pd.DataFrame(data)
    df["fundingRate"] = df["fundingRate"].astype(float)
    df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms")
    df.set_index("fundingTime", inplace=True)
    return df

def fetch_order_book(symbol="BTCUSDT", limit=100):
    """Fetch L2 order book snapshot."""
    url = f"https://api.binance.com/api/v3/depth?symbol={symbol}&limit={limit}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode())
    bids = pd.DataFrame(data["bids"], columns=["price", "qty"], dtype=float)
    asks = pd.DataFrame(data["asks"], columns=["price", "qty"], dtype=float)
    return bids, asks

def fetch_open_interest(symbol="BTCUSDT"):
    """Fetch current open interest."""
    url = f"https://fapi.binance.com/fapi/v1/openInterest?symbol={symbol}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode())
    return {
        "symbol": data["symbol"],
        "open_interest": float(data["openInterest"]),
        "time": pd.to_datetime(data["time"], unit="ms")
    }

if __name__ == "__main__":
    print("Fetching BTC Daily Spot...")
    btc_1d = fetch_klines("BTCUSDT", "1d", limit=1000)
    btc_1d.to_csv(os.path.join(DATA_DIR, "btc_1d_spot.csv"))
    print(f"Saved {len(btc_1d)} daily bars: {btc_1d.index[0].date()} to {btc_1d.index[-1].date()}")
    
    print("Fetching BTC 15m Spot (last 1000 bars)...")
    btc_15m = fetch_klines("BTCUSDT", "15m", limit=1000)
    btc_15m.to_csv(os.path.join(DATA_DIR, "btc_15m_spot.csv"))
    print(f"Saved {len(btc_15m)} 15m bars: {btc_15m.index[0]} to {btc_15m.index[-1]}")

    print("Fetching BTC Perpetual Funding History...")
    funding = fetch_funding_history("BTCUSDT", limit=1000)
    funding.to_csv(os.path.join(DATA_DIR, "btc_funding_rates.csv"))
    print(f"Saved {len(funding)} funding records: {funding.index[0]} to {funding.index[-1]}")

    print("Fetching L2 Order Book snapshot...")
    bids, asks = fetch_order_book("BTCUSDT", limit=100)
    bids.to_csv(os.path.join(DATA_DIR, "btc_orderbook_bids.csv"), index=False)
    asks.to_csv(os.path.join(DATA_DIR, "btc_orderbook_asks.csv"), index=False)
    
    best_bid = bids.iloc[0]["price"]
    best_ask = asks.iloc[0]["price"]
    spread = best_ask - best_bid
    spread_bps = (spread / best_bid) * 10000
    print(f"L2 Snapshot: Best Bid={best_bid:,.2f}, Best Ask={best_ask:,.2f}, Spread=${spread:.2f} ({spread_bps:.2f} bps)")

    oi = fetch_open_interest("BTCUSDT")
    print(f"Open Interest: {oi['open_interest']:,.2f} BTC (~${oi['open_interest'] * best_bid / 1e9:.2f}B USD)")
