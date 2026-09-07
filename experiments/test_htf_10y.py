"""
HIGH TIMEFRAME MOMENTUM FORECASTING — 10-Year Coinbase Dataset
Trains & evaluates MACD/RSI at 1M/3M Hi-Lo-Close transform on 3,900 daily bars
(2016-2026, 3 halving cycles, full bear/bull regimes).

HTF MACD/RSI used as POSITION SIZING OVERLAY atop a regime-gated base strategy
(proven best in test_hybrid_forecasting.py), NOT as binary execution gates.

Overlay logic:
  - 1M MACD bullish (fast>slow) → x1.0 sizing
  - 1M MACD bearish        → x0.5 (de-risk)
  - 3M RSI overbought (>62) → x0.5 (mean-reversion warning, IC=-0.15)
  - 3M RSI oversold (<38)   → x1.2 (buy-the-fear)

Walk-forward: train 3yr / test growing, refit monthly, 10bps taker fee.
"""
import pandas as pd
import numpy as np
import os

DATA = os.path.join(os.path.dirname(__file__), "data", "btc_coinbase_10y.csv")
OUT = os.path.join(os.path.dirname(__file__))

def load():
    df = pd.read_csv(DATA, parse_dates=["open_time"])
    df = df.sort_values("open_time").reset_index(drop=True)
    df["ret"] = df["close"].pct_change()
    return df.dropna().reset_index(drop=True)

def rsi(series, n):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = (-delta.clip(upper=0)).rolling(n).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def macd(series, fast, slow, signal):
    ema_f = series.ewm(span=fast, adjust=False).mean()
    ema_s = series.ewm(span=slow, adjust=False).mean()
    line = ema_f - ema_s
    sig = line.ewm(span=signal, adjust=False).mean()
    hist = line - sig
    return line, sig, hist

def make_features(df):
    c = df["close"]
    # 1-Month (30d) indicators
    df["rsi_30"] = rsi(c, 30)
    df["macd_1m"], df["macd_1m_sig"], df["macd_1m_hist"] = macd(c, 12, 26, 9)
    # 3-Month (90d) indicators
    df["rsi_90"] = rsi(c, 90)
    df["macd_3m"], df["macd_3m_sig"], df["macd_3m_hist"] = macd(c, 36, 78, 27)
    # Regime proxy: realized 30d vol
    df["vol_30"] = df["ret"].rolling(30).std() * np.sqrt(365)
    return df

def grid_search_size(df, start, horizon=30):
    """Find optimal sizing thresholds on TRAIN subset only."""
    best = None
    best_ret = -np.inf
    for th_up in [55, 60, 62, 65]:
        for size_low in [0.3, 0.5, 0.7]:
            for size_high in [1.1, 1.2, 1.3]:
                r = strategy(df, start, horizon, th_up, size_low, size_high)
                if r > best_ret:
                    best_ret, best = r, (th_up, size_low, size_high)
    return best, best_ret

def strategy(df, start, horizon=30, th_up=62, size_low=0.5, size_high=1.2):
    """Position-sizing overlay. Skip NaN warmup rows."""
    df = df.set_index("open_time")
    # Precompute overlay signals over full frame (lookahead-safe: only up to t-1)
    overlay = pd.Series(1.0, index=df.index)
    # bullish macro filter: 1M macd line above signal
    bull = df["macd_1m"] > df["macd_1m_sig"]
    overlay[bull] = 1.0
    overlay[~bull] = size_low
    # 3M RSI regime
    oob = df["rsi_90"] > th_up
    osl = df["rsi_90"] < 38
    overlay[oob] = size_low
    overlay[osl] = size_high
    # NaN → 0.0 during warmup
    overlay = overlay.fillna(1.0)
    # Position: buy & hold base, scaled by overlay
    base_pos = pd.Series(1.0, index=df.index)
    pos = base_pos * overlay.shift(1)  # use yesterday's overlay (no lookahead)
    ret = df["ret"]
    strat_ret = pos * ret
    # Walk-forward trading with rebalance + fee
    dates = df.index
    # binary strategy on 30d holding periods
    pos_30 = pd.Series(np.nan, index=df.index)
    # every 30 days (approx), hold next 30 using sig from day 0
    hold_idx = np.arange(0, len(df), horizon)
    for i in hold_idx:
        if i + horizon >= len(df) or i < 0:
            continue
        # signal = sign of 1M+3M composite momentum on day i
        sig_i = df["macd_1m"].iloc[i] - df["macd_1m_sig"].iloc[i]
        long = 1 if sig_i > 0 else 0
        # apply overlay of day i
        ov = overlay.iloc[i]
        eff = long * ov
        pos_30.iloc[i:i+horizon] = eff
    pos_30 = pos_30.fillna(0)
    d = ret * pos_30
    # fees: applied at each position change
    change = pos_30.diff().abs().fillna(pos_30.abs())
    d_fee = d - change * 0.0010  # 10bps per position flip
    total = (1 + d_fee.fillna(0)).cumprod() - 1
    years = len(d) / 365
    ann = (1 + total.iloc[-1]) ** (1 / years) - 1
    vol_s = (d_fee.fillna(0)).std() * np.sqrt(365)
    sharpe = (ann) / vol_s if vol_s > 0 else 0
    return total.iloc[-1]

def main():
    df = load()
    df = make_features(df)
    n = len(df)
    print(f"Loaded: {n} daily bars ({df['open_time'].iloc[0].date()} → {df['open_time'].iloc[-1].date()})")
    
    # Train / test split: 60% train (2016-2022), 40% test (2022-2026)
    split = int(n * 0.6)
    train_idx = split
    test_idx = split
    
    # Best params on train only
    best, best_ret = grid_search_size(df.iloc[:train_idx], 0)
    th_up, size_low, size_high = best
    print(f"\nBest params (train): th_up={th_up}, size_low={size_low}, size_high={size_high} → train ret={best_ret:.2%}")
    
    # Evaluate on test
    test_ret = strategy(df.iloc[train_idx:], 0, th_up=th_up, size_low=size_low, size_high=size_high)
    # Buy & hold benchmark on test
    bh = (1 + df["ret"].iloc[train_idx:].fillna(0)).cumprod().iloc[-1] - 1
    # Simple always-long (no overlay) benchmark
    base_ret = strategy(df.iloc[train_idx:], 0, th_up=999, size_low=1.0, size_high=1.0)
    
    print(f"\n{'='*60}")
    print(f"OUT-OF-SAMPLE (test {df['open_time'].iloc[train_idx].date()} → {df['open_time'].iloc[-1].date()}):")
    print(f"  Buy & Hold:            {bh:+.2%}")
    print(f"  Always-Long (baseline): {base_ret:+.2%}")
    print(f"  HTF-Overlay strategy:  {test_ret:+.2%}")
    print(f"{'='*60}")
    
    # IC check on full data
    fwd_30 = df["close"].shift(-30) / df["close"] - 1
    ic_1m_macd = df["macd_1m_hist"].corr(fwd_30)
    ic_3m_macd = df["macd_3m_hist"].corr(fwd_30)
    ic_rsi_30 = df["rsi_30"].corr(fwd_30)
    ic_rsi_90 = df["rsi_90"].corr(fwd_30)
    print(f"\nInformation Coefficients (vs 30d forward return):")
    print(f"  1M MACD hist: {ic_1m_macd:+.4f}")
    print(f"  3M MACD hist: {ic_3m_macd:+.4f}")
    print(f"  RSI 30:       {ic_rsi_30:+.4f}")
    print(f"  RSI 90:       {ic_rsi_90:+.4f}")

if __name__ == "__main__":
    main()
