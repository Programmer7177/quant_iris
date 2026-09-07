"""
DYNAMIC ADAPTIVE REGIME GRID TRADING (10-YEAR BTC COINBASE DATA)
State Machine:
  - WAIT: Bear breakdown (MACD 1M < Signal & RSI 90 falling). Hold cash.
  - ACCUMULATE: Value / consolidation zone (RSI 90 < 45 or low-vol base). Dynamic grid buys at P10/P25/P50.
  - RIDE / HARVEST: Bull expansion (MACD 1M > Signal & RSI 90 > 55). Hold, scale out into P75/P90.

Evaluates on 2016-2026 daily data with 10bps fee.
"""
import os
import pandas as pd
import numpy as np

DATA_PATH = r"C:\Mirza Personal\crypto quant\data\btc_coinbase_10y.csv"

def rsi(series, n):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = (-delta.clip(upper=0)).rolling(n).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def macd(series, fast=12, slow=26, signal=9):
    ema_f = series.ewm(span=fast, adjust=False).mean()
    ema_s = series.ewm(span=slow, adjust=False).mean()
    line = ema_f - ema_s
    sig = line.ewm(span=signal, adjust=False).mean()
    hist = line - sig
    return line, sig, hist

def max_drawdown(nav_series):
    peak = nav_series.cummax()
    dd = (nav_series - peak) / peak
    return dd.min()

def main():
    df = pd.read_csv(DATA_PATH, parse_dates=["open_time"])
    df = df.sort_values("open_time").reset_index(drop=True)
    c = df["close"]
    
    # HTF metrics
    df["rsi_90"] = rsi(c, 90)
    df["rsi_30"] = rsi(c, 30)
    df["macd_1m"], df["macd_1m_sig"], df["macd_1m_hist"] = macd(c, 12, 26, 9)
    df["vol_30"] = c.pct_change().rolling(30).std() * np.sqrt(365)
    
    # Rolling 90d FHS-like empirical quantiles for adaptive grid
    # Lookback 180 days to compute empirical quantiles of recent price window
    df["p10"] = c.rolling(180).quantile(0.10)
    df["p25"] = c.rolling(180).quantile(0.25)
    df["p50"] = c.rolling(180).quantile(0.50)
    df["p75"] = c.rolling(180).quantile(0.75)
    df["p90"] = c.rolling(180).quantile(0.90)
    
    df = df.dropna().reset_index(drop=True)
    n = len(df)
    
    # Simulation
    initial_capital = 10000.0
    cash = initial_capital
    btc = 0.0
    fee_rate = 0.0010  # 10 bps
    
    nav_history = []
    state_history = []
    trades = 0
    grid_fills = {"L1": 0, "L2": 0, "L3": 0, "TP": 0}
    
    for i in range(n):
        row = df.iloc[i]
        price = row["close"]
        rsi90 = row["rsi_90"]
        macd_bull = row["macd_1m"] > row["macd_1m_sig"]
        macd_hist = row["macd_1m_hist"]
        
        # State Classification
        # 1. WAIT: deep downtrend / bear breakdown
        if (not macd_bull) and (rsi90 < 42) and (macd_hist < 0):
            state = "WAIT"
        # 2. RIDE / HARVEST: strong bull expansion
        elif macd_bull and (rsi90 >= 55):
            state = "RIDE"
        # 3. ACCUMULATE: value zone / base building
        else:
            state = "ACCUMULATE"
            
        state_history.append(state)
        
        # Execution logic per state
        if state == "WAIT":
            # If holding, preserve capital or do not buy new
            pass
            
        elif state == "ACCUMULATE":
            # Adaptive grid allocation based on rolling quantiles
            # Level 1: price <= p50 (allocate 20% remaining cash)
            if price <= row["p50"] and price > row["p25"] and cash > 200:
                buy_val = cash * 0.20
                qty = (buy_val * (1 - fee_rate)) / price
                btc += qty
                cash -= buy_val
                trades += 1
                grid_fills["L1"] += 1
            # Level 2: price <= p25 (allocate 35% remaining cash)
            elif price <= row["p25"] and price > row["p10"] and cash > 200:
                buy_val = cash * 0.35
                qty = (buy_val * (1 - fee_rate)) / price
                btc += qty
                cash -= buy_val
                trades += 1
                grid_fills["L2"] += 1
            # Level 3: price <= p10 (allocate 50% remaining cash - deep value)
            elif price <= row["p10"] and cash > 200:
                buy_val = cash * 0.50
                qty = (buy_val * (1 - fee_rate)) / price
                btc += qty
                cash -= buy_val
                trades += 1
                grid_fills["L3"] += 1
                
        elif state == "RIDE":
            # Take-Profit Scaling into strength
            # Level TP1: price >= p75 and holding btc
            if price >= row["p75"] and price < row["p90"] and btc > 0.005:
                sell_btc = btc * 0.25
                sell_val = sell_btc * price * (1 - fee_rate)
                btc -= sell_btc
                cash += sell_val
                trades += 1
                grid_fills["TP"] += 1
            # Level TP2: price >= p90 (euphoria / harvest)
            elif price >= row["p90"] and btc > 0.005:
                sell_btc = btc * 0.40
                sell_val = sell_btc * price * (1 - fee_rate)
                btc -= sell_btc
                cash += sell_val
                trades += 1
                grid_fills["TP"] += 1
                
        nav = cash + btc * price
        nav_history.append(nav)
        
    df["nav"] = nav_history
    df["state"] = state_history
    
    # Benchmarks
    first_price = df["close"].iloc[0]
    last_price = df["close"].iloc[-1]
    bh_nav = initial_capital * (df["close"] / first_price)
    
    strat_ret = (df["nav"].iloc[-1] / initial_capital) - 1
    bh_ret = (last_price / first_price) - 1
    
    strat_mdd = max_drawdown(df["nav"])
    bh_mdd = max_drawdown(bh_nav)
    
    # Sharpe
    daily_ret = df["nav"].pct_change().dropna()
    ann_ret = (1 + strat_ret) ** (365 / len(df)) - 1
    ann_vol = daily_ret.std() * np.sqrt(365)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    
    # State distribution
    st_counts = pd.Series(state_history).value_counts(normalize=True)
    
    print("=== HASIL ADAPTIVE REGIME GRID BACKTEST (10 TAHUN) ===")
    print(f"Periode: {df['open_time'].iloc[0].date()} s.d. {df['open_time'].iloc[-1].date()} ({len(df)} bar)")
    print(f"Harga BTC: ${first_price:,.0f} -> ${last_price:,.0f}")
    print("---")
    print(f"Return Adaptive Grid Strategy: {strat_ret:+.2%} (Final: ${df['nav'].iloc[-1]:,.2f})")
    print(f"Return Buy & Hold:            {bh_ret:+.2%} (Final: ${bh_nav.iloc[-1]:,.2f})")
    print("---")
    print(f"Max Drawdown Strategy:        {strat_mdd:.2%}")
    print(f"Max Drawdown Buy & Hold:       {bh_mdd:.2%}")
    print(f"Annualized Return Strategy:   {ann_ret:.2%}")
    print(f"Annualized Volatility:        {ann_vol:.2%}")
    print(f"Sharpe Ratio:                 {sharpe:.2f}")
    print("---")
    print(f"Total Eksekusi Trades:        {trades}")
    print(f"Grid Fills Breakdown:         {grid_fills}")
    print("Distribusi State:")
    for k, v in st_counts.items():
        print(f"  {k:12s}: {v:.1%}")

if __name__ == "__main__":
    main()
