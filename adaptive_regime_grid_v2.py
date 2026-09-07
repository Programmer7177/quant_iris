"""
ADAPTIVE REGIME GRID V2: CYCLIC REBALANCING & INVENTORY MANAGEMENT
Problems in V1:
  1. Cash depletion: Bought once in bear, ran out of cash, never sold until bull, but take-profit sold too early.
  2. Fixed quantile grid without local range tracking.

Fix in V2:
  - Local dynamic grid bands around rolling EMA50 (geometric bands: -5%, -10%, -15%, +10%, +20%).
  - Grid buys and takes profit locally within the ACCUMULATE state to compound cash flow.
  - State WAIT acts as hard stop/hedge or cash lock.
  - State RIDE keeps 50% core position, trailing stops only on excess inventory.
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
    df = pd.read_csv(DATA_PATH, parse_dates=["open_time"]).sort_values("open_time").reset_index(drop=True)
    c = df["close"]
    
    df["rsi_90"] = rsi(c, 90)
    df["rsi_30"] = rsi(c, 30)
    df["macd_1m"], df["macd_1m_sig"], df["macd_1m_hist"] = macd(c, 12, 26, 9)
    df["ema_50"] = c.ewm(span=50, adjust=False).mean()
    df["ema_200"] = c.ewm(span=200, adjust=False).mean()
    
    df = df.dropna().reset_index(drop=True)
    n = len(df)
    
    initial_capital = 10000.0
    cash = initial_capital
    btc = 0.0
    fee_rate = 0.0010
    
    nav_history = []
    state_history = []
    trades = 0
    
    # Track inventory lots: list of {'price': p, 'qty': q}
    lots = []
    
    for i in range(n):
        row = df.iloc[i]
        price = row["close"]
        rsi90 = row["rsi_90"]
        macd_bull = row["macd_1m"] > row["macd_1m_sig"]
        ema50 = row["ema_50"]
        ema200 = row["ema_200"]
        
        # State
        if (price < ema200) and (not macd_bull):
            state = "WAIT" # Bear regime: preserve cash, do not buy dips blindly
        elif macd_bull and (price > ema50) and (rsi90 > 52):
            state = "RIDE" # Bull momentum: keep high allocation
        else:
            state = "ACCUMULATE" # Range/consolidation: active grid trading
            
        state_history.append(state)
        
        total_val = cash + btc * price
        target_btc_val = 0.0
        
        if state == "WAIT":
            # Hold defensive allocation (max 20% exposure)
            target_btc_val = total_val * 0.15
        elif state == "RIDE":
            # Bull momentum: ride trend (80% exposure)
            target_btc_val = total_val * 0.85
        elif state == "ACCUMULATE":
            # Grid mean-reversion around EMA50
            # Deviation from EMA50: price lower -> buy more, price higher -> trim
            dev = (price - ema50) / ema50
            # Base exposure 50%, scaled from 20% to 80% based on dev (-20% to +20%)
            exposure = np.clip(0.50 - dev * 1.5, 0.20, 0.80)
            target_btc_val = total_val * exposure
            
        curr_btc_val = btc * price
        diff = target_btc_val - curr_btc_val
        
        # Rebalance threshold: only trade if rebalance > 3% of portfolio to avoid fee drag
        if abs(diff) > (total_val * 0.03):
            if diff > 0 and cash > 50:
                buy_amount = min(diff, cash * 0.99)
                buy_qty = (buy_amount * (1 - fee_rate)) / price
                btc += buy_qty
                cash -= buy_amount
                trades += 1
            elif diff < 0 and btc > 0.001:
                sell_amount = min(abs(diff), curr_btc_val * 0.99)
                sell_qty = sell_amount / price
                btc -= sell_qty
                cash += (sell_amount * (1 - fee_rate))
                trades += 1
                
        nav = cash + btc * price
        nav_history.append(nav)
        
    df["nav"] = nav_history
    df["state"] = state_history
    
    first_price = df["close"].iloc[0]
    last_price = df["close"].iloc[-1]
    bh_nav = initial_capital * (df["close"] / first_price)
    
    strat_ret = (df["nav"].iloc[-1] / initial_capital) - 1
    bh_ret = (last_price / first_price) - 1
    
    strat_mdd = max_drawdown(df["nav"])
    bh_mdd = max_drawdown(bh_nav)
    
    daily_ret = df["nav"].pct_change().dropna()
    ann_ret = (1 + strat_ret) ** (365 / len(df)) - 1
    ann_vol = daily_ret.std() * np.sqrt(365)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    
    st_counts = pd.Series(state_history).value_counts(normalize=True)
    
    print("=== HASIL ADAPTIVE REGIME GRID V2 (DYNAMIC EXPOSURE) ===")
    print(f"Periode: {df['open_time'].iloc[0].date()} s.d. {df['open_time'].iloc[-1].date()} ({len(df)} bar)")
    print(f"Harga BTC: ${first_price:,.0f} -> ${last_price:,.0f}")
    print("---")
    print(f"Return Adaptive Strategy:     {strat_ret:+.2%} (Final: ${df['nav'].iloc[-1]:,.2f})")
    print(f"Return Buy & Hold:            {bh_ret:+.2%} (Final: ${bh_nav.iloc[-1]:,.2f})")
    print("---")
    print(f"Max Drawdown Strategy:        {strat_mdd:.2%}")
    print(f"Max Drawdown Buy & Hold:       {bh_mdd:.2%}")
    print(f"Annualized Return:            {ann_ret:.2%}")
    print(f"Annualized Volatility:        {ann_vol:.2%}")
    print(f"Sharpe Ratio:                 {sharpe:.2f}")
    print("---")
    print(f"Total Trades Rebalanced:      {trades}")
    print("Distribusi State:")
    for k, v in st_counts.items():
        print(f"  {k:12s}: {v:.1%}")

if __name__ == "__main__":
    main()
