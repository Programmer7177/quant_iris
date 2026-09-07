"""
FASE 5: ON-CHAIN ANALYTICS & BITCOIN VALUATION SIGNALS
1. MVRV Z-Score Framework (Market Top / Bottom Detection)
2. NVT Ratio (Network Value to Transactions - Fundamental PE)
3. Puell Multiple (Miner Revenue Stress & Halving Shock)
4. Structural Halving Clock Analysis (Molnar 2026 benchmark)
"""

import os
import numpy as np
import pandas as pd

DATA_DIR = r"C:\Users\mirza\quant-crypto\data"

def compute_onchain_proxies():
    """
    Constructs on-chain valuation models using daily spot price & volume proxies:
    - MVRV proxy: Price / 200-day Volume Weighted Price (Realized Price proxy)
    - NVT proxy: Market Cap / (Volume * Price)
    - Puell Multiple proxy: Daily Miner Revenue (Issuance) / 365-day MA
    """
    df = pd.read_csv(os.path.join(DATA_DIR, "btc_1d_spot.csv"), index_col="open_time", parse_dates=True)
    
    # 1. Realized Price Proxy (VWAP over 200-day rolling window)
    df["vwap_200"] = (df["close"] * df["volume"]).rolling(200).sum() / df["volume"].rolling(200).sum()
    df["mvrv_proxy"] = df["close"] / df["vwap_200"]
    df["mvrv_zscore"] = (df["mvrv_proxy"] - df["mvrv_proxy"].rolling(365).mean()) / df["mvrv_proxy"].rolling(365).std()
    
    # 2. NVT (Network Value to Transactions) Proxy
    # Supply estimated around 19.75M BTC (2024-2026)
    circulating_supply = 19.75e6
    df["mcap"] = df["close"] * circulating_supply
    # Daily transfer volume proxy = 3x spot volume (typical on-chain multiplier)
    df["daily_tx_vol_usd"] = df["quote_asset_volume"] * 3.0
    df["nvt_proxy"] = df["mcap"] / df["daily_tx_vol_usd"].rolling(14).mean()
    
    # 3. Puell Multiple Proxy
    # Halving 4 (April 2024): Block reward = 3.125 BTC per block (144 blocks/day = 450 BTC/day)
    daily_issuance_btc = 450.0
    df["daily_miner_rev_usd"] = daily_issuance_btc * df["close"]
    df["puell_multiple"] = df["daily_miner_rev_usd"] / df["daily_miner_rev_usd"].rolling(365).mean()
    
    clean_df = df.dropna()
    latest = clean_df.iloc[-1]
    
    return {
        "current_price": latest["close"],
        "realized_price_proxy": latest["vwap_200"],
        "mvrv_proxy": latest["mvrv_proxy"],
        "mvrv_zscore": latest["mvrv_zscore"],
        "nvt_proxy": latest["nvt_proxy"],
        "puell_multiple": latest["puell_multiple"],
        "mvrv_history": clean_df["mvrv_proxy"],
        "puell_history": clean_df["puell_multiple"]
    }

def halving_clock_analysis():
    """
    Molnar (2026, arXiv:2607.26188):
    'Bitcoin Runs on a Clock: Why Every Price Indicator Dies and the Halving Clock Doesn't'
    Tracks epoch progression: Epoch 4 began April 19, 2024 (~Block 840,000).
    Expected next halving: ~April 2028.
    """
    halving_dates = [
        pd.Timestamp("2012-11-28"),
        pd.Timestamp("2016-07-09"),
        pd.Timestamp("2020-05-11"),
        pd.Timestamp("2024-04-19"),
        pd.Timestamp("2028-04-15") # Estimated
    ]
    current_date = pd.Timestamp("2026-09-06")
    
    last_halving = halving_dates[3]
    next_halving = halving_dates[4]
    
    days_since_halving = (current_date - last_halving).days
    total_epoch_days = (next_halving - last_halving).days
    epoch_progress = days_since_halving / total_epoch_days
    
    return {
        "days_since_halving": days_since_halving,
        "days_to_next": (next_halving - current_date).days,
        "epoch_progress_pct": epoch_progress * 100.0,
        "cycle_phase": "Post-Halving Expansion / Parabolic Phase Window" if epoch_progress < 0.65 else "Late Epoch Consolidation"
    }

if __name__ == "__main__":
    print("=" * 60)
    print("FASE 5: ON-CHAIN VALUATION & HALVING CLOCK DYNAMICS")
    print("=" * 60)

    onchain = compute_onchain_proxies()
    print(f"\n[1] On-Chain Valuation Metrics (BTC = ${onchain['current_price']:,.2f}):")
    print(f"  Realized Price Proxy:   ${onchain['realized_price_proxy']:,.2f}")
    print(f"  MVRV Ratio:             {onchain['mvrv_proxy']:.2f}")
    print(f"  MVRV Z-Score:           {onchain['mvrv_zscore']:+.2f}")
    print(f"  NVT Ratio (PE Proxy):   {onchain['nvt_proxy']:.1f}")
    print(f"  Puell Multiple:         {onchain['puell_multiple']:.2f}")
    
    # Valuation interpretation
    print("\n  -> Interpretation:")
    if onchain["mvrv_proxy"] > 3.0:
        print("     MVRV Alert: Extreme Greed / Distribution Phase (MVRV > 3.0)")
    elif onchain["mvrv_proxy"] < 1.0:
        print("     MVRV Alert: Undervalued Accumulation Zone (MVRV < 1.0)")
    else:
        print(f"     MVRV Status: Healthy Mid-Cycle Expansion ({onchain['mvrv_proxy']:.2f}x basis cost)")

    if onchain["puell_multiple"] < 0.5:
        print("     Puell Alert: Severe Miner Capitulation (Historic Buy Zone)")
    elif onchain["puell_multiple"] > 2.0:
        print("     Puell Alert: Miner Over-Profitability (Selling Overhead)")
    else:
        print(f"     Puell Status: Sustainable Miner Revenue Regime ({onchain['puell_multiple']:.2f})")

    # Halving Clock
    h_clock = halving_clock_analysis()
    print(f"\n[2] Structural Halving Clock (Molnar 2026):")
    print(f"  Days Since 4th Halving: {h_clock['days_since_halving']} days")
    print(f"  Days to 5th Halving:    {h_clock['days_to_next']} days")
    print(f"  Epoch 4 Progress:       {h_clock['epoch_progress_pct']:.1f}%")
    print(f"  Cycle Classification:   {h_clock['cycle_phase']}")
    print("=" * 60)
