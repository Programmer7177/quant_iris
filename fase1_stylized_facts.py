"""
FASE 1: STYLIZED FACTS ANALYSIS (BITCOIN EMPIRICAL DYNAMICS)
Evaluates core empirical properties on real Binance BTCUSDT data:
1. Heavy/Fat Tails: Kurtosis, Jarque-Bera test, Hill tail-index estimate
2. Volatility Clustering: Autocorrelation of |r_t| and r_t^2 vs r_t
3. Leverage Effect / Asymmetry: corr(r_t, sigma_{t+1})
4. Microstructure: Order book imbalance, bid-ask spread
5. Perpetual Funding Dynamics: Carry yield, distribution, persistence
"""

import os
import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.tsa.stattools import acf

DATA_DIR = r"C:\Users\mirza\quant-crypto\data"

def analyze_stylized_facts():
    # Load daily spot data
    df_1d = pd.read_csv(os.path.join(DATA_DIR, "btc_1d_spot.csv"), index_col="open_time", parse_dates=True)
    df_15m = pd.read_csv(os.path.join(DATA_DIR, "btc_15m_spot.csv"), index_col="open_time", parse_dates=True)
    funding_df = pd.read_csv(os.path.join(DATA_DIR, "btc_funding_rates.csv"), index_col="fundingTime", parse_dates=True)
    
    # 1. Log Returns
    df_1d["log_ret"] = np.log(df_1d["close"] / df_1d["close"].shift(1))
    df_15m["log_ret"] = np.log(df_15m["close"] / df_15m["close"].shift(1))
    ret_1d = df_1d["log_ret"].dropna()
    ret_15m = df_15m["log_ret"].dropna()

    print("=" * 60)
    print("EMPIRICAL STYLIZED FACTS — REAL BITCOIN DATA")
    print("=" * 60)

    # 1. Return Distributions & Fat Tails
    mean_1d = ret_1d.mean() * 365
    std_1d = ret_1d.std() * np.sqrt(365)
    skew_1d = stats.skew(ret_1d)
    kurt_1d = stats.kurtosis(ret_1d) # excess kurtosis (Normal = 0)
    jb_stat, jb_p = stats.jarque_bera(ret_1d)

    print(f"\n[1] Return Distribution (Daily: {len(ret_1d)} obs)")
    print(f"  Annualized Return: {mean_1d:+.2%}")
    print(f"  Annualized Vol:    {std_1d:.2%}")
    print(f"  Skewness:          {skew_1d:.3f} (Symmetric ≈ 0)")
    print(f"  Excess Kurtosis:   {kurt_1d:.3f} (Normal = 0, Fat-tail > 0)")
    print(f"  Jarque-Bera:       stat={jb_stat:,.1f}, p={jb_p:.4e} (Reject Normal: {jb_p < 0.01})")

    # Hill Estimator for Tail Index alpha (Power law P(R > x) ~ x^(-alpha))
    # Top 5% tail
    sorted_losses = np.sort(np.abs(ret_1d[ret_1d < 0]))[::-1]
    k_tail = int(0.05 * len(sorted_losses))
    hill_alpha = 1.0 / (np.mean(np.log(sorted_losses[:k_tail] / sorted_losses[k_tail])))
    print(f"  Tail Index (alpha, Hill top 5%): {hill_alpha:.2f}")
    if hill_alpha > 2:
        print("  -> Finite variance (alpha > 2), but heavy tails!")
    else:
        print("  -> Infinite variance regime (alpha <= 2)!")

    # 2. Autocorrelation & Volatility Clustering
    print("\n[2] Autocorrelation & Volatility Clustering")
    acf_raw = acf(ret_1d, nlags=10)
    acf_abs = acf(np.abs(ret_1d), nlags=10)
    acf_sq = acf(ret_1d**2, nlags=10)

    print("  Lag | Raw Return ACF | |Return| ACF | Return^2 ACF")
    print("  ----+----------------+--------------+-------------")
    for lag in range(1, 6):
        print(f"   {lag:2d} |    {acf_raw[lag]:+8.4f}    |   {acf_abs[lag]:8.4f}   |   {acf_sq[lag]:8.4f}")
    print("  Notice: Raw return ACF ≈ 0 (efficient market), but |r| & r^2 ACF > 0 (volatility clustering!)")

    # 3. 15-Minute Reversal Check (Kitron & Wengrowicz 2026 paper)
    print("\n[3] Short-Horizon Dynamics (15-min bars, 1000 obs)")
    sign_ret = np.sign(ret_15m)
    sign_reversal = (sign_ret * sign_ret.shift(1) < 0).mean()
    acf_15m_1 = acf(ret_15m, nlags=3)[1]
    print(f"  15m Lag-1 Autocorrelation: {acf_15m_1:+8.4f}")
    print(f"  Directional Reversal Frequency: {sign_reversal:.2%} (Theoretical random = 50.0%)")

    # 4. Asymmetric Volatility / Leverage Effect
    # In equities, corr(r_t, vol_{t+1}) < 0 strongly. In crypto, often symmetric or weak.
    realized_vol_5d = ret_1d.rolling(5).std().shift(-5)
    leverage_corr = ret_1d.corr(realized_vol_5d)
    print(f"\n[4] Leverage Effect (Return vs Future 5d Volatility)")
    print(f"  Correlation(r_t, Vol_{{t+5}}): {leverage_corr:+.4f}")
    if abs(leverage_corr) < 0.15:
        print("  -> Weak/Insignificant leverage effect (typical for crypto vs traditional equities).")

    # 5. Funding Rate Dynamics (Binance BTCUSDT Perp)
    print(f"\n[5] Perpetual Funding Rates ({len(funding_df)} observations)")
    # Funding paid 3x daily (every 8 hours)
    fr_annualized = funding_df["fundingRate"].mean() * 3 * 365
    fr_std_ann = funding_df["fundingRate"].std() * np.sqrt(3 * 365)
    pct_positive = (funding_df["fundingRate"] > 0).mean()
    max_rate = funding_df["fundingRate"].max()
    min_rate = funding_df["fundingRate"].min()
    
    print(f"  Mean 8h Funding Rate: {funding_df['fundingRate'].mean():+.6f}")
    print(f"  Annualized Carry Yield: {fr_annualized:+.2%}")
    print(f"  Annualized Vol of Rate: {fr_std_ann:.2%}")
    print(f"  % Positive (Longs pay Shorts): {pct_positive:.1%}")
    print(f"  Max 8h Rate: {max_rate:+.4f} ({max_rate*3*365:+.1%} ann)")
    print(f"  Min 8h Rate: {min_rate:+.4f} ({min_rate*3*365:+.1%} ann)")

    # 6. Order Book Imbalance (L2 Depth)
    bids = pd.read_csv(os.path.join(DATA_DIR, "btc_orderbook_bids.csv"))
    asks = pd.read_csv(os.path.join(DATA_DIR, "btc_orderbook_asks.csv"))
    bid_vol = bids["qty"].sum()
    ask_vol = asks["qty"].sum()
    obi = (bid_vol - ask_vol) / (bid_vol + ask_vol)
    
    print(f"\n[6] L2 Order Book Microstructure (Top 100 levels)")
    print(f"  Total Bid Liquidity: {bid_vol:,.2f} BTC (~${bid_vol*bids.iloc[0]['price']/1e6:.1f}M)")
    print(f"  Total Ask Liquidity: {ask_vol:,.2f} BTC (~${ask_vol*asks.iloc[0]['price']/1e6:.1f}M)")
    print(f"  Order Book Imbalance (OBI): {obi:+.3f} (Range [-1, 1])")
    print("=" * 60)

if __name__ == "__main__":
    analyze_stylized_facts()
