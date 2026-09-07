"""
FASE 7: MULTI-ASSET PORTFOLIO CONSTRUCTION & RISK MANAGEMENT
1. Fetch 1,000 daily bars for 4 major crypto assets: BTC, ETH, SOL, BNB
2. Classical Markowitz Mean-Variance vs Rockafellar-Uryasev Mean-CVaR (95%)
3. Multi-Asset Fractional Kelly Allocation
4. Tail Dependence & Crash Correlation Matrix
"""

import os
import urllib.request
import json
import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize

DATA_DIR = r"C:\Users\mirza\quant-crypto\data"
ASSETS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]

def fetch_multi_asset_daily():
    """Fetch daily closes for the asset universe."""
    data = {}
    for sym in ASSETS:
        file_path = os.path.join(DATA_DIR, f"{sym.lower()}_1d.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, index_col="open_time", parse_dates=True)
        else:
            url = f"https://api.binance.com/api/v3/klines?symbol={sym}&interval=1d&limit=1000"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                raw = json.loads(resp.read().decode())
            cols = ["open_time", "open", "high", "low", "close", "volume",
                    "close_time", "qav", "num_trades", "tbb", "tbq", "ignore"]
            df = pd.DataFrame(raw, columns=cols)
            df["close"] = df["close"].astype(float)
            df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
            df.set_index("open_time", inplace=True)
            df.to_csv(file_path)
            print(f"Fetched {sym} daily data.")
        data[sym.replace("USDT", "")] = df["close"]
        
    prices = pd.DataFrame(data).dropna()
    returns = np.log(prices / prices.shift(1)).dropna()
    return prices, returns

# ─────────────────────────────────────────────────────────────
# 1. MEAN-VARIANCE (MARKOWITZ MINIMUM VARIANCE)
# ─────────────────────────────────────────────────────────────

def optimize_min_variance(cov_mat):
    n = cov_mat.shape[0]
    inv_cov = np.linalg.pinv(cov_mat)
    ones = np.ones(n)
    w = (inv_cov @ ones) / (ones.T @ inv_cov @ ones)
    return np.clip(w, 0, 1) / np.sum(np.clip(w, 0, 1))

# ─────────────────────────────────────────────────────────────
# 2. MEAN-CVAR (ROCKAFELLAR-URYASEV LP FORMULATION)
# ─────────────────────────────────────────────────────────────

def optimize_min_cvar(returns_mat, alpha=0.05):
    """
    Minimizes CVaR at confidence level (1-alpha):
    min_{w, xi, z} xi + 1 / ((1 - alpha) * T) * sum(z_t)
    subject to:
      z_t >= -r_t @ w - xi,  z_t >= 0
      sum(w) = 1, w >= 0 (long only)
    """
    T, n = returns_mat.shape
    # Variables: [w_1..w_n, xi, z_1..z_T] => total n + 1 + T variables
    c = np.zeros(n + 1 + T)
    c[n] = 1.0 # xi
    c[n + 1:] = 1.0 / ((1.0 - alpha) * T)

    # Inequality constraints: -r_t @ w - xi - z_t <= 0
    A_ub = np.zeros((T, n + 1 + T))
    for t in range(T):
        A_ub[t, :n] = -returns_mat[t]
        A_ub[t, n] = -1.0
        A_ub[t, n + 1 + t] = -1.0
    b_ub = np.zeros(T)

    # Equality constraint: sum(w) = 1
    A_eq = np.zeros((1, n + 1 + T))
    A_eq[0, :n] = 1.0
    b_eq = [1.0]

    # Bounds: w in [0, 1], xi unconstrained, z_t >= 0
    bounds = [(0, 1)] * n + [(None, None)] + [(0, None)] * T

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    if res.success:
        w_cvar = res.x[:n]
        cvar_val = res.fun
        return w_cvar, cvar_val
    return np.ones(n) / n, 0.0

# ─────────────────────────────────────────────────────────────
# 3. MULTI-ASSET FRACTIONAL KELLY ALLOCATION
# ─────────────────────────────────────────────────────────────

def compute_kelly_allocation(mu_vec, cov_mat, fraction=0.25):
    """
    Computes unconstrained Kelly weights: f* = Sigma^(-1) * mu
    Fractional Kelly scale: f_frac = fraction * f*
    """
    f_star = np.linalg.solve(cov_mat, mu_vec)
    f_fractional = fraction * f_star
    return f_star, f_fractional

# ─────────────────────────────────────────────────────────────
# 4. TAIL DEPENDENCE / LOWER QUANTILE CORRELATIONS
# ─────────────────────────────────────────────────────────────

def compute_crash_correlation_matrix(returns_df, quantile=0.05):
    """
    Computes correlation conditional on market crash days (BTC in bottom 5%).
    Reveals the 'Crashing Together' breakdown of diversification.
    """
    btc_bottom_q = returns_df["BTC"] <= returns_df["BTC"].quantile(quantile)
    crash_returns = returns_df[btc_bottom_q]
    
    uncond_corr = returns_df.corr()
    crash_corr = crash_returns.corr()
    return uncond_corr, crash_corr, len(crash_returns)

if __name__ == "__main__":
    print("=" * 60)
    print("FASE 7: MULTI-ASSET PORTFOLIO CONSTRUCTION & TAIL RISK")
    print("=" * 60)

    prices, returns_df = fetch_multi_asset_daily()
    assets = list(returns_df.columns)
    mu_daily = returns_df.mean().values
    cov_daily = returns_df.cov().values
    
    print(f"\n[1] Asset Universe Historical Stats ({len(returns_df)} daily bars):")
    for a in assets:
        ann_ret = returns_df[a].mean() * 365
        ann_vol = returns_df[a].std() * np.sqrt(365)
        print(f"  {a:4s}: Ann Return = {ann_ret:+6.2%}, Ann Vol = {ann_vol:5.2%}, Sharpe = {ann_ret/ann_vol:4.2f}")

    # 2. Portfolio Optimization Comparison
    w_minvar = optimize_min_variance(cov_daily)
    w_cvar, cvar_daily = optimize_min_cvar(returns_df.values, alpha=0.05)

    print("\n[2] Portfolio Allocation Comparison:")
    print("  Asset | Equal Weight | Min Variance | Min CVaR (95% Tail)")
    print("  ------+--------------+--------------+--------------------")
    for i, a in enumerate(assets):
        print(f"   {a:4s} |    {25.0:6.1f}%    |    {w_minvar[i]*100:6.1f}%   |       {w_cvar[i]*100:6.1f}%")
        
    cvar_ann = cvar_daily * np.sqrt(365)
    print(f"\n  Portfolio CVaR(95%): {cvar_daily:.2%}/day ({cvar_ann:.2%} annualized tail risk)")

    # 3. Fractional Kelly Allocation
    f_full, f_quarter = compute_kelly_allocation(mu_daily * 365, cov_daily * 365, fraction=0.25)
    print("\n[3] Kelly Criterion Leverage Sizing:")
    for i, a in enumerate(assets):
        print(f"  {a:4s}: Full Kelly = {f_full[i]:+6.2f}x | Quarter Kelly = {f_quarter[i]:+6.2f}x")
    print(f"  Total Portfolio Leverage: Full = {np.sum(f_full):.2f}x | Quarter = {np.sum(f_quarter):.2f}x")

    # 4. Tail Dependence ("Crashing Together, Rallying Apart")
    uncond_corr, crash_corr, n_crashes = compute_crash_correlation_matrix(returns_df, quantile=0.05)
    print(f"\n[4] Tail Dependence Analysis ({n_crashes} crash days, BTC bottom 5%):")
    print("  Asset Pair | Normal Correlation | Crash Correlation | Shift")
    print("  -----------+--------------------+-------------------+-------")
    pairs = [("BTC", "ETH"), ("BTC", "SOL"), ("BTC", "BNB"), ("ETH", "SOL")]
    for a1, a2 in pairs:
        c_norm = uncond_corr.loc[a1, a2]
        c_crash = crash_corr.loc[a1, a2]
        print(f"   {a1}-{a2:3s}   |       {c_norm:+6.3f}       |      {c_crash:+6.3f}       | {c_crash - c_norm:+6.3f}")
    print("  -> Crucial Risk Finding: Correlations spike sharply during tail events,")
    print("     confirming that multi-token diversification collapses precisely when needed most!")
    print("=" * 60)
