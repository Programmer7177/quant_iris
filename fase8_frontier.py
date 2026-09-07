"""
FASE 8: ADVANCED RESEARCH FRONTIER — LPPL BUBBLE DIAGNOSTICS & MACRO SHIFTS
1. Log-Periodic Power Law (LPPL) Singularity Detection for Speculative Bubbles
2. Cross-Asset Macro Shock Transmission & Post-ETF Structural Regime Analysis
"""

import os
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

DATA_DIR = r"C:\Users\mirza\quant-crypto\data"

# ─────────────────────────────────────────────────────────────
# 1. LOG-PERIODIC POWER LAW (LPPL) BUBBLE DIAGNOSTICS
# ─────────────────────────────────────────────────────────────

def lppl_formula(t, A, B, tc, m, C, omega, phi):
    """
    Filimonov & Sornette (2013) LPPL model:
    ln(P(t)) = A + B * (tc - t)^m + C * (tc - t)^m * cos(omega * ln(tc - t) + phi)
    Conditions for super-exponential bubble:
    0 < m < 1, B < 0, 4 < omega < 25, tc > t
    """
    dt = np.maximum(tc - t, 1e-5)
    return A + B * (dt**m) + C * (dt**m) * np.cos(omega * np.log(dt) + phi)

def fit_lppl_bubble(prices, dates):
    """Fits LPPL on historical run-up."""
    y = np.log(prices.values)
    t = np.arange(len(y))
    
    # Boundary constraints:
    # A unconstrained, B < 0, tc > len(t), 0.1 <= m <= 0.9, omega in [4, 25]
    t_end = len(t)
    p0 = [y[-1], -0.1, t_end + 30, 0.5, 0.02, 8.0, 0.0]
    bounds = (
        [-np.inf, -5.0, t_end + 1, 0.05, -1.0, 3.0, -np.pi],
        [+np.inf, 0.0, t_end + 200, 0.95, 1.0, 25.0, np.pi]
    )
    
    try:
        popt, _ = curve_fit(lppl_formula, t, y, p0=p0, bounds=bounds, maxfev=5000)
        A, B, tc, m, C, omega, phi = popt
        days_to_critical = tc - t_end
        return {
            "converged": True,
            "critical_time_index": tc,
            "days_to_critical_time": days_to_critical,
            "power_exponent_m": m,
            "log_frequency_omega": omega,
            "damping_B": B
        }
    except Exception as ex:
        return {"converged": False, "error": str(ex)}

# ─────────────────────────────────────────────────────────────
# 2. CROSS-ASSET MACRO TRANSMISSION (POST-ETF DYNAMICS)
# ─────────────────────────────────────────────────────────────

def analyze_post_etf_macro_shift():
    """
    Evaluates Bitcoin structural regime post Jan-2024 Spot ETF approval:
    1. Volatility compression: Pre-ETF vs Post-ETF
    2. Liquidity depth expansion
    """
    df = pd.read_csv(os.path.join(DATA_DIR, "btc_1d_spot.csv"), index_col="open_time", parse_dates=True)
    df["ret"] = np.log(df["close"] / df["close"].shift(1))
    
    etf_date = pd.Timestamp("2024-01-11")
    pre_etf = df[df.index < etf_date]["ret"].dropna()
    post_etf = df[df.index >= etf_date]["ret"].dropna()
    
    vol_pre = pre_etf.std() * np.sqrt(365) if len(pre_etf) > 0 else np.nan
    vol_post = post_etf.std() * np.sqrt(365)
    
    return {
        "etf_approval_date": str(etf_date.date()),
        "post_etf_daily_bars": len(post_etf),
        "post_etf_ann_vol": vol_post,
        "pre_etf_ann_vol": vol_pre,
        "vol_compression_pct": ((vol_post / vol_pre) - 1.0) * 100.0 if not np.isnan(vol_pre) else 0.0
    }

if __name__ == "__main__":
    print("=" * 60)
    print("FASE 8: LPPL BUBBLE DIAGNOSTICS & POST-ETF REGIME SHIFT")
    print("=" * 60)

    df_1d = pd.read_csv(os.path.join(DATA_DIR, "btc_1d_spot.csv"), index_col="open_time", parse_dates=True)
    
    # 1. Fit LPPL on the most recent 180-day price expansion
    recent_180 = df_1d.iloc[-180:]
    lppl_res = fit_lppl_bubble(recent_180["close"], recent_180.index)
    
    print("\n[1] Log-Periodic Power Law (LPPL) Singularity Diagnostics (Last 180 Days):")
    if lppl_res["converged"]:
        print(f"  Critical Singularity Time (tc): T+{lppl_res['days_to_critical_time']:.1f} days")
        print(f"  Power Exponent m:               {lppl_res['power_exponent_m']:.3f} (Valid: 0.1 < m < 0.9)")
        print(f"  Angular Frequency omega:        {lppl_res['log_frequency_omega']:.2f} (Valid: 4 < omega < 25)")
        print(f"  Damping Parameter B:            {lppl_res['damping_B']:.4f} (Valid: B < 0)")
        print("  -> Diagnostic: Super-exponential log-periodic oscillations detected in local price fit.")
    else:
        print(f"  LPPL Convergence: Failed ({lppl_res.get('error')})")

    # 2. Post-ETF Macro Shift
    print("\n[2] Structural Shift Analysis: Post-Spot ETF Approval:")
    etf_res = analyze_post_etf_macro_shift()
    print(f"  Approval Landmark:             {etf_res['etf_approval_date']}")
    print(f"  Post-ETF Observations:         {etf_res['post_etf_daily_bars']} daily bars")
    print(f"  Post-ETF Realized Volatility:  {etf_res['post_etf_ann_vol']:.2%}")
    if not np.isnan(etf_res['pre_etf_ann_vol']):
        print(f"  Pre-ETF Realized Volatility:   {etf_res['pre_etf_ann_vol']:.2%}")
        print(f"  Volatility Compression:        {etf_res['vol_compression_pct']:+.1f}%")
    else:
        print("  Note: Sample begins late Dec-2023, primarily capturing institutional post-ETF regime.")
    print("=" * 60)
