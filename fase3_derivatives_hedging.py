"""
FASE 3: DERIVATIVES PRICING & DYNAMIC OPTION HEDGING
1. Inverse Options Pricer (Deribit Mechanics: settled in BTC)
2. Model Pricing Comparison: BSM vs Merton Jump vs Heston Stoch Vol
3. Dynamic Hedging Simulation under 5 bps Transaction Friction:
   - Standard Discrete Delta Hedging
   - Whalley-Wilmott Asymptotic No-Trade Band (Kumar 2026 benchmark)
"""

import numpy as np
import scipy.stats as stats
import scipy.integrate as integrate

# ─────────────────────────────────────────────────────────────
# 1. INVERSE OPTIONS PRICER (DERIBIT MECHANICAL SPECIFICATION)
# ─────────────────────────────────────────────────────────────

def inverse_call_bsm(S, K, T, r, sigma):
    """
    Prices an Inverse European Call option settled in cryptocurrency (Deribit).
    Payoff in crypto at T: max(S_T - K, 0) / S_T = max(1 - K/S_T, 0)
    Value in USD: S_0 * Inverse_Price
    """
    if T <= 0:
        return max(1.0 - K / S, 0.0)
        
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # In crypto units:
    # Price = N(d1) - (K/S) * exp(-r*T) * N(d2)
    crypto_price = stats.norm.cdf(d1) - (K / S) * np.exp(-r * T) * stats.norm.cdf(d2)
    usd_price = crypto_price * S
    return crypto_price, usd_price

# ─────────────────────────────────────────────────────────────
# 2. HESTON STOCHASTIC VOLATILITY PRICER (FOURIER INTEGRATION)
# ─────────────────────────────────────────────────────────────

def heston_char_func(phi, S0, v0, kappa, theta, xi, rho, r, T, j):
    """Heston characteristic functions f_1 and f_2."""
    u = 0.5 if j == 1 else -0.5
    b = kappa - rho * xi if j == 1 else kappa
    
    d = np.sqrt((rho * xi * phi * 1j - b)**2 - xi**2 * (2 * u * phi * 1j - phi**2))
    g = (b - rho * xi * phi * 1j + d) / (b - rho * xi * phi * 1j - d)
    
    C = r * phi * 1j * T + (kappa * theta / xi**2) * (
        (b - rho * xi * phi * 1j + d) * T - 2 * np.log((1 - g * np.exp(d * T)) / (1 - g))
    )
    D = ((b - rho * xi * phi * 1j + d) / xi**2) * ((1 - np.exp(d * T)) / (1 - g * np.exp(d * T)))
    
    return np.exp(C + D * v0 + 1j * phi * np.log(S0))

def heston_price_call(S0, K, T, r, v0, kappa, theta, xi, rho):
    """Heston European Call via numerical integration."""
    def integrand(phi, j):
        cf = heston_char_func(phi, S0, v0, kappa, theta, xi, rho, r, T, j)
        return np.real(np.exp(-1j * phi * np.log(K)) * cf / (1j * phi))

    P1 = 0.5 + (1 / np.pi) * integrate.quad(lambda phi: integrand(phi, 1), 0, 100)[0]
    P2 = 0.5 + (1 / np.pi) * integrate.quad(lambda phi: integrand(phi, 2), 0, 100)[0]
    
    return S0 * P1 - K * np.exp(-r * T) * P2

# ─────────────────────────────────────────────────────────────
# 3. DYNAMIC HEDGING UNDER MARKET FRICTIONS (WHALLEY-WILMOTT)
# ─────────────────────────────────────────────────────────────

def simulate_hedging_friction_experiment(
    S0=80000.0, K=80000.0, T=30/365, r=0.04, sigma=0.52,
    gamma_risk_aversion=1e-5, cost_bps=5.0, n_steps=720, n_paths=500, seed=42
):
    """
    Simulates dynamic delta hedging of a short ATM Call option:
    Strategy A: Periodic Discrete BS Rebalancing (e.g. hourly, 720 steps in 30 days)
    Strategy B: Whalley-Wilmott No-Trade Band (rebalance only when Delta breaches optimal band):
        Band half-width: H_t = ( (3/2) * (c * S_t * Gamma^2) / gamma )^(1/3)
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    c_fee = cost_bps * 1e-4  # 5 bps fee
    
    # Pre-simulate asset price paths
    Z = rng.standard_normal((n_paths, n_steps))
    S = np.zeros((n_paths, n_steps + 1))
    S[:, 0] = S0
    for t in range(n_steps):
        S[:, t+1] = S[:, t] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, t])
        
    initial_call_price = stats.norm.cdf(
        (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    )*S0 - K*np.exp(-r*T)*stats.norm.cdf(
        (np.log(S0/K) + (r - 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    )
    
    cost_bs_all = []
    cost_ww_all = []
    rebal_bs_all = []
    rebal_ww_all = []
    
    for p in range(n_paths):
        # Path p
        pos_bs = 0.0
        fee_bs = 0.0
        rebal_bs = 0
        
        pos_ww = 0.0
        fee_ww = 0.0
        rebal_ww = 0
        
        for t in range(n_steps):
            St = S[p, t]
            tau = max(T - t * dt, 1e-5)
            
            d1 = (np.log(St/K) + (r + 0.5*sigma**2)*tau) / (sigma*np.sqrt(tau))
            target_delta = stats.norm.cdf(d1)
            gamma_bs = stats.norm.pdf(d1) / (St * sigma * np.sqrt(tau))
            
            # --- Strategy A: Periodic BS Rebalance
            trade_bs = target_delta - pos_bs
            fee_bs += np.abs(trade_bs) * St * c_fee
            pos_bs = target_delta
            rebal_bs += 1
            
            # --- Strategy B: Whalley-Wilmott No-Trade Band
            # Half-width h = ( (3/2) * (c * S * Gamma^2) / gamma )^(1/3)
            h = ((1.5 * c_fee * St * (gamma_bs**2)) / gamma_risk_aversion)**(1/3)
            h = min(max(h, 0.01), 0.25) # Cap band for numerical stability
            
            if np.abs(pos_ww - target_delta) > h:
                # Rebalance to the boundary of the band
                new_pos = target_delta - np.sign(pos_ww - target_delta) * h
                trade_ww = new_pos - pos_ww
                fee_ww += np.abs(trade_ww) * St * c_fee
                pos_ww = new_pos
                rebal_ww += 1
                
        cost_bs_all.append(fee_bs)
        cost_ww_all.append(fee_ww)
        rebal_bs_all.append(rebal_bs)
        rebal_ww_all.append(rebal_ww)
        
    return {
        "mean_fee_bs": np.mean(cost_bs_all),
        "mean_fee_ww": np.mean(cost_ww_all),
        "savings_usd": np.mean(cost_bs_all) - np.mean(cost_ww_all),
        "avg_rebal_bs": np.mean(rebal_bs_all),
        "avg_rebal_ww": np.mean(rebal_ww_all),
        "trade_reduction_ratio": np.mean(rebal_bs_all) / np.mean(rebal_ww_all)
    }

if __name__ == "__main__":
    print("=" * 60)
    print("FASE 3: DERIVATIVES PRICING & FRICTION HEDGING RESULTS")
    print("=" * 60)

    S0 = 80000.0
    K = 80000.0
    T = 30 / 365
    r = 0.04
    sigma = 0.5261 # Calibrated from Fase 2 GARCH

    # 1. Inverse Option
    c_crypto, c_usd = inverse_call_bsm(S0, K, T, r, sigma)
    print(f"[1] Deribit Inverse ATM Call (S0={S0:,.0f}, K={K:,.0f}, T=30d, Vol={sigma:.1%})")
    print(f"  Price in Crypto (BTC): {c_crypto:.4f} BTC")
    print(f"  Equivalent Value USD:  ${c_usd:,.2f}")

    # 2. Model Comparison: BSM vs Heston
    v0 = sigma**2
    kappa = 2.0
    theta = sigma**2
    xi = 0.8
    rho = -0.10 # Crypto leverage is mild
    
    c_heston = heston_price_call(S0, K, T, r, v0, kappa, theta, xi, rho)
    c_bsm = c_usd
    print(f"\n[2] Pricing Model Comparison:")
    print(f"  BSM Call Price:     ${c_bsm:,.2f}")
    print(f"  Heston Call Price:  ${c_heston:,.2f}")
    print(f"  Model Discrepancy:  ${abs(c_heston - c_bsm):,.2f} ({abs(c_heston - c_bsm)/c_bsm:.2%})")

    # 3. Dynamic Hedging Friction Benchmark
    print(f"\n[3] Dynamic Option Hedging under 5 bps Friction (30 Days, 720 Steps):")
    res_hedge = simulate_hedging_friction_experiment(
        S0=S0, K=K, T=T, r=r, sigma=sigma, cost_bps=5.0, n_steps=720, n_paths=300
    )
    print(f"  BS Hourly Rebalancing Friction Cost:  ${res_hedge['mean_fee_bs']:,.2f} ({res_hedge['avg_rebal_bs']:.0f} trades)")
    print(f"  Whalley-Wilmott Band Friction Cost:    ${res_hedge['mean_fee_ww']:,.2f} ({res_hedge['avg_rebal_ww']:.0f} trades)")
    print(f"  Transaction Cost Saved per Episode:   ${res_hedge['savings_usd']:,.2f}")
    print(f"  Trade Frequency Reduction:            {res_hedge['trade_reduction_ratio']:.1f}x fewer trades")
    print("=" * 60)
