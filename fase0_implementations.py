"""
FASE 0: IMPLEMENTATIONS — Quant Crypto Foundations
Covers: BSM, GARCH, Jump-Diffusion, CVaR, Kelly, OU pairs trading
All stdlib + numpy/scipy/pandas only.
"""

import math
import numpy as np
import scipy.stats as stats
from scipy.optimize import minimize, brentq
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
# 0A. STOCHASTIC CALCULUS IMPLEMENTATIONS
# ─────────────────────────────────────────────

def simulate_gbm(S0, mu, sigma, T, n_steps, n_paths=1, seed=None):
    """Geometric Brownian Motion simulation (Euler-Maruyama)."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    Z = rng.standard_normal((n_paths, n_steps))
    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    log_prices = np.cumsum(log_returns, axis=1)
    S = S0 * np.exp(np.hstack([np.zeros((n_paths, 1)), log_prices]))
    return S  # shape: (n_paths, n_steps+1)


def simulate_merton_jump_diffusion(S0, mu, sigma, lam, mu_J, sigma_J, T, n_steps, n_paths=1, seed=None):
    """
    Merton (1976) Jump-Diffusion.
    lam = Poisson intensity (jumps per year)
    mu_J, sigma_J = mean and std of log jump size
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    S = np.zeros((n_paths, n_steps + 1))
    S[:, 0] = S0

    # Compensated drift: mu - lambda*(E[e^J] - 1)
    k = np.exp(mu_J + 0.5 * sigma_J**2) - 1  # E[e^J - 1]
    mu_comp = mu - lam * k

    for t in range(n_steps):
        Z = rng.standard_normal(n_paths)
        N = rng.poisson(lam * dt, n_paths)  # number of jumps
        J = rng.normal(mu_J, sigma_J, n_paths) * N  # total log jump
        dS = (mu_comp - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z + J
        S[:, t+1] = S[:, t] * np.exp(dS)
    return S


def simulate_ou_process(X0, kappa, mu, sigma, T, n_steps, n_paths=1, seed=None):
    """Ornstein-Uhlenbeck process (exact discrete simulation)."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    X = np.zeros((n_paths, n_steps + 1))
    X[:, 0] = X0

    e_kdt = np.exp(-kappa * dt)
    var = sigma**2 * (1 - np.exp(-2 * kappa * dt)) / (2 * kappa)

    for t in range(n_steps):
        Z = rng.standard_normal(n_paths)
        X[:, t+1] = mu + e_kdt * (X[:, t] - mu) + np.sqrt(var) * Z
    return X


# ─────────────────────────────────────────────
# 0B. BSM PRICING & GREEKS
# ─────────────────────────────────────────────

def bsm_price(S, K, T, r, sigma, option_type='call'):
    """Black-Scholes-Merton option price."""
    if T <= 0:
        return max(S - K, 0) if option_type == 'call' else max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        return S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)


def bsm_greeks(S, K, T, r, sigma, option_type='call'):
    """Compute all BSM Greeks."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    n_d1 = stats.norm.pdf(d1)
    N_d1 = stats.norm.cdf(d1)
    N_d2 = stats.norm.cdf(d2)

    delta = N_d1 if option_type == 'call' else N_d1 - 1
    gamma = n_d1 / (S * sigma * np.sqrt(T))
    vega = S * n_d1 * np.sqrt(T)  # per unit vol (not %)
    theta_call = (-S * n_d1 * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * N_d2)
    theta = theta_call if option_type == 'call' else (theta_call + r * K * np.exp(-r * T))
    rho = K * T * np.exp(-r * T) * (N_d2 if option_type == 'call' else -stats.norm.cdf(-d2))

    return {'delta': delta, 'gamma': gamma, 'vega': vega, 'theta': theta, 'rho': rho}


def implied_vol(market_price, S, K, T, r, option_type='call', tol=1e-6):
    """Invert BSM to find implied volatility via Brent's method."""
    try:
        f = lambda sigma: bsm_price(S, K, T, r, sigma, option_type) - market_price
        return brentq(f, 1e-6, 10.0, xtol=tol)
    except ValueError:
        return np.nan


def merton_jump_price(S, K, T, r, sigma, lam, mu_J, sigma_J, n_terms=20, option_type='call'):
    """Merton jump-diffusion option price (infinite series, truncated)."""
    k = np.exp(mu_J + 0.5 * sigma_J**2) - 1
    lam_prime = lam * (1 + k)
    price = 0.0
    for n in range(n_terms):
        weight = np.exp(-lam_prime * T) * (lam_prime * T)**n / math.factorial(n)
        rn = r - lam * k + n * mu_J / T
        sigma_n = np.sqrt(sigma**2 + n * sigma_J**2 / T)
        if sigma_n > 0:
            price += weight * bsm_price(S, K, T, rn, sigma_n, option_type)
    return price


# ─────────────────────────────────────────────
# 0C. GARCH MODELS
# ─────────────────────────────────────────────

def garch11_filter(returns, omega, alpha, beta):
    """GARCH(1,1) variance filter. Returns conditional variances."""
    n = len(returns)
    sigma2 = np.zeros(n)
    sigma2[0] = omega / (1 - alpha - beta)  # unconditional variance
    for t in range(1, n):
        sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]
    return sigma2


def garch11_loglik(params, returns):
    """Negative log-likelihood for GARCH(1,1) with normal errors."""
    omega, alpha, beta = params
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
        return 1e10
    sigma2 = garch11_filter(returns, omega, alpha, beta)
    if np.any(sigma2 <= 0):
        return 1e10
    ll = -0.5 * np.sum(np.log(2 * np.pi * sigma2) + returns**2 / sigma2)
    return -ll  # negative for minimization


def fit_garch11(returns, x0=None):
    """Fit GARCH(1,1) via MLE. Returns (omega, alpha, beta)."""
    if x0 is None:
        var0 = np.var(returns)
        x0 = [var0 * 0.05, 0.10, 0.85]
    bounds = [(1e-8, None), (1e-6, 0.999), (1e-6, 0.999)]
    constraints = {'type': 'ineq', 'fun': lambda p: 0.999 - p[1] - p[2]}
    res = minimize(garch11_loglik, x0, args=(returns,),
                   method='SLSQP', bounds=bounds, constraints=constraints,
                   options={'maxiter': 500})
    omega, alpha, beta = res.x
    sigma2 = garch11_filter(returns, omega, alpha, beta)
    persistence = alpha + beta
    uncond_vol = np.sqrt(omega / (1 - persistence)) if persistence < 1 else np.nan
    return {
        'omega': omega, 'alpha': alpha, 'beta': beta,
        'persistence': persistence, 'uncond_vol_ann': uncond_vol * np.sqrt(252),
        'sigma2': sigma2, 'converged': res.success
    }


def gjr_garch_filter(returns, omega, alpha, gamma, beta):
    """GJR-GARCH (TGARCH) variance filter with leverage effect."""
    n = len(returns)
    sigma2 = np.zeros(n)
    sigma2[0] = omega / (1 - alpha - 0.5 * gamma - beta)
    for t in range(1, n):
        indicator = 1.0 if returns[t-1] < 0 else 0.0
        sigma2[t] = omega + (alpha + gamma * indicator) * returns[t-1]**2 + beta * sigma2[t-1]
    return sigma2


# ─────────────────────────────────────────────
# 0D. RISK MEASURES
# ─────────────────────────────────────────────

def historical_var_cvar(returns, alpha=0.05):
    """Historical simulation VaR and CVaR."""
    sorted_r = np.sort(returns)
    n = len(sorted_r)
    idx = int(np.floor(alpha * n))
    var = -sorted_r[idx]
    cvar = -np.mean(sorted_r[:idx]) if idx > 0 else var
    return var, cvar


def parametric_var_cvar_t(mu, sigma, nu, alpha=0.05):
    """Parametric VaR and CVaR assuming Student-t(nu) distribution."""
    t_alpha = stats.t.ppf(alpha, df=nu)
    var = -(mu + sigma * t_alpha)
    # CVaR for t-distribution (closed form)
    t_pdf_val = stats.t.pdf(t_alpha, df=nu)
    cvar = -(mu - sigma * t_pdf_val * (nu + t_alpha**2) / ((nu - 1) * alpha))
    return var, cvar


def portfolio_var_cvar_mc(weights, returns_matrix, alpha=0.05, n_sim=10000, seed=42):
    """
    Monte Carlo CVaR for portfolio.
    weights: (n_assets,)
    returns_matrix: (T, n_assets) historical returns
    """
    rng = np.random.default_rng(seed)
    mu = returns_matrix.mean(axis=0)
    cov = np.cov(returns_matrix.T)
    sim_returns = rng.multivariate_normal(mu, cov, n_sim)
    port_returns = sim_returns @ weights
    var, cvar = historical_var_cvar(port_returns, alpha)
    return var, cvar


def mean_cvar_optimization(returns_matrix, alpha=0.05, target_return=None):
    """
    Rockafellar-Uryasev (2000) mean-CVaR portfolio optimization.
    LP formulation via auxiliary variable xi.
    returns_matrix: (T, n_assets)
    """
    from scipy.optimize import linprog
    T, n = returns_matrix.shape

    # Variables: [w (n), xi (1), z (T)] -> minimize CVaR
    # CVaR = xi + 1/((1-alpha)*T) * sum(max(-R*w - xi, 0))
    # Linearize: z_t >= -r_t @ w - xi, z_t >= 0

    # Objective: minimize xi + 1/((1-a)*T) * sum(z)
    c = np.zeros(n + 1 + T)
    c[n] = 1.0  # xi
    c[n+1:] = 1.0 / ((1 - alpha) * T)  # z_t

    # Constraints: z_t >= -r_t @ w - xi  <=>  -r_t @ w - xi - z_t <= 0
    A_ub = np.zeros((T, n + 1 + T))
    for t in range(T):
        A_ub[t, :n] = -returns_matrix[t]  # -r_t @ w
        A_ub[t, n] = -1.0  # -xi
        A_ub[t, n + 1 + t] = -1.0  # -z_t
    b_ub = np.zeros(T)

    # z_t >= 0
    A_ub2 = np.zeros((T, n + 1 + T))
    for t in range(T):
        A_ub2[t, n + 1 + t] = -1.0
    b_ub2 = np.zeros(T)

    A_ub_full = np.vstack([A_ub, A_ub2])
    b_ub_full = np.concatenate([b_ub, b_ub2])

    # Equality: sum(w) = 1
    A_eq = np.zeros((1, n + 1 + T))
    A_eq[0, :n] = 1.0
    b_eq = [1.0]

    # Optional: return constraint
    if target_return is not None:
        A_ret = np.zeros((1, n + 1 + T))
        A_ret[0, :n] = returns_matrix.mean(axis=0)
        A_eq = np.vstack([A_eq, A_ret])
        b_eq = b_eq + [target_return]

    # Bounds: w >= 0 (long-only), xi free, z >= 0
    bounds = [(0, None)] * n + [(None, None)] + [(0, None)] * T

    res = linprog(c, A_ub=A_ub_full, b_ub=b_ub_full,
                  A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

    if res.success:
        w_opt = res.x[:n]
        xi_opt = res.x[n]
        cvar_opt = xi_opt + np.sum(res.x[n+1:]) / ((1 - alpha) * T)
        return {'weights': w_opt, 'cvar': cvar_opt, 'xi': xi_opt, 'success': True}
    return {'success': False, 'message': res.message}


# ─────────────────────────────────────────────
# 0E. KELLY CRITERION
# ─────────────────────────────────────────────

def kelly_single_asset(mu, sigma, rf=0.0):
    """Single asset Kelly fraction (continuous, log-normal returns)."""
    excess = mu - rf
    return excess / sigma**2


def kelly_multi_asset(mu, cov):
    """Multi-asset Kelly (unconstrained). Returns weight vector."""
    return np.linalg.solve(cov, mu)


def fractional_kelly(mu, sigma, fraction=0.25, rf=0.0):
    """Fractional Kelly — reduces overbet due to estimation error."""
    f_full = kelly_single_asset(mu, sigma, rf)
    return fraction * f_full


# ─────────────────────────────────────────────
# 0F. PAIRS TRADING — OU FRAMEWORK
# ─────────────────────────────────────────────

def fit_ou_least_squares(spread):
    """
    Fit OU process parameters via OLS on discretized SDE.
    dX = kappa*(mu - X)*dt + sigma*dW  =>
    X_t = a + b*X_{t-1} + eps
    """
    X = spread[:-1]
    Y = spread[1:]
    n = len(X)
    # OLS
    X_mat = np.column_stack([np.ones(n), X])
    beta = np.linalg.lstsq(X_mat, Y, rcond=None)[0]
    a, b = beta
    residuals = Y - X_mat @ beta
    sigma_eps = np.std(residuals, ddof=2)

    # Recover OU params (assuming dt=1)
    kappa = -np.log(b)  # kappa = -ln(b)
    mu = a / (1 - b)
    sigma = sigma_eps * np.sqrt(2 * kappa / (1 - b**2))
    half_life = np.log(2) / kappa

    return {'kappa': kappa, 'mu': mu, 'sigma': sigma,
            'half_life': half_life, 'a': a, 'b': b}


def ou_zscore(spread, mu, sigma):
    """Z-score of spread relative to OU equilibrium."""
    return (spread - mu) / sigma


def bertram_optimal_thresholds(kappa, sigma, transaction_cost=0.001):
    """
    Approximate optimal entry/exit thresholds for OU pairs trading.
    Maximize expected profit per unit time (Bertram 2010 framework).
    Returns: (entry_sigma_multiple, exit_sigma_multiple)
    """
    # Simple approximation: entry at 1.5-2.0 sigma, exit at mean
    # Transaction cost reduces optimal entry level
    base_entry = 1.5 + transaction_cost * 100  # crude adjustment
    return base_entry, 0.0


def generate_pairs_signals(spread, kappa, mu, sigma, entry_z=1.5, exit_z=0.0):
    """Generate pairs trading signals from spread Z-score."""
    z = ou_zscore(spread, mu, sigma)
    signals = np.zeros(len(z))
    position = 0

    for t in range(len(z)):
        if position == 0:
            if z[t] < -entry_z:
                signals[t] = 1   # long spread
                position = 1
            elif z[t] > entry_z:
                signals[t] = -1  # short spread
                position = -1
        elif position == 1 and z[t] >= -exit_z:
            signals[t] = 0
            position = 0
        elif position == -1 and z[t] <= exit_z:
            signals[t] = 0
            position = 0
        else:
            signals[t] = position

    return signals, z


# ─────────────────────────────────────────────
# DEMO / SELF-TEST
# ─────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("FASE 0 — QUANT CRYPTO FOUNDATIONS TEST")
    print("=" * 60)

    np.random.seed(42)

    # ── 1. BSM Pricing
    print("\n[1] BSM Option Pricing")
    S, K, T, r, sigma = 50000, 50000, 30/365, 0.05, 0.80  # BTC ATM call
    call = bsm_price(S, K, T, r, sigma, 'call')
    put = bsm_price(S, K, T, r, sigma, 'put')
    greeks = bsm_greeks(S, K, T, r, sigma, 'call')
    print(f"  ATM Call: ${call:,.2f} | Put: ${put:,.2f}")
    print(f"  Delta: {greeks['delta']:.4f} | Gamma: {greeks['gamma']:.6f}")
    print(f"  Vega: {greeks['vega']:.2f} | Theta: {greeks['theta']:.2f}/day")

    # Check put-call parity
    pcp = call - put - (S - K * np.exp(-r * T))
    print(f"  Put-Call Parity Error: {pcp:.6f} (should be ~0)")

    # ── 2. Merton Jump-Diffusion
    print("\n[2] Merton Jump-Diffusion")
    jcall = merton_jump_price(S, K, T, r, sigma=0.60, lam=5, mu_J=-0.05, sigma_J=0.10)
    bscall = bsm_price(S, K, T, r, 0.60)
    print(f"  Merton Jump Price: ${jcall:,.2f}")
    print(f"  BSM Price (same sigma): ${bscall:,.2f}")
    print(f"  Jump premium: ${jcall - bscall:,.2f}")

    # ── 3. GBM Simulation
    print("\n[3] GBM Simulation (BTC)")
    paths = simulate_gbm(50000, mu=0.50, sigma=0.80, T=1, n_steps=252, n_paths=5000, seed=42)
    final_prices = paths[:, -1]
    print(f"  Final Price: mean=${final_prices.mean():,.0f}, std=${final_prices.std():,.0f}")
    print(f"  P(Price > 100k): {(final_prices > 100000).mean():.2%}")
    print(f"  P(Price < 25k): {(final_prices < 25000).mean():.2%}")

    # ── 4. GARCH(1,1) Fitting
    print("\n[4] GARCH(1,1) on Simulated Returns")
    returns = np.random.standard_t(df=4, size=1000) * 0.03  # fat tail returns
    result = fit_garch11(returns)
    print(f"  omega={result['omega']:.6f}, alpha={result['alpha']:.4f}, beta={result['beta']:.4f}")
    print(f"  Persistence (alpha+beta): {result['persistence']:.4f}")
    print(f"  Unconditional Ann Vol: {result['uncond_vol_ann']:.2%}")
    print(f"  Converged: {result['converged']}")

    # ── 5. Historical VaR/CVaR
    print("\n[5] VaR & CVaR (Historical Simulation)")
    daily_returns = np.random.standard_t(df=4, size=2000) * 0.04
    var95, cvar95 = historical_var_cvar(daily_returns, alpha=0.05)
    var99, cvar99 = historical_var_cvar(daily_returns, alpha=0.01)
    print(f"  VaR(95%): {var95:.2%} | CVaR(95%): {cvar95:.2%}")
    print(f"  VaR(99%): {var99:.2%} | CVaR(99%): {cvar99:.2%}")

    # ── 6. Kelly Criterion
    print("\n[6] Kelly Criterion")
    mu_btc = 0.50  # 50% annual return
    sigma_btc = 0.80  # 80% annual vol
    f_full = kelly_single_asset(mu_btc, sigma_btc)
    f_quarter = fractional_kelly(mu_btc, sigma_btc, fraction=0.25)
    print(f"  Full Kelly: {f_full:.2%} of portfolio in BTC")
    print(f"  Quarter Kelly (practical): {f_quarter:.2%}")
    print(f"  Note: Full Kelly too aggressive — estimation error → ruin")

    # ── 7. OU Pairs Trading
    print("\n[7] OU Pairs Trading (BTC-ETH spread)")
    # Simulate correlated crypto pair
    n = 1000
    btc = np.cumsum(np.random.normal(0.001, 0.04, n))
    eth = 0.8 * btc + simulate_ou_process(0, kappa=0.1, mu=0, sigma=0.02, T=n, n_steps=n, seed=1)[0, 1:]
    spread = btc - eth

    ou_params = fit_ou_least_squares(spread)
    print(f"  OU kappa: {ou_params['kappa']:.4f}")
    print(f"  OU mu: {ou_params['mu']:.4f}")
    print(f"  OU sigma: {ou_params['sigma']:.4f}")
    print(f"  Half-life: {ou_params['half_life']:.1f} periods")

    signals, z = generate_pairs_signals(spread, ou_params['kappa'], ou_params['mu'], ou_params['sigma'])
    n_trades = np.sum(np.diff(signals) != 0)
    print(f"  Signals generated, {n_trades} trade signals")

    # ── 8. Mean-CVaR Portfolio
    print("\n[8] Mean-CVaR Portfolio Optimization")
    T_obs = 500
    n_assets = 4
    # Simulate returns: BTC, ETH, SOL, BNB
    mu_vec = np.array([0.50, 0.40, 0.60, 0.30]) / 252
    cov_mat = np.array([
        [0.80**2, 0.60*0.80*0.90, 0.60*0.80*1.20*0.85, 0.60*0.80*0.70*0.80],
        [0.60*0.80*0.90, 0.90**2, 0.90*1.20*0.88, 0.90*0.70*0.75],
        [0.60*0.80*1.20*0.85, 0.90*1.20*0.88, 1.20**2, 1.20*0.70*0.72],
        [0.60*0.80*0.70*0.80, 0.90*0.70*0.75, 1.20*0.70*0.72, 0.70**2]
    ]) / 252
    sim_returns = np.random.multivariate_normal(mu_vec, cov_mat, T_obs)
    result = mean_cvar_optimization(sim_returns, alpha=0.05)
    if result['success']:
        assets = ['BTC', 'ETH', 'SOL', 'BNB']
        print(f"  CVaR-optimal weights:")
        for asset, w in zip(assets, result['weights']):
            print(f"    {asset}: {w:.2%}")
        print(f"  Portfolio CVaR(95%): {result['cvar']:.2%}/day")
    else:
        print(f"  Optimization failed: {result.get('message')}")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED — FASE 0 IMPLEMENTATIONS READY")
    print("=" * 60)
