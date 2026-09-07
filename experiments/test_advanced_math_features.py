"""
Benchmark Advanced Mathematical Features on 10Y Coinbase BTC Data:
1. Fractional Differentiation (Lopez de Prado, expanding window weights)
   - Test d in [0.2, 0.6], step 0.05
   - Evaluate ADF p-value, ADF stat, correlation with log-price
   - Select optimal minimum d retaining maximum memory while stationary
2. Realized Jump Variation Decomposition (Barndorff-Nielsen & Shephard):
   - Daily proxy via Parkinson RV & Bipower variation proxy (or return-based rolling BV vs RV)
   - Continuous variance vs Jump variance: J_t = max(0, RV_t - BV_t), Jump ratio = J_t / RV_t
3. Yang-Zhang Volatility Estimator:
   - Overnight jump variance + Rogers-Satchell open-to-close drift-independent variance
   - Optimal k weight formulation
4. High-Frequency Microstructure Proxies on Daily Bars:
   - Garman-Klass Volatility
   - Rogers-Satchell Volatility
   - Parkinson Ratio (Parkinson Vol / Close-to-Close Vol)
   - Amihud Illiquidity Ratio & Rolling Kyle's Lambda proxy
   - Daily VPIN proxy (Volume-Synchronized Probability of Toxicity: |V_buy - V_sell| / V_total via tick rule proxy)
   - Signed Volume Flow & Roll Spread proxy
5. Statistical Benchmark & Feature Selection:
   - Forward returns at 7d, 14d, 30d
   - Spearman Rank Information Coefficient (IC) and p-values
   - Random Forest Feature Importance
"""

import sys
import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.tsa.stattools import adfuller
from sklearn.ensemble import RandomForestRegressor

DATA_PATH = r"C:\Mirza Personal\crypto quant\data\btc_coinbase_10y.csv"

def get_fracdiff_weights(d, size):
    """
    Generate weights for fractional differentiation using binomial series expansion.
    w_0 = 1, w_k = -w_{k-1} * (d - k + 1) / k
    """
    w = [1.0]
    for k in range(1, size):
        w.append(-w[-1] / k * (d - k + 1))
    return np.array(w[::-1])

def frac_diff_ffd(series, d, thres=1e-4):
    """
    Fixed-width window fractional differentiation (Marcos Lopez de Prado).
    """
    # 1. Compute weights until threshold
    w = [1.0]
    k = 1
    while True:
        w_ = -w[-1] / k * (d - k + 1)
        if abs(w_) < thres:
            break
        w.append(w_)
        k += 1
    w = np.array(w[::-1])
    width = len(w)
    
    # 2. Apply weights via rolling window dot product
    vals = series.values
    res = np.full(len(series), np.nan)
    for i in range(width - 1, len(series)):
        res[i] = np.dot(w, vals[i - width + 1 : i + 1])
        
    return pd.Series(res, index=series.index), width

def yang_zhang_vol(df, window=20):
    """
    Yang-Zhang volatility estimator (2000):
    Handles open jumps and continuous drift.
    sigma_YZ^2 = sigma_open^2 + k * sigma_close^2 + (1 - k) * sigma_RS^2
    where k = 0.34 / (1.34 + (n + 1) / (n - 1))
    """
    log_ho = np.log(df['high'] / df['open'])
    log_lo = np.log(df['low'] / df['open'])
    log_co = np.log(df['close'] / df['open'])
    
    log_oc = np.log(df['open'] / df['close'].shift(1))
    log_cc = np.log(df['close'] / df['close'].shift(1))
    
    # Rogers-Satchell open-to-close term
    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
    sigma_rs_sq = rs.rolling(window).mean()
    
    # Overnight variance (open / close_{t-1})
    sigma_open_sq = log_oc.rolling(window).var()
    
    # Close-to-close variance
    sigma_close_sq = log_cc.rolling(window).var()
    
    k = 0.34 / (1.34 + (window + 1.0) / (window - 1.0))
    sigma_yz_sq = sigma_open_sq + k * sigma_close_sq + (1 - k) * sigma_rs_sq
    return np.sqrt(np.maximum(0, sigma_yz_sq) * 365.25)

def garman_klass_vol(df, window=20):
    """
    Garman-Klass volatility estimator (1980):
    0.5 * (ln(H/L))^2 - (2*ln(2) - 1) * (ln(C/O))^2
    """
    log_hl = np.log(df['high'] / df['low'])
    log_co = np.log(df['close'] / df['open'])
    gk = 0.5 * (log_hl ** 2) - (2 * np.log(2) - 1) * (log_co ** 2)
    return np.sqrt(gk.rolling(window).mean() * 365.25)

def parkinson_vol(df, window=20):
    """
    Parkinson volatility estimator (1980):
    (ln(H/L))^2 / (4 * ln(2))
    """
    log_hl = np.log(df['high'] / df['low'])
    pv = (log_hl ** 2) / (4.0 * np.log(2))
    return np.sqrt(pv.rolling(window).mean() * 365.25)

def rogers_satchell_vol(df, window=20):
    """
    Rogers-Satchell volatility estimator (1991):
    ln(H/C)*ln(H/O) + ln(L/C)*ln(L/O)
    """
    log_ho = np.log(df['high'] / df['open'])
    log_lo = np.log(df['low'] / df['open'])
    log_co = np.log(df['close'] / df['open'])
    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
    return np.sqrt(np.maximum(0, rs.rolling(window).mean()) * 365.25)

def barndorff_nielsen_shephard_jump(df, window=20):
    """
    Barndorff-Nielsen & Shephard (2004, 2006) Bipower Variation Decomposition:
    Realized Variance (RV) vs Bipower Variation (BV).
    BV = (mu_1)^{-2} * sum(|r_t| * |r_{t-1}|), where mu_1 = sqrt(2/pi) ~ 0.79788
    Jump Variation: J_t = max(0, RV_t - BV_t)
    Relative Jump Ratio: RJ_t = J_t / (RV_t + eps)
    """
    ret = np.log(df['close'] / df['close'].shift(1))
    mu1 = np.sqrt(2.0 / np.pi)
    
    # Daily squared return sum over rolling window as RV
    rv = (ret ** 2).rolling(window).sum()
    
    # Bipower variation: product of consecutive absolute returns
    abs_ret = ret.abs()
    bv = (1.0 / (mu1 ** 2)) * (abs_ret * abs_ret.shift(1)).rolling(window - 1).sum()
    
    jump = np.maximum(0.0, rv - bv)
    jump_ratio = jump / (rv + 1e-12)
    
    # Jump z-test statistic proxy (Huang & Tauchen 2005)
    # Tripower quarticity for asymptotic variance
    mu_43 = 2**(2/3) * stats.gamma.pdf(7/6, 1) if hasattr(stats, 'gamma') else 0.8309
    # Simplified relative jump ratio & continuous component
    continuous_vol = np.sqrt(bv * (365.25 / window))
    jump_vol = np.sqrt(jump * (365.25 / window))
    
    return jump, jump_ratio, continuous_vol, jump_vol

def daily_vpin_proxy(df, window=20):
    """
    Daily VPIN proxy (Volume-Synchronized Probability of Toxicity):
    Easley, Lopez de Prado, O'Hara (2012).
    Classify volume into buy/sell using tick rule / price change proxy:
    v_buy = V * (C - L) / (H - L)
    v_sell = V * (H - C) / (H - L)
    Imbalance: |v_buy - v_sell| / (v_buy + v_sell)
    Rolling average gives daily VPIN proxy.
    """
    hl_range = df['high'] - df['low']
    hl_range = np.where(hl_range == 0, 1e-8, hl_range)
    
    buy_share = (df['close'] - df['low']) / hl_range
    v_buy = df['volume'] * buy_share
    v_sell = df['volume'] * (1.0 - buy_share)
    
    imbalance = (v_buy - v_sell).abs()
    vpin = imbalance.rolling(window).sum() / (df['volume'].rolling(window).sum() + 1e-12)
    return vpin

def amihud_and_kyle(df, window=20):
    """
    Amihud illiquidity: |r_t| / (P_t * V_t)
    Kyle's lambda proxy: Cov(r_t, Signed_Volume_t) / Var(Signed_Volume_t)
    """
    ret = df['close'].pct_change()
    dollar_vol = df['close'] * df['volume']
    amihud = (ret.abs() / (dollar_vol + 1e-12)) * 1e8 # scaled
    amihud_ma = amihud.rolling(window).mean()
    
    # Signed volume
    sign = np.sign(ret)
    signed_vol = sign * df['volume']
    
    # Kyle lambda rolling OLS slope: Cov(ret, signed_vol) / Var(signed_vol)
    cov = (ret * signed_vol).rolling(window).mean() - ret.rolling(window).mean() * signed_vol.rolling(window).mean()
    var_sv = signed_vol.rolling(window).var()
    kyle_lambda = (cov / (var_sv + 1e-12)) * 1e6 # scaled
    
    return amihud_ma, kyle_lambda

def run_benchmark():
    print("================================================================================")
    print(" ADVANCED MATHEMATICAL VOLATILITY, JUMP & FRACDIFF BENCHMARK ON 10Y COINBASE BTC")
    print("================================================================================")
    
    df = pd.read_csv(DATA_PATH)
    df['open_time'] = pd.to_datetime(df['open_time'])
    df = df.sort_values('open_time').reset_index(drop=True)
    log_p = np.log(df['close'])
    
    print(f"Loaded {len(df)} daily bars from {df['open_time'].iloc[0].strftime('%Y-%m-%d')} to {df['open_time'].iloc[-1].strftime('%Y-%m-%d')}")
    
    # --------------------------------------------------------------------------
    # 1. FRACDIFF OPTIMIZATION (Marcos Lopez de Prado)
    # --------------------------------------------------------------------------
    print("\n--- 1. FRACDIFF MEMORY VS STATIONARITY GRID SEARCH (d in [0.10, 0.70]) ---")
    d_grid = np.arange(0.10, 0.75, 0.05)
    frac_results = []
    
    best_d = None
    min_stat_d = None
    
    for d in d_grid:
        d = round(d, 2)
        fd_series, width = frac_diff_ffd(log_p, d, thres=1e-4)
        valid = fd_series.dropna()
        
        # ADF Test
        adf_res = adfuller(valid, maxlag=1, autolag=None)
        adf_stat = adf_res[0]
        p_val = adf_res[1]
        crit_95 = adf_res[4]['5%']
        crit_99 = adf_res[4]['1%']
        
        # Correlation with raw log price
        corr = np.corrcoef(valid, log_p.loc[valid.index])[0, 1]
        is_stat = p_val < 0.05
        
        frac_results.append({
            'd': d,
            'adf_stat': adf_stat,
            'p_value': p_val,
            'crit_95': crit_95,
            'corr_with_logP': corr,
            'width': width,
            'is_stationary': is_stat
        })
        
        stat_mark = "[STATIONARY 95%]" if is_stat else "[NON-STAT]"
        if is_stat and min_stat_d is None:
            min_stat_d = d
            
        print(f"d={d:.2f} | ADF stat: {adf_stat:8.3f} (p-val: {p_val:8.4e}) | Corr(fd, logP): {corr:.4f} | Window: {width:4d} {stat_mark}")
    
    frac_df = pd.DataFrame(frac_results)
    best_d = min_stat_d if min_stat_d is not None else 0.40
    print(f"\nOptimal Lopez de Prado FracDiff d*: {best_d:.2f} (Retains {frac_df.loc[frac_df['d']==best_d, 'corr_with_logP'].values[0]*100:.2f}% memory with ADF p={frac_df.loc[frac_df['d']==best_d, 'p_value'].values[0]:.4e})")
    
    # Generate optimal and comparison FracDiff series
    df['fracdiff_optimal'], _ = frac_diff_ffd(log_p, best_d, thres=1e-4)
    df['fracdiff_0_3'], _ = frac_diff_ffd(log_p, 0.30, thres=1e-4)
    df['fracdiff_0_5'], _ = frac_diff_ffd(log_p, 0.50, thres=1e-4)
    
    # --------------------------------------------------------------------------
    # 2. VOLATILITY ESTIMATORS & JUMP DECOMPOSITION
    # --------------------------------------------------------------------------
    print("\n--- 2. COMPUTING ADVANCED VOLATILITY & JUMP DECOMPOSITIONS ---")
    df['vol_yang_zhang_20'] = yang_zhang_vol(df, window=20)
    df['vol_garman_klass_20'] = garman_klass_vol(df, window=20)
    df['vol_parkinson_20'] = parkinson_vol(df, window=20)
    df['vol_rogers_satchell_20'] = rogers_satchell_vol(df, window=20)
    
    # Close-to-close rolling vol
    ret = np.log(df['close'] / df['close'].shift(1))
    df['vol_close_close_20'] = ret.rolling(20).std() * np.sqrt(365.25)
    
    # Parkinson efficiency ratio: Parkinson Vol / Close-to-Close Vol
    # Values >> 1 indicate intraday range expansion / micro jumps
    df['parkinson_ratio_20'] = df['vol_parkinson_20'] / (df['vol_close_close_20'] + 1e-12)
    
    # Barndorff-Nielsen & Shephard Bipower Jump decomposition
    jump, jump_ratio, cont_vol, jump_vol = barndorff_nielsen_shephard_jump(df, window=20)
    df['jump_var_20'] = jump
    df['jump_ratio_20'] = jump_ratio
    df['vol_bipower_continuous_20'] = cont_vol
    df['vol_jump_component_20'] = jump_vol
    
    # Microstructure proxies: VPIN, Amihud, Kyle's Lambda
    df['vpin_daily_20'] = daily_vpin_proxy(df, window=20)
    df['amihud_illiq_20'], df['kyle_lambda_20'] = amihud_and_kyle(df, window=20)
    
    # Signed Volume Flow Momentum (5-day vs 20-day)
    sign_vol = np.sign(df['close'].pct_change()) * df['volume']
    df['signed_volume_flow_20'] = sign_vol.rolling(20).mean() / (df['volume'].rolling(20).mean() + 1e-12)
    
    # --------------------------------------------------------------------------
    # 3. FORWARD RETURNS & INFORMATION COEFFICIENT (SPEARMAN IC)
    # --------------------------------------------------------------------------
    print("\n--- 3. BENCHMARKING INFORMATION COEFFICIENT (SPEARMAN IC) ON FORWARD RETURNS ---")
    horizons = [7, 14, 30]
    for h in horizons:
        df[f'fwd_ret_{h}d'] = df['close'].shift(-h) / df['close'] - 1.0
        
    candidate_features = [
        'fracdiff_optimal',
        'fracdiff_0_3',
        'fracdiff_0_5',
        'vol_yang_zhang_20',
        'vol_garman_klass_20',
        'vol_parkinson_20',
        'vol_rogers_satchell_20',
        'vol_close_close_20',
        'parkinson_ratio_20',
        'jump_var_20',
        'jump_ratio_20',
        'vol_bipower_continuous_20',
        'vol_jump_component_20',
        'vpin_daily_20',
        'amihud_illiq_20',
        'kyle_lambda_20',
        'signed_volume_flow_20'
    ]
    
    ic_summary = []
    for feat in candidate_features:
        row = {'feature': feat}
        for h in horizons:
            target = f'fwd_ret_{h}d'
            sub = df[[feat, target]].dropna()
            if len(sub) > 100:
                spearman_corr, p_val = stats.spearmanr(sub[feat], sub[target])
                row[f'IC_{h}d'] = spearman_corr
                row[f'p_{h}d'] = p_val
            else:
                row[f'IC_{h}d'] = np.nan
                row[f'p_{h}d'] = np.nan
        ic_summary.append(row)
        
    ic_df = pd.DataFrame(ic_summary)
    
    print("\nSpearman Information Coefficients (IC):")
    print("-" * 90)
    print(f"{'Feature':<28} | {'IC (7d)':<9} {'p-val':<10} | {'IC (14d)':<9} {'p-val':<10} | {'IC (30d)':<9} {'p-val':<10}")
    print("-" * 90)
    for _, r in ic_df.iterrows():
        print(f"{r['feature']:<28} | {r['IC_7d']:9.4f} {r['p_7d']:<10.4e} | {r['IC_14d']:9.4f} {r['p_14d']:<10.4e} | {r['IC_30d']:9.4f} {r['p_30d']:<10.4e}")
    print("-" * 90)
    
    # --------------------------------------------------------------------------
    # 4. RANDOM FOREST FEATURE IMPORTANCE RANKING
    # --------------------------------------------------------------------------
    print("\n--- 4. NON-LINEAR RANDOM FOREST FEATURE IMPORTANCE RANKING ---")
    rf_ranks = {}
    for h in horizons:
        target = f'fwd_ret_{h}d'
        sub = df[candidate_features + [target]].dropna()
        X = sub[candidate_features]
        y = sub[target]
        
        rf = RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_leaf=20, random_state=42)
        rf.fit(X, y)
        importances = pd.Series(rf.feature_importances_, index=candidate_features).sort_values(ascending=False)
        rf_ranks[f'Horizon_{h}d'] = importances
        
    rf_df = pd.DataFrame(rf_ranks)
    rf_df['Mean_Importance'] = rf_df.mean(axis=1)
    rf_df = rf_df.sort_values('Mean_Importance', ascending=False)
    
    print("\nFeature Importance Across Horizons (Mean Gini Impurity Decrease):")
    print("-" * 80)
    print(f"{'Feature':<28} | {'7d Imp':<10} | {'14d Imp':<10} | {'30d Imp':<10} | {'Mean Imp':<10}")
    print("-" * 80)
    for feat, row in rf_df.iterrows():
        print(f"{feat:<28} | {row['Horizon_7d']:10.4f} | {row['Horizon_14d']:10.4f} | {row['Horizon_30d']:10.4f} | {row['Mean_Importance']:10.4f}")
    print("-" * 80)
    
    # --------------------------------------------------------------------------
    # 5. SUMMARY KEY FINDINGS
    # --------------------------------------------------------------------------
    print("\n================================================================================")
    print(" QUANTITATIVE FINDINGS & MICROSTRUCTURE SYNTHESIS")
    print("================================================================================")
    print(f"1. FracDiff Stationarity Bound:")
    print(f"   - Minimum stationary d: {best_d:.2f}")
    print(f"   - Memory preserved (correlation with raw log P): {frac_df.loc[frac_df['d']==best_d, 'corr_with_logP'].values[0]*100:.2f}%")
    print(f"   - Standard return diff (d=1.0) completely wipes long memory; FracDiff retains persistent trend signals.")
    
    print(f"\n2. Realized Jump Decomposition:")
    mean_jump_ratio = df['jump_ratio_20'].mean()
    print(f"   - Average Jump Ratio (J_t / RV_t) over 10Y: {mean_jump_ratio*100:.2f}%")
    top_jump_ic = ic_df.loc[ic_df['feature'] == 'jump_ratio_20', 'IC_30d'].values[0]
    print(f"   - Jump Ratio 30d Spearman IC: {top_jump_ic:.4f} (p-val: {ic_df.loc[ic_df['feature'] == 'jump_ratio_20', 'p_30d'].values[0]:.4e})")
    
    print(f"\n3. Volatility Estimator Efficiency:")
    yz_mean = df['vol_yang_zhang_20'].mean()
    gk_mean = df['vol_garman_klass_20'].mean()
    pv_mean = df['vol_parkinson_20'].mean()
    cc_mean = df['vol_close_close_20'].mean()
    print(f"   - Mean Annualized Vol - Yang-Zhang: {yz_mean*100:.2f}% | Garman-Klass: {gk_mean*100:.2f}% | Parkinson: {pv_mean*100:.2f}% | Close-Close: {cc_mean*100:.2f}%")
    print(f"   - Yang-Zhang captures overnight gap jumps between UTC boundaries that Parkinson misses.")
    
    top_feature = rf_df.index[0]
    print(f"\n4. Leading Predictive Feature:")
    print(f"   - '{top_feature}' achieves highest aggregate non-linear predictive power (Mean Importance: {rf_df.iloc[0]['Mean_Importance']:.4f})")
    print("================================================================================\n")

if __name__ == '__main__':
    run_benchmark()
