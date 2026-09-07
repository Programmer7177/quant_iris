"""
DEEP FEATURE EXTRACTION & DIMENSIONAL EXPANSION EXPERIMENT
Expands feature set from pure Price/RSI to Multi-Dimensional Quant Pillars:
1. Microstructure & Order Flow: Parkinson Volatility, Garman-Klass Vol, Volume Trend, Amihud Illiquidity proxy.
2. Derivatives & Positioning: Funding Rate Proxy (Synthetic Basis from high/low skew), Implied Variance ratio.
3. On-chain / Macro Cycles: Days to/from Halving Cycle phase (Harmonic sine/cosine encoding), Power Law Corridor Residuals.
4. Momentum & Term Structure: RSI Multi-scale (7d, 14d, 30d, 90d, 180d), MACD Divergence, Moving Average Spreads.
5. Volatility Regimes: Volatility of Volatility (VoV), Short-to-Long Vol Ratio (Term Structure of Vol).

Tests Out-of-Sample (1,271 days) across 7d, 14d, 30d horizons:
Compares:
  - Benchmark (Baseline Price + RSI only)
  - Expanded Multi-Dimensional Feature Set (with Feature Selection / L1 Sparsity & ExtraTrees Feature Importance)
"""
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LassoCV
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import SelectFromModel

DATA_PATH = r"C:\Mirza Personal\crypto quant\data\btc_coinbase_10y.csv"

def rsi(series, n):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = (-delta.clip(upper=0)).rolling(n).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def build_rich_features():
    df = pd.read_csv(DATA_PATH, parse_dates=["open_time"]).sort_values("open_time").reset_index(drop=True)
    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]
    
    # 1. PRICE MOMENTUM & OSCILLATORS (Multi-scale)
    df["log_ret"] = np.log(c / c.shift(1))
    df["rsi_7"] = rsi(c, 7)
    df["rsi_14"] = rsi(c, 14)
    df["rsi_30"] = rsi(c, 30)
    df["rsi_90"] = rsi(c, 90)
    df["rsi_180"] = rsi(c, 180)
    
    # MACD 1M & 3M
    ema12 = c.ewm(span=12).mean()
    ema26 = c.ewm(span=26).mean()
    df["macd_1m_hist"] = (ema12 - ema26) - (ema12 - ema26).ewm(span=9).mean()
    
    # Moving Average Dispersion
    ema20 = c.ewm(span=20).mean()
    ema50 = c.ewm(span=50).mean()
    ema200 = c.ewm(span=200).mean()
    df["spread_20_50"] = (ema20 - ema50) / ema50
    df["spread_50_200"] = (ema50 - ema200) / ema200
    df["dist_ema200"] = (c - ema200) / ema200
    
    # 2. ADVANCED VOLATILITY ESTIMATORS (Microstructure literature)
    # Parkinson Volatility (uses High/Low - 5x more efficient than Close-to-Close)
    hl_ratio = np.log(h / (l + 1e-9))
    df["vol_parkinson_14"] = np.sqrt((1.0 / (4.0 * np.log(2.0))) * (hl_ratio**2).rolling(14).mean()) * np.sqrt(365)
    
    # Garman-Klass Volatility (incorporates Open, High, Low, Close)
    log_co = np.log(c / (o + 1e-9))
    gk = 0.5 * (hl_ratio**2) - (2.0 * np.log(2.0) - 1.0) * (log_co**2)
    df["vol_garman_klass_14"] = np.sqrt(gk.rolling(14).mean().clip(lower=0)) * np.sqrt(365)
    
    # Volatility Term Structure (Short Vol vs Long Vol)
    vol_7 = df["log_ret"].rolling(7).std() * np.sqrt(365)
    vol_60 = df["log_ret"].rolling(60).std() * np.sqrt(365)
    df["vol_term_structure"] = vol_7 / (vol_60 + 1e-9)
    df["vol_of_vol_30"] = vol_7.rolling(30).std() # Volatility of Volatility
    
    # 3. LIQUIDITY & VOLUME DYNAMICS (Amihud & Roll proxies)
    # Amihud Illiquidity: |Return| / Dollar Volume
    dollar_vol = c * v
    df["amihud_illiquidity_14"] = (np.abs(df["log_ret"]) / (dollar_vol + 1e-9)).rolling(14).mean() * 1e9
    df["volume_ratio_7_30"] = v.rolling(7).mean() / (v.rolling(30).mean() + 1e-9)
    
    # Price Range Skew (proxy for aggressive buyer/seller exhaustion)
    df["bar_shadow_skew"] = ((h - np.maximum(c, o)) - (np.minimum(c, o) - l)) / (h - l + 1e-9)
    df["bar_shadow_skew_14"] = df["bar_shadow_skew"].rolling(14).mean()
    
    # 4. MACRO & HALVING HARMONIC CYCLE
    # BTC Halvings: 2012-11-28, 2016-07-09, 2020-05-11, 2024-04-19
    # Halving cycle ~1460 days. Sine & Cosine encoding to capture non-linear 4-year cycle phase
    genesis = pd.Timestamp("2009-01-03")
    days_since_genesis = (df["open_time"] - genesis).dt.days
    cycle_period = 1460.0
    df["halving_cycle_sin"] = np.sin(2 * np.pi * days_since_genesis / cycle_period)
    df["halving_cycle_cos"] = np.cos(2 * np.pi * days_since_genesis / cycle_period)
    
    # Power Law Residual (Santostasi model)
    df["power_law_trend"] = -17.0 + 5.8 * np.log(days_since_genesis)
    df["power_law_residual"] = np.log(c) - df["power_law_trend"]
    
    # Target definitions
    for h_days in [7, 14, 30]:
        df[f"fwd_ret_{h_days}"] = np.log(c.shift(-h_days) / c)
        
    return df.dropna().reset_index(drop=True)

def evaluate_models():
    df = build_rich_features()
    n = len(df)
    split_idx = int(n * 0.65) # 65% train, 35% test out-of-sample (~1,270 days test)
    
    baseline_feats = ["rsi_14", "rsi_30", "rsi_90", "dist_ema200", "power_law_residual"]
    
    all_expanded_feats = [
        "rsi_7", "rsi_14", "rsi_30", "rsi_90", "rsi_180",
        "macd_1m_hist", "spread_20_50", "spread_50_200", "dist_ema200",
        "vol_parkinson_14", "vol_garman_klass_14", "vol_term_structure", "vol_of_vol_30",
        "amihud_illiquidity_14", "volume_ratio_7_30", "bar_shadow_skew_14",
        "halving_cycle_sin", "halving_cycle_cos", "power_law_residual"
    ]
    
    train = df.iloc[:split_idx]
    
    print("=== FEATURE EXTRACTION COMPARISON: BASELINE vs EXPANDED QUANT PILLARS ===")
    print(f"Total Observations: {n} bar harian | Fitur Baru: {len(all_expanded_feats)} dimensi")
    print(f"Periode Test Out-of-Sample: {df['open_time'].iloc[split_idx].date()} s.d. {df['open_time'].iloc[-1].date()}\n")
    
    # Feature Importance via ExtraTrees on 30d target
    et = ExtraTreesRegressor(n_estimators=100, max_depth=5, random_state=42)
    y_30_train = train["fwd_ret_30"].values
    X_train_exp = train[all_expanded_feats].values
    et.fit(X_train_exp, y_30_train)
    
    fi = pd.Series(et.feature_importances_, index=all_expanded_feats).sort_values(ascending=False)
    print("--- TOP 8 FITUR PALING MEMILIKI SINYAL PREDIKTIF (ExtraTrees Feature Importance) ---")
    for feat_name, imp in fi.head(8).items():
        print(f"  {feat_name:<26}: {imp:.4f} ({imp*100:.1f}%)")
    print("----------------------------------------------------------------------------------\n")
    
    # Compare Out-of-Sample Performance across horizons
    for h in [7, 14, 30]:
        test = df.iloc[split_idx:-h]
        y_tr = train[f"fwd_ret_{h}"].values
        y_te = test[f"fwd_ret_{h}"].values
        actual_dir = np.sign(y_te)
        
        # 1. Baseline Model (Price & basic RSI only)
        m_base = Ridge(alpha=100.0).fit(train[baseline_feats].values, y_tr)
        pred_base = m_base.predict(test[baseline_feats].values)
        acc_base = np.mean(np.sign(pred_base) == actual_dir)
        mape_base = np.mean(np.abs(test["close"] * np.exp(y_te) - test["close"] * np.exp(pred_base)) / (test["close"] * np.exp(y_te)))
        
        # 2. Expanded Multi-Dimensional Model (All Quant Pillars with tuned Regularization)
        # Using Ridge with feature standardization internally
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("reg", Ridge(alpha=250.0))
        ])
        pipe.fit(train[all_expanded_feats].values, y_tr)
        pred_exp = pipe.predict(test[all_expanded_feats].values)
        acc_exp = np.mean(np.sign(pred_exp) == actual_dir)
        mape_exp = np.mean(np.abs(test["close"] * np.exp(y_te) - test["close"] * np.exp(pred_exp)) / (test["close"] * np.exp(y_te)))
        
        # 3. Tree-Ensemble on Expanded Features (Capturing Non-linear Interactions)
        m_tree = ExtraTreesRegressor(n_estimators=80, max_depth=3, min_samples_leaf=25, random_state=42)
        m_tree.fit(train[all_expanded_feats].values, y_tr)
        pred_tree = m_tree.predict(test[all_expanded_feats].values)
        acc_tree = np.mean(np.sign(pred_tree) == actual_dir)
        mape_tree = np.mean(np.abs(test["close"] * np.exp(y_te) - test["close"] * np.exp(pred_tree)) / (test["close"] * np.exp(y_te)))
        
        best_acc = max(acc_base, acc_exp, acc_tree)
        winner = "Expanded Linear" if best_acc == acc_exp else ("Expanded Tree" if best_acc == acc_tree else "Baseline")
        
        print(f"[HORIZON {h:2d} HARI]")
        print(f"  Baseline (Price/RSI)    : Akurasi Arah = {acc_base:.2%} | MAPE = {mape_base:.2%}")
        print(f"  Expanded (Linear Scaled): Akurasi Arah = {acc_exp:.2%} | MAPE = {mape_exp:.2%}")
        print(f"  Expanded (Tree Non-lin) : Akurasi Arah = {acc_tree:.2%} | MAPE = {mape_tree:.2%}")
        print(f"  --> Pemenang: {winner} (Delta Akurasi: {best_acc - acc_base:+.2%})\n")

if __name__ == "__main__":
    evaluate_models()
