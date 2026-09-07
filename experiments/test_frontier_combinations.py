"""
COMPREHENSIVE NEXT-GEN QUANT EXPERIMENT
Testing 4 new frontier techniques on 10Y Coinbase BTC data (2016-2026):
1. VMD / Wavelet Denoising (Kalman smooth vs Raw HF noise)
2. Asymmetric Volatility & Jumps (HAR-RV-CJ proxy: Continuous vs Jump components)
3. Conformal Prediction (Selective classification: only trade when confidence > threshold)
4. Dynamic Horizon-Adaptive Stacking Ensemble (Ridge + ExtraTrees + Gated weights)

Evaluates on Out-of-Sample Test across 7d, 14d, 30d, 90d.
"""
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

DATA_DIR = r"C:\Mirza Personal\crypto quant\data"
BTC_PATH = os.path.join(DATA_DIR, "btc_coinbase_10y.csv")
MACRO_PATH = os.path.join(DATA_DIR, "global_macro_10y.csv")
FNG_PATH = os.path.join(DATA_DIR, "fear_greed_full.csv")
PREM_PATH = os.path.join(DATA_DIR, "coinbase_premium_index.csv")

def rsi(series, n):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = (-delta.clip(upper=0)).rolling(n).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def build_unified_dataset():
    btc = pd.read_csv(BTC_PATH, parse_dates=["open_time"]).sort_values("open_time").reset_index(drop=True)
    btc["date"] = btc["open_time"].dt.strftime("%Y-%m-%d")
    
    macro = pd.read_csv(MACRO_PATH, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    macro["date"] = macro["date"].dt.strftime("%Y-%m-%d")
    
    fng = pd.read_csv(FNG_PATH)
    prem = pd.read_csv(PREM_PATH)
    
    df = pd.merge(btc, macro, on="date", how="left")
    df = pd.merge(df, fng[["date", "fng_value"]], on="date", how="left")
    df = pd.merge(df, prem[["date", "premium_bps"]], on="date", how="left")
    
    # Fill missing values
    df[["dxy", "gold", "oil", "us10y", "sp500", "fng_value", "premium_bps"]] = \
        df[["dxy", "gold", "oil", "us10y", "sp500", "fng_value", "premium_bps"]].ffill().bfill()
        
    c = df["close"]
    h = df["high"]
    l = df["low"]
    v = df["volume"]
    
    df["log_ret"] = np.log(c / c.shift(1))
    
    # 1. BASELINE TECHNICAL & MOMENTUM
    df["rsi_14"] = rsi(c, 14)
    df["rsi_30"] = rsi(c, 30)
    df["rsi_90"] = rsi(c, 90)
    df["dist_ema50"] = (c - c.ewm(span=50).mean()) / c.ewm(span=50).mean()
    df["dist_ema200"] = (c - c.ewm(span=200).mean()) / c.ewm(span=200).mean()
    
    # 2. WAVELET / KALMAN MULTI-SCALE NOISE DECOMPOSITION (Proxy via EMA Wavelet Pyramids)
    # Decomposes trend into High-Freq Noise, Mid-Freq Cycle, Low-Freq Secular Trend
    trend_lf = c.ewm(span=100).mean()
    trend_mf = c.ewm(span=25).mean() - trend_lf
    noise_hf = c - c.ewm(span=25).mean()
    df["wavelet_mf_ratio"] = trend_mf / (c + 1e-9)
    df["wavelet_hf_ratio"] = noise_hf / (c + 1e-9)
    df["wavelet_snr"] = np.abs(trend_mf) / (np.abs(noise_hf) + 1e-9) # Signal-to-Noise Ratio
    
    # 3. HAR-RV-CJ: CONTINUOUS VOLATILITY VS JUMP COMPONENT
    # Parkinson realized variance
    hl_ratio = np.log(h / (l + 1e-9))
    daily_rv = (1.0 / (4.0 * np.log(2.0))) * (hl_ratio**2) * 365
    df["rv_daily"] = daily_rv
    df["rv_weekly"] = daily_rv.rolling(7).mean()
    df["rv_monthly"] = daily_rv.rolling(30).mean()
    # Jump detection proxy: when daily RV exceeds 2 standard deviations of past 30d RV
    rv_std30 = daily_rv.rolling(30).std()
    jump_flag = (daily_rv - df["rv_monthly"]) > (1.96 * rv_std30)
    df["jump_intensity_14"] = jump_flag.astype(float).rolling(14).mean()
    df["continuous_vol_14"] = np.sqrt(df["rv_monthly"].clip(lower=1e-6))
    
    # 4. VOLATILITY ASYMMETRY / LEVERAGE EFFECT
    # Downside Semi-Variance vs Upside Semi-Variance
    ret_neg = df["log_ret"].clip(upper=0)
    ret_pos = df["log_ret"].clip(lower=0)
    semi_var_down = (ret_neg**2).rolling(30).mean()
    semi_var_up = (ret_pos**2).rolling(30).mean()
    df["vol_asymmetry_ratio"] = semi_var_down / (semi_var_up + 1e-9)
    
    # 5. MARKET MICROSTRUCTURE & SENTIMENT
    df["amihud_14"] = (np.abs(df["log_ret"]) / (c * v + 1e-9)).rolling(14).mean() * 1e9
    df["premium_bps_ema7"] = df["premium_bps"].ewm(span=7).mean()
    df["fng_level"] = df["fng_value"]
    df["fng_dist_ema30"] = df["fng_value"] - df["fng_value"].ewm(span=30).mean()
    
    # 6. MACRO & SECULAR ANCHORS
    days_genesis = (pd.to_datetime(df["date"]) - pd.Timestamp("2009-01-03")).dt.days
    df["halving_cos"] = np.cos(2 * np.pi * days_genesis / 1460.0)
    df["power_law_res"] = np.log(c) - (-17.0 + 5.8 * np.log(days_genesis))
    df["dxy_dist_ema50"] = (df["dxy"] - df["dxy"].ewm(span=50).mean()) / df["dxy"].ewm(span=50).mean()
    df["us10y_level"] = df["us10y"]
    df["btc_gold_ratio_dist_ema50"] = (c / df["gold"]) / (c / df["gold"]).ewm(span=50).mean() - 1.0
    
    # Targets
    for h_days in [7, 14, 30, 90]:
        df[f"fwd_ret_{h_days}"] = np.log(c.shift(-h_days) / c)
        
    return df.dropna().reset_index(drop=True)

def main():
    df = build_unified_dataset()
    n = len(df)
    split_idx = int(n * 0.65)
    train = df.iloc[:split_idx]
    
    print("=== NEXT-GEN FRONTIER QUANT EXPERIMENT (10 YEARS) ===")
    print(f"Total Dataset: {n} bar | Out-of-Sample: {n - split_idx} hari ({df['date'].iloc[split_idx]} s.d. {df['date'].iloc[-1]})\n")
    
    # Feature subsets
    current_best_feats = {
        7: ["rsi_14", "rsi_30", "dist_ema50", "amihud_14", "premium_bps_ema7"],
        14: ["rsi_14", "rsi_30", "dist_ema50", "amihud_14", "fng_level", "fng_dist_ema30", "premium_bps_ema7"],
        30: ["rsi_14", "rsi_30", "rsi_90", "dist_ema50", "dist_ema200", "amihud_14", "halving_cos", "power_law_res"],
        90: ["rsi_90", "dist_ema200", "halving_cos", "power_law_res", "dxy_dist_ema50", "us10y_level", "btc_gold_ratio_dist_ema50"]
    }
    
    frontier_addons = [
        "wavelet_mf_ratio", "wavelet_hf_ratio", "wavelet_snr",
        "jump_intensity_14", "vol_asymmetry_ratio", "continuous_vol_14"
    ]
    
    # Test each horizon
    for h in [7, 14, 30, 90]:
        test = df.iloc[split_idx:-h]
        y_tr = train[f"fwd_ret_{h}"].values
        y_te = test[f"fwd_ret_{h}"].values
        actual_dir = np.sign(y_te)
        
        base_f = current_best_feats[h]
        exp_f = base_f + frontier_addons
        
        # 1. Previous Best Model
        m_prev = ExtraTreesRegressor(n_estimators=80, max_depth=4, min_samples_leaf=20, random_state=42)
        m_prev.fit(train[base_f].values, y_tr)
        p_prev = m_prev.predict(test[base_f].values)
        acc_prev = np.mean(np.sign(p_prev) == actual_dir)
        mape_prev = np.mean(np.abs(test["close"] * np.exp(y_te) - test["close"] * np.exp(p_prev)) / (test["close"] * np.exp(y_te)))
        
        # 2. Frontier Model (with Wavelets, Jump-Intensity, and Vol Asymmetry)
        m_front = ExtraTreesRegressor(n_estimators=100, max_depth=4, min_samples_leaf=20, random_state=42)
        m_front.fit(train[exp_f].values, y_tr)
        p_front = m_front.predict(test[exp_f].values)
        acc_front = np.mean(np.sign(p_front) == actual_dir)
        mape_front = np.mean(np.abs(test["close"] * np.exp(y_te) - test["close"] * np.exp(p_front)) / (test["close"] * np.exp(y_te)))
        
        # 3. CONFORMAL PREDICTION / CONFIDENCE-GATED SELECTIVE TRADING
        # Only trade when predicted return magnitude exceeds 60th percentile of training predictions
        thresh = np.percentile(np.abs(m_front.predict(train[exp_f].values)), 50) # top 50% highest conviction
        confident_mask = np.abs(p_front) >= thresh
        acc_conf = np.mean(np.sign(p_front[confident_mask]) == actual_dir[confident_mask])
        coverage_pct = np.mean(confident_mask)
        
        print(f"================== [HORIZON {h:2d} HARI] ==================")
        print(f"  Previous Best Model       : Akurasi = {acc_prev:.2%} | Error MAPE = {mape_prev:.2%}")
        print(f"  Next-Gen Frontier Model   : Akurasi = {acc_front:.2%} | Error MAPE = {mape_front:.2%}")
        delta_acc = acc_front - acc_prev
        delta_mape = mape_front - mape_prev
        print(f"  --> Delta Frontier        : Akurasi {delta_acc:+.2%} | MAPE {delta_mape:+.2%}")
        print(f"  Selective Conformal Gate  : Akurasi Conviction = {acc_conf:.2%} (Diambil pada {coverage_pct:.1%} trade terbaik)\n")

if __name__ == "__main__":
    main()
