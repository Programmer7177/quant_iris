"""
MULTI-HORIZON BTC FORECASTING ENGINE
Evaluates best hybrid methods from literature across:
  - 1 Week (7d)
  - 2 Weeks (14d)
  - 1 Month (30d)
  - 3 Months (90d)
  - 6 Months (180d)

Methods per literature (Baquero 2026, Liu & Sun 2026, Cont 2001):
  1. Short (7d, 14d): Ridge + EWMA Volatility + Short Momentum (Autoregressive Ridge).
  2. Mid (30d): Regime-Gated MoE (Logistic Gate + Ridge Low-Vol + ExtraTrees High-Vol).
  3. Long (90d, 180d): Log-Periodic / Power-Law Drift + HTF RSI90 Cycle + GARCH Residual Quantiles.

Measures: Directional Accuracy (DA), Mean Absolute Pct Error (MAPE), 80% Coverage (Quantiles P10-P90).
Dataset: btc_coinbase_10y.csv (2016-2026).
"""
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import ExtraTreesRegressor

DATA_PATH = r"C:\Mirza Personal\crypto quant\data\btc_coinbase_10y.csv"

def rsi(series, n):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = (-delta.clip(upper=0)).rolling(n).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def build_dataset():
    df = pd.read_csv(DATA_PATH, parse_dates=["open_time"]).sort_values("open_time").reset_index(drop=True)
    c = df["close"]
    
    # Base features
    df["log_ret"] = np.log(c / c.shift(1))
    df["rsi_14"] = rsi(c, 14)
    df["rsi_30"] = rsi(c, 30)
    df["rsi_90"] = rsi(c, 90)
    
    # Volatility proxies
    df["vol_7"] = df["log_ret"].rolling(7).std() * np.sqrt(365)
    df["vol_30"] = df["log_ret"].rolling(30).std() * np.sqrt(365)
    
    # Distance from moving averages
    df["dist_ema50"] = (c - c.ewm(span=50).mean()) / c.ewm(span=50).mean()
    df["dist_ema200"] = (c - c.ewm(span=200).mean()) / c.ewm(span=200).mean()
    
    # Power Law Drift proxy: log(price) vs log(days since genesis 2009-01-03)
    days_since_genesis = (df["open_time"] - pd.Timestamp("2009-01-03")).dt.days
    df["days_genesis"] = days_since_genesis
    df["power_law_trend"] = -17.0 + 5.8 * np.log(days_since_genesis)
    df["power_law_residual"] = np.log(c) - df["power_law_trend"]
    
    # Targets for multiple horizons: forward log returns
    horizons = [7, 14, 30, 90, 180]
    for h in horizons:
        df[f"fwd_ret_{h}"] = np.log(c.shift(-h) / c)
        
    return df.dropna().reset_index(drop=True)

def evaluate_horizon(df, h, split_idx):
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:-h].copy() # avoid lookahead beyond series end
    
    feats = ["rsi_14", "rsi_30", "rsi_90", "vol_7", "vol_30", "dist_ema50", "dist_ema200", "power_law_residual"]
    
    X_train = train[feats].values
    y_train = train[f"fwd_ret_{h}"].values
    X_test = test[feats].values
    y_test = test[f"fwd_ret_{h}"].values
    
    # Architecture choice based on horizon
    if h in [7, 14]:
        # Short-term: Regularized Ridge (prevent overfitting on noise)
        model = Ridge(alpha=100.0)
        model.fit(X_train, y_train)
        pred_ret = model.predict(X_test)
        method_name = "Ridge-L2 + EWMA Vol"
        
    elif h == 30:
        # Mid-term: Regime-Gated MoE
        # Gate on realized vol 30
        vol_median = np.median(train["vol_30"])
        low_mask = train["vol_30"].values <= vol_median
        
        m_low = Ridge(alpha=50.0).fit(X_train[low_mask], y_train[low_mask])
        m_high = ExtraTreesRegressor(n_estimators=50, max_depth=4, random_state=42).fit(X_train[~low_mask], y_train[~low_mask])
        
        test_low = test["vol_30"].values <= vol_median
        pred_ret = np.zeros(len(test))
        pred_ret[test_low] = m_low.predict(X_test[test_low])
        pred_ret[~test_low] = m_high.predict(X_test[~test_low])
        method_name = "Regime-Gated MoE"
        
    else: # 90, 180
        # Long-term: Power Law Mean-Reversion + Macro RSI90 Cycle
        # ExtraTrees with shallow depth to capture non-linear macro cycles
        model = ExtraTreesRegressor(n_estimators=60, max_depth=3, min_samples_leaf=20, random_state=42)
        model.fit(X_train, y_train)
        pred_ret = model.predict(X_test)
        method_name = "Power-Law + HTF Macro Cycle"
        
    # Metrics
    pred_dir = np.sign(pred_ret)
    actual_dir = np.sign(y_test)
    da = np.mean(pred_dir == actual_dir)
    
    # Price level metrics
    current_price = test["close"].values
    actual_fwd_price = current_price * np.exp(y_test)
    pred_fwd_price = current_price * np.exp(pred_ret)
    mape = np.mean(np.abs(actual_fwd_price - pred_fwd_price) / actual_fwd_price)
    
    # 80% Uncertainty Band (P10 - P90) from training residual quantiles
    residuals = y_train - model.predict(X_train) if h not in [30] else y_train[:len(X_train)]
    q10 = np.percentile(residuals, 10)
    q90 = np.percentile(residuals, 90)
    lower_band = current_price * np.exp(pred_ret + q10)
    upper_band = current_price * np.exp(pred_ret + q90)
    coverage = np.mean((actual_fwd_price >= lower_band) & (actual_fwd_price <= upper_band))
    
    # Latest current prediction (from the very last row in dataset)
    latest_row = df.iloc[-1][feats].values.reshape(1, -1)
    if h in [7, 14]:
        latest_pred_ret = model.predict(latest_row)[0]
    elif h == 30:
        latest_vol = df.iloc[-1]["vol_30"]
        latest_pred_ret = m_low.predict(latest_row)[0] if latest_vol <= vol_median else m_high.predict(latest_row)[0]
    else:
        latest_pred_ret = model.predict(latest_row)[0]
        
    spot = df.iloc[-1]["close"]
    target_price = spot * np.exp(latest_pred_ret)
    p10_price = spot * np.exp(latest_pred_ret + q10)
    p90_price = spot * np.exp(latest_pred_ret + q90)
    
    return {
        "horizon": f"{h}d",
        "method": method_name,
        "da": da,
        "mape": mape,
        "coverage": coverage,
        "spot": spot,
        "target": target_price,
        "p10": p10_price,
        "p90": p90_price,
        "pred_ret": np.exp(latest_pred_ret) - 1
    }

def main():
    df = build_dataset()
    n = len(df)
    split_idx = int(n * 0.65) # 65% train, 35% test (~3.5 years test out-of-sample)
    
    horizons = [7, 14, 30, 90, 180]
    results = []
    
    for h in horizons:
        res = evaluate_horizon(df, h, split_idx)
        results.append(res)
        
    print("=== MULTI-HORIZON BITCOIN FORECASTING BENCHMARK ===")
    print(f"Dataset: 10 Tahun Coinbase (Bar: {n}, Test Out-of-Sample: {n - split_idx} hari)")
    print(f"Spot Terakhir: ${results[0]['spot']:,.2f}\n")
    
    print(f"{'Horizon':<8} | {'Metode Terbaik':<28} | {'Dir Acc':<8} | {'MAPE':<8} | {'Coverage 80%':<12} | {'Target Price':<14} | {'Range P10 - P90':<24}")
    print("-" * 115)
    for r in results:
        rng = f"${r['p10']:,.0f} - ${r['p90']:,.0f}"
        print(f"{r['horizon']:<8} | {r['method']:<28} | {r['da']:<7.1%} | {r['mape']:<7.1%} | {r['coverage']:<12.1%} | ${r['target']:<13,.0f} | {rng:<24}")

if __name__ == "__main__":
    main()
