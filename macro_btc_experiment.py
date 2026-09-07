"""
MACRO-AUGMENTED MULTI-HORIZON BTC FORECASTING ENGINE
Combines:
  1. BTC Native Features: High-Low Parkinson Vol, Amihud Illiquidity, Multi-scale RSI, Moving Avg Spreads.
  2. Harmonic Halving Cycle & Power Law Residual.
  3. Global Macro Signals:
     - DXY: Level, 30d Return, Distance from EMA50 (Dollar liquidity cycle).
     - Gold: Level, 30d Return, Gold/BTC ratio trend (Store of value competition).
     - Oil: Level, 30d Return (Cost-push inflation & real economy demand proxy).
     - US 10Y Yield (^TNX): Level, 30d Change in bps (Risk-free rate & discount factor).
     - S&P 500: Level, 30d Return, Correlation with BTC (Risk-on/Risk-off regime).

Tests Out-of-Sample (10 Years, 2016-2026, 1200+ days test).
Evaluates:
  - Macro Feature Importance.
  - Information Coefficients (IC) of Macro factors vs BTC 30d/90d returns.
  - Directional Accuracy & MAPE improvement vs Crypto-only models.
"""
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

BTC_PATH = r"C:\Mirza Personal\crypto quant\data\btc_coinbase_10y.csv"
MACRO_PATH = r"C:\Mirza Personal\crypto quant\data\global_macro_10y.csv"

def rsi(series, n):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = (-delta.clip(upper=0)).rolling(n).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def build_merged_dataset():
    btc = pd.read_csv(BTC_PATH, parse_dates=["open_time"]).sort_values("open_time").reset_index(drop=True)
    btc["date"] = btc["open_time"].dt.strftime("%Y-%m-%d")
    
    macro = pd.read_csv(MACRO_PATH, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    macro["date"] = macro["date"].dt.strftime("%Y-%m-%d")
    
    # Merge on date, then forward fill macro values (crypto trades weekends, tradfi does not)
    df = pd.merge(btc, macro, on="date", how="left")
    df[["dxy", "gold", "oil", "us10y", "sp500"]] = df[["dxy", "gold", "oil", "us10y", "sp500"]].ffill().bfill()
    
    c = df["close"]
    h = df["high"]
    l = df["low"]
    o = df["open"]
    v = df["volume"]
    
    # 1. CRYPTO NATIVE PILLARS
    df["log_ret"] = np.log(c / c.shift(1))
    df["rsi_14"] = rsi(c, 14)
    df["rsi_30"] = rsi(c, 30)
    df["rsi_90"] = rsi(c, 90)
    
    hl_ratio = np.log(h / (l + 1e-9))
    df["vol_parkinson_14"] = np.sqrt((1.0 / (4.0 * np.log(2.0))) * (hl_ratio**2).rolling(14).mean()) * np.sqrt(365)
    
    ema20 = c.ewm(span=20).mean()
    ema50 = c.ewm(span=50).mean()
    ema200 = c.ewm(span=200).mean()
    df["spread_50_200"] = (ema50 - ema200) / ema200
    df["dist_ema200"] = (c - ema200) / ema200
    
    # Amihud Illiquidity
    df["amihud_14"] = (np.abs(df["log_ret"]) / (c * v + 1e-9)).rolling(14).mean() * 1e9
    
    # Halving & Power Law
    days_genesis = (pd.to_datetime(df["date"]) - pd.Timestamp("2009-01-03")).dt.days
    df["halving_sin"] = np.sin(2 * np.pi * days_genesis / 1460.0)
    df["halving_cos"] = np.cos(2 * np.pi * days_genesis / 1460.0)
    power_law = -17.0 + 5.8 * np.log(days_genesis)
    df["power_law_res"] = np.log(c) - power_law
    
    # 2. GLOBAL MACRO EXPANSIONS
    # DXY (US Dollar Strength)
    df["dxy_ret_30"] = np.log(df["dxy"] / df["dxy"].shift(30))
    df["dxy_dist_ema50"] = (df["dxy"] - df["dxy"].ewm(span=50).mean()) / df["dxy"].ewm(span=50).mean()
    
    # Gold (Store of Value & Debasement Hedge)
    df["gold_ret_30"] = np.log(df["gold"] / df["gold"].shift(30))
    df["btc_gold_ratio"] = c / (df["gold"] + 1e-9)
    df["btc_gold_ratio_dist_ema50"] = (df["btc_gold_ratio"] - df["btc_gold_ratio"].ewm(span=50).mean()) / df["btc_gold_ratio"].ewm(span=50).mean()
    
    # Crude Oil (Inflation / Commodity cycle)
    df["oil_ret_30"] = np.log(df["oil"] / df["oil"].shift(30))
    
    # US 10-Year Treasury Yield (Risk-free hurdle & Fed discount rate)
    df["us10y_change_30"] = df["us10y"] - df["us10y"].shift(30)
    df["us10y_level"] = df["us10y"]
    
    # S&P 500 (Global Equity Risk Appetite)
    df["sp500_ret_30"] = np.log(df["sp500"] / df["sp500"].shift(30))
    # Rolling 30d correlation between BTC and SP500 returns
    df["btc_sp500_corr_30"] = df["log_ret"].rolling(30).corr(np.log(df["sp500"] / df["sp500"].shift(1)))
    
    # Forward Target log returns
    for h_days in [7, 14, 30, 90]:
        df[f"fwd_ret_{h_days}"] = np.log(c.shift(-h_days) / c)
        
    return df.dropna().reset_index(drop=True)

def main():
    df = build_merged_dataset()
    n = len(df)
    split_idx = int(n * 0.65)
    
    crypto_only_feats = [
        "rsi_14", "rsi_30", "rsi_90", "vol_parkinson_14",
        "spread_50_200", "dist_ema200", "amihud_14",
        "halving_sin", "halving_cos", "power_law_res"
    ]
    
    macro_feats = [
        "dxy_ret_30", "dxy_dist_ema50",
        "gold_ret_30", "btc_gold_ratio_dist_ema50",
        "oil_ret_30",
        "us10y_change_30", "us10y_level",
        "sp500_ret_30", "btc_sp500_corr_30"
    ]
    
    all_feats = crypto_only_feats + macro_feats
    
    train = df.iloc[:split_idx]
    
    print("=== MULTI-ASSET MACRO INTEGRATION BENCHMARK (10 YEARS) ===")
    print(f"Total Dataset: {n} hari ({df['date'].iloc[0]} s.d. {df['date'].iloc[-1]})")
    print(f"Fitur Crypto-Native: {len(crypto_only_feats)} | Fitur Makro: {len(macro_feats)} | Total: {len(all_feats)}")
    print(f"Out-of-Sample Test: {n - split_idx} hari ({df['date'].iloc[split_idx]} s.d. {df['date'].iloc[-1]})\n")
    
    # 1. Information Coefficients (Macro signals vs BTC 30d forward return)
    fwd_30 = df["fwd_ret_30"]
    fwd_90 = df["fwd_ret_90"]
    print("--- KORELASI STATISTIK / INFORMATION COEFFICIENT (IC) FAKTOR MAKRO ---")
    for mf in macro_feats:
        ic_30 = df[mf].corr(fwd_30)
        ic_90 = df[mf].corr(fwd_90)
        print(f"  {mf:<28}: IC (30h) = {ic_30:+.4f} | IC (90h) = {ic_90:+.4f}")
    print("-" * 75 + "\n")
    
    # 2. ExtraTrees Feature Importance across All Features
    et = ExtraTreesRegressor(n_estimators=100, max_depth=5, random_state=42)
    et.fit(train[all_feats].values, train["fwd_ret_30"].values)
    fi = pd.Series(et.feature_importances_, index=all_feats).sort_values(ascending=False)
    
    print("--- TOP 10 FEATURE IMPORTANCE (Crypto + Makro) ---")
    for f_name, imp in fi.head(10).items():
        tag = "[MAKRO]" if f_name in macro_feats else "[CRYPTO]"
        print(f"  {tag:<8} {f_name:<28}: {imp:.4f} ({imp*100:.1f}%)")
    print("-" * 75 + "\n")
    
    # 3. Out-of-Sample Comparison across Horizons
    for h in [7, 14, 30, 90]:
        test = df.iloc[split_idx:-h]
        y_tr = train[f"fwd_ret_{h}"].values
        y_te = test[f"fwd_ret_{h}"].values
        actual_dir = np.sign(y_te)
        
        # Model A: Crypto Only
        mA = ExtraTreesRegressor(n_estimators=80, max_depth=4, min_samples_leaf=20, random_state=42)
        mA.fit(train[crypto_only_feats].values, y_tr)
        predA = mA.predict(test[crypto_only_feats].values)
        accA = np.mean(np.sign(predA) == actual_dir)
        mapeA = np.mean(np.abs(test["close"] * np.exp(y_te) - test["close"] * np.exp(predA)) / (test["close"] * np.exp(y_te)))
        
        # Model B: Crypto + Macro
        mB = ExtraTreesRegressor(n_estimators=80, max_depth=4, min_samples_leaf=20, random_state=42)
        mB.fit(train[all_feats].values, y_tr)
        predB = mB.predict(test[all_feats].values)
        accB = np.mean(np.sign(predB) == actual_dir)
        mapeB = np.mean(np.abs(test["close"] * np.exp(y_te) - test["close"] * np.exp(predB)) / (test["close"] * np.exp(y_te)))
        
        delta_acc = accB - accA
        delta_mape = mapeB - mapeA
        
        print(f"[HORIZON {h:2d} HARI]")
        print(f"  Crypto-Only : Akurasi Arah = {accA:.2%} | Error MAPE = {mapeA:.2%}")
        print(f"  Crypto+Makro: Akurasi Arah = {accB:.2%} | Error MAPE = {mapeB:.2%}")
        status = "NAIK TINGKAT" if delta_acc > 0 else "NETRAL/TURUN"
        print(f"  --> Dampak Makro: Akurasi {delta_acc:+.2%} | MAPE {delta_mape:+.2%} ({status})\n")

if __name__ == "__main__":
    main()
