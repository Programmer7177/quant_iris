"""
BENCHMARK: CHINA MACRO, US DEBT STRESS, INFLATION EXPECTATIONS, AND CREDIT RISK
Tests whether adding:
  1. China PBOC liquidity & currency proxy (USD/CNY & FXI China Large-Cap)
  2. US Long-term Debt Stress & Treasury Duration (TLT 20+ Year Treasury)
  3. Market Inflation Expectations (TIP vs Treasuries)
  4. High-Yield Corporate Credit Spread (HYG credit risk appetite)
improves Bitcoin forecasting accuracy on 1-month and 3-month horizons.
"""
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

BTC_PATH = r"C:\Mirza Personal\crypto quant\data\btc_coinbase_10y.csv"
GLOBAL_MACRO_PATH = r"C:\Mirza Personal\crypto quant\data\global_macro_10y.csv"
ADV_MACRO_PATH = r"C:\Mirza Personal\crypto quant\data\advanced_macro_10y.csv"

def rsi(series, n):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = (-delta.clip(upper=0)).rolling(n).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def build_dataset():
    btc = pd.read_csv(BTC_PATH, parse_dates=["open_time"]).sort_values("open_time").reset_index(drop=True)
    btc["date"] = btc["open_time"].dt.strftime("%Y-%m-%d")
    
    macro1 = pd.read_csv(GLOBAL_MACRO_PATH, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    macro1["date"] = macro1["date"].dt.strftime("%Y-%m-%d")
    
    macro2 = pd.read_csv(ADV_MACRO_PATH, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    macro2["date"] = macro2["date"].dt.strftime("%Y-%m-%d")
    
    df = pd.merge(btc, macro1, on="date", how="left")
    df = pd.merge(df, macro2, on="date", how="left")
    
    fill_cols = ["dxy", "gold", "oil", "us10y", "sp500", "usdcny", "china_fxi", "infl_tip", "us_debt_tlt", "credit_hyg"]
    df[fill_cols] = df[fill_cols].ffill().bfill()
    
    c = df["close"]
    
    # Target 30d and 90d forward returns
    df["fwd_ret_30"] = np.log(c.shift(-30) / c)
    df["fwd_ret_90"] = np.log(c.shift(-90) / c)
    
    # 1. Base Crypto Features
    df["rsi_90"] = rsi(c, 90)
    df["dist_ema200"] = (c - c.ewm(span=200).mean()) / c.ewm(span=200).mean()
    days_genesis = (pd.to_datetime(df["date"]) - pd.Timestamp("2009-01-03")).dt.days
    df["halving_cos"] = np.cos(2 * np.pi * days_genesis / 1460.0)
    df["power_law_res"] = np.log(c) - (-17.0 + 5.8 * np.log(days_genesis))
    
    # 2. Existing Macro (DXY & Gold)
    df["dxy_dist_ema50"] = (df["dxy"] - df["dxy"].ewm(span=50).mean()) / df["dxy"].ewm(span=50).mean()
    df["gold_ret_30"] = np.log(df["gold"] / df["gold"].shift(30))
    df["us10y_level"] = df["us10y"]
    
    # 3. New Advanced Macro (China, Debt, Inflation, Credit)
    df["usdcny_ret_30"] = np.log(df["usdcny"] / df["usdcny"].shift(30)) # Yuan Devaluation
    df["china_fxi_ret_30"] = np.log(df["china_fxi"] / df["china_fxi"].shift(30)) # China Equities / Stimulus
    df["tlt_ret_30"] = np.log(df["us_debt_tlt"] / df["us_debt_tlt"].shift(30)) # US Bond Market Stress
    df["tip_tlt_ratio"] = df["infl_tip"] / (df["us_debt_tlt"] + 1e-9) # Inflation vs Nominal Bond Ratio
    df["tip_tlt_dist_ema50"] = (df["tip_tlt_ratio"] - df["tip_tlt_ratio"].ewm(span=50).mean()) / df["tip_tlt_ratio"].ewm(span=50).mean()
    df["credit_hyg_ret_30"] = np.log(df["credit_hyg"] / df["credit_hyg"].shift(30)) # Credit Appetite
    
    return df.dropna().reset_index(drop=True)

def main():
    df = build_dataset()
    n = len(df)
    split_idx = int(n * 0.65)
    
    train = df.iloc[:split_idx]
    
    adv_macro_feats = [
        "usdcny_ret_30", "china_fxi_ret_30",
        "tlt_ret_30", "tip_tlt_dist_ema50", "credit_hyg_ret_30"
    ]
    
    print("=== PENGUJIAN FAKTOR MAKRO BARU: CHINA, INFLATION RATIO, & US DEBT BOND STRESS ===")
    print(f"Total Dataset: {n} bar | Out-of-Sample Test: {n - split_idx} hari\n")
    
    print("--- INFORMATION COEFFICIENT (IC) FAKTOR MAKRO BARU ---")
    for f in adv_macro_feats:
        ic30 = df[f].corr(df["fwd_ret_30"])
        ic90 = df[f].corr(df["fwd_ret_90"])
        print(f"  {f:<24}: IC (30h) = {ic30:+.4f} | IC (90h) = {ic90:+.4f}")
    print("-" * 75 + "\n")
    
    # Feature Importance with all features
    all_f = ["halving_cos", "power_law_res", "rsi_90", "dist_ema200", "dxy_dist_ema50", "us10y_level", "gold_ret_30"] + adv_macro_feats
    et = ExtraTreesRegressor(n_estimators=100, max_depth=5, random_state=42)
    et.fit(train[all_f].values, train["fwd_ret_90"].values)
    fi = pd.Series(et.feature_importances_, index=all_f).sort_values(ascending=False)
    
    print("--- RANKING BOBOT FITUR PREDIKSI 90 HARI (CRYPTO + MAKRO LENGKAP) ---")
    for fname, imp in fi.items():
        tag = "[NEW MACRO]" if fname in adv_macro_feats else ("[MACRO]" if "dxy" in fname or "us10" in fname or "gold" in fname else "[CRYPTO]")
        print(f"  {tag:<12} {fname:<24}: {imp:.4f} ({imp*100:.1f}%)")
    print("-" * 75 + "\n")
    
    # Backtest Horizon 90d: Model Standard Macro vs Model dengan New Macro
    y_tr = train["fwd_ret_90"].values
    test = df.iloc[split_idx:-90]
    y_te = test["fwd_ret_90"].values
    actual_dir = np.sign(y_te)
    
    m_old = ExtraTreesRegressor(n_estimators=80, max_depth=4, min_samples_leaf=20, random_state=42)
    m_old.fit(train[["halving_cos", "power_law_res", "rsi_90", "dist_ema200", "dxy_dist_ema50", "us10y_level", "gold_ret_30"]].values, y_tr)
    pred_old = m_old.predict(test[["halving_cos", "power_law_res", "rsi_90", "dist_ema200", "dxy_dist_ema50", "us10y_level", "gold_ret_30"]].values)
    acc_old = np.mean(np.sign(pred_old) == actual_dir)
    mape_old = np.mean(np.abs(test["close"] * np.exp(y_te) - test["close"] * np.exp(pred_old)) / (test["close"] * np.exp(y_te)))
    
    m_new = ExtraTreesRegressor(n_estimators=80, max_depth=4, min_samples_leaf=20, random_state=42)
    m_new.fit(train[all_f].values, y_tr)
    pred_new = m_new.predict(test[all_f].values)
    acc_new = np.mean(np.sign(pred_new) == actual_dir)
    mape_new = np.mean(np.abs(test["close"] * np.exp(y_te) - test["close"] * np.exp(pred_new)) / (test["close"] * np.exp(y_te)))
    
    print("[HASIL EVALUASI OUT-OF-SAMPLE HORIZON 90 HARI]")
    print(f"  Model Makro Standar  : Akurasi Arah = {acc_old:.2%} | Error MAPE = {mape_old:.2%}")
    print(f"  Model + China & Debt : Akurasi Arah = {acc_new:.2%} | Error MAPE = {mape_new:.2%}")
    print(f"  --> Delta Akurasi Arah: {acc_new - acc_old:+.2%} | Delta Error: {mape_new - mape_old:+.2%}")

if __name__ == "__main__":
    main()
