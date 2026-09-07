"""
BENCHMARKING LIQUID VS ILLIQUID SUPPLY & WALLET DRIFT
Tests 3 Empirical On-Chain Flow & Supply Squeeze Metrics (2017 - 2026):
  1. Coinbase Institutional Accumulation Proxy (Coinbase Premium Spread).
  2. Spot ETF Cold Wallet Absorption (IBIT/FBTC Net Inflow Volume).
  3. Mining / Network Velocity Stress (Hashrate Growth vs Active Transactions).

Evaluates Information Coefficient (IC) and Directional Accuracy on 30d & 90d Horizons.
"""
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge

DATA_DIR = r"C:\Mirza Personal\crypto quant\data"
BTC_PATH = os.path.join(DATA_DIR, "btc_coinbase_10y.csv")
PREM_PATH = os.path.join(DATA_DIR, "coinbase_premium_index.csv")
ETF_PATH = os.path.join(DATA_DIR, "btc_spot_etf_flows.csv")

def rsi(series, n):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = (-delta.clip(upper=0)).rolling(n).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def main():
    btc = pd.read_csv(BTC_PATH, parse_dates=["open_time"]).sort_values("open_time").reset_index(drop=True)
    btc["date"] = btc["open_time"].dt.strftime("%Y-%m-%d")
    
    prem = pd.read_csv(PREM_PATH)
    etf = pd.read_csv(ETF_PATH)
    etf = etf.rename(columns={"Unnamed: 0": "date"}) if "Unnamed: 0" in etf.columns else etf
    
    df = pd.merge(btc, prem[["date", "premium_bps", "premium_bps_ema7", "premium_bps_ema30"]], on="date", how="left")
    df = pd.merge(df, etf[["date", "etf_total_dollar_vol"]], on="date", how="left")
    
    df["premium_bps"] = df["premium_bps"].ffill().bfill()
    df["premium_bps_ema30"] = df["premium_bps_ema30"].ffill().bfill()
    # ETF volume before 2024 is 0 (pre-launch)
    df["etf_total_dollar_vol"] = df["etf_total_dollar_vol"].fillna(0.0)
    
    c = df["close"]
    
    # Supply Absorption Metrics:
    # 1. Cumulative Institutional Squeeze: Rolling 30d sum of positive Coinbase Premium
    df["inst_cold_absorption_30"] = df["premium_bps"].clip(lower=0).rolling(30).mean()
    # 2. Hot-to-Cold Net Drain Proxy: Premium EMA7 vs EMA30 divergence
    df["wallet_drift_divergence"] = df["premium_bps_ema7"] - df["premium_bps_ema30"]
    # 3. ETF Absorption Intensity: Daily ETF Dollar Volume / (BTC Market Cap proxy)
    df["etf_absorption_ratio"] = df["etf_total_dollar_vol"] / (c * df["volume"] + 1e-9)
    df["etf_absorption_ratio_ema7"] = df["etf_absorption_ratio"].ewm(span=7).mean()
    
    # Macro & Technical Baselines
    df["rsi_90"] = rsi(c, 90)
    days_genesis = (pd.to_datetime(df["date"]) - pd.Timestamp("2009-01-03")).dt.days
    df["halving_cos"] = np.cos(2 * np.pi * days_genesis / 1460.0)
    df["power_law_res"] = np.log(c) - (-17.0 + 5.8 * np.log(days_genesis))
    
    # Target forward returns
    for h in [30, 90]:
        df[f"fwd_ret_{h}"] = np.log(c.shift(-h) / c)
        
    df = df.dropna().reset_index(drop=True)
    n = len(df)
    split_idx = int(n * 0.65)
    train = df.iloc[:split_idx]
    
    print("=== PENGUJIAN KUANTITATIF: COLD/HOT WALLET & SUPPLY ABSORPTION ===")
    print(f"Total Dataset: {n} bar ({df['date'].iloc[0]} s.d. {df['date'].iloc[-1]})\n")
    
    flow_feats = ["inst_cold_absorption_30", "wallet_drift_divergence", "etf_absorption_ratio_ema7"]
    
    print("--- KORELASI STATISTIK (IC) ALIRAN SUPPLY ABSORPTION ---")
    for f in flow_feats:
        ic30 = df[f].corr(df["fwd_ret_30"])
        ic90 = df[f].corr(df["fwd_ret_90"])
        print(f"  {f:<28}: IC (30h) = {ic30:+.4f} | IC (90h) = {ic90:+.4f}")
    print("-" * 75 + "\n")
    
    base_f = ["rsi_90", "halving_cos", "power_law_res"]
    flow_augmented_f = base_f + flow_feats
    
    # Evaluate Out-of-Sample Performance
    for h in [30, 90]:
        test = df.iloc[split_idx:-h]
        y_tr = train[f"fwd_ret_{h}"].values
        y_te = test[f"fwd_ret_{h}"].values
        actual_dir = np.sign(y_te)
        
        # Model Baseline
        m_base = ExtraTreesRegressor(n_estimators=80, max_depth=4, min_samples_leaf=20, random_state=42)
        m_base.fit(train[base_f].values, y_tr)
        p_base = m_base.predict(test[base_f].values)
        acc_base = np.mean(np.sign(p_base) == actual_dir)
        mape_base = np.mean(np.abs(test["close"] * np.exp(y_te) - test["close"] * np.exp(p_base)) / (test["close"] * np.exp(y_te)))
        
        # Model + Supply Absorption Flows
        m_flow = ExtraTreesRegressor(n_estimators=80, max_depth=4, min_samples_leaf=20, random_state=42)
        m_flow.fit(train[flow_augmented_f].values, y_tr)
        p_flow = m_flow.predict(test[flow_augmented_f].values)
        acc_flow = np.mean(np.sign(p_flow) == actual_dir)
        mape_flow = np.mean(np.abs(test["close"] * np.exp(y_te) - test["close"] * np.exp(p_flow)) / (test["close"] * np.exp(y_te)))
        
        print(f"[HORIZON {h:2d} HARI]")
        print(f"  Tanpa Supply Absorption  : Akurasi Arah = {acc_base:.2%} | Error MAPE = {mape_base:.2%}")
        print(f"  + Cold Wallet Absorption : Akurasi Arah = {acc_flow:.2%} | Error MAPE = {mape_flow:.2%}")
        print(f"  --> Delta: Akurasi {acc_flow - acc_base:+.2%} | MAPE {mape_flow - mape_base:+.2%}\n")

if __name__ == "__main__":
    main()
