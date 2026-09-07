"""
HYBRID FORECASTING ARCHITECTURE FOR BITCOIN
Compares 4 Paradigms under Strict Walk-Forward Protocol:
1. Baseline: Pure Lagged Technicals
2. Spectral / Fourier Frequency Decomposition (MoFE concept, Liu & Sun 2026)
3. On-Chain / Macro Hybrid (Technicals + Funding Rate + MVRV + Vol Ratio)
4. Regime-Gated Mixture-of-Experts (MoE): Separate sub-models for Low-Vol vs High-Vol regimes
"""

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score

DATA_DIR = r"C:\Users\mirza\quant-crypto\data"

def prepare_hybrid_dataset():
    df_1d = pd.read_csv(os.path.join(DATA_DIR, "btc_1d_spot.csv"), index_col="open_time", parse_dates=True)
    funding_df = pd.read_csv(os.path.join(DATA_DIR, "btc_funding_rates.csv"), index_col="fundingTime", parse_dates=True)
    
    # Base returns
    df = df_1d.copy()
    df["ret"] = np.log(df["close"] / df["close"].shift(1))
    
    # 1. Technical features
    for lag in [1, 2, 3, 5]:
        df[f"ret_lag_{lag}"] = df["ret"].shift(lag)
        
    df["vol_5d"] = df["ret"].rolling(5).std()
    df["vol_20d"] = df["ret"].rolling(20).std()
    df["vol_ratio"] = df["vol_5d"] / (df["vol_20d"] + 1e-8)
    
    # 2. Fourier / Spectral dominant cycle components (MoFE concept)
    # FFT on rolling 30-day window
    def rolling_fft_dominant_freq(series, window=30):
        out = np.zeros(len(series))
        vals = series.values
        for i in range(window, len(series)):
            chunk = vals[i-window:i]
            fft_vals = np.abs(np.fft.rfft(chunk - np.mean(chunk)))
            out[i] = np.argmax(fft_vals[1:]) + 1 # Dominant harmonic
        return out
        
    df["fft_dominant_mode"] = rolling_fft_dominant_freq(df["close"], window=30)
    
    # 3. On-Chain / Macro features
    # VWAP MVRV proxy
    df["vwap_200"] = (df["close"] * df["volume"]).rolling(200).sum() / (df["volume"].rolling(200).sum() + 1e-8)
    df["mvrv_proxy"] = df["close"] / df["vwap_200"]
    
    # Resample daily funding rate
    daily_fr = funding_df["fundingRate"].resample("D").mean()
    df["funding_rate"] = daily_fr
    df["funding_rate"] = df["funding_rate"].ffill().bfill()
    
    # 4. Target: Direction at t+1
    df["target"] = (df["ret"].shift(-1) > 0).astype(int)
    df["next_ret"] = df["ret"].shift(-1)
    
    # Regime indicator: 1 if high vol, 0 if low vol
    df["regime_high_vol"] = (df["vol_5d"] > df["vol_20d"]).astype(int)
    
    return df.dropna()

def run_hybrid_walk_forward(df, train_window=250):
    feature_tech = ["ret_lag_1", "ret_lag_2", "ret_lag_3", "ret_lag_5", "vol_ratio"]
    feature_spectral = feature_tech + ["fft_dominant_mode"]
    feature_multimodal = feature_spectral + ["mvrv_proxy", "funding_rate"]
    
    y = df["target"].values
    next_ret = df["next_ret"].values
    regimes = df["regime_high_vol"].values
    
    n = len(df)
    preds = {
        "1. Baseline (Technicals)": [],
        "2. Spectral (Fourier+Tech)": [],
        "3. Multimodal (OnChain+Fund)": [],
        "4. Regime-Gated MoE": []
    }
    actuals = []
    returns_test = []
    
    # Initialize sub-models for MoE
    moe_low = LogisticRegression(C=0.1)
    moe_high = GradientBoostingClassifier(n_estimators=30, max_depth=2, random_state=42)
    
    clf_base = LogisticRegression(C=0.1)
    clf_spec = LogisticRegression(C=0.1)
    clf_multi = GradientBoostingClassifier(n_estimators=40, max_depth=2, random_state=42)
    
    for t in range(train_window, n):
        idx_train = range(t - train_window, t)
        
        y_train = y[idx_train]
        actuals.append(y[t])
        returns_test.append(next_ret[t])
        
        # Periodic refit every 7 days
        if (t - train_window) % 7 == 0:
            # Model 1
            X1_train = df[feature_tech].iloc[idx_train].values
            clf_base.fit(X1_train, y_train)
            
            # Model 2
            X2_train = df[feature_spectral].iloc[idx_train].values
            clf_spec.fit(X2_train, y_train)
            
            # Model 3
            X3_train = df[feature_multimodal].iloc[idx_train].values
            clf_multi.fit(X3_train, y_train)
            
            # Model 4: Regime MoE
            r_train = regimes[idx_train]
            idx_low = np.where(r_train == 0)[0]
            idx_high = np.where(r_train == 1)[0]
            
            if len(idx_low) > 30:
                moe_low.fit(X3_train[idx_low], y_train[idx_low])
            if len(idx_high) > 30:
                moe_high.fit(X3_train[idx_high], y_train[idx_high])

        # Inference at step t
        x1_t = df[feature_tech].iloc[t:t+1].values
        x2_t = df[feature_spectral].iloc[t:t+1].values
        x3_t = df[feature_multimodal].iloc[t:t+1].values
        
        p1 = clf_base.predict(x1_t)[0]
        p2 = clf_spec.predict(x2_t)[0]
        p3 = clf_multi.predict(x3_t)[0]
        
        # MoE gated routing
        current_regime = regimes[t]
        if current_regime == 0 and hasattr(moe_low, "classes_"):
            p4 = moe_low.predict(x3_t)[0]
        elif current_regime == 1 and hasattr(moe_high, "classes_"):
            p4 = moe_high.predict(x3_t)[0]
        else:
            p4 = p3
            
        preds["1. Baseline (Technicals)"].append(p1)
        preds["2. Spectral (Fourier+Tech)"].append(p2)
        preds["3. Multimodal (OnChain+Fund)"].append(p3)
        preds["4. Regime-Gated MoE"].append(p4)

    actuals = np.array(actuals)
    returns_test = np.array(returns_test)
    fee = 0.0005 # 5 bps fee
    
    summary = {}
    for name, p_arr in preds.items():
        p_arr = np.array(p_arr)
        acc = accuracy_score(actuals, p_arr)
        trades = np.diff(p_arr, prepend=0) != 0
        strat_net = (p_arr * returns_test) - (trades * fee)
        cum_net = np.exp(np.sum(strat_net)) - 1.0
        sharpe = (np.mean(strat_net) / (np.std(strat_net) + 1e-8)) * np.sqrt(365)
        
        summary[name] = {
            "accuracy": acc,
            "net_return": cum_net,
            "sharpe": sharpe,
            "trades": np.sum(trades)
        }
        
    return summary, len(actuals)

if __name__ == "__main__":
    df = prepare_hybrid_dataset()
    results, test_days = run_hybrid_walk_forward(df, train_window=250)
    
    print("=" * 70)
    print(f"HYBRID FORECASTING COMPARISON ON BITCOIN (Walk-Forward, {test_days} Days)")
    print("=" * 70)
    print(f"{'Method / Architecture':<28} | {'Accuracy':<8} | {'Net Return':<10} | {'Sharpe':<6} | {'Trades'}")
    print("-" * 70)
    for model, m in results.items():
        print(f"{model:<28} | {m['accuracy']*100:6.2f}%  | {m['net_return']*100:+8.2f}%  | {m['sharpe']:6.2f} | {m['trades']}")
    print("=" * 70)
