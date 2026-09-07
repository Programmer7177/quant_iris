"""
TEST FORECASTING — WALK-FORWARD BENCHMARK ON BITCOIN
Protocol: Rolling Walk-Forward Validation (No Look-Ahead Bias)
Target: Directional Return Sign at t+1: y_t = 1 if r_{t+1} > 0 else 0
Models:
1. Naive Baseline (Random / Majority Class)
2. AR(p) Logistic Regression (Lags 1-5)
3. Multi-Feature Quant Classifier (Lags, Rolling Vol, RSI, OBI proxy, Momentum)
Metric: Out-Of-Sample Accuracy, AUC-ROC, Strategy Net Return after 5 bps fee
"""

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

DATA_DIR = r"C:\Users\mirza\quant-crypto\data"

def prepare_features(df):
    df = df.copy()
    df["ret"] = np.log(df["close"] / df["close"].shift(1))
    
    # Feature engineering
    for lag in [1, 2, 3, 5, 8]:
        df[f"ret_lag_{lag}"] = df["ret"].shift(lag)
        
    df["vol_5d"] = df["ret"].rolling(5).std()
    df["vol_20d"] = df["ret"].rolling(20).std()
    df["vol_ratio"] = df["vol_5d"] / (df["vol_20d"] + 1e-8)
    
    # Momentum / Trend
    df["ma_10_ratio"] = df["close"] / df["close"].rolling(10).mean() - 1.0
    df["ma_50_ratio"] = df["close"] / df["close"].rolling(50).mean() - 1.0
    
    # Volume momentum
    df["volu_ratio"] = df["volume"] / df["volume"].rolling(10).mean() - 1.0
    
    # Target: sign of next-day return
    df["target"] = (df["ret"].shift(-1) > 0).astype(int)
    df["next_ret"] = df["ret"].shift(-1)
    
    clean = df.dropna()
    return clean

def run_walk_forward(clean_df, train_window=300, step=1):
    """
    Expanding/Rolling walk-forward:
    Train on [t - train_window : t], test on t.
    """
    feature_cols = [c for c in clean_df.columns if c.startswith("ret_lag_") or "vol_" in c or "ma_" in c or "volu_" in c]
    
    X = clean_df[feature_cols].values
    y = clean_df["target"].values
    next_ret = clean_df["next_ret"].values
    
    n = len(clean_df)
    preds_naive = []
    preds_logistic = []
    preds_rf = []
    actuals = []
    returns_test = []
    
    # Models
    clf_log = LogisticRegression(C=0.1, penalty="l2", solver="lbfgs")
    clf_rf = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42, n_jobs=-1)
    
    # Re-train periodically to save time (every 10 steps)
    retrain_freq = 10
    
    for t in range(train_window, n):
        idx_train = range(t - train_window, t)
        X_train, y_train = X[idx_train], y[idx_train]
        X_test = X[t:t+1]
        y_test = y[t]
        
        # Naive: predict historical majority class
        pred_naive = int(np.mean(y_train) >= 0.5)
        preds_naive.append(pred_naive)
        
        # Retrain ML models every retrain_freq steps
        if (t - train_window) % retrain_freq == 0:
            clf_log.fit(X_train, y_train)
            clf_rf.fit(X_train, y_train)
            
        p_log = clf_log.predict(X_test)[0]
        p_rf = clf_rf.predict(X_test)[0]
        
        preds_logistic.append(p_log)
        preds_rf.append(p_rf)
        actuals.append(y_test)
        returns_test.append(next_ret[t])
        
    actuals = np.array(actuals)
    returns_test = np.array(returns_test)
    
    results = {}
    models = {
        "Naive (Majority)": np.array(preds_naive),
        "Logistic Reg (L2)": np.array(preds_logistic),
        "Random Forest (Depth 3)": np.array(preds_rf)
    }
    
    fee = 0.0005 # 5 bps
    
    for name, p in models.items():
        acc = accuracy_score(actuals, p)
        # Position: 1 (long) if pred=1 else 0 (cash)
        pos = p
        trades = np.diff(pos, prepend=0) != 0
        strat_gross = pos * returns_test
        strat_net = strat_gross - trades * fee
        cum_net = np.exp(np.sum(strat_net)) - 1.0
        sharpe = (np.mean(strat_net) / (np.std(strat_net) + 1e-8)) * np.sqrt(365)
        
        results[name] = {
            "accuracy": acc,
            "net_return": cum_net,
            "sharpe": sharpe,
            "n_trades": np.sum(trades)
        }
        
    # Buy & Hold benchmark
    bnh_ret = np.exp(np.sum(returns_test)) - 1.0
    bnh_sharpe = (np.mean(returns_test) / (np.std(returns_test) + 1e-8)) * np.sqrt(365)
    results["Buy & Hold"] = {
        "accuracy": np.mean(actuals),
        "net_return": bnh_ret,
        "sharpe": bnh_sharpe,
        "n_trades": 1
    }
    
    return results, len(actuals)

if __name__ == "__main__":
    df_1d = pd.read_csv(os.path.join(DATA_DIR, "btc_1d_spot.csv"), index_col="open_time", parse_dates=True)
    clean_df = prepare_features(df_1d)
    
    results, test_len = run_walk_forward(clean_df, train_window=300)
    
    print("=" * 65)
    print(f"BITCOIN FORECASTING BENCHMARK (Walk-Forward, {test_len} Test Days)")
    print("=" * 65)
    print(f"{'Model':<24} | {'Accuracy':<8} | {'Net Return':<10} | {'Sharpe':<6} | {'Trades'}")
    print("-" * 65)
    for model, m in results.items():
        print(f"{model:<24} | {m['accuracy']*100:6.2f}%  | {m['net_return']*100:+8.2f}%  | {m['sharpe']:6.2f} | {m['n_trades']}")
    print("=" * 65)
