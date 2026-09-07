"""
STUDY CASE LOG: MULTI-ORIGIN WALK-FORWARD EVALUATION
Selects concrete dates across different market regimes (Bull peak, Bear crash, Sideways base, Recovery)
Shows:
  - Date & Spot (Titik Awal)
  - Horizon (7d, 14d, 30d)
  - Predicted Target & Direction
  - Actual Price & Direction
  - Direction Match (TRUE/FALSE) & Error %
"""
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor

DATA_PATH = r"C:\Mirza Personal\crypto quant\data\btc_coinbase_10y.csv"

def rsi(series, n):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = (-delta.clip(upper=0)).rolling(n).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def main():
    df = pd.read_csv(DATA_PATH, parse_dates=["open_time"]).sort_values("open_time").reset_index(drop=True)
    c = df["close"]
    
    df["log_ret"] = np.log(c / c.shift(1))
    df["rsi_14"] = rsi(c, 14)
    df["rsi_30"] = rsi(c, 30)
    df["rsi_90"] = rsi(c, 90)
    df["vol_7"] = df["log_ret"].rolling(7).std() * np.sqrt(365)
    df["vol_30"] = df["log_ret"].rolling(30).std() * np.sqrt(365)
    df["dist_ema50"] = (c - c.ewm(span=50).mean()) / c.ewm(span=50).mean()
    df["dist_ema200"] = (c - c.ewm(span=200).mean()) / c.ewm(span=200).mean()
    
    days_since_genesis = (df["open_time"] - pd.Timestamp("2009-01-03")).dt.days
    df["power_law_trend"] = -17.0 + 5.8 * np.log(days_since_genesis)
    df["power_law_residual"] = np.log(c) - df["power_law_trend"]
    
    feats = ["rsi_14", "rsi_30", "rsi_90", "vol_7", "vol_30", "dist_ema50", "dist_ema200", "power_law_residual"]
    
    # Concrete case study dates in test set:
    # 1. 2022-11-05 (Pre-FTX Crash, bear plunge)
    # 2. 2023-01-01 (Bear Market Bottom / Accumulation base)
    # 3. 2023-10-15 (Pre-ETF Rally breakout)
    # 4. 2024-03-10 (ATH break / Euphoria peak around $70k)
    # 5. 2024-08-01 (Summer unwind / Yen carry trade crash dip)
    case_dates = ["2022-11-05", "2023-01-01", "2023-10-15", "2024-03-10", "2024-08-01"]
    
    horizons = [7, 14, 30]
    
    results = []
    
    for d_str in case_dates:
        origin_idx = df[df["open_time"] == pd.Timestamp(d_str)].index
        if len(origin_idx) == 0:
            continue
        idx = origin_idx[0]
        
        train_df = df.iloc[:idx].dropna().copy()
        current_row = df.iloc[idx]
        current_spot = current_row["close"]
        current_feat = current_row[feats].values.reshape(1, -1)
        
        for h in horizons:
            # target in train
            y_train = np.log(train_df["close"].shift(-h) / train_df["close"]).dropna()
            X_train = train_df.iloc[:len(y_train)][feats].values
            
            # fit model
            if h in [7, 14]:
                model = Ridge(alpha=100.0).fit(X_train, y_train)
                pred_log_ret = model.predict(current_feat)[0]
                method = "Ridge-L2"
            else:
                # MoE 30d
                vol_med = np.median(train_df["vol_30"].dropna())
                low_m = train_df["vol_30"].iloc[:len(y_train)].values <= vol_med
                m_low = Ridge(alpha=50.0).fit(X_train[low_m], y_train[low_m])
                m_high = ExtraTreesRegressor(n_estimators=50, max_depth=4, random_state=42).fit(X_train[~low_m], y_train[~low_m])
                curr_vol = current_row["vol_30"]
                pred_log_ret = m_low.predict(current_feat)[0] if curr_vol <= vol_med else m_high.predict(current_feat)[0]
                method = "MoE-Hybrid"
                
            pred_price = current_spot * np.exp(pred_log_ret)
            pred_dir = "NAIK" if pred_log_ret > 0 else "TURUN"
            
            actual_row = df.iloc[idx + h]
            actual_price = actual_row["close"]
            actual_date = actual_row["open_time"].date()
            actual_log_ret = np.log(actual_price / current_spot)
            actual_dir = "NAIK" if actual_log_ret > 0 else "TURUN"
            
            correct_dir = (pred_dir == actual_dir)
            err_pct = (actual_price - pred_price) / actual_price
            
            results.append({
                "origin_date": d_str,
                "spot": current_spot,
                "horizon": f"{h}d",
                "method": method,
                "pred_price": pred_price,
                "pred_dir": pred_dir,
                "target_date": actual_date,
                "actual_price": actual_price,
                "actual_dir": actual_dir,
                "correct": "BENAR" if correct_dir else "SALAH",
                "err_pct": err_pct
            })
            
    res_df = pd.DataFrame(results)
    for d_str in case_dates:
        sub = res_df[res_df["origin_date"] == d_str]
        first_row = sub.iloc[0]
        print(f"\n=========================================================================================")
        print(f"STUDY CASE: Tanggal Titik Awal: {first_row['origin_date']} | Spot Awal: ${first_row['spot']:,.2f}")
        print(f"=========================================================================================")
        print(f"{'Horizon':<8} | {'Prediksi Harga':<16} | {'Arah Pred':<10} | {'Tgl Aktual':<12} | {'Harga Aktual':<16} | {'Arah Akt':<10} | {'Status'}")
        print("-" * 95)
        for _, r in sub.iterrows():
            print(f"{r['horizon']:<8} | ${r['pred_price']:<15,.2f} | {r['pred_dir']:<10} | {str(r['target_date']):<12} | ${r['actual_price']:<15,.2f} | {r['actual_dir']:<10} | {r['correct']}")

if __name__ == "__main__":
    main()
