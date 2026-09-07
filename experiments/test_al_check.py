import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, f1_score
from forecast_engine import load_data, fit_hmm, FEAT_SHORT, FEAT_MEDIUM

def run_test():
    df = load_data()
    log_rets = np.log(df['close'] / df['close'].shift(1)).fillna(0).values
    states, _, _ = fit_hmm(log_rets)
    df['hmm_state'] = states
    df['hmm_norm'] = states.astype(float) / 3.0

    print("Data loaded. Total rows:", len(df))

if __name__ == '__main__':
    run_test()
