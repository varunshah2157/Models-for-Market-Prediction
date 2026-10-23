"""
ml_return_prediction.py
Machine Learning Return Prediction using Decision Trees, Random Forest, and XGBoost.
Evaluates out-of-fold R^2 performance using purged TimeSeriesSplit cross-validation.
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

def engineer_features(df):
    data = df.copy()
    log_hl = np.log(data['High'] / data['Low']) ** 2
    log_co = np.log(data['Close'] / data['Open']) ** 2
    gk_daily = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
    data['gk_vol_20d'] = gk_daily.rolling(20).mean()

    rolling_252d_high = data['High'].rolling(252, min_periods=20).max()
    data['dist_52w_high'] = (data['Close'] - rolling_252d_high) / rolling_252d_high
    data['mom_skip5_60d'] = (data['Close'].shift(5) / data['Close'].shift(65)) - 1
    data = data.bfill()

    feature_cols = ['dist_52w_high', 'gk_vol_20d', 'mom_skip5_60d']
    return data[feature_cols]

def evaluate_models(train_path='train.csv'):
    df = pd.read_csv(train_path)
    X = engineer_features(df)
    y = df['target']

    tscv = TimeSeriesSplit(n_splits=5, gap=10)
    scaler = StandardScaler()

    models = {
        'Decision Tree': DecisionTreeRegressor(max_depth=5, min_samples_leaf=20, random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=6, min_samples_leaf=15, random_state=42, n_jobs=-1)
    }
    if HAS_XGBOOST:
        models['XGBoost'] = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=42)

    results = {}
    for name, model in models.items():
        fold_scores = []
        for train_idx, val_idx in tscv.split(X):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            X_tr_s = scaler.fit_transform(X_tr)
            X_val_s = scaler.transform(X_val)

            model.fit(X_tr_s, y_tr)
            preds = model.predict(X_val_s)
            fold_scores.append(r2_score(y_val, preds))

        results[name] = np.mean(fold_scores)
        print(f"{name} Mean CV R^2: {results[name]:.6f}")

    return results

if __name__ == '__main__':
    evaluate_models('C:/Users/varun/OneDrive/Desktop/IIT/CSOT - Quant/week2/data/train.csv')
