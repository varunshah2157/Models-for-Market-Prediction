import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline

# =============================================================================
# 0. CONFIGURATION
# =============================================================================
TRAIN_PATH = 'train.csv'
TEST_PATH = 'test.csv'
SUBMISSION_PATH = 'my_submission.csv'

# Shrinkage scalar for normal days
BULL_MARKET_SCALAR = 0.10

# =============================================================================
# 1. FEATURE ENGINEERING
# =============================================================================


def engineer_features(df):
    data = df.copy()

    log_hl = np.log(data['High'] / data['Low']) ** 2
    log_co = np.log(data['Close'] / data['Open']) ** 2
    gk_daily = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
    data['gk_vol_20d'] = gk_daily.rolling(20).mean()

    rolling_252d_high = data['High'].rolling(252, min_periods=20).max()
    data['dist_52w_high'] = (
        data['Close'] - rolling_252d_high) / rolling_252d_high

    data['mom_skip5_60d'] = (data['Close'].shift(5) /
                             data['Close'].shift(65)) - 1

    data = data.bfill()
    keep_cols = ['id', 'Date', 'target', 'gk_vol_20d',
                 'dist_52w_high', 'mom_skip5_60d']
    final_cols = [c for c in keep_cols if c in data.columns]

    return data[final_cols]


# =============================================================================
# 2. LOAD & ENGINEER DATA
# =============================================================================
print("Loading data...")
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

print("Engineering features...")
df_train_features = engineer_features(train_df)

lookback_days = 260
history_for_test = train_df.tail(lookback_days)
stitched_test = pd.concat([history_for_test, test_df], ignore_index=True)

stitched_test_features = engineer_features(stitched_test)
df_test_features = stitched_test_features.iloc[-len(test_df):].copy()

feature_cols = ['dist_52w_high', 'gk_vol_20d', 'mom_skip5_60d']
X_train_full = df_train_features[feature_cols]
y_train_full = df_train_features['target']
X_test_features = df_test_features[feature_cols]

# =============================================================================
# 3. GMM REGIME DETECTION
# =============================================================================
print("\nTraining Unsupervised Regime Detector (GMM)...")
gmm_scaler = StandardScaler()
X_train_scaled = gmm_scaler.fit_transform(X_train_full)
X_test_scaled = gmm_scaler.transform(X_test_features)

gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
regime_preds_train = gmm.fit_predict(X_train_scaled)

# Dynamically figure out which regime is the "Safe/Bull" regime (Lower Volatility)
vol_regime_0 = X_train_full.loc[regime_preds_train == 0, 'gk_vol_20d'].mean()
vol_regime_1 = X_train_full.loc[regime_preds_train == 1, 'gk_vol_20d'].mean()

safe_regime_idx = 0 if vol_regime_0 < vol_regime_1 else 1

safe_prob_train = gmm.predict_proba(X_train_scaled)[:, safe_regime_idx]
safe_prob_test = gmm.predict_proba(X_test_scaled)[:, safe_regime_idx]

# =============================================================================
# 4. CROSS-VALIDATION (WITH HARD CIRCUIT BREAKER)
# =============================================================================
print("\nEvaluating Hard Circuit Breaker Model via TimeSeriesSplit...")

EMBARGO_DAYS = 10
tscv = TimeSeriesSplit(n_splits=5, gap=EMBARGO_DAYS)

model = Pipeline([
    ('scaler', StandardScaler()),
    # Massive alpha = 500.0 forces the model's base predictions to be incredibly flat and safe
    ('huber', HuberRegressor(epsilon=1.15, alpha=500.0, max_iter=1000))
])

fold_scores = []

for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_full)):
    X_train_fold, X_val_fold = X_train_full.iloc[train_idx], X_train_full.iloc[val_idx]
    y_train_fold, y_val_fold = y_train_full.iloc[train_idx], y_train_full.iloc[val_idx]

    val_safe_probs = safe_prob_train[val_idx]

    y_train_clipped = np.clip(
        y_train_fold,
        y_train_fold.mean() - 3 * y_train_fold.std(),
        y_train_fold.mean() + 3 * y_train_fold.std()
    )

    model.fit(X_train_fold, y_train_clipped)
    raw_preds = model.predict(X_val_fold)

    # 🎯 THE HARD CUTOFF
    # If the market is Safe (>50% probability), trade normally with the 10% scalar.
    # If it is Unsafe (<50% probability), predict exactly 0.0.
    final_cv_preds = np.where(
        val_safe_probs > 0.50,
        raw_preds * BULL_MARKET_SCALAR,
        0.0
    )

    score = r2_score(y_val_fold, final_cv_preds)
    fold_scores.append(score)
    print(f"Fold {fold+1} R²: {score:.4f}")

print(f"Average CV R²: {np.mean(fold_scores):.4f}")

# =============================================================================
# 5. RETRAIN & GENERATE FINAL SUBMISSION
# =============================================================================
print("\nRetraining and generating final predictions...")

y_train_full_clipped = np.clip(
    y_train_full,
    y_train_full.mean() - 3 * y_train_full.std(),
    y_train_full.mean() + 3 * y_train_full.std()
)

model.fit(X_train_full, y_train_full_clipped)
raw_test_predictions = model.predict(X_test_features)

# 🎯 APPLY HARD CUTOFF TO THE TEST SET
final_predictions = np.where(
    safe_prob_test > 0.50,
    raw_test_predictions * BULL_MARKET_SCALAR,
    0.0
)

submission = pd.DataFrame({
    'id': test_df['id'],
    'target': final_predictions
})

submission.to_csv(SUBMISSION_PATH, index=False)
print(
    f"🎉 Success! Hard Cutoff predictions saved to '{SUBMISSION_PATH}'. Submit this!")
