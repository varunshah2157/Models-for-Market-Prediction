import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

plt.style.use('seaborn-v0_8-whitegrid')

# Load the raw data
print("Loading data...")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

print(f"Training data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}")
print(train_df.head())


def engineer_features(df):
    """
    Apply advanced quantitative feature engineering.
    """
    data = df.copy()

    # Baseline daily return for internal calculations
    daily_ret = data['Close'].pct_change()

    # Garman-Klass Volatility Proxy (20-day rolling)
    # Uses High, Low, Open, and Close for a truer variance estimate
    log_hl = np.log(data['High'] / data['Low'])
    log_co = np.log(data['Close'] / data['Open'])
    gk_variance = 0.5 * (log_hl ** 2) - (2 * np.log(2) - 1) * (log_co ** 2)
    data['feat_garman_klass_vol_20d'] = np.sqrt(
        gk_variance.rolling(window=20).mean())

    # -------------------------------------------------------------
    # 🧹 CLEANUP: Drop raw absolute prices
    # -------------------------------------------------------------
    columns_to_drop = ['Open', 'High', 'Low', 'Close', 'Volume', 'Date']
    data = data.drop(columns=columns_to_drop, errors='ignore')

    return data


print("Processing new quantitative features...")
X_train_full = engineer_features(train_df)
X_test = engineer_features(test_df)

# Isolate the target variable
y_train_full = X_train_full['target']

# Isolate just the predictive features
feature_cols = [
    'feat_garman_klass_vol_20d'
]
X_train_features = X_train_full[feature_cols]
X_test_features = X_test[feature_cols]

print(
    f"Using {len(feature_cols)} features. Check for NaNs and run your pipeline!")


split_index = [int(len(X_train_features) * (0.6+i*0.01)) for i in range(0, 31)]

X_train = [X_train_features.iloc[:split_index[i]] for i in range(0, 31)]
y_train = [y_train_full.iloc[:split_index[i]] for i in range(0, 31)]

X_val = [X_train_features.iloc[split_index[i]:] for i in range(0, 31)]
y_val = [y_train_full.iloc[split_index[i]:] for i in range(0, 31)]

for i in range(0, 31):
    print(
        f"Training on {len(X_train[i])} rows, Validating on {len(X_val[i])} rows.")

# Define the pipeline
model = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('regressor', LinearRegression())
])

best_val_r2 = -float('inf')
best_split_pct = 0

print("Training Linear Regression model...")
print("-" * 50)
for i in range(0, 31):
    # Train the model
    model.fit(X_train[i], y_train[i])

    # Predict on validation fold
    val_predictions = model.predict(X_val[i])

    # Evaluate
    val_r2 = r2_score(y_val[i], val_predictions)

    # Calculate what percentage this split represents (for clean printing)
    current_split_pct = 60 + i

    print(f"Split {current_split_pct}% | R²: {val_r2:.6f}")

    # Track the best score
    if val_r2 > best_val_r2:
        best_val_r2 = val_r2
        best_split_pct = current_split_pct
print("-" * 50)
print(
    f"🏆 BEST R²: {best_val_r2:.6f} (achieved at a {best_split_pct}% train split)")

# Retrain on the FULL training set
model.fit(X_train_features, y_train_full)

# Predict on the hidden test set
test_predictions = model.predict(X_test_features)

# Format the submission exactly as required
submission = pd.DataFrame({
    'id': test_df['id'],
    'target': test_predictions
})

# Save to CSV
submission_filename = 'my_submission.csv'
submission.to_csv(submission_filename, index=False)

print(f"🎉 Success! Predictions saved to '{submission_filename}'.")
