# Models for Market Prediction & Quantitative Alpha Pipeline

End-to-end quantitative trading pipeline in Python covering technical factor engineering, alpha signal backtesting, cross-sectional portfolio construction, and statistical/machine learning return prediction.

---

### Architecture Overview

```
CSOT - Quant / Models-for-Market-Prediction
├── week1/                  # Alpha signal backtesting & statistical independence
│   ├── part1.py            # Simple Moving Average (SMA) baseline
│   ├── part2-1.py          # 3-Moving-Average crossover signal
│   ├── part2-2.py          # Fast/slow MA crossover
│   ├── part2-3.py          # 200-day trend-filter strategy
│   └── part3.md            # Empirical drawdown analysis and independence de-correlation
├── week2/                  # Feature engineering & ML return prediction
│   ├── ml_return_prediction.py  # Out-of-fold CV: Decision Tree, Random Forest, XGBoost
│   ├── data/
│   │   ├── linear_regression.py # OLS baseline on Garman-Klass volatility
│   │   ├── ridge_regression.py  # Ridge regularization with recency weighting
│   │   └── test.py              # Unsupervised GMM market regime detector & Huber loss
│   └── scoring/score.py    # Out-of-sample R^2 scoring metric
└── week3/                  # Daily cross-sectional alpha research challenge
    ├── notebooks/
    │   ├── attemp2.py      # Composite reversal + volume flow strategy (get_weights)
    │   └── main.ipynb      # Strategy research notebook & execution simulation
    └── src/utils.py        # Metrics: Net Sharpe Ratio, unit-capital & dollar-neutral checks
```

---

### 1. Alpha Signal Backtesting & Risk Diagnostics (`week1/`)
- **Signal Formulations:** Implementation and event-driven backtesting of dual moving-average crossovers (10-day, 20-day, 60-day) and 200-day long-term trend regimes.
- **Statistical Independence:** Explicit separation of overlapping observation windows ($753 \text{ days} / 200 \approx 3.7$ independent sections) to evaluate Sharpe ratio reliability and mitigate backtest overfitting.
- **Drawdown Control:** Comparative analysis of benchmark Buy-and-Hold drawdown ($-24.50\%$) vs. trend-filtered regimes ($-9.47\%$) against opportunity cost of holding cash.

### 2. Feature Engineering & Return Prediction (`week2/`)
- **Microstructure Features:**
  - **Garman-Klass Volatility Proxy:** Continuous intraday volatility estimate computed over 20-day rolling windows via High, Low, Open, and Close:
    $$\sigma_{GK}^2 = 0.5 \left(\ln \frac{H}{L}\right)^2 - (2\ln 2 - 1)\left(\ln \frac{C}{O}\right)^2$$
  - **52-Week High Distance:** Normalized distance from 252-day trailing peak price.
  - **Momentum Skip-5:** 60-day momentum shifted by 5 trading days to prevent 1-week microstructure mean-reversion contamination.
- **Regime Detection & Model Training:**
  - **Gaussian Mixture Models (GMM):** 2-component clustering isolating high-volatility vs. low-volatility regimes to trigger an adaptive circuit breaker.
  - **Cross-Validation:** Purged `TimeSeriesSplit` with a 10-day embargo gap.
  - **Model Architecture:** Comparative evaluation of Linear Regression, Ridge ($L_2$), Huber Robust Regressor, Decision Trees, Random Forest, and XGBoost.

### 3. Cross-Sectional Alpha Portfolio Engine (`week3/`)
- **Universe & Constraints:** Daily cross-sectional portfolio construction over 2,167 anonymized assets:
  - Unit-capital bounded: $\sum |w_i| \le 1.0$
  - Dollar-neutral: $|\sum w_i| \le 10^{-4}$
  - Single-name position constraint: $\max |w_i| \le 0.10$
  - Transaction cost friction: 1 bps (0.01%) per turnover
- **Composite Alpha Signal:** Equalized z-scored composite combining short-term mean-reversion (`trend_5_20`, `williams_r`, `relative_strength_index`) and volume flow accumulation (`on_balance_volume`, `volume`), traded across 20% tail quantiles with 10% EMA execution blending.
