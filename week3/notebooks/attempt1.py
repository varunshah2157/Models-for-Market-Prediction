import sys
from pathlib import Path
import pandas as pd
import numpy as np

current_file_path = Path(__file__).resolve()
REPO_ROOT = current_file_path.parent.parent
sys.path.append(str(REPO_ROOT / "src"))
utils = __import__("utils")


def get_weights(features: pd.DataFrame, today_universe: pd.Series) -> dict[str, float]:
    """
    Optimal Parameters: 5% Tail, 20% EMA Blend
    """
    TAIL_PCT = 0.05
    BLEND_FACTOR = 0.20

    if not hasattr(get_weights, "previous_weights"):
        get_weights.previous_weights = pd.Series(dtype=float)

    tradable_stocks = today_universe[today_universe == 1].index.tolist()
    if len(tradable_stocks) < 2 or len(features) == 0:
        return {s: 0.0 for s in tradable_stocks}

    latest = features.iloc[-1]

    # --- ALPHA MODEL ---
    vol = latest['volatility_20'].loc[tradable_stocks]
    williams = latest['williams_r'].loc[tradable_stocks]
    trend_short = latest['trend_1_3'].loc[tradable_stocks]

    df = pd.DataFrame({'vol': vol, 'williams': williams,
                      'trend_short': trend_short})
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.median(), inplace=True)

    ranks = df.rank(pct=True)
    raw_signal = ranks['vol'] - ranks['williams'] - ranks['trend_short']

    comp_rank = raw_signal.rank(pct=True)
    centered = comp_rank - 0.5

    # --- TAIL ALLOCATION ---
    cutoff = 0.5 - TAIL_PCT
    centered[centered.abs() < cutoff] = 0.0

    longs = centered[centered > 0]
    shorts = centered[centered < 0]

    target_weights = pd.Series(0.0, index=tradable_stocks)
    if len(longs) > 0 and len(shorts) > 0:
        long_weights = longs / longs.sum() * 0.5
        short_weights = shorts / shorts.abs().sum() * 0.5
        target_weights.update(long_weights)
        target_weights.update(short_weights)

    # --- EXECUTION BUFFER ---
    prev_weights = get_weights.previous_weights
    all_active = list(set(target_weights.index).union(set(prev_weights.index)))

    t_weights = target_weights.reindex(all_active).fillna(0.0)
    p_weights = prev_weights.reindex(all_active).fillna(0.0)

    tradable_mask = today_universe.reindex(all_active).fillna(0) == 1
    p_weights = p_weights.where(tradable_mask, 0.0)

    final_weights = (BLEND_FACTOR * t_weights) + \
        ((1.0 - BLEND_FACTOR) * p_weights)

    # --- CAPITAL LEAK FIX ---
    final_weights = final_weights.clip(lower=-0.1, upper=0.1)

    l_sum = final_weights[final_weights > 0].sum()
    s_sum = final_weights[final_weights < 0].abs().sum()

    target_leg = min(l_sum, s_sum)
    if l_sum > 0:
        final_weights[final_weights > 0] *= (target_leg / l_sum)
    if s_sum > 0:
        final_weights[final_weights < 0] *= (target_leg / s_sum)

    gross = final_weights.abs().sum()
    if 0 < gross < 0.999:
        final_weights *= (0.999 / gross)
        final_weights = final_weights.clip(lower=-0.1, upper=0.1)

    get_weights.previous_weights = final_weights[final_weights != 0]
    return final_weights[final_weights != 0].to_dict()


def main():
    print("Loading full datasets...")
    features, universe, returns = utils.load_data()

    print("Running Backtest on full dataset (This will take ~35 minutes)...")
    weights = utils.backtest_strategy(
        get_weights, features, universe, show_progress=True)

    print("\nValidating Constraints:")
    checks = utils.validate_weights(weights, universe)
    for name, passed in checks.items():
        print(f"{name:20s} {'PASS' if passed else 'FAIL'}")

    print("\nScoring Strategy (Training Period 2005-2019):")
    metrics = utils.summarize_performance(weights, returns)
    for k, v in metrics.items():
        print(f"{k:16s} {v:.4f}")


if __name__ == "__main__":
    main()
