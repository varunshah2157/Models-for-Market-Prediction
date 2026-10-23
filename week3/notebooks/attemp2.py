import sys
from pathlib import Path
import pandas as pd
import numpy as np

current_file_path = Path(__file__).resolve()
REPO_ROOT = current_file_path.parent.parent
sys.path.append(str(REPO_ROOT / "src"))
utils = __import__("utils")


# Signed feature weights for the composite signal.
#   negative -> long LOW values (reversal: fade recent strength)
#   positive -> long HIGH values (volume / flow accumulation)
# Chosen from measured IC (research_diagnostics) + de-correlation + split-half
# stability. The two volume features are the independent, positively-signed
# second bet that lifted Sharpe the most.
FEATS = {
    "trend_5_20": -1, "trend_20_60": -1, "williams_r": -1,
    "relative_strength_index": -1, "average_true_range": -1,
    "accumulation_distribution_index": -1, "ease_of_movement": -1,
    "on_balance_volume": +1, "volume": +1,
}
TAIL_PCT = 0.20    # trade the top/bottom 20% of the composite score
# EMA execution blend: ~10-day holding (matched to IC horizon)
BLEND = 0.10


def _equalize_series(f: pd.Series) -> pd.Series:
    """Scale long and short legs to the smaller leg -> dollar neutral."""
    pos, neg = f > 0, f < 0
    l_sum = f[pos].sum()
    s_sum = -f[neg].sum()
    leg = min(l_sum, s_sum)
    if l_sum > 0:
        f.loc[pos] = f.loc[pos] * (leg / l_sum)
    if s_sum > 0:
        f.loc[neg] = f.loc[neg] * (leg / s_sum)
    return f


def get_weights(features: pd.DataFrame, today_universe: pd.Series) -> dict[str, float]:
    """
    Cross-sectional reversal + volume composite, z-scored over today's tradable
    universe, traded in the 20% tails, smoothed with a 10%/90% EMA execution
    blend, then made dollar-neutral and constraint-compliant.
    """
    if not hasattr(get_weights, "previous_weights"):
        get_weights.previous_weights = pd.Series(dtype=float)

    tradable = today_universe[today_universe == 1].index
    if len(tradable) < 2 or len(features) == 0:
        get_weights.previous_weights = pd.Series(dtype=float)
        return {s: 0.0 for s in tradable}

    # feature values at t-1 (last row strictly before t)
    latest = features.iloc[-1]

    # --- composite signal: sum of signed cross-sectional z-scores ---
    composite = pd.Series(0.0, index=tradable)
    for feat, w in FEATS.items():
        x = latest[feat].reindex(tradable).replace([np.inf, -np.inf], np.nan)
        sd = x.std()
        if sd and not np.isnan(sd):
            z = ((x - x.mean()) / sd).fillna(0.0)
        else:
            z = pd.Series(0.0, index=tradable)
        composite = composite.add(z * w, fill_value=0.0)

    # --- keep only the extreme tails ---
    centered = composite.rank(pct=True) - 0.5
    centered[centered.abs() < (0.5 - TAIL_PCT)] = 0.0
    longs = centered[centered > 0]
    shorts = centered[centered < 0]

    target = pd.Series(0.0, index=tradable)
    if len(longs) > 0 and len(shorts) > 0:
        target.update(longs / longs.sum() * 0.5)
        target.update(shorts / shorts.abs().sum() * 0.5)

    # --- EMA execution blend (previous book masked to today's universe) ---
    prev = get_weights.previous_weights
    active = target.index.union(prev.index)
    t_w = target.reindex(active).fillna(0.0)
    p_w = prev.reindex(active).fillna(0.0)
    p_w = p_w.where(today_universe.reindex(active).fillna(0) == 1, 0.0)
    final = BLEND * t_w + (1.0 - BLEND) * p_w

    # --- constraint compliance ---
    final = final.clip(-0.1, 0.1)
    final = _equalize_series(final)
    gross = final.abs().sum()
    if 0 < gross < 0.999:
        final *= (0.999 / gross)
        final = final.clip(-0.1, 0.1)
        final = _equalize_series(final)   # re-balance in case the clip bit

    final = final[final != 0]
    get_weights.previous_weights = final
    return final.to_dict()


def main():
    print("Loading full datasets...")
    features, universe, returns = utils.load_data()

    print("Running OFFICIAL backtest on full dataset (~35 min)...")
    weights = utils.backtest_strategy(
        get_weights, features, universe, show_progress=True)

    print("\nValidating Constraints:")
    checks = utils.validate_weights(weights, universe)
    for name, passed in checks.items():
        print(f"{name:20s} {'PASS' if passed else 'FAIL'}")

    print("\nScoring Strategy:")
    metrics = utils.summarize_performance(weights, returns)
    for k, v in metrics.items():
        print(f"{k:16s} {v:.4f}")

    out = REPO_ROOT / "submission.csv"
    weights.to_csv(out)
    print(f"\nSaved {out} with shape {weights.shape}")


if __name__ == "__main__":
    main()
