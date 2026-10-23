# CSOT Quant — Week 3 Alpha Research Challenge — FULL CONTEXT DOSSIER

> **Purpose of this file:** This is a single, self-contained context document
> compiled from *every* file in the `Week 3` repository archive. It exists so
> that an AI assistant (Gemini) — which cannot read the original `.zip` — has
> complete knowledge of the challenge, the data schema, the rules, the
> evaluation math, the code API, and the deliverable formats, in order to help
> implement and submit a strategy. Nothing from the original repo has been
> omitted. Where the original repo *references* a file that is **not present**
> in the archive, that is flagged explicitly (see Section 11).

---

## 0. TL;DR — What this challenge is, in one screen

| Item | Value |
|---|---|
| **Task** | Build a daily, cross-sectional, market-neutral **long/short equity** strategy |
| **How** | Implement a single function `get_weights(features, today_universe) -> dict[str, float]` in `notebooks/main.ipynb` |
| **Universe** | 2167 anonymized stocks; tradable set changes daily via `universe.parquet` |
| **Inputs allowed** | Only the 22 pre-computed features in `features.parquet` (no raw OHLC, no external data) |
| **Primary metric** | **Annualized Net Sharpe Ratio** (after trading costs) |
| **Trading cost** | 0.01% (1 bp) of traded capital per day |
| **Hard constraints** | Unit-capital ≤ 1+1e-4 · max \|weight\| ≤ 0.1 · dollar-neutral \|Σw\| ≤ 1e-4 · universe-respected |
| **Local data** | 2005–2019 returns provided for backtesting; final scoring on held-out later period |
| **Date range (features)** | 2005-01-03 → 2025-02-07 |
| **Deliverables** | `submission.csv` + completed `main.ipynb` + `strategy_description.txt` + source repo |
| **Opens / Closes** | 16 June → 23 June, 11:59 PM (final deadline) |
| **Team size / submissions** | Up to 4 people; unlimited submissions, only the **last valid** one counts |

---

## 1. Repository structure (as shipped in the archive)

```
Week 3/
├── README.md                          # Challenge overview (master)
├── .gitignore                         # Ignores data, submission.csv, venvs, caches
├── requirements.txt                   # Python dependencies
├── strategy_description_template.txt  # Template to fill in for submission
├── data/
│   ├── README.md                      # Dataset download instructions (short)
│   ├── download_data.md               # Dataset download instructions (detailed)
│   ├── features.parquet               # (NOT in repo — download from Google Drive)
│   ├── returns.parquet                # (NOT in repo — download from Google Drive)
│   └── universe.parquet               # (NOT in repo — download from Google Drive)
├── docs/
│   ├── competition_rules.md           # Task + data-access rules + constraints
│   ├── evaluation.md                  # Metric definitions & formulas
│   ├── submission_format.md           # get_weights contract + submission.csv format
│   └── faq.md                         # Frequently asked questions
├── notebooks/
│   └── main.ipynb                     # ★ Where you implement get_weights + run backtest
└── src/
    └── utils.py                       # Backtest loop, metrics, constraint validators
```

> **Note on the outer archive:** the `.zip` nests this as
> `week-3/Week 3/...`, and there is an outer top-level `week-3/README.md` that
> describes the whole "CSOT Quant Finance & Research" course (Week 1 & Week 2
> marked *Completed*, Week 3 = this Alpha Research Challenge). That outer README
> adds no task detail beyond what is captured here.

---

## 2. The challenge — objective & framing

- **Goal:** Engage in **alpha research** to develop systematic strategies for
  trading stocks **daily**, using **price/volume-derived features**.
- **Deliverable strategy:** Identify predictive signals and construct a
  **market-neutral long-short portfolio** that **maximizes Net Sharpe Ratio**.
- **Implementation surface:** The entire strategy is expressed through **one
  function**, `get_weights`, in `notebooks/main.ipynb`. It is called **once per
  trading day** and must return a dictionary of stock → weight for that day.
- **No raw prices:** Only derived indicators are provided. There is **no**
  open/high/low/close data to work with directly.

### Timeline

| Event | Date |
|---|---|
| Competition Opens | 16 June |
| Competition Closes | 23 June |
| Final Submission Deadline | 23 June, 11:59 PM |

---

## 3. Dataset — schema & description (critical for any implementation)

The dataset is **three parquet files**, hosted on Google Drive (not committed to
the repo). They are placed in `data/`. *(Per the user's instruction, the
mechanics of downloading are out of scope here — but the schema below is
essential context.)*

### 3.1 Files & roles

| File | Role | Usage rule |
|---|---|---|
| `features.parquet` | The 22 derived indicators — **your only model input** | Allowed as input |
| `universe.parquet` | 0/1 daily tradability mask per stock | Determines who is tradable each day |
| `returns.parquet` | Realized daily returns | **Backtest/validation ONLY — NEVER a model input** |

### 3.2 Dimensions

- **Stocks:** 2167 anonymized stock identifiers.
- **Date range:** 2005-01-03 → 2025-02-07.
- **Local returns available:** 2005–2019 (training/validation window). Final
  scoring is on a **later held-out period** for which returns are *not* provided.

### 3.3 DataFrame shapes / index conventions (from `utils.py` + notebook)

- `features`: **MultiIndex columns** `(feature_name, stock_id)`; **rows = dates**
  (chronological). So `features[("macd", "STOCK_123")]` is one feature's series
  for one stock.
- `universe`: rows = dates, columns = stock_ids, values ∈ {0, 1}. Its index and
  columns define the canonical date set and stock set for the whole problem.
- `returns`: rows = dates, columns = stock_ids, values = daily returns.
- The weights matrix produced by the backtest has the **same shape, index, and
  columns as `universe`**, filled with `0.0` everywhere a stock isn't traded.

### 3.4 The 22 available features

Features are derived from Open, High, Low, Close, Volume. The full list:

| # | Feature | # | Feature |
|---|---|---|---|
| 1 | `relative_strength_index` | 12 | `chande_momentum_oscillator` |
| 2 | `williams_r` | 13 | `ichimoku` |
| 3 | `volatility_20` | 14 | `know_sure_thing` |
| 4 | `volatility_60` | 15 | `ultimate_oscillator` |
| 5 | `trend_1_3` | 16 | `aroon` |
| 6 | `trend_5_20` | 17 | `stochastic_oscillator` |
| 7 | `trend_20_60` | 18 | `on_balance_volume` |
| 8 | `average_true_range` | 19 | `ease_of_movement` |
| 9 | `macd` | 20 | `chaikin_money_flow` |
| 10 | `trix` | 21 | `accumulation_distribution_index` |
| 11 | `commodity_channel_index` | 22 | `volume` |

These span **momentum** (RSI, Williams %R, MACD, TRIX, CCI, CMO, KST, ultimate
oscillator, stochastic, aroon, trend_*), **volatility** (volatility_20/60, ATR),
and **volume/flow** (OBV, ease_of_movement, Chaikin money flow, accumulation/
distribution, volume) categories.

---

## 4. Portfolio constraints (HARD — violation = not scored)

For date `t` and stock `s`, let `w(t, s)` be the fraction of unit capital
allocated (positive = long, negative = short).

| Constraint | Exact rule | Meaning |
|---|---|---|
| **Unit capital** | `sum(|w[t, s]|) <= 1 + 1e-4` | Total gross exposure ≤ 1 (with tiny tolerance) |
| **Max position size** | `|w[t, s]| <= 0.1` for every stock | No single name exceeds 10% of book |
| **Dollar neutral** | `|sum(w[t, s])| <= 1e-4` | Long $ exposure equals short $ exposure |
| **Universe** | `w[t, s] = 0` whenever `universe[t, s] = 0` | Only trade today's eligible stocks |

- The numerical tolerances above are **authoritative** (from
  `docs/competition_rules.md`). They are encoded as constants in `utils.py`:
  `MAX_WEIGHT = 0.1`, `UNIT_CAPITAL_TOL = 1e-4`, `DOLLAR_NEUTRAL_TOL = 1e-4`.
- The **max_weight** check in code uses `MAX_WEIGHT + 1e-9` and the
  **universe** check uses a `1e-9` tolerance (floating-point slack).
- A submission violating **any** constraint **will not be scored**. Always run
  the validation cell (`utils.validate_weights`) before submitting.

### No look-ahead (anti-lookahead) rule

- When `get_weights` is called for date `t`, it **only receives feature history
  strictly before `t`** (the backtest passes `features.loc[features.index < t]`).
- You must **not** use any information dated `>= t`.
- `returns.parquet` may **never** be used as a model input — it exists solely
  for you to score yourself locally.
- **No external data** of any kind (sector maps, macro, alt-data). The dataset
  is anonymized precisely so identities can't be recovered and external joins
  are impossible.

---

## 5. Evaluation math (the metric that ranks you)

**Net Sharpe Ratio** (risk-adjusted return after costs) is the sole ranking
metric. Higher is better.

### 5.1 Daily quantities

Given the daily weight matrix `w[t, s]`:

| Quantity | Formula |
|---|---|
| **Book Value** (capital deployed) | `BookValue[t] = sum(|w[t, s]|)` |
| **Traded Capital** (capital turned over) | `Traded[t] = sum(|w[t, s] - w[t-1, s]|)` |
| **Turnover** (avg daily traded as % of book) | `Turnover = sum(Traded) / sum(BookValue) * 100` |
| **Gross PnL** (return before costs) | `GrossPnL[t] = sum(w[t, s] * returns[t, s])` |
| **Net PnL** (return after costs) | `NetPnL[t] = GrossPnL[t] - 0.0001 * Traded[t]` |

- Trading cost = **0.01% (1 basis point) of traded capital per day** (`COST_BPS = 0.0001`).

### 5.2 Sharpe ratios (annualized with √252)

```
Gross Sharpe = sqrt(252) * mean(GrossPnL) / std(GrossPnL)
Net Sharpe   = sqrt(252) * mean(NetPnL)   / std(NetPnL)
```

- `TRADING_DAYS_PER_YEAR = 252`.
- **Net Sharpe Ratio determines final ranking.**
- Edge case in code: if `std(pnl) == 0` or NaN, `sharpe_ratio` returns `0.0`.

### 5.3 Training vs. test split (overfitting warning)

- **2005–2019 returns** are provided → treat as your **training/validation**
  period.
- **Final scoring** is on a **later held-out period** with **no returns
  provided**. Use cross-validation and out-of-sample discipline; a strategy
  that only looks good on 2005–2019 is unlikely to hold up.

---

## 6. The `get_weights` contract (the one function you write)

Signature (must match exactly):

```python
def get_weights(features: pd.DataFrame, today_universe: pd.Series) -> dict[str, float]:
    ...
```

**Inputs**

- `features` — every feature's **full history strictly before today** (no
  look-ahead). MultiIndex columns `(feature_name, stock_id)`; index = dates
  before `t`.
- `today_universe` — a **0/1 `pd.Series` indexed by stock id**; value `1` =
  tradable today.

**Output**

- `dict[str, float]` mapping **stock id → weight for today**.
- **Only include stock ids where `today_universe == 1`.**
- Output should respect the constraints in Section 4 (the backtest will not
  "fix" your weights — it only drops illegal/non-universe entries to 0).

**How it's driven**

- `utils.backtest_strategy` walks forward over **every date in `universe.index`**,
  calls `get_weights` per day with the pre-`t` history slice, and assembles all
  daily outputs into one weights DataFrame.
- The loop **silently drops** any returned stock that is not in `stock_ids` or
  not tradable today (`today_universe.get(stock_id, 0) != 1`), setting it to 0.
  Frequent drops indicate a bug worth fixing.

---

## 7. `src/utils.py` — complete API reference

All paths default relative to the repo root (`utils.py` lives in `src/`, and
`DATA_DIR = repo_root/data`), so the functions work from `notebooks/`, repo
root, or CI.

**Module constants**

| Constant | Value | Meaning |
|---|---|---|
| `TRADING_DAYS_PER_YEAR` | 252 | Annualization factor |
| `COST_BPS` | 0.0001 | 0.01% trading cost per unit traded capital |
| `MAX_WEIGHT` | 0.1 | Max abs weight per stock |
| `UNIT_CAPITAL_TOL` | 1e-4 | Tolerance for gross-exposure ≤ 1 |
| `DOLLAR_NEUTRAL_TOL` | 1e-4 | Tolerance for net exposure ≈ 0 |

**Functions**

| Function | Signature (abridged) | What it does |
|---|---|---|
| `load_data` | `(features_path, universe_path, returns_path) -> (features, universe, returns)` | Reads the three parquet files (defaults to `data/*.parquet`) |
| `backtest_strategy` | `(get_weights_fn, features, universe, show_progress=True) -> pd.DataFrame` | Walk-forward loop; builds the full weights matrix (same shape as `universe`, zero-filled). Uses `tqdm` progress bar |
| `validate_weights` | `(weights, universe) -> dict[str, bool]` | Checks the 4 hard constraints → keys: `unit_capital`, `max_weight`, `dollar_neutral`, `universe_respected` |
| `validate_shape` | `(weights, universe) -> dict[str, bool]` | Checks `same_columns`, `same_index`, `no_missing_values` against `universe` |
| `book_value` | `(weights) -> pd.Series` | `weights.abs().sum(axis=1)` |
| `traded_capital` | `(weights) -> pd.Series` | `weights.diff().abs().sum(axis=1)`, first row filled with day-0 gross |
| `turnover` | `(weights) -> float` | `traded.sum() / book.sum() * 100` |
| `gross_pnl` | `(weights, returns) -> pd.Series` | `(w * r).sum(axis=1)` over common index/cols |
| `net_pnl` | `(weights, returns) -> pd.Series` | `gross_pnl - COST_BPS * traded_capital` |
| `sharpe_ratio` | `(pnl, annualization=252) -> float` | `sqrt(252)*mean/std`; returns 0.0 if std is 0/NaN |
| `summarize_performance` | `(weights, returns) -> dict` | Returns `turnover_pct`, `gross_sharpe`, `net_sharpe`, `mean_book_value` |

**Implementation details worth knowing**

- `backtest_strategy` history slice: `features.loc[features.index < t]` — this is
  the mechanism enforcing no look-ahead.
- `gross_pnl` / `net_pnl` align on `weights.index ∩ returns.index` and
  `weights.columns ∩ returns.columns`, so mismatched coverage is handled by
  intersection (silently).
- `traded_capital` fills the first day's NaN diff with that day's gross book
  (`weights.iloc[0].abs().sum()`), i.e. day 0 is treated as trading in from flat.

---

## 8. `notebooks/main.ipynb` — cell-by-cell map

The notebook is `nbformat 4.5`, kernel **Python 3** (language_info version
"3.11"). 16 cells:

| Cell | Type | Content |
|---|---|---|
| 0 | md | Title + instructions; pointers to docs; reminder to download data into `data/` |
| 1 | code | Imports (`sys, pathlib, pandas as pd, numpy as np, matplotlib.pyplot as plt`); sets `REPO_ROOT = Path("..").resolve()`; appends `src/` to path; `import utils` |
| 2 | md | "## 1. Load data" |
| 3 | code | `features, universe, returns = utils.load_data()`; prints the three shapes |
| 4 | code | `features.head()` |
| 5 | md | "## 2. Implement your strategy" — explains the `get_weights` inputs/outputs + restates the 3 numeric constraints |
| 6 | code | **`get_weights` definition — currently a placeholder** (see Section 9). ★ This is the cell to replace |
| 7 | md | "## 3. Backtest" |
| 8 | code | `weights = utils.backtest_strategy(get_weights, features, universe)`; `weights.head()` |
| 9 | md | "## 4. Validate constraints" |
| 10 | code | `checks = utils.validate_weights(weights, universe)`; prints PASS/FAIL per constraint |
| 11 | md | "## 5. Score the strategy (training period only)" |
| 12 | code | `metrics = utils.summarize_performance(weights, returns)`; prints each metric to 4 dp |
| 13 | code | `net_pnl = utils.net_pnl(weights, returns)`; `net_pnl.cumsum().plot(...)`; cumulative Net PnL chart |
| 14 | md | "## 6. Export submission"; notes `submission.csv` saved to repo root, must match `universe` shape; mentions running `src/validate_submission.py` |
| 15 | code | `output_path = REPO_ROOT / "submission.csv"`; `weights.to_csv(output_path)`; prints saved shape |

> **The only cell you must change** is **cell 6** (`get_weights`). Everything else
> is plumbing that runs the backtest, validates, scores, plots, and exports.

---

## 9. The placeholder `get_weights` (what ships, and why it must be replaced)

Cell 6 currently contains a **trivial, alpha-free placeholder** whose only job
is to make the backtest run end-to-end while satisfying every constraint:

- Takes today's tradable stocks (`today_universe == 1`).
- If fewer than 2 tradable, returns all-zero weights.
- Splits the tradable list in half: **first half = longs, second half = shorts**.
- Sizes each leg so gross exposure is split evenly and dollar-neutral:
  - `max_gross = min(1.0, 0.1 * min(len(longs), len(shorts)) * 2)`
  - `w_long = (max_gross / 2) / len(longs)`
  - `w_short = (max_gross / 2) / len(shorts)`
  - longs get `+w_long`, shorts get `-w_short`.
- It is **dollar-neutral and constraint-satisfying but carries no real signal**
  (the long/short split is arbitrary ordering, not based on any feature).

**Replacing it** means: compute a per-stock score from the feature history,
rank cross-sectionally, go long the top / short the bottom, and size weights to
satisfy all four constraints.

---

## 10. Submission — what to hand in & exact format

### 10.1 Deliverables

1. `notebooks/main.ipynb` — with a **completed** `get_weights`.
2. `submission.csv` — in the format below.
3. `strategy_description.txt` — filled in from
   `strategy_description_template.txt`.
4. Your **source code repository**.

### 10.2 `submission.csv` format

- **Rows:** every date in `data/universe.parquet` — no more, no fewer.
- **Columns:** every stock id in `data/universe.parquet`, **in the same order**.
- **Values:** float weight `w(t, s)` per date/stock; `0` for any stock not
  tradable that day.
- It is written by notebook cell 15 (`weights.to_csv(REPO_ROOT/"submission.csv")`).

### 10.3 Validating before submitting

Run the **Validate constraints** cell (cell 10). You can also validate from a
script:

```python
import sys; sys.path.append("src")
import pandas as pd, utils
universe = pd.read_parquet("data/universe.parquet")
weights  = pd.read_csv("submission.csv", index_col=0, parse_dates=True)
weights.columns = weights.columns.astype(universe.columns.dtype)
print({**utils.validate_shape(weights, universe), **utils.validate_weights(weights, universe)})
```

### 10.4 Submission integrity rules

- Your submitted **code must reproduce** the `submission.csv` you hand in.
  **Mismatches disqualify** the entry — re-run end-to-end before final submit.
- Only your **last valid** submission before the deadline counts.
- Unlimited submissions allowed; teams of up to 4.

---

## 11. ⚠️ Files referenced in the repo but MISSING from the archive

These are mentioned by the docs/notebook but are **not present** in the
extracted archive. An implementer should be aware they don't exist yet:

| Referenced path | Referenced where | Status |
|---|---|---|
| `sample_submission/submission.csv` | `Week 3/README.md` ("Repository Files"), `docs/submission_format.md` (as an all-zero shape example) | **MISSING** — no `sample_submission/` directory exists |
| `src/validate_submission.py` | `notebooks/main.ipynb` cell 14, `docs/submission_format.md` ("run `src/validate_submission.py`") | **MISSING** — `src/` contains only `utils.py` |

**Implication:** the standalone validation script doesn't exist; use the
in-notebook validation cell (`utils.validate_shape` + `utils.validate_weights`)
or the inline script in Section 10.3 instead. There is no provided sample
submission file to copy the exact column dtype/order from — derive it directly
from `universe.parquet`.

---

## 12. Rules summary — Allowed vs. Not Allowed

**Allowed**

- Any modeling approach: rule-based signals, statistical models, ML, ensembles.
- Combining/transforming the 22 provided features however you like.
- Unlimited local iteration and submissions before the deadline.

**Not Allowed**

- Using `returns.parquet` as a feature/input.
- Using future feature values (anything dated `>= t`) inside `get_weights(t)`.
- Submitting weights for stocks not in that day's `universe`.
- Using **any** external data source — only the provided anonymized features.

---

## 13. FAQ (compiled, paraphrased from `docs/faq.md`)

- **ML or rule-based?** Either — anything using only strictly-pre-`t` feature
  history is fine.
- **External data (sectors, macro, alt-data)?** No. Anonymization also makes it
  technically impossible.
- **Why dollar-neutral?** To force a *relative* view (which names beat which)
  rather than a directional market bet; the book can profit even in a flat
  market.
- **Code doesn't reproduce my `submission.csv`?** Disqualified — re-run
  end-to-end before final submit.
- **Can the tradable set change daily?** Yes — that's exactly what
  `universe.parquet` encodes; stocks enter/leave on any date, surfaced via
  `today_universe`.
- **I return a weight for a non-tradable stock — what happens?**
  `backtest_strategy` silently drops it to 0; frequent occurrences signal a bug.
- **Where do I check my Sharpe?** Run `main.ipynb` end-to-end; the last cells
  compute Gross/Net Sharpe + turnover from `returns.parquet`.
- **Deadline?** 23 June, 11:59 PM. Opens 16 June. Only the last valid submission
  is scored.
- **Submissions / team size?** Unlimited submissions (last valid counts); teams
  up to 4. *(The repo notes organizers may adjust this line.)*
- **Where's the data?** `data/download_data.md` — parquet files on Google Drive,
  not committed.

---

## 14. `strategy_description_template.txt` — structure to fill in

The submission's `strategy_description.txt` is produced by completing this
template. Its sections:

1. **Core Idea** — 1–2 sentences: which inefficiency/pattern (e.g. short-term
   mean reversion, cross-sectional momentum, vol risk premium).
2. **Signal Construction** — features used (columns from `features.parquet`),
   how combined (avg / weighted z-score / ML), lookback/smoothing choices + why.
3. **Portfolio Construction** — ranking/scoring method, selection rule (e.g.
   top/bottom X%), weighting scheme (equal / score / inverse-vol), how
   dollar-neutral + max-weight constraints are enforced.
4. **Risk Controls** — position limits, vol filters, exposure caps, turnover
   limits.
5. **Known Limitations / Things You'd Try Next** — honest account of what didn't
   work / future ideas.
6. **Backtest Summary (training period)** — Net Sharpe, Gross Sharpe, Turnover %.

---

## 15. Environment / dependencies

From `requirements.txt`:

```
pandas>=2.0
numpy>=1.24
pyarrow>=14.0
scipy>=1.10
matplotlib>=3.7
jupyter>=1.0
notebook>=7.0
tqdm>=4.65
```

`.gitignore` excludes: `__pycache__/`, `*.pyc`, `.ipynb_checkpoints/`, `.venv/`,
`venv/`, `.DS_Store`, `data/*.parquet`, and `submission.csv` (regenerate it,
never commit a stale one).

Quick start (from the repo's own README):

```bash
pip install -r requirements.txt
# download the dataset into data/  (see data/download_data.md)
jupyter notebook notebooks/main.ipynb
```

---

## 16. Companion files provided alongside this README

The following original files are provided **as-is** for the assistant to read
and edit directly (these are the templates to be updated):

- **`main.ipynb`** — the notebook with the placeholder `get_weights` (cell 6) to
  be replaced; all backtest/validate/score/export plumbing intact.
- **`utils.py`** — the backtest engine, metrics, and constraint validators
  (reference; generally not edited).
- **`strategy_description_template.txt`** — the strategy write-up template to
  fill in for submission.

*End of context dossier. Every file and detail in the original `Week 3` archive
is represented above.*
