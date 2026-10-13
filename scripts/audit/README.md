# Reproducibility & P&L audit

Standalone scripts that re-check NB1–NB4. Run from this folder with
`python3.11` (torch, pandas, scipy). Outputs (models, CSVs, pickles) go to `out/`
(git-ignored).

Data used:
- `DATA_TAG=orig` reads `data/df_multi_unaligned.csv`, the dataset as shipped on `main`, before the fix.
- `DATA_TAG=realigned` reads `data/df_multi.csv`, produced by NB1 with the §2b re-alignment.

`exp1`, `exp3` and `exp8*` reload the weights of the original (unaligned) NB3 run from
`results/original_backup_2026-10-13/`. This folder is local only, because `results/` is not versioned.

| Script | What it does | Runtime |
|---|---|---|
| `exp1_reproduce_saved.py` | Reloads `results/vae_results_extended.pkl` and checks the saved weights reproduce NB3's RMSE table and NB4's portfolio Sharpes exactly | seconds |
| `exp3_diagnostics.py` | Autocorrelation and stale quotes, CVAE vs PCA/NS/naive benchmarks, execution-lag and cost sensitivity | seconds |
| `realign.py` | Detects and undoes the row misalignment between tenor columns in `data/data_bloomberg.xlsx`, writing `out/df_multi_realigned.csv` | ~1 min |
| `exp7_realigned_quick.py` | Cheap signals (PCA, NS, naive) on original vs realigned data | seconds |
| `exp8_aligned_segments.py`, `exp8b_all_tenors.py` | Non-circular test: P&L on days where the raw columns happen to be aligned vs misaligned | seconds |
| `exp9_pen_sensitivity.py` | Robustness of the realignment to the Viterbi penalty | ~1 min |
| `exp10_placebo.py` | Synthetic placebo: misalignment manufactures Sharpe from random walks, and realignment keeps genuine edges | ~5 min |
| `exp2_train_seeds.py [seeds]` | Retrains CVAE and MultiVAE, full-sample and train-only, per seed (`DATA_TAG=orig` or `realigned`) | ~8 min per batch of 8 |
| `exp4_seed_analysis.py` | Seed stability of the RMSE and NS comparison, IS vs OOS RMSE, leak-free trading Sharpe | seconds |
| `exp5_walkforward.py train` and `eval` | Expanding-window quarterly-refit CVAE, 511 OOS days, causal residuals | ~25 min train |
| `exp6_significance.py` | Block-bootstrap CI for the walk-forward Sharpe, break-even transaction cost | ~1 min |

Typical order: `exp1`, `realign`, `exp7`, `exp8b`, `exp10`, then
`DATA_TAG=realigned python3.11 exp2_train_seeds.py 0 1 2`, then
`DATA_TAG=realigned python3.11 exp4_seed_analysis.py`, then the
`exp5 train` / `exp5 eval` / `exp6` sequence.

## Key finding
The tenor columns of each Bloomberg sheet are offset from the shared `Date`
column by a row shift that changes stepwise over time (up to ±12 rows). Each
curve therefore mixes rates from different days. This alone produces the NB4
Sharpe ratios (3.99 slope, 4.57 cross-currency, 5.12 butterfly). On realigned
data, and with no look-ahead, no fair-value signal is distinguishable from zero
after costs. The proper fix is to re-pull the data with one common date axis,
for example a single BDH call on all tickers with a fixed calendar and
`Fill=P`.
