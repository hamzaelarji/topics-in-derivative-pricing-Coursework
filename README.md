# Autoencoder Market Models for Interest Rates

Replication and extension of Sokol (2022), *"Autoencoder Market Models for Interest Rates"* (SSRN 4300756), for the *Topics in Derivative Pricing* coursework (Prof. V. Piterbarg).

**Authors:** Diane Murzi, Chedi Mnif, Hamza El Arji

## What this project does

1. **Replicates Sokol's core empirical claim** in-sample: a 2D Conditional VAE (CVAE) reconstructs swap curves more accurately than the 3D Nelson-Siegel (NS) basis in all 7 currencies.
2. **Implements three AEMM variants** from Chapters 3–4 of the paper: forward rate, multi-factor short rate, and autoregressive P-measure.
3. **Tests a CVAE-driven trading strategy** on the 2s10s swap slope, plus extensions (cross-currency, per-tenor, carry, 2s5s10s butterfly). After fixing a data defect and removing look-ahead, **none of these strategies shows a robust out-of-sample edge**.

## Data

- Source: Bloomberg OIS swap rates, `data/data_bloomberg.xlsx`.
- 8 currencies are loaded. CHF is dropped in NB2 because its ticker is not SARON-based. The analysis uses **7 currencies**: AUD, CAD, DKK, EUR, GBP, JPY, USD.
- **7 tenors**: 2, 3, 5, 10, 15, 20, 30Y.
- Daily data, 30 Jan 2023 → 15 Jan 2026 (774 business days). This is the longest window where every currency and tenor is present.
- Train/test split at 31 Mar 2025: 566 train days and 208 test days (about 73/27).

### Data defect and fix (NB1 §2b)

In the Excel export, each sheet's tenor columns are **not aligned with the shared `Date` column**. Each column is shifted by a number of rows that changes stepwise over time, by up to 12 rows. Every "curve" therefore mixed rates from different days.

- **Symptom:** in the raw sheets, same-day correlation of daily 2Y vs 3Y changes is 0.05–0.21 in every currency except EUR. It should be about 0.95.
- **Effect on trading:** the misalignment creates spurious next-day predictability. A tenor that lags catches up with the others, so any fair-value residual appears to mean-revert.
- **Fix:** NB1 now re-aligns each tenor to the 2Y column with a Viterbi shift-path search. Same-day 2Y/3Y correlations rise to 0.84–0.99 (`figures/fig_01e_realignment_shifts.png`). `data/df_multi.csv` is the re-aligned dataset, and `data/df_multi_unaligned.csv` keeps the original.
- **Limitation:** this is an estimated repair. On synthetic data with known shifts it recovers about 97% of rows (`scripts/audit/exp10_placebo.py`). The proper fix is to **re-pull the data with one common date axis**, for example one BDH call on all tickers with a fixed calendar.

### Second defect: EUR sheet offset (NB1 §2c)

Re-aligning tenors within a sheet cannot detect an offset of a **whole sheet** against the other currencies.

- **Symptom:** daily level changes in the EUR sheet led USD, DKK and GBP by one row from April 2023 and by two rows from May 2024. Real end-of-day rates cannot lead another market by days, so each EUR value was labelled 1–2 business days too early. Every other sheet sits at lag 0.
- **Fix:** NB1 shifts the whole EUR sheet onto DKK's calendar, using the same shift-path search on level changes. DKK is pegged to EUR and closes in the same time zone. The same-day correlation of level changes rises from 0.13 to 0.90 for EUR-DKK, 0.17 to 0.64 for EUR-USD and 0.12 to 0.76 for EUR-GBP.
- **Scope:** this only matters for cross-currency comparisons. Within the EUR curve the tenors were already consistent.

## Key results (re-aligned data)

**Reconstruction, in-sample.** Models are trained on all 774 days. Values are mean RMSE per currency (range across the 7 currencies).

| Model | Latent dim | Mean RMSE (bp) |
|---|---|---|
| Nelson-Siegel (NB3 benchmark) | 3 | 3.05–9.08 |
| Single-currency VAE | 2 | 2.29–14.58 |
| Multi-currency VAE | 2 | 3.37–6.78 (beats NS in 6/7 currencies, not GBP) |
| **Multi-currency CVAE** | 2 | **2.18–2.45** (beats NS in 7/7 currencies) |
| Nelson-Siegel-Svensson (NB2) | 4 | 0.53 pooled |

**Reconstruction, out of sample.** Models are trained on dates up to 31 Mar 2025 and evaluated on the 208 test days.

| Model | In-sample RMSE (bp) | Out-of-sample RMSE (bp) |
|---|---|---|
| CVAE (NB3 §13b) | 2.31 | 4.90 |
| Multi-VAE (NB3 §13) | 4.00 | 6.09 |
| NS, λ = 0.5, fitted per date | — | 5.27 |

The CVAE's out-of-sample RMSE depends on the seed: 4.9–7.6 bp across four runs (`scripts/audit/exp4_seed_analysis.py`). The "VAE beats NS" result therefore holds in-sample but not reliably out of sample with only about 2 years of training data.

**Trading.** Equal-weight portfolio across 7 currencies, OOS (208 days), raw-bp P&L, 0.5 bp per position change, signals from train-only models (NB4).

| Signal | Sharpe, fixed HP | Sharpe, tuned HP |
|---|---|---|
| CVAE slope residual | 0.34 | −0.05 |
| NS slope residual (λ = 0.5, chosen on the train window) | 0.46 | 0.27 |
| Naive slope z-score | −0.07 | −0.16 |
| CVAE z₂ | −2.23 | −1.80 |
| 2s5s10s butterfly, CVAE residual (Ext. E) | 0.42 | — |
| Ext. B1: cross-currency slope spread, 21 pairs (model-free) | 0.55 (0.46 without EUR pairs) | — |
| Ext. B2: B1 gated by multi-VAE latent distance | −2.51 | — |

With 208 days the standard error of a Sharpe ratio is about 1.1, so none of these differs significantly from zero. DV01-weighted P&L gives the same Sharpe as raw bp to within 0.05. Adding carry (Ext. D) makes the CVAE strategy worse. Gating the cross-currency trade on latent distance (B2) makes it worse, not better. Before the EUR sheet was re-dated (§2c), B1 showed 1.21, almost entirely from the six EUR pairs.

A stricter walk-forward test (`scripts/audit/exp5_walkforward.py`, run before the §2c EUR fix, which barely affects single-currency signals) re-fits the CVAE quarterly on an expanding window over 511 OOS days. The seed-ensemble Sharpe is **0.03**, with a block-bootstrap 95% CI of [−1.6, 1.4], and the break-even cost is about 0.5 bp per position change.

**Earlier results.** Previous versions of this README reported an OOS Sharpe of 3.99 for the CVAE strategy, 4.57 for cross-currency and 5.12 for the butterfly. Those numbers came from the misaligned data combined with a CVAE trained on the full sample, test window included. On untouched data, days where every tenor happens to be aligned earn 0.006 bp/day (t = 0.06), against 0.77 bp/day (t = 9.7) on misaligned days (`scripts/audit/exp8b_all_tenors.py`).

## Known limitations and open issues

- **Short sample.** About 3 years of data versus 22 in the paper, and one train/test split.
- **Data repairs are estimated.** Both data fixes (tenor shifts in §2b, the EUR sheet offset in §2c) are estimated from the data rather than read from true timestamps. A clean re-pull with one common date axis would replace both.
- **Sensitivity to small data changes.** The single-seed CVAE strategy moved from 0.12 to −0.05 (tuned HP) when only the EUR dates changed.
- **NS benchmark parametrisation in NB3.** NB2 and NB4 use the `exp(−λτ)` form with λ = 0.5. NB3's in-sample comparison still uses `NelsonSiegel(lambda_param=1.5)` in the `exp(−τ/1.5)` form, which equals λ ≈ 0.67.
- **Wilcoxon p-values in NB3** treat strongly autocorrelated daily RMSEs as independent, so they overstate significance.
- **eNS/NSS λ often sits at the optimiser bounds** [0.05, 15] in NB2. For example, NSS λ₁ is at the lower bound on 9% of dates.
- **Carry in NB4 Ext. D is a simplified approximation**: `(r₂ − r₁₀)/252` per unit of spread DV01. It ignores the floating legs and the 2Y/10Y notional ratio, and there is no roll-down.
- **Training stability.** Single-seed VAE training is occasionally unstable: one multi-VAE run in the audit stalled on a plateau with a loss about 10× higher.

NB4 conventions:
- Transaction costs are charged per unit change of the spread position, in bp of the spread, identically in raw-bp and DV01 mode.
- `NumTrades` counts position changes.
- `HitRate` is the share of winning trades. `DayHitRate` is the share of winning days.

## Repository layout

```
├── 01_data_loading.ipynb      # load Bloomberg data, re-align tenors (§2b) and the EUR sheet (§2c), clean
├── 02_nelson_siegel.ipynb     # NS / eNS / NSS benchmark + λ cross-validation
├── 03_vae_training.ipynb      # single/multi VAE, CVAE, train-only models (§13, §13b), AEMM variants
├── 04_trading_signals.ipynb   # 2s10s strategies + extensions B–E, signals from train-only models
├── data/
│   ├── data_bloomberg.xlsx        # raw Bloomberg export
│   ├── df_multi.csv               # produced by NB1 (re-aligned)
│   └── df_multi_unaligned.csv     # original dataset before the fix
├── figures/                   # all PNGs produced by the notebooks
├── results/                   # pickles / weights shared between notebooks (not versioned)
├── scripts/
│   ├── audit/                     # reproducibility & P&L audit (see scripts/audit/README.md)
│   └── *.py                       # helpers for the report values
├── tests/                     # scratch work during the implementation
├── papers and reports/        # paper and report
└── README.md
```

## How to run

1. **Install dependencies** (Python ≥ 3.10):
   ```bash
   pip install numpy pandas scipy matplotlib torch openpyxl
   ```
2. **Place the Bloomberg Excel file** at `data/data_bloomberg.xlsx`.
3. **Run the notebooks in order**: NB1 → NB2 → NB3 → NB4. Each notebook saves its outputs into `results/` and `figures/` for the next one. NB3 takes about 10 minutes on a laptop CPU.
4. **Optional:** run the audit scripts in `scripts/audit/` (see its README).

## References

- Sokol, A. (2022). *Autoencoder Market Models for Interest Rates.* SSRN 4300756.
- Kingma, D.P. & Welling, M. (2013). *Auto-Encoding Variational Bayes.* arXiv:1312.6114.
- Nelson, C.R. & Siegel, A.F. (1987). *Parsimonious Modeling of Yield Curves.* J. Business, 60(4).
