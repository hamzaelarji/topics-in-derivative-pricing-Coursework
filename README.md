# Autoencoder Market Models for Interest Rates

Replication and extension of Sokol (2022), *"Autoencoder Market Models for Interest Rates"* (SSRN 4300756), for the *Topics in Derivative Pricing* coursework (Prof. V. Piterbarg).

**Authors:** Diane Murzi, Chedi Mnif, Hamza El Arji

## What this project does

1. **Replicates Sokol's core empirical claim** in-sample: a 2D Conditional VAE (CVAE) reconstructs OIS swap curves more accurately than the 3D Nelson-Siegel (NS) basis in 5 of 7 currencies (not CHF or JPY). Out of sample, the VAEs do **not** beat NS.
2. **Implements three AEMM variants** from Chapters 3–4 of the paper: forward rate, multi-factor short rate, and autoregressive P-measure.
3. **Tests CVAE-driven trading strategies** on the 2s10s slope and the 2s5s10s butterfly, plus carry and cross-currency extensions. With clean data and no look-ahead, **none shows a statistically significant out-of-sample edge**.

## Data

- **Source:** fresh Bloomberg pull, `data/bloomberg_swap_download_comlet.xlsx`. Every ticker shares one weekday calendar, so the columns are aligned by construction.
- **7 currencies, current OIS benchmarks:**
  - USD SOFR (`USOSFR`)
  - EUR €STR (`EESWE`)
  - GBP SONIA (`BPSWS`)
  - JPY TONA (`JYSO`)
  - AUD AONIA (`ADSO`)
  - CAD CORRA (`CDSO`)
  - CHF SARON (`SFSNT`)

  DKK is not used because `DKSW` is a CIBOR swap, not an OIS.
- **7 tenors:** 2, 3, 5, 10, 15, 20, 30Y.
- **Window:** daily, 9 Sep 2020 → 23 Oct 2026 (1,598 business days). This is the longest window where every currency and tenor is present. The start is set by CAD 15–30Y CORRA swaps; holiday gaps of up to 5 days are forward-filled, which covers the Japanese New Year.
- **Train/test split:** at 31 Dec 2024. 1,125 train days and 473 test days (about 70/30).
- **Same-day correlation of daily changes:** 0.85–0.99 between 2Y and 3Y. Every currency lines up with USD at lag 0, except JPY and AUD, which sometimes react to US moves one day later because they close before New York.
- The Excel file also contains IBOR-swap histories back to 2000 and overnight rates (`data/bloomberg_extras_*.xlsx`), which are not used yet.

### The first export, and why the results changed

The project originally used `data/data_bloomberg.xlsx`, which had two defects:

1. **Tenor columns were shifted against the `Date` column,** by up to 12 rows, changing stepwise over time. Each "curve" mixed rates from different days.
2. **The EUR sheet was dated 1–2 days early.** Its 2023–2026 values were actually €STR OIS; the old `EUSWE` (EONIA OIS) ticker stopped in December 2021.

NB1 can still load that file (`DATA_SOURCE = "original"`) and repairs both defects (§2b, §2c). The fresh pull confirmed the repair: in 2023–2026 the repaired values matched Bloomberg exactly on **97.9%** of (date, tenor) points, against **40.9%** for the unrepaired export. The repaired 2023–2026 dataset is kept as `data/df_multi_repaired_2023_2026.csv`, and the raw export as `data/df_multi_unaligned.csv`.

The misalignment created spurious next-day predictability. On the original export the CVAE slope strategy showed an out-of-sample Sharpe of **3.99** (butterfly 5.12, cross-currency 4.57). On days where the raw columns happened to be aligned, the same strategy earned 0.006 bp/day (t = 0.06), so those Sharpes were artefacts (`scripts/audit/`).

## Key results (clean data, NB2–NB4)

**Reconstruction, in-sample.** Models are trained on all 1,598 days. Values are mean RMSE per currency (range across the 7 currencies).

| Model | Latent dim | Mean RMSE (bp) |
|---|---|---|
| Nelson-Siegel, λ = 0.05 (NB2 cross-validation) | 3 | 2.09–7.94 |
| Single-currency VAE | 2 | 3.10–9.55 |
| Multi-currency VAE | 2 | 3.61–7.67 (beats NS in 4/7 currencies) |
| **Multi-currency CVAE** | 2 | **2.63–3.35** (beats NS in 5/7; not CHF, JPY) |
| eNS (λ re-fitted per date) | 3 + λ | 2.92 pooled |
| Nelson-Siegel-Svensson | 4 | 0.45 pooled |

**Reconstruction, out of sample.** Models are trained on dates up to 31 Dec 2024 and evaluated on the 473 test days.

| Model | In-sample RMSE (bp) | Out-of-sample RMSE (bp) |
|---|---|---|
| CVAE (NB3 §13b) | 2.90 | 7.79 |
| Multi-VAE (NB3 §13) | 4.79 | 13.28 |
| NS, λ = 0.05, fitted per date | 5.98 | 3.16 |

With about four years of training data, the autoencoders do not generalise to 2025–2026 curve shapes. Per-date NS does not need training and is much more accurate out of sample.

**Trading.** Equal-weight portfolios across 7 currencies, OOS (473 days), raw-bp P&L, 0.5 bp per position change (1 bp for the butterfly), signals from train-only models (NB4).

| Strategy | OOS Sharpe |
|---|---|
| A: CVAE slope residual, fixed / tuned HP | 0.34 / 0.77 |
| NS slope residual | −0.92 |
| Naive slope z-score | −0.39 |
| B: CVAE butterfly residual | −0.25 |
| NS / naive butterfly | −2.10 / −0.01 |
| Ext. 2: cross-currency, all 21 pairs | 0.29 |
| Ext. 2: close / far pairs (train-window latent distance) | 0.77 / −0.23 |

With 473 days the standard error of a Sharpe ratio is about 0.73, so nothing here differs significantly from zero. The best numbers come after choosing among many variants. The CVAE ranks above the NS and naive benchmarks, but by less than one standard error. Carry dominates the slope trade: in Extension 1 the average carry Sharpe is 0.80 against 0.04 for the signal.

## Known limitations

- **Sample length.** About 6 years of data versus 22 in the paper, and one train/test split.
- **Data before the benchmark launches.** SOFR and €STR swap histories before their launch (2018/2019) are Bloomberg back-fills. They lie outside the window used here.
- **Wilcoxon p-values in NB3** treat strongly autocorrelated daily RMSEs as independent, so they overstate significance.
- **NS λ.** The cross-validation curve is flat and has two local minima (λ ≈ 0.05 and ≈ 0.5). NB2 picks 0.05 on the full sample; NB4 re-selects λ on the train window only and gets 0.5.
- **Carry (NB4 Ext. 1)** is a simplified approximation, `(r₂ − r₁₀)/252` per unit of spread DV01, with no floating legs or roll-down.
- **Single-seed training.** VAE training is seed-dependent and occasionally stalls on a plateau (`scripts/audit/exp4_seed_analysis.py`).
- **Stale commentary.** Some interpretive commentary in NB2/NB3 was written for the first export; those cells are flagged with a warning note.

NB4 conventions:
- Transaction costs are charged per unit change of the spread position, identically in raw-bp and DV01 mode.
- `NumTrades` counts position changes.
- `HitRate` is the share of winning trades; `DayHitRate` is the share of winning days.

## Repository layout

```
├── 01_data_loading.ipynb      # load Bloomberg data, alignment checks (§2b/§2c repair the first export), clean
├── 02_nelson_siegel.ipynb     # NS / eNS / NSS benchmark + λ cross-validation
├── 03_vae_training.ipynb      # single/multi VAE, CVAE, train-only models (§13, §13b), AEMM variants
├── 04_trading_signals.ipynb   # 2s10s strategies + extensions B–E, signals from train-only models
├── data/
│   ├── bloomberg_swap_download_comlet.xlsx   # fresh Bloomberg pull (default source)
│   ├── bloomberg_extras_*.xlsx                # overnight rates, IBOR fixings (not used yet)
│   ├── data_bloomberg.xlsx                    # first export (misaligned; DATA_SOURCE="original")
│   ├── df_multi.csv                           # produced by NB1 (clean OIS data)
│   ├── df_multi_repaired_2023_2026.csv        # first export after the NB1 §2b/§2c repair
│   └── df_multi_unaligned.csv                 # first export, unrepaired
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
2. **The Bloomberg Excel file** is `data/bloomberg_swap_download_comlet.xlsx`.
3. **Run the notebooks in order**: NB1 → NB2 → NB3 → NB4. Each notebook saves its outputs into `results/` and `figures/` for the next one. NB3 takes about 35 minutes on a laptop CPU.
4. **Optional:** run the audit scripts in `scripts/audit/` (see its README).

## References

- Sokol, A. (2022). *Autoencoder Market Models for Interest Rates.* SSRN 4300756.
- Kingma, D.P. & Welling, M. (2013). *Auto-Encoding Variational Bayes.* arXiv:1312.6114.
- Nelson, C.R. & Siegel, A.F. (1987). *Parsimonious Modeling of Yield Curves.* J. Business, 60(4).
