# Autoencoder Market Models for Interest Rates

Replication and extension of Sokol (2022), *"Autoencoder Market Models for Interest Rates"* (SSRN 4300756), for the *Topics in Derivative Pricing* coursework (Prof. V. Piterbarg).

**Authors:** Diane Murzi, Chedi Mnif, Hamza El Arji

## What this project does

1. **Replicates Sokol's core empirical claim** in-sample on 9 of his 13 currencies: a 2D Conditional VAE (CVAE), with the paper's post-processing step, reconstructs swap curves more accurately than the 3D Nelson-Siegel (NS) basis in 7 of 9 currencies. The two exceptions (CZK, GBP) are within 0.2 bp. Out of sample, the VAEs do **not** beat NS.
2. **Implements three AEMM variants** from Chapters 3–4 of the paper: forward rate, multi-factor short rate, and autoregressive P-measure.
3. **Tests CVAE-driven trading strategies** on the 2s10s slope and the 2s5s10s butterfly, plus carry and cross-currency extensions (NB4).
4. **Builds a full relative-value curve-arbitrage & anomaly-detection process** on the AEMM (NB5): monitoring, model-neutral trade construction, execution and regime-aware risk controls.
5. **Asks whether the autoencoder residual is a tradable mispricing or a model error** (NB6): predictive power at 1–20 days against PCA and Nelson-Siegel, a carry/rolldown P&L decomposition and residual alpha, Markov-switching regimes, and reconstruction vs prediction training objectives.

With clean data and no look-ahead, **no strategy shows a statistically significant out-of-sample edge**.

## Data

- **Source:** fresh Bloomberg pull, `data/bloomberg_swap_download_comlet.xlsx`. Every ticker shares one weekday calendar, so the columns are aligned by construction.
- **9 currencies** (the 9 of Sokol's 13 that are available in the pull):
  - **OIS swaps (current benchmarks):**
    - USD SOFR (`USOSFR`)
    - EUR €STR (`EESWE`)
    - GBP SONIA (`BPSWS`)
    - JPY TONA (`JYSO`)
    - AUD AONIA (`ADSO`)
    - CAD CORRA (`CDSO`)
    - CHF SARON (`SFSNT`)
  - **IBOR swaps:** DKK (`DKSW`, vs CIBOR) and CZK (`CKSW`, vs PRIBOR). Like Sokol, we mix OIS and IBOR swaps.
- **7 tenors:** 2, 3, 5, 10, 15, 20, 30Y.
- **Window:** daily, 9 Sep 2020 → 23 Oct 2026 (1,598 business days). This is the longest window where every currency and tenor is present. The start is set by CAD 15–30Y CORRA swaps; holiday gaps of up to 5 days are forward-filled, which covers the Japanese New Year.
- **Train/test split (NB3/NB4):** at 31 Dec 2024. 1,125 train days and 473 test days (about 70/30).
- **Same-day correlation of daily changes:** 0.85–0.99 between 2Y and 3Y in every currency. Every currency lines up with USD at lag 0, except JPY and AUD, which sometimes react to US moves one day later because they close before New York.
- The Excel file also contains IBOR-swap histories back to 2000 and overnight rates (`data/bloomberg_extras_*.xlsx`), which are not used yet.

### The first export, and why the results changed

The project originally used `data/data_bloomberg.xlsx`, which had two defects:

1. **Tenor columns were shifted against the `Date` column,** by up to 12 rows, changing stepwise over time. Each "curve" mixed rates from different days.
2. **The EUR sheet was dated 1–2 days early.** Its 2023–2026 values were actually €STR OIS; the old `EUSWE` (EONIA OIS) ticker stopped in December 2021.

NB1 can still load that file (`DATA_SOURCE = "original"`) and repairs both defects (§2b, §2c). The fresh pull confirmed the repair: in 2023–2026 the repaired values matched Bloomberg exactly on **97.9%** of (date, tenor) points, against **40.9%** for the unrepaired export. The repaired 2023–2026 dataset is kept as `data/df_multi_repaired_2023_2026.csv`, and the raw export as `data/df_multi_unaligned.csv`.

The misalignment created spurious next-day predictability. On the original export the CVAE slope strategy showed an out-of-sample Sharpe of **3.99** (butterfly 5.12, cross-currency 4.57). On days where the raw columns happened to be aligned, the same strategy earned 0.006 bp/day (t = 0.06), so those Sharpes were artefacts (`scripts/audit/`).

## Key results (clean data, NB2–NB4)

**Reconstruction, in-sample.** Models are trained on all 1,598 days. Values are mean RMSE per currency (range across the 9 currencies).

| Model | Latent dim | Mean RMSE (bp) |
|---|---|---|
| Nelson-Siegel, λ = 0.45 (NB2 cross-validation) | 3 | 2.34–9.07 |
| Single-currency VAE | 2 | 2.55–6.02 |
| Multi-currency VAE (+ post-processing) | 2 | 3.89–7.33 (beats NS in 5/9 currencies) |
| **Multi-currency CVAE (+ post-processing)** | 2 | **2.53–4.47** (beats NS in 7/9; not CZK 4.47 vs 4.35, GBP 2.53 vs 2.34) |
| eNS (λ re-fitted per date) | 3 + λ | 2.94 pooled |
| Nelson-Siegel-Svensson | 4 | 0.56 pooled |

The per-currency table is in NB3 ("Per-currency replication check"). The two currencies where NS wins are close calls that depend on the training seed: an earlier run with a different seed had the CVAE winning in 8/9.

**Reconstruction, out of sample.** Models are trained on dates up to 31 Dec 2024 and evaluated on the 473 test days.

| Model | In-sample RMSE (bp) | Out-of-sample RMSE (bp) |
|---|---|---|
| CVAE (NB3 §13b) | 3.12 | 7.50 |
| Multi-VAE (NB3 §13) | 5.53 | 10.18 |
| NS, λ = 0.45, fitted per date | 5.67 | 4.45 |

With about four years of training data, the autoencoders do not generalise well to 2025–2026 curve shapes. Per-date NS does not need training and is much more accurate out of sample.

**Trading (NB4).** Equal-weight portfolios across 9 currencies, OOS (473 days), raw-bp P&L, 0.5 bp per position change (1 bp for the butterfly), signals from train-only models.

| Strategy | OOS Sharpe |
|---|---|
| A: CVAE slope residual, fixed / tuned HP | 0.33 / 0.71 |
| NS slope residual | −1.16 |
| Naive slope z-score | −0.49 |
| B: CVAE butterfly residual | −0.28 |
| NS / naive butterfly | −1.85 / 0.16 |
| Ext. 2: cross-currency, all 36 pairs | 0.41 |
| Ext. 2: close / far pairs (train-window latent distance) | 1.14 / −0.26 |

With 473 days the standard error of a Sharpe ratio is about 0.73, so nothing here differs significantly from zero, and the best numbers come after choosing among many variants. In Extension 1 the average carry Sharpe (0.17) is comparable to the signal's (0.10).

## Relative-value curve arbitrage & anomaly detection (NB5)

`05_rv_anomaly_strategy.ipynb` turns the AEMM into an institutional RV process.

**Timeline (no look-ahead):** the CVAE and multi-VAE are fitted on data up to 2023. 2024 is a calibration year with no trading. The strategy trades 2025–2026, the same 473 days as NB4. Every threshold is a trailing statistic of out-of-sample model outputs.

1. **Signal generation & monitoring**
   - The fair value S' is the CVAE reconstruction with Sokol's post-processing gradient descent (§2.3.1). The residual is r = S − S' per tenor, and D_L2 is reported as an RMSE in bp.
   - D_L2 is ranked against its trailing 250-day out-of-sample history: the 90th percentile marks a dislocation and the 99th an extreme.
   - A **drift ratio** compares the 20-day D_L2 with its 2024 level.
   - Latent **zones** come from the multi-VAE's shared space, following Sokol §2.3.3 and Fig. 7:
     - Zone 1: inside the currency's own historical envelope.
     - Zone 2: inside another currency's envelope.
     - Zone 3: outside every envelope, i.e. extrapolation ("terra incognita").
2. **Residual decomposition & trade construction**
   - Per-tenor residual z-scores and a concentration measure show where the curve leaves the manifold.
   - The residual is projected on 8 butterflies and 4 steepeners/flatteners.
   - Butterfly weights are **model-neutral**: they solve J_legsᵀ w = 0, where J is the decoder Jacobian, so they are immune to moves along the manifold. 79% of flies are well-conditioned; the rest use 50/50 weights.
3. **Mean reversion & execution**
   - Enter the most dislocated package (|z| ≥ 2, z and dislocation of the same sign, dislocation ≥ 2× round-trip cost) when the curve is flagged.
   - Fade the dislocation. Fill at the next close, size each trade to 1 bp/day of volatility, and pay 0.25 bp per leg per unit DV01 each way.
   - Take profit at |z| ≤ 0.5; time stop at 20 days.
4. **Risk controls**
   - Each currency-day is classified as one of three regimes:

     | Class | Conditions | Action |
     |---|---|---|
     | **Liquidity dislocation** | Zone 1, localised residual | full size |
     | **Regime shift** | Zone 3, or 5 extreme days, or drift ratio > 2, or a broad multi-currency distortion | no entries, exit, 20-day cooldown, retraining flag |
     | **Uncertain** | anything else | half size |

   - Per-trade stops: a signal stop (|z| ≥ 4) and a P&L stop (loss > dislocation).
   - Portfolio stop: a 15 bp-risk drawdown from the high-water mark flattens the book.

**Results.** Trading window 2025–2026, 9 currencies, Sharpe SE ≈ 0.73.

| Variant | Sharpe | Total (bp-risk) | Trades |
|---|---|---|---|
| AEMM RV, full | −1.70 | −31.6 | 34 |
| AEMM RV, no risk controls | −2.72 | −206.1 | 95 |
| Classical engine (NS fair value, 50/50 flies, no zones) | −1.06 | −30.6 | 50 |
| AEMM RV, same-close execution | −1.02 | −21.9 | 33 |
| AEMM RV, zero costs | −0.56 | −11.0 | 36 |

- **Out of sample the model drifts.** Mean D_L2 rises from 2–5 bp in the fit window to 6–18 bp in 2025–26. The regime classifier stands aside on 49% of currency-days.
- **The risk controls work as intended:** they cut the loss from −206 to −32 bp-risk units.
- **The dislocations do not mean-revert in price.** Even with zero costs the strategy loses, and it does not beat the classical engine. Against a drifting model, a residual can close because the fair value moves rather than the market.
- **The framework is the deliverable; there is no alpha on this sample.** The natural next step is the governance loop the regime flags call for: periodic walk-forward retraining.
- **Caveats:**
  - The multi-VAE latent space is close to one-dimensional (corr(z₁, z₂) ≈ 0.98 for most currencies), so the zone map is coarse.
  - The drift detector and the cost-aware edge were added after an independent code review that had seen the 2025–26 model drift.


## Research: is the autoencoder residual a tradable mispricing? (NB6)

`06_rv_research.ipynb` uses the same timeline as NB5. Models are fitted on 2020–2023 and evaluated on 2024 → Oct 2026 (735 days, all out of sample).

1. **Predictive power.** We compare residual signals (VAE, PCA-2, PCA-3, NS) against *hedged* future moves at 1, 5, 10 and 20 days:
   - per tenor, the move net of level and slope;
   - per package: DV01-neutral spreads, and butterflies neutral to level and slope.

   The **immediate IC**, measured from the signal's own close, is +0.10 to +0.13. The **tradable IC**, measured from the fill one day later, is only **+0.02 to +0.03**, and the tradable hit rate is about 49–50%.
   - About 75% of the apparent reversion happens on the next print, before a position can be filled.
   - The VAE is no better than PCA or NS.
2. **Portfolios with a full P&L decomposition.** The rates P&L is split into rates, carry, rolldown and costs:
   - **carry** uses the actual floating index: overnight rates for OIS, 6M CIBOR/PRIBOR for DKK/CZK;
   - **rolldown** comes from curve interpolation, including the 1Y tenor.

   Gross (rates-only) Sharpe is about 0 for every model (VAE −0.02). After removing the known factor exposure, the residual alpha is −0.32 for the VAE (SE ≈ 0.6). Costs dominate the net result. Even at zero cost the net Sharpe is −0.3 to −0.6.
3. **Regimes.** A Markov-switching model with switching variance is fitted on the VAE latent factors, using **filtered** probabilities only. The results run against the usual intuition:
   - residuals revert *faster* in the high-volatility regime (half-life 36 days vs 69 days);
   - the tradable IC is higher there;
   - trading only in the low-volatility regime makes results worse.
4. **Training objective.** The same encoder is trained on reconstruction, on prediction of 5- and 20-day moves, or on both (multitask):
   - The prediction-trained model overfits: its OOS R² against a random walk is −4.0 at 5 days, and its tradable IC is negative.
   - The multitask model reconstructs slightly better out of sample (8.3 vs 8.9 bp) but forecasts nothing (IC ≈ 0).
   - The best forecaster is a plain ridge regression, with a tiny edge.

**Answer.** On this sample the autoencoder residual behaves like **model error plus next-day quote noise, not a tradable mispricing**:
- it is persistent, with a half-life of 40–70 days, which fits slow model drift;
- the part that reverts quickly is not capturable;
- it is no more informative than PCA or Nelson-Siegel residuals.

Considered and left out:
- **HJM / no-arbitrage constrained models:** a separate pricing project, with no alpha question.
- **FX-hedged cross-currency trades:** already covered by NB4's Extension 2.

## Known limitations

- **Sample length.** About 6 years of data versus 22 in the paper, and one train/test split.
- **Data before the benchmark launches.** SOFR and €STR swap histories before their launch (2018/2019) are Bloomberg back-fills. They lie outside the window used here.
- **Wilcoxon p-values in NB3** treat strongly autocorrelated daily RMSEs as independent, so they overstate significance.
- **NS λ.** The cross-validation curve is flat and can have two local minima (around 0.05 and 0.5). With 9 currencies, NB2 picks λ = 0.45 on the full sample, and NB4/NB5 re-select λ on their training windows only (0.45 in both).
- **Carry (NB4 Ext. 1)** is a simplified approximation, `(r₂ − r₁₀)/252` per unit of spread DV01, with no floating legs or roll-down.
- **Training stability.** VAE training is seed-dependent and can stall on a plateau: a single-currency VAE once reached 114 bp RMSE. NB3 now retries a stalled single-currency VAE with another seed (up to 3 attempts) and re-seeds before each multi-currency model, so results no longer depend on cell order. Close per-currency comparisons with NS can still flip with the seed (`scripts/audit/exp4_seed_analysis.py`).
- **Latent space.** The multi-VAE's two latent coordinates are highly correlated (≈ 0.98 for most currencies), so the latent "world map" and NB5's zones are close to one-dimensional.
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
├── 04_trading_signals.ipynb   # 2s10s slope / butterfly strategies + carry and cross-currency extensions
├── 05_rv_anomaly_strategy.ipynb  # relative-value curve arbitrage & anomaly detection (NB5)
├── 06_rv_research.ipynb          # is the AE residual a tradable mispricing? IC, P&L decomposition, regimes, objectives (NB6)
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
3. **Run the notebooks in order**: NB1 → NB2 → NB3 → NB4 → NB5 → NB6. Each notebook saves its outputs into `results/` and `figures/` for the next one. NB3 takes about 40 minutes on a laptop CPU, NB5 about 10 (it trains its own models on data up to 2023), and NB6 about 6 (it reuses NB5's CVAE and trains two more models).
4. **Optional:** run the audit scripts in `scripts/audit/` (see its README).

## References

- Sokol, A. (2022). *Autoencoder Market Models for Interest Rates.* SSRN 4300756.
- Kingma, D.P. & Welling, M. (2013). *Auto-Encoding Variational Bayes.* arXiv:1312.6114.
- Nelson, C.R. & Siegel, A.F. (1987). *Parsimonious Modeling of Yield Curves.* J. Business, 60(4).
