# Autoencoder Market Models for Interest Rates

Replication and extension of Sokol (2022), *"Autoencoder Market Models for Interest Rates"* (SSRN 4300756), for the *Topics in Derivative Pricing* coursework (Prof. V. Piterbarg).

**Authors:** Diane Murzi, Chedi Mnif, Hamza El Arji

## What this project does

1. Replicates Sokol's core empirical claim: a **2D Conditional VAE** matches or beats the **3D Nelson-Siegel** basis on every currency, with Wilcoxon $p<10^{-10}$.
2. Implements three AEMM variants from Chapters 3–4 of the paper (forward rate, multi-factor short rate, autoregressive P-measure).
3. Extends the framework with a **CVAE-driven trading strategy** on the 2s10s swap slope: OOS portfolio Sharpe **3.99** vs 2.18 (naive) vs −1.38 (NS residual), on 208 business days across 7 currencies.

## Data

- Source: Bloomberg OIS swap rates
- **7 currencies**: AUD, CAD, DKK, EUR, GBP, JPY, USD
- **7 tenors**: 2, 3, 5, 10, 15, 20, 30 Y
- Daily, 30 Jan 2023 → 15 Jan 2026 (774 business days)
- Train/test split: 31 Mar 2025 (70/30)

## Repository layout

```
├── data/                      # raw + cleaned data (not versioned)
│   ├── data_bloomberg.xlsx
│   └── df_multi.csv           # produced by NB1
├── figures/                   # all PNGs produced by the notebooks
├── tests/                     # tests during the implementation
├── results/                   # pickles and others shared between notebooks
├── notebooks/
│   ├── 01_data_loading.ipynb      # load, clean, align Bloomberg data
│   ├── 02_nelson_siegel.ipynb     # NS / eNS / NSS benchmark + λ cross-validation
│   ├── 03_vae_training.ipynb      # VAE, Multi-VAE, CVAE + 3 AEMM variants
│   └── 04_trading_signals.ipynb   # flagship strategy + multi-strategy extensions
├── papers and reports/            # full LaTeX report
│                 
└── README.md
```

## How to run

1. **Install dependencies**
   ```bash
   pip install numpy pandas scipy matplotlib torch openpyxl
   ```
2. **Place the Bloomberg Excel** at `data/data_bloomberg.xlsx`.
3. **Run the notebooks in order** (NB1 → NB2 → NB3 → NB4). Each notebook saves its outputs into `results/` and `figures/` for the next one to consume.


## References

- Sokol, A. (2022). *Autoencoder Market Models for Interest Rates.* SSRN 4300756.
- Kingma, D.P. & Welling, M. (2013). *Auto-Encoding Variational Bayes.* arXiv:1312.6114.
- Nelson, C.R. & Siegel, A.F. (1987). *Parsimonious Modeling of Yield Curves.* J. Business, 60(4).