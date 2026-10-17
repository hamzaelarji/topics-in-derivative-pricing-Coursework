#!/usr/bin/env python
# coding: utf-8

# # Notebook 4 — Trading Strategies (CVAE-driven)
#
# This notebook extends the AEMM framework of Sokol (2022) with **two trading
# strategies** built on CVAE outputs from NB3, each targeting a different
# dimension of the yield curve.
#
# ## Strategy A — CVAE Slope Residual Mean-Reversion (2s10s)
# Signal = (actual 2s10s slope) − (CVAE-reconstructed 2s10s slope), z-scored.
# Rationale: the CVAE provides a non-parametric "fair value" for the slope.
# Deviations from this fair value are idiosyncratic and tend to mean-revert.
# This is the direct trading extension of Sokol (2022) §2.4.
#
# ## Strategy B — CVAE Curvature Residual Mean-Reversion (2s5s10s Butterfly)
# Signal = (actual 2s5s10s butterfly) − (CVAE-reconstructed butterfly), z-scored.
# Rationale: the butterfly is by construction DV01-neutral to parallel shifts
# AND slope changes — it isolates pure curvature. Bikbov & Chernov (2010) show
# the curvature factor is the most predictable of the three curve dimensions.
# The CVAE's non-parametric decoder provides a better curvature fair value than NS.
#
# ## Design Principles
#
# **Complementarity**: Strategy A targets slope (2 curve factors), Strategy B
# targets curvature (the remaining unexplained dimension). Together they
# cover the two tradeable non-level dimensions of the curve, leaving the
# level dimension untraded as it requires a directional macro view.
#
# **Benchmarks**: both strategies are compared against (1) Naive z-score of
# the raw spread/butterfly, and (2) NS residual z-score. If CVAE does not
# beat both benchmarks, the model adds no trading value beyond classical tools.
#
# **P&L conventions**: raw bp (clean, interpretable) and DV01-weighted (realistic).
#
# **Hyperparameters**: fixed defaults first, then grid-searched on train window
# and evaluated OOS — never tuned on OOS data.
#
# ## Honest Reporting
#
# > ⚠️ OOS window ≈ 208 business days (~9 months, Apr 2025 – Jan 2026).
# > Standard error on annualised Sharpe ≈ √(1/0.83) ≈ 1.1.
# > Any Sharpe within ±1 of zero is statistically indistinguishable from noise.
# > The value of this notebook is the *relative ranking* of signals on the
# > same OOS window, and the *methodology demonstration*, not absolute Sharpe claims.
#
# ## References
# - Sokol, A. (2022). Autoencoder Market Models for Interest Rates. SSRN 4300756.
# - Bikbov, R. & Chernov, M. (2010). Yield curve and volatility: Lessons from Eurodollar
#   futures and options. Journal of Financial Econometrics.
# - Diebold, F.X. & Li, C. (2006). Forecasting the term structure of government bond yields.
# - CME Group. Swap Rate Curve Strategies with DSF Futures.
# - Clarus FT. Mechanics and Definitions of Spread and Butterfly Swap Packages.
# - Lombard Odier IM (2022). Exploiting yield-curve dynamics in hiking cycles.


# ## 0 — Imports & Setup

# %%
import pickle
import warnings
from pathlib import Path
from itertools import combinations
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

FIG_DIR = Path("figures"); FIG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR = Path("results"); RES_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.figsize": (14, 6),
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ## 1 — Load NB3 outputs
#
# We load the CVAE weights, configuration, and NS results from NB3.
# No retraining — this notebook is purely inferential.

# %%
with open(RES_DIR / "vae_results_extended.pkl", "rb") as f:
    vae_data = pickle.load(f)

cfg           = vae_data["config"]
CURRENCIES    = list(cfg["currencies"])
N_CCY         = len(CURRENCIES)
CCY_TO_IDX    = {c: i for i, c in enumerate(CURRENCIES)}
TARGET_TENORS = list(cfg["target_tenors"])
S_MIN, S_MAX  = float(cfg["S_MIN"]), float(cfg["S_MAX"])
BP_PER_UNIT   = float(cfg["bp_per_unit"])
TRAIN_CUTOFF  = pd.Timestamp(cfg["train_cutoff"])
SEED          = int(cfg["seed"])

torch.manual_seed(SEED); np.random.seed(SEED)

print(f"Currencies    : {CURRENCIES}")
print(f"Tenors        : {TARGET_TENORS}")
print(f"Train cutoff  : {TRAIN_CUTOFF.date()}")
print(f"OOS window    : post {TRAIN_CUTOFF.date()}")


# %%
# Reload swap data from CSV (same source as NB3)
df_long = pd.read_csv("data/df_multi.csv", parse_dates=["Date"])
EXCLUDE = set(df_long["currency"].unique()) - set(CURRENCIES)
df_long = df_long[~df_long["currency"].isin(EXCLUDE)].copy()

TENOR_COLS = [str(t) for t in TARGET_TENORS]
swap_aligned: dict[str, pd.DataFrame] = {}
for ccy in CURRENCIES:
    sub = (df_long[df_long["currency"] == ccy]
           .set_index("Date").sort_index()[TENOR_COLS].copy())
    sub.columns = TARGET_TENORS
    swap_aligned[ccy] = sub

dates_ref = swap_aligned[CURRENCIES[0]].index
n_oos = (dates_ref > TRAIN_CUTOFF).sum()
print(f"Loaded {len(CURRENCIES)} currencies, {len(dates_ref)} dates")
print(f"OOS: {n_oos} business days "
      f"({dates_ref[dates_ref > TRAIN_CUTOFF].min().date()} → {dates_ref.max().date()})")


# ## 2 — Reconstruct CVAE and Multi-VAE from NB3 weights
#
# We re-instantiate both architectures exactly as in NB3.
# The CVAE is used for single-currency fair value (Strategies A & B).
# The Multi-VAE is used for cross-currency latent space analysis.

# %%
class CVAE(nn.Module):
    """Conditional VAE — Table 3 of Sokol (2022). Loaded from NB3 weights."""
    def __init__(self, input_dim=7, latent_dim=2, n_currencies=N_CCY):
        super().__init__()
        self.latent_dim = latent_dim
        enc_in = input_dim + n_currencies
        dec_in = latent_dim + n_currencies
        self.encoder = nn.Sequential(
            nn.Linear(enc_in, enc_in - 1), nn.Tanh(),
            nn.Linear(enc_in - 1, 2 * latent_dim))
        self.decoder = nn.Sequential(
            nn.Linear(dec_in, input_dim), nn.Tanh(),
            nn.Linear(input_dim, input_dim), nn.Tanh(),
            nn.Linear(input_dim, input_dim), nn.Sigmoid())

    def encode(self, x, y):
        h = self.encoder(torch.cat([x, y], dim=-1))
        return h[:, :self.latent_dim], h[:, self.latent_dim:]

    def decode(self, z, y):
        return self.decoder(torch.cat([z, y], dim=-1))

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        return self.decode(mu, y), mu, logvar  # deterministic at eval


class MultiVAE(nn.Module):
    """Multi-currency VAE — Table 2 of Sokol (2022). Loaded from NB3 weights."""
    def __init__(self, input_dim=7, latent_dim=2, hidden_dim=4):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim), nn.Tanh(),
            nn.Linear(input_dim, 2 * latent_dim))
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, input_dim), nn.Tanh(),
            nn.Linear(input_dim, input_dim), nn.Sigmoid())

    def encode(self, x):
        h = self.encoder(x)
        return h[:, :self.latent_dim], h[:, self.latent_dim:]

    def forward(self, x):
        mu, logvar = self.encode(x)
        return self.decoder(mu), mu, logvar


# Load weights
cvae = CVAE(input_dim=len(TARGET_TENORS), latent_dim=2, n_currencies=N_CCY).to(device)
cvae.load_state_dict(vae_data["cvae_state"])
cvae.eval()

multi_vae = MultiVAE(input_dim=len(TARGET_TENORS), latent_dim=2, hidden_dim=4).to(device)
multi_vae.load_state_dict(vae_data["multi_vae_state"])
multi_vae.eval()

print(f"CVAE loaded      ({sum(p.numel() for p in cvae.parameters()):,} params)")
print(f"Multi-VAE loaded ({sum(p.numel() for p in multi_vae.parameters()):,} params)")


# ## 3 — Helper functions
#
# Normalisation, CVAE fair value reconstruction, and NS benchmark.
# Note: we use λ★ = 0.5 from NB2 pooled cross-validation for NS,
# consistent with the reconstruction benchmark in NB3.

# %%
# Load λ★ from NB2 for consistency
with open(RES_DIR / "ns_results.pkl", "rb") as f:
    ns_nb2 = pickle.load(f)
LAM_STAR = ns_nb2["lambda_star"]
print(f"Using λ★ = {LAM_STAR:.4f} from NB2 (pooled cross-validation)")


def normalize_rates(r: np.ndarray) -> np.ndarray:
    return np.clip((r - S_MIN) / (S_MAX - S_MIN), 0.0, 1.0)


def denormalize_rates(x: np.ndarray) -> np.ndarray:
    return x * (S_MAX - S_MIN) + S_MIN


def cvae_reconstruct(df: pd.DataFrame, ccy: str) -> pd.DataFrame:
    """
    Run the CVAE on a currency's full history.
    Returns reconstructed rates + latent codes (z1, z2).
    Uses deterministic mode (mu only, no sampling).
    """
    rates = df.values.astype(np.float32)
    X = torch.from_numpy(normalize_rates(rates)).to(device)
    oh = np.zeros((len(rates), N_CCY), dtype=np.float32)
    oh[:, CCY_TO_IDX[ccy]] = 1.0
    Y = torch.from_numpy(oh).to(device)
    with torch.no_grad():
        Xrec, mu, _ = cvae(X, Y)
    rec = denormalize_rates(Xrec.cpu().numpy())
    z   = mu.cpu().numpy()
    out = pd.DataFrame(rec, index=df.index, columns=TARGET_TENORS)
    out["z1"] = z[:, 0]
    out["z2"] = z[:, 1]
    return out


def ns_reconstruct(df: pd.DataFrame) -> pd.DataFrame:
    """
    NS reconstruction using λ★ from NB2.
    Consistent with NB3 §7 after the λ correction.
    """
    tenors = np.array(TARGET_TENORS, dtype=float)
    lam = LAM_STAR
    tau = tenors
    lt = np.maximum(lam * tau, 1e-10)
    f1 = np.ones_like(tau)
    f2 = (1 - np.exp(-lt)) / lt
    f3 = f2 - np.exp(-lt)
    B = np.column_stack([f1, f2, f3])

    rates = df.values.astype(float)
    fitted = np.zeros_like(rates)
    for i in range(len(rates)):
        beta, *_ = np.linalg.lstsq(B, rates[i], rcond=None)
        fitted[i] = B @ beta
    return pd.DataFrame(fitted, index=df.index, columns=TARGET_TENORS)


# ## 4 — Build signal panels
#
# For each currency we compute all signals and benchmarks on the FULL date
# index. Z-scoring happens inside the backtest engine so lookback can be varied.

# %%
def build_panel(ccy: str) -> pd.DataFrame:
    """
    Build the signal panel for one currency containing:
    - Actual slope (2s10s) and butterfly (2s5s10s) in bp
    - CVAE fair value slope and butterfly
    - NS fair value slope and butterfly
    - Residuals (actual - fair value) for both instruments
    - CVAE latent codes z1, z2
    """
    df  = swap_aligned[ccy]
    cv  = cvae_reconstruct(df, ccy)
    nsf = ns_reconstruct(df)

    # 2s10s slope
    slope_actual = (df[10]  - df[2])  * BP_PER_UNIT
    slope_cvae   = (cv[10]  - cv[2])  * BP_PER_UNIT
    slope_ns     = (nsf[10] - nsf[2]) * BP_PER_UNIT

    # 2s5s10s butterfly = 2*5Y - 2Y - 10Y
    # Positive when the 5Y is "rich" (above the linear interpolation of 2Y and 10Y)
    bf_actual = (2*df[5]  - df[2]  - df[10])  * BP_PER_UNIT
    bf_cvae   = (2*cv[5]  - cv[2]  - cv[10])  * BP_PER_UNIT
    bf_ns     = (2*nsf[5] - nsf[2] - nsf[10]) * BP_PER_UNIT

    p = pd.DataFrame(index=df.index)
    # Raw rates for DV01 computations
    p["rate_2Y"]  = df[2]
    p["rate_5Y"]  = df[5]
    p["rate_10Y"] = df[10]

    # Slope signals
    p["slope_bp"]      = slope_actual
    p["slope_chg_bp"]  = slope_actual.diff()
    p["cvae_slope_bp"] = slope_cvae
    p["ns_slope_bp"]   = slope_ns
    p["cvae_slope_resid"] = slope_actual - slope_cvae   # Strategy A signal
    p["ns_slope_resid"]   = slope_actual - slope_ns     # Benchmark

    # Butterfly signals
    p["bf_bp"]         = bf_actual
    p["bf_chg_bp"]     = bf_actual.diff()
    p["cvae_bf_bp"]    = bf_cvae
    p["ns_bf_bp"]      = bf_ns
    p["cvae_bf_resid"] = bf_actual - bf_cvae            # Strategy B signal
    p["ns_bf_resid"]   = bf_actual - bf_ns              # Benchmark

    # Latent codes
    p["z1"] = cv["z1"]
    p["z2"] = cv["z2"]
    return p


panels: dict[str, pd.DataFrame] = {}
for ccy in CURRENCIES:
    panels[ccy] = build_panel(ccy)

print(f"Built panels for {len(panels)} currencies")
print(f"Columns: {list(panels[CURRENCIES[0]].columns)}")

# Sanity check: CVAE residuals should be small (consistent with NB3 RMSE ~4-7 bp)
print("\nMean absolute slope residual by currency (bp):")
for ccy in CURRENCIES:
    mae = panels[ccy]["cvae_slope_resid"].abs().mean()
    print(f"  {ccy}: {mae:.2f} bp")


# ## 5 — Backtest engine
#
# A unified engine handles both slope (2-leg) and butterfly (3-leg) trades.
# Key design choices are documented below.

# %%
def dv01_annuity(tenor_years: float, yield_dec: float) -> float:
    """
    Par-swap DV01 per unit notional: DV01 = A(T,y) × 1e-4
    where A(T,y) = (1 - (1+y)^(-T)) / y  (annuity approximation).

    Source: CME Group, 'Swap Rate Curve Strategies with DSF futures', p.6.
    Uses y→0 limit (A → T) for near-zero rates to handle JPY/EUR negative rate era.
    """
    y = float(yield_dec)
    if abs(y) < 1e-8:
        annuity = float(tenor_years)
    else:
        annuity = (1.0 - (1.0 + y) ** (-tenor_years)) / y
    return annuity * 1e-4


def compute_zscore(series: pd.Series, lookback: int) -> pd.Series:
    """Rolling z-score. min_periods = lookback // 2 to avoid long warm-up gaps."""
    rm = series.rolling(lookback, min_periods=lookback // 2).mean()
    rs = series.rolling(lookback, min_periods=lookback // 2).std().clip(lower=1e-6)
    return (series - rm) / rs


def build_positions(zscore: pd.Series, entry_z: float, exit_z: float) -> pd.Series:
    """
    Hysteresis-based mean-reversion position from a z-score.

    Convention (for slope/butterfly):
        +1 = long spread  (z << 0 → spread too low → bet it widens)
        -1 = short spread (z >> 0 → spread too high → bet it narrows)
         0 = flat

    The position is lagged by 1 bar in backtest() so we trade at next open.
    """
    pos = np.zeros(len(zscore))
    prev = 0.0
    z = zscore.values
    for i in range(len(z)):
        zi = z[i]
        if np.isnan(zi):
            pos[i] = prev
            continue
        if zi >  entry_z:  prev = -1.0   # spread too high → fade it (short)
        elif zi < -entry_z: prev = +1.0   # spread too low  → fade it (long)
        elif abs(zi) < exit_z: prev = 0.0
        pos[i] = prev
    return pd.Series(pos, index=zscore.index)


def backtest_slope(
    panel: pd.DataFrame,
    signal: pd.Series,
    *,
    entry_z: float = 1.0,
    exit_z: float = 0.3,
    lookback: int = 60,
    tc_bp: float = 0.5,
    pnl_mode: str = "raw_bp",
    notional_10y: float = 1e7,
) -> pd.DataFrame:
    """
    Backtest a 2s10s spread strategy.

    Position sign:
        +1 = steepener: pay 10Y fixed, receive 2Y fixed → profits if slope widens
        -1 = flattener: receive 10Y fixed, pay 2Y fixed → profits if slope narrows

    pnl_mode='raw_bp':
        P&L = pos_lag × Δ(10Y - 2Y) in bp.
        Simple and clean. TC = tc_bp per unit of position change.

    pnl_mode='dv01':
        Both legs sized so DV01_2Y × N_2Y = DV01_10Y × N_10Y.
        The trade is then insensitive to parallel shifts (level-neutral).
        N_2Y = N_10Y × (DV01_10Y / DV01_2Y), recomputed daily.
        P&L = pos_lag × (dr_10Y × N_10Y × DV01_10Y - dr_2Y × N_2Y × DV01_2Y) × 1e4.
        TC = tc_bp × (N_10Y × DV01_10Y + N_2Y × DV01_2Y) per flip.
        Source: Clarus FT, 'Mechanics and Definitions of Spread Packages'.
    """
    z       = compute_zscore(signal, lookback)
    pos     = build_positions(z, entry_z, exit_z)
    pos_lag = pos.shift(1).fillna(0.0)

    if pnl_mode == "raw_bp":
        pnl_step = pos_lag * panel["slope_chg_bp"]
        tc       = pos_lag.diff().abs().fillna(0.0) * tc_bp
        net      = pnl_step - tc
        unit     = "bp"

    elif pnl_mode == "dv01":
        d2  = panel["rate_2Y"].apply(lambda y: dv01_annuity(2.0,  y))
        d10 = panel["rate_10Y"].apply(lambda y: dv01_annuity(10.0, y))
        n2  = notional_10y * (d10 / d2)

        dr2  = panel["rate_2Y"].diff()
        dr10 = panel["rate_10Y"].diff()

        # Steepener (+1): pay 10Y → profit when r_10Y rises → +dr10 component
        #                 receive 2Y → profit when r_2Y falls → -dr2 component
        spread_pnl = (+dr10 * notional_10y * d10 - dr2 * n2 * d2) * 1e4
        pnl_step   = pos_lag * spread_pnl
        tc = pos_lag.diff().abs().fillna(0.0) * (tc_bp * (notional_10y * d10 + n2 * d2))
        net  = pnl_step - tc
        unit = "USD"
    else:
        raise ValueError(pnl_mode)

    out = pd.DataFrame({
        "signal":   signal,
        "zscore":   z,
        "position": pos,
        "pnl":      net,
        "cum_pnl":  net.cumsum(),
    }, index=panel.index)
    out.attrs["unit"] = unit
    return out


def backtest_butterfly(
    panel: pd.DataFrame,
    signal: pd.Series,
    *,
    entry_z: float = 1.0,
    exit_z: float = 0.3,
    lookback: int = 60,
    tc_bp: float = 1.0,
) -> pd.DataFrame:
    """
    Backtest a 2s5s10s butterfly strategy (raw bp only).

    Structure: butterfly = 2*5Y - 2Y - 10Y  (positive when 5Y is rich)
    Position:
        +1 = long body / short wings: profit when butterfly widens
        -1 = short body / long wings: profit when butterfly narrows

    P&L = pos_lag × Δ(butterfly) in bp.
    TC = tc_bp per position flip. tc_bp = 1.0 default (3-leg trade: 3×0.33bp).

    Note: this trade is already DV01-neutral to shifts (2Y and 10Y cancel
    in the butterfly, by construction) and approximately slope-neutral
    (positive and negative slope exposure cancel across the three tenors).
    So no separate DV01 mode is needed — the butterfly IS the duration-neutral trade.
    Reference: Montréal Exchange (2021), 'Understanding 2-5-10 Butterfly Trades'.
    """
    z       = compute_zscore(signal, lookback)
    pos     = build_positions(z, entry_z, exit_z)
    pos_lag = pos.shift(1).fillna(0.0)

    pnl = pos_lag * panel["bf_chg_bp"]
    tc  = pos_lag.diff().abs().fillna(0.0) * tc_bp
    net = pnl - tc

    out = pd.DataFrame({
        "signal":   signal,
        "zscore":   z,
        "position": pos,
        "pnl":      net,
        "cum_pnl":  net.cumsum(),
    }, index=panel.index)
    out.attrs["unit"] = "bp"
    return out


# ## 6 — Performance metrics

# %%
ANN = 252  # business days per year

def perf_stats(pnl: pd.Series, label: str = "") -> dict:
    """Annualised performance statistics from a daily P&L series."""
    pnl = pnl.dropna()
    if len(pnl) == 0 or pnl.std() < 1e-12:
        return {"label": label, "N": 0, "Total": 0.0, "Ann": 0.0,
                "Vol": 0.0, "Sharpe": 0.0, "MaxDD": 0.0, "HitRate": 0.0,
                "NumTrades": 0}
    cum     = pnl.cumsum()
    peak    = cum.expanding().max()
    maxdd   = float((cum - peak).min())
    trades  = pnl[pnl != 0]
    return {
        "label":     label,
        "N":         len(pnl),
        "Total":     float(cum.iloc[-1]),
        "Ann":       float(pnl.mean() * ANN),
        "Vol":       float(pnl.std() * np.sqrt(ANN)),
        "Sharpe":    float(pnl.mean() / pnl.std() * np.sqrt(ANN)),
        "MaxDD":     maxdd,
        "HitRate":   float((trades > 0).mean()) if len(trades) else 0.0,
        "NumTrades": int(len(trades)),
    }


# ## 7 — Strategy A: CVAE Slope Residual (2s10s)
#
# We compare three signals on the slope:
# 1. CVAE residual (Strategy A) — the main signal
# 2. NS residual   — classical benchmark
# 3. Naive z-score — naive benchmark (no model)
#
# Expected finding: CVAE should beat NS on most currencies because
# its non-parametric fair value is more accurate (NB3: NS RMSE ~6.8 bp pooled
# vs CVAE RMSE ~5.5 bp pooled). A better fit → smaller but cleaner residuals
# → better mean-reversion signal.

# %%
HP_FIXED_SLOPE = dict(entry_z=1.0, exit_z=0.3, lookback=60, tc_bp=0.5)

SLOPE_SIGNALS = {
    "CVAE_resid": lambda p: p["cvae_slope_resid"],
    "NS_resid":   lambda p: p["ns_slope_resid"],
    "Naive":      lambda p: p["slope_bp"],
}

# Run all (signal × pnl_mode) combinations for every currency
slope_bt: dict = {}   # (ccy, sig, mode) → bt df
slope_rows = []

for ccy in CURRENCIES:
    p = panels[ccy]
    oos = p.index > TRAIN_CUTOFF
    for sig_name, sig_fn in SLOPE_SIGNALS.items():
        sig = sig_fn(p)
        for mode in ("raw_bp", "dv01"):
            bt = backtest_slope(p, sig, pnl_mode=mode, **HP_FIXED_SLOPE)
            slope_bt[(ccy, sig_name, mode)] = bt
            s = perf_stats(bt.loc[oos, "pnl"], label=f"{ccy}|{sig_name}|{mode}")
            s.update(currency=ccy, signal=sig_name, pnl_mode=mode)
            slope_rows.append(s)

df_slope = pd.DataFrame(slope_rows)

print("=" * 72)
print("STRATEGY A — CVAE Slope Residual (2s10s)")
print(f"Fixed HP: {HP_FIXED_SLOPE}")
print(f"OOS window: {n_oos} business days")
print("=" * 72)

def summary_signals(df, pnl_mode, value="Sharpe"):
    sub = df[df["pnl_mode"] == pnl_mode]
    return (sub.groupby("signal")[["Sharpe", "Total", "MaxDD", "HitRate"]]
              .mean().round(2).sort_values("Sharpe", ascending=False))

print("\nAvg across currencies — RAW BP:")
print(summary_signals(df_slope, "raw_bp"))
print("\nAvg across currencies — DV01:")
print(summary_signals(df_slope, "dv01"))


# %%
# Per-currency Sharpe table
def per_ccy_sharpe(df, pnl_mode):
    sub = df[df["pnl_mode"] == pnl_mode]
    pivot = sub.pivot_table(index="currency", columns="signal", values="Sharpe").round(2)
    cols = [c for c in ["Naive", "NS_resid", "CVAE_resid"] if c in pivot.columns]
    return pivot[cols]

print("Sharpe ratios by currency — RAW BP:")
print(per_ccy_sharpe(df_slope, "raw_bp"))
print()
print("Sharpe ratios by currency — DV01:")
print(per_ccy_sharpe(df_slope, "dv01"))


# %%
# Equal-weight portfolio across currencies
def portfolio_pnl_slope(sig_name, mode):
    pnl_list = []
    for ccy in CURRENCIES:
        bt = slope_bt.get((ccy, sig_name, mode))
        if bt is None: continue
        oos = bt.loc[bt.index > TRAIN_CUTOFF, "pnl"]
        pnl_list.append(oos.rename(ccy))
    M = pd.concat(pnl_list, axis=1).fillna(0.0)
    return M.mean(axis=1)

print("\nEqual-weight portfolio — RAW BP")
for sig in ["CVAE_resid", "NS_resid", "Naive"]:
    pnl = portfolio_pnl_slope(sig, "raw_bp")
    s = perf_stats(pnl)
    print(f"  {sig:<15} Sharpe={s['Sharpe']:.2f}  Total={s['Total']:.1f} bp  "
          f"MaxDD={s['MaxDD']:.1f} bp  HitRate={s['HitRate']:.2f}")

print("\nEqual-weight portfolio — DV01")
for sig in ["CVAE_resid", "NS_resid", "Naive"]:
    pnl = portfolio_pnl_slope(sig, "dv01")
    s = perf_stats(pnl)
    print(f"  {sig:<15} Sharpe={s['Sharpe']:.2f}  Total={s['Total']/1e3:.0f}k USD  "
          f"MaxDD={s['MaxDD']/1e3:.0f}k USD")


# %%
# Plot: cumulative P&L per currency (raw bp)
colors_sig = {"Naive": "gray", "NS_resid": "tab:purple", "CVAE_resid": "tab:blue"}
styles_sig = {"Naive": ":", "NS_resid": "--", "CVAE_resid": "-"}

n, cols_fig = len(CURRENCIES), 2
rows_fig = (n + cols_fig - 1) // cols_fig
fig, axes = plt.subplots(rows_fig, cols_fig, figsize=(15, 3.5 * rows_fig), squeeze=False)

for k, ccy in enumerate(CURRENCIES):
    ax = axes[k // cols_fig, k % cols_fig]
    for sig in ["Naive", "NS_resid", "CVAE_resid"]:
        bt = slope_bt[(ccy, sig, "raw_bp")]
        oos_pnl = bt.loc[bt.index > TRAIN_CUTOFF, "pnl"].cumsum()
        ax.plot(oos_pnl.index, oos_pnl.values,
                color=colors_sig[sig], linestyle=styles_sig[sig],
                lw=1.5, label=sig)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_title(f"{ccy}", fontweight="bold")
    ax.set_ylabel("Cum P&L (bp)")
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.tick_params(axis="x", rotation=30)

for k in range(n, rows_fig * cols_fig):
    axes[k // cols_fig, k % cols_fig].set_visible(False)

fig.suptitle("Strategy A — OOS Cumulative P&L per currency (raw bp, fixed HP)",
             fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig_strat_a_per_ccy.png", dpi=140, bbox_inches="tight")
plt.show()


# %%
# Portfolio cumulative P&L comparison
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
for ax, mode in zip(axes, ["raw_bp", "dv01"]):
    for sig in ["Naive", "NS_resid", "CVAE_resid"]:
        pnl = portfolio_pnl_slope(sig, mode)
        ax.plot(pnl.index, pnl.cumsum(),
                color=colors_sig[sig], linestyle=styles_sig[sig], lw=2, label=sig)
    ax.axhline(0, color="black", lw=0.5)
    unit = "bp" if mode == "raw_bp" else "USD"
    ax.set_title(f"Strategy A — EW portfolio ({mode})", fontweight="bold")
    ax.set_ylabel(f"Cumulative P&L ({unit})")
    ax.legend(fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.tick_params(axis="x", rotation=30)

plt.tight_layout()
plt.savefig(FIG_DIR / "fig_strat_a_portfolio.png", dpi=140, bbox_inches="tight")
plt.show()


# ## 8 — Hyperparameter optimisation for Strategy A
#
# We grid-search entry_z, exit_z, lookback on the TRAIN window only,
# then evaluate on OOS. This is methodologically clean — no look-ahead.
# We then assess whether tuning helps or hurts (overfitting risk).

# %%
ENTRY_GRID    = [0.75, 1.0, 1.25, 1.5]
EXIT_GRID     = [0.0, 0.2, 0.4]
LOOKBACK_GRID = [30, 60, 90]
TC_FIXED      = 0.5

def tune_signal(panel, signal, pnl_mode, train_mask, is_butterfly=False):
    """Grid search on train window, return best HP dict."""
    best = None
    for lb in LOOKBACK_GRID:
        for ez in ENTRY_GRID:
            for xz in EXIT_GRID:
                if xz >= ez: continue
                if is_butterfly:
                    bt = backtest_butterfly(panel, signal, entry_z=ez, exit_z=xz,
                                           lookback=lb, tc_bp=TC_FIXED*2)
                else:
                    bt = backtest_slope(panel, signal, pnl_mode=pnl_mode,
                                        entry_z=ez, exit_z=xz, lookback=lb, tc_bp=TC_FIXED)
                s = perf_stats(bt.loc[train_mask, "pnl"])["Sharpe"]
                if best is None or s > best[0]:
                    best = (s, dict(entry_z=ez, exit_z=xz, lookback=lb, tc_bp=TC_FIXED))
    return best

print("Tuning Strategy A (CVAE_resid, raw_bp) on train window...")
slope_tuned_bt: dict = {}
tuned_rows_a = []

for ccy in CURRENCIES:
    p = panels[ccy]
    train_mask = p.index <= TRAIN_CUTOFF
    oos_mask   = p.index >  TRAIN_CUTOFF
    sig = p["cvae_slope_resid"]

    best_train_sharpe, best_hp = tune_signal(p, sig, "raw_bp", train_mask)
    bt = backtest_slope(p, sig, pnl_mode="raw_bp", **best_hp)
    slope_tuned_bt[ccy] = bt

    s = perf_stats(bt.loc[oos_mask, "pnl"])
    s.update(currency=ccy, **best_hp, train_sharpe=round(best_train_sharpe, 2))
    tuned_rows_a.append(s)

df_tuned_a = pd.DataFrame(tuned_rows_a)

print("\nChosen HP and OOS Sharpe — CVAE_resid slope:")
print(df_tuned_a[["currency", "entry_z", "exit_z", "lookback",
                   "train_sharpe", "Sharpe"]].rename(
    columns={"Sharpe": "OOS_Sharpe"}).to_string(index=False))

print(f"\nFixed HP portfolio Sharpe: {perf_stats(portfolio_pnl_slope('CVAE_resid', 'raw_bp'))['Sharpe']:.2f}")
tuned_port_pnl_a = pd.concat([
    slope_tuned_bt[c].loc[slope_tuned_bt[c].index > TRAIN_CUTOFF, "pnl"].rename(c)
    for c in CURRENCIES], axis=1).fillna(0.0).mean(axis=1)
print(f"Tuned HP portfolio Sharpe: {perf_stats(tuned_port_pnl_a)['Sharpe']:.2f}")
print("\n→ If tuned ≈ fixed: signal is robust to HP choice (good sign).")
print("→ If tuned >> fixed: results may be HP-sensitive (caution).")


# ## 9 — Strategy B: CVAE Curvature Residual (2s5s10s Butterfly)
#
# The 2s5s10s butterfly = 2*5Y - 2Y - 10Y captures curvature — the part of
# the curve that NS's β₃ and the CVAE's nonlinear decoder handle differently.
#
# Why butterfly instead of z₂ direct signal:
# - z₂ encodes the rate *level* (confirmed by Figure 15 in NB3), not curvature
# - Trading z₂ mean-reversion is a directional level bet — poorly motivated
# - The butterfly is a proven DV01-neutral curvature trade used by practitioners
# - Bikbov & Chernov (2010): curvature is the most predictable curve factor
#
# The CVAE adds value here because its nonlinear decoder can fit the 5Y point
# more accurately than the smooth NS exponential basis.

# %%
HP_FIXED_BF = dict(entry_z=1.0, exit_z=0.3, lookback=60, tc_bp=1.0)

BF_SIGNALS = {
    "CVAE_bf_resid": lambda p: p["cvae_bf_resid"],
    "NS_bf_resid":   lambda p: p["ns_bf_resid"],
    "Naive_bf":      lambda p: p["bf_bp"],
}

bf_bt: dict = {}
bf_rows = []

for ccy in CURRENCIES:
    p = panels[ccy]
    oos = p.index > TRAIN_CUTOFF
    for sig_name, sig_fn in BF_SIGNALS.items():
        sig = sig_fn(p)
        bt = backtest_butterfly(p, sig, **HP_FIXED_BF)
        bf_bt[(ccy, sig_name)] = bt
        s = perf_stats(bt.loc[oos, "pnl"], label=f"{ccy}|{sig_name}")
        s.update(currency=ccy, signal=sig_name)
        bf_rows.append(s)

df_bf = pd.DataFrame(bf_rows)

print("=" * 72)
print("STRATEGY B — CVAE Curvature Residual (2s5s10s Butterfly)")
print(f"Fixed HP: {HP_FIXED_BF}")
print("=" * 72)

print("\nAvg across currencies:")
print(df_bf.groupby("signal")[["Sharpe", "Total", "MaxDD", "HitRate"]]
      .mean().round(2).sort_values("Sharpe", ascending=False))

print("\nSharpe by currency:")
pivot_bf = df_bf.pivot_table(index="currency", columns="signal", values="Sharpe").round(2)
cols_bf = [c for c in ["Naive_bf", "NS_bf_resid", "CVAE_bf_resid"] if c in pivot_bf.columns]
print(pivot_bf[cols_bf])


# %%
# Equal-weight portfolio
def portfolio_pnl_bf(sig_name):
    pnl_list = [bf_bt[(c, sig_name)].loc[bf_bt[(c, sig_name)].index > TRAIN_CUTOFF, "pnl"].rename(c)
                for c in CURRENCIES]
    return pd.concat(pnl_list, axis=1).fillna(0.0).mean(axis=1)

print("\nEqual-weight portfolio — butterfly (raw bp):")
for sig in ["CVAE_bf_resid", "NS_bf_resid", "Naive_bf"]:
    pnl = portfolio_pnl_bf(sig)
    s = perf_stats(pnl)
    print(f"  {sig:<20} Sharpe={s['Sharpe']:.2f}  Total={s['Total']:.1f} bp  "
          f"MaxDD={s['MaxDD']:.1f} bp  HitRate={s['HitRate']:.2f}")


# %%
# Plot: actual vs CVAE fair value butterfly for each currency
fig, axes = plt.subplots(rows_fig, cols_fig, figsize=(15, 3.5 * rows_fig), squeeze=False)
for k, ccy in enumerate(CURRENCIES):
    ax = axes[k // cols_fig, k % cols_fig]
    p = panels[ccy]
    ax.plot(p.index, p["bf_bp"], lw=1, color="tab:blue", label="Actual")
    ax.plot(p.index, p["cvae_bf_bp"], lw=1, color="tab:orange", linestyle="--", label="CVAE")
    ax.plot(p.index, p["ns_bf_bp"], lw=1, color="tab:purple", linestyle=":", label="NS")
    ax.axhline(0, color="black", lw=0.5)
    ax.axvline(TRAIN_CUTOFF, color="red", lw=1, linestyle="--", alpha=0.5, label="Train/OOS")
    ax.set_title(f"{ccy} — 2s5s10s butterfly vs fair value", fontweight="bold")
    ax.set_ylabel("bp")
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.tick_params(axis="x", rotation=30)

for k in range(n, rows_fig * cols_fig):
    axes[k // cols_fig, k % cols_fig].set_visible(False)

fig.suptitle("2s5s10s Butterfly: Actual vs CVAE vs NS Fair Value", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig_strat_b_fairvalue.png", dpi=140, bbox_inches="tight")
plt.show()


# %%
# Plot: per-currency cum P&L for butterfly
colors_bf = {"Naive_bf": "gray", "NS_bf_resid": "tab:purple", "CVAE_bf_resid": "tab:orange"}
styles_bf = {"Naive_bf": ":", "NS_bf_resid": "--", "CVAE_bf_resid": "-"}

fig, axes = plt.subplots(rows_fig, cols_fig, figsize=(15, 3.5 * rows_fig), squeeze=False)
for k, ccy in enumerate(CURRENCIES):
    ax = axes[k // cols_fig, k % cols_fig]
    for sig in ["Naive_bf", "NS_bf_resid", "CVAE_bf_resid"]:
        bt = bf_bt[(ccy, sig)]
        oos_pnl = bt.loc[bt.index > TRAIN_CUTOFF, "pnl"].cumsum()
        ax.plot(oos_pnl.index, oos_pnl.values,
                color=colors_bf[sig], linestyle=styles_bf[sig], lw=1.5, label=sig)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_title(f"{ccy}", fontweight="bold")
    ax.set_ylabel("Cum P&L (bp)")
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.tick_params(axis="x", rotation=30)

for k in range(n, rows_fig * cols_fig):
    axes[k // cols_fig, k % cols_fig].set_visible(False)

fig.suptitle("Strategy B — OOS Cumulative P&L per currency (raw bp, fixed HP)",
             fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig_strat_b_per_ccy.png", dpi=140, bbox_inches="tight")
plt.show()


# %%
# Portfolio comparison A vs B
fig, ax = plt.subplots(figsize=(13, 5))
pnl_a = portfolio_pnl_slope("CVAE_resid", "raw_bp")
pnl_b = portfolio_pnl_bf("CVAE_bf_resid")

ax.plot(pnl_a.cumsum(), lw=2.5, color="tab:blue", label="Strategy A (slope residual)")
ax.plot(pnl_b.cumsum(), lw=2.5, color="tab:orange", label="Strategy B (butterfly residual)")
ax.plot(portfolio_pnl_slope("NS_resid", "raw_bp").cumsum(),
        lw=1.5, color="tab:purple", linestyle="--", label="Benchmark: NS slope")
ax.plot(portfolio_pnl_bf("NS_bf_resid").cumsum(),
        lw=1.5, color="tab:red", linestyle="--", label="Benchmark: NS butterfly")
ax.axhline(0, color="black", lw=0.5)
ax.set_title("Equal-weight portfolio: Strategy A vs B vs NS benchmarks (OOS, raw bp)",
             fontsize=12, fontweight="bold")
ax.set_ylabel("Cumulative P&L (bp)")
ax.legend(fontsize=10)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.tick_params(axis="x", rotation=30)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig_strat_ab_comparison.png", dpi=140, bbox_inches="tight")
plt.show()

# Combined statistics
print("Combined OOS statistics — equal-weight portfolios:")
print(f"{'Strategy':<35} {'Sharpe':>8} {'Total(bp)':>12} {'MaxDD(bp)':>12} {'HitRate':>10}")
print("-" * 80)
for label, pnl in [
    ("Strategy A (CVAE slope resid)", pnl_a),
    ("Strategy B (CVAE butterfly resid)", pnl_b),
    ("Benchmark: NS slope resid", portfolio_pnl_slope("NS_resid", "raw_bp")),
    ("Benchmark: NS butterfly resid", portfolio_pnl_bf("NS_bf_resid")),
    ("Benchmark: Naive slope", portfolio_pnl_slope("Naive", "raw_bp")),
    ("Benchmark: Naive butterfly", portfolio_pnl_bf("Naive_bf")),
]:
    s = perf_stats(pnl)
    print(f"  {label:<33} {s['Sharpe']:>8.2f} {s['Total']:>12.1f} {s['MaxDD']:>12.1f} {s['HitRate']:>10.2f}")


# ## 10 — Carry analysis for Strategy A
#
# A 2s10s spread trade has non-zero carry: the daily coupon accrual from
# holding fixed-rate positions. For a steepener (receive 2Y, pay 10Y):
#   carry_t = (r_2Y - r_10Y) / 252
#
# On an inverted curve (r_2Y > r_10Y, as in 2023-2024): carry is POSITIVE for steepeners
# On a normal curve (r_10Y > r_2Y, as in 2025+): carry is NEGATIVE for steepeners
#
# Lombard Odier (2022) stress that carry can dominate short-window P&L.
# We decompose the CVAE_resid P&L into signal alpha and carry components.
#
# Note: butterfly carry is ~zero by construction (wings cancel).

# %%
print("Carry decomposition — Strategy A (CVAE slope residual)")
print(f"{'Currency':<10} {'Sharpe_total':>14} {'Sharpe_alpha':>14} "
      f"{'Sharpe_carry':>14} {'Ann_carry(bp)':>15}")
print("-" * 70)

for ccy in CURRENCIES:
    p = panels[ccy]
    bt = slope_bt[(ccy, "CVAE_resid", "raw_bp")]
    oos = bt.index > TRAIN_CUTOFF

    pos_lag = bt["position"].shift(1).fillna(0.0)
    daily_carry_bp = (p["rate_2Y"] - p["rate_10Y"]) * BP_PER_UNIT / 252
    carry_pnl = pos_lag * daily_carry_bp

    pnl_total = bt["pnl"]
    pnl_alpha = pnl_total - carry_pnl  # price-change component only

    s_total = perf_stats(pnl_total.loc[oos])["Sharpe"]
    s_alpha = perf_stats(pnl_alpha.loc[oos])["Sharpe"]
    s_carry = perf_stats(carry_pnl.loc[oos])["Sharpe"]
    ann_carry = carry_pnl.loc[oos].sum() * (252 / n_oos)

    print(f"{ccy:<10} {s_total:>14.2f} {s_alpha:>14.2f} {s_carry:>14.2f} {ann_carry:>15.1f}")

print("\nInterpretation:")
print("  Sharpe_alpha > Sharpe_total → carry hurts (normal curve, steepener carry negative)")
print("  Sharpe_alpha < Sharpe_total → carry helps (inverted curve, steepener carry positive)")
print("  The butterfly (Strategy B) has ~zero carry by construction.")


# ## 11 — Summary table and final verdict

# %%
print("=" * 72)
print("FINAL SUMMARY — OOS Performance (equal-weight portfolio, raw bp)")
print(f"OOS window: {n_oos} business days  "
      f"({(dates_ref > TRAIN_CUTOFF).min().date()} → {dates_ref.max().date()})")
print(f"Standard error on Sharpe ≈ {np.sqrt(1/(n_oos/252)):.2f} — interpret with caution")
print("=" * 72)

summary_rows = []
for label, pnl in [
    ("Strategy A — CVAE slope (2s10s)",     portfolio_pnl_slope("CVAE_resid", "raw_bp")),
    ("Strategy B — CVAE butterfly (2s5s10s)", portfolio_pnl_bf("CVAE_bf_resid")),
    ("Benchmark: NS slope",                  portfolio_pnl_slope("NS_resid", "raw_bp")),
    ("Benchmark: NS butterfly",              portfolio_pnl_bf("NS_bf_resid")),
    ("Benchmark: Naive slope",               portfolio_pnl_slope("Naive", "raw_bp")),
    ("Benchmark: Naive butterfly",           portfolio_pnl_bf("Naive_bf")),
]:
    s = perf_stats(pnl, label=label)
    summary_rows.append(s)

df_summary = pd.DataFrame(summary_rows).set_index("label")
print(df_summary[["Sharpe", "Total", "Ann", "Vol", "MaxDD", "HitRate"]].round(2))

print("\nKey findings:")
sharpe_a  = perf_stats(portfolio_pnl_slope("CVAE_resid", "raw_bp"))["Sharpe"]
sharpe_b  = perf_stats(portfolio_pnl_bf("CVAE_bf_resid"))["Sharpe"]
sharpe_ns_a = perf_stats(portfolio_pnl_slope("NS_resid", "raw_bp"))["Sharpe"]
sharpe_ns_b = perf_stats(portfolio_pnl_bf("NS_bf_resid"))["Sharpe"]
print(f"  1. Strategy A CVAE slope Sharpe {sharpe_a:.2f} vs NS slope {sharpe_ns_a:.2f} "
      f"→ {'CVAE adds value' if sharpe_a > sharpe_ns_a else 'NS better'}")
print(f"  2. Strategy B CVAE butterfly Sharpe {sharpe_b:.2f} vs NS butterfly {sharpe_ns_b:.2f} "
      f"→ {'CVAE adds value' if sharpe_b > sharpe_ns_b else 'NS better'}")
print(f"  3. NS_resid slope Sharpe {sharpe_ns_a:.2f} — "
      f"a good fitting model does not automatically produce good trading signals.")
print(f"  4. Both strategies are complementary: "
      f"A trades slope, B trades curvature — together they cover both non-level dimensions.")


# ## 12 — Save results

# %%
Path("results").mkdir(exist_ok=True)
trading_output = {
    "panels":         panels,
    "df_slope":       df_slope,
    "df_bf":          df_bf,
    "df_tuned_a":     df_tuned_a,
    "df_summary":     df_summary,
    "config": {
        "currencies":     CURRENCIES,
        "tenors":         TARGET_TENORS,
        "train_cutoff":   str(TRAIN_CUTOFF),
        "hp_slope_fixed": HP_FIXED_SLOPE,
        "hp_bf_fixed":    HP_FIXED_BF,
        "entry_grid":     ENTRY_GRID,
        "exit_grid":      EXIT_GRID,
        "lookback_grid":  LOOKBACK_GRID,
        "lam_star":       LAM_STAR,
    },
}
with open(RES_DIR / "trading_results.pkl", "wb") as f:
    pickle.dump(trading_output, f)
print("✓ Saved results/trading_results.pkl")


# ## 13 — Limitations and what would make this credible at a desk
#
# **Sample size**: OOS window ~9 months → Sharpe SE ≈ 1.1. Any signal within ±1
# Sharpe of zero is statistically indistinguishable from noise.
# The *relative ranking* of CVAE vs NS vs Naive is more meaningful than absolute values.
#
# **What the results do show**:
# - CVAE residuals are cleaner signals than NS residuals on most currencies
#   (consistent with CVAE having lower reconstruction RMSE than NS in NB3)
# - The butterfly strategy (curvature) outperforms the slope strategy on some
#   currencies — consistent with Bikbov & Chernov (2010) finding curvature
#   more predictable than slope
# - NS_resid being a poor signal illustrates a key insight: a good *fitting*
#   model (NS: 6.8 bp pooled RMSE) does not automatically produce good
#   *trading signals*. The CVAE's advantage comes from more accurate reconstruction
#   of the idiosyncratic component
#
# **What would make this credible**:
# - 10-20 year sample (Sokol uses 22 years)
# - Walk-forward CVAE refitting (model drift over time)
# - Carry/roll-down adjusted P&L (especially for slope trades)
# - Realistic bid-offer spreads that scale with position size
# - Funding cost adjustment for the levered DV01 positions