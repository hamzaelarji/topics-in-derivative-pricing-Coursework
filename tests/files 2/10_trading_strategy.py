# %% [markdown]
# # Notebook 10 — Trading Strategy: Mispricing Detection & Regime Analysis
#
# This notebook **extends beyond Sokol (2022)** into practical trading applications:
#
# 1. **Mispricing detection via VAE reconstruction error** — If the VAE cannot
#    accurately reconstruct a curve, the residual identifies mispricings at
#    specific tenors relative to the VAE-learned fair-value manifold.
#
# 2. **Latent space regime detection** — Cluster historical z trajectories to
#    identify yield curve regimes (e.g., low-rate/steep, high-rate/flat, inverted).
#    Monitor real-time regime transitions for tactical allocation.
#
# 3. **Carry & roll-down analysis** — Decompose expected return into carry
#    (coupon income) and roll-down (maturity shortening). Neutralise these
#    components to isolate VAE-driven alpha.
#
# 4. **Signal construction** — Combine mispricing z-scores, regime indicators,
#    and momentum in latent space into a composite trading signal.
#
# 5. **Backtesting** — Walk-forward evaluation of the signal on swap rate
#    changes, including transaction costs, Sharpe ratio, and drawdown analysis.

# %% [markdown]
# ## 0 — Imports

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from typing import Dict, Tuple, List, Optional
from pathlib import Path

Path("figs").mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.figsize": (14, 6),
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% [markdown]
# ## 1 — Load data & model

# %%
with open("clean_swap_data.pkl", "rb") as f:
    data = pickle.load(f)
with open("vae_results.pkl", "rb") as f:
    vae_data = pickle.load(f)

swap_aligned = data["swap_aligned"]
TARGET_TENORS = data["target_tenors"]
TENORS = np.array(TARGET_TENORS, dtype=float)

cfg = vae_data["config"]
S_MIN, S_MAX = float(cfg["S_MIN"]), float(cfg["S_MAX"])
BP_PER_UNIT = float(cfg["bp_per_unit"])
CURRENCIES = cfg["currencies"]

def normalize_rates(r):
    return np.clip((r - S_MIN) / (S_MAX - S_MIN), 0, 1)
def denormalize_rates(x):
    return x * (S_MAX - S_MIN) + S_MIN

# %%
class VAE(nn.Module):
    def __init__(self, input_dim=7, latent_dim=2, hidden_dim=4, multi_currency=False):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        if not multi_currency:
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim), nn.Tanh(),
                nn.Linear(hidden_dim, 2 * latent_dim))
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim), nn.Tanh(),
                nn.Linear(hidden_dim, input_dim), nn.Sigmoid())
        else:
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
    def decode(self, z):
        return self.decoder(z)
    def forward(self, x):
        mu, logvar = self.encode(x)
        return self.decode(mu), mu, logvar
    def get_latent(self, x):
        self.eval()
        with torch.no_grad():
            mu, _ = self.encode(x.to(device))
        return mu.cpu().numpy()

multi_vae = VAE(7, 2, 4, multi_currency=True).to(device)
multi_vae.load_state_dict(torch.load("multi_vae_weights.pt", map_location=device))
multi_vae.eval()
print("Multi-currency VAE loaded")

# %% [markdown]
# ---
# ## 2 — Mispricing Detection via VAE Reconstruction Error
#
# **Idea**: The VAE learns a 2D manifold of "plausible" swap curves. When the
# market curve deviates from this manifold, the residual
#
# $$\epsilon_n(t) = S_n^{market}(t) - \hat{S}_n^{VAE}(t)$$
#
# measures how much tenor $n$ is "cheap" (ε > 0, market rate above fair value)
# or "rich" (ε < 0, below fair value) relative to the cross-currency norm.
#
# The z-score of this residual, normalised by its rolling standard deviation,
# provides a mean-reverting trading signal.

# %%
def compute_mispricing(
    model: VAE,
    swap_data: pd.DataFrame,
    tenors: np.ndarray,
) -> pd.DataFrame:
    """
    Compute per-tenor mispricing = market - VAE reconstruction (in bp).
    
    Returns a DataFrame with same index as swap_data and columns for each tenor.
    """
    model.eval()
    rates = swap_data.values.astype(np.float32)
    rates_norm = normalize_rates(rates)
    
    with torch.no_grad():
        X = torch.from_numpy(rates_norm).to(device)
        X_recon, _, _ = model(X)
    
    recon = denormalize_rates(X_recon.cpu().numpy())
    residuals = (rates - recon) * BP_PER_UNIT
    
    return pd.DataFrame(residuals, index=swap_data.index, columns=swap_data.columns)


def compute_mispricing_zscore(
    residuals: pd.DataFrame,
    lookback: int = 63,
    min_periods: int = 21,
) -> pd.DataFrame:
    """
    Standardise mispricing by rolling mean and std.
    z = (ε - rolling_mean(ε)) / rolling_std(ε)
    """
    rolling_mean = residuals.rolling(lookback, min_periods=min_periods).mean()
    rolling_std = residuals.rolling(lookback, min_periods=min_periods).std()
    rolling_std = rolling_std.clip(lower=0.5)  # floor to avoid division by zero
    
    return (residuals - rolling_mean) / rolling_std

# %% [markdown]
# ### 2.1 — Compute mispricing for all currencies

# %%
mispricing: Dict[str, pd.DataFrame] = {}
mispricing_z: Dict[str, pd.DataFrame] = {}

for ccy in CURRENCIES:
    residuals = compute_mispricing(multi_vae, swap_aligned[ccy], TENORS)
    z_scores = compute_mispricing_zscore(residuals, lookback=63)
    mispricing[ccy] = residuals
    mispricing_z[ccy] = z_scores
    
    print(f"{ccy}: mean |ε|={residuals.abs().mean().mean():.2f}bp, "
          f"max |ε|={residuals.abs().max().max():.1f}bp")

# %% [markdown]
# ### 2.2 — Visualise mispricing time series

# %%
ccy_demo = "USD"
res = mispricing[ccy_demo]
zs = mispricing_z[ccy_demo]

fig, axes = plt.subplots(3, 1, figsize=(18, 14), sharex=True)

# (a) Raw mispricing residuals
ax = axes[0]
for col in res.columns:
    ax.plot(res.index, res[col], alpha=0.6, lw=0.8, label=col)
ax.set_ylabel("Residual (bp)")
ax.set_title(f"(a) {ccy_demo} — Raw mispricing ε = Market − VAE (bp)")
ax.legend(ncol=4, fontsize=9)
ax.axhline(0, color="black", lw=0.5)

# (b) Z-scores
ax = axes[1]
for col in zs.columns:
    ax.plot(zs.index, zs[col], alpha=0.6, lw=0.8, label=col)
ax.set_ylabel("Z-score")
ax.set_title(f"(b) {ccy_demo} — Mispricing z-scores (63d rolling)")
ax.legend(ncol=4, fontsize=9)
ax.axhline(0, color="black", lw=0.5)
ax.axhline(2, color="red", lw=0.5, ls="--")
ax.axhline(-2, color="red", lw=0.5, ls="--")

# (c) RMSE heatmap over time
ax = axes[2]
rmse_ts = np.sqrt((res ** 2).mean(axis=1))
ax.plot(res.index, rmse_ts, color="steelblue", lw=1)
ax.fill_between(res.index, 0, rmse_ts, alpha=0.2)
ax.set_ylabel("RMSE (bp)")
ax.set_xlabel("Date")
ax.set_title(f"(c) {ccy_demo} — Total reconstruction RMSE over time")

plt.tight_layout()
plt.savefig("figs/fig_41_mispricing_timeseries.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ### 2.3 — Autocorrelation of mispricing (mean reversion signal)

# %%
fig, axes = plt.subplots(1, len(CURRENCIES), figsize=(6 * len(CURRENCIES), 5), squeeze=False)

for j, ccy in enumerate(CURRENCIES):
    ax = axes[0, j]
    res_ccy = mispricing[ccy].dropna()
    
    lags = range(1, 22)
    for col in res_ccy.columns:
        acf = [res_ccy[col].autocorr(lag=l) for l in lags]
        ax.plot(lags, acf, alpha=0.7, label=col)
    
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("Lag (days)")
    ax.set_ylabel("Autocorrelation")
    ax.set_title(f"{ccy}", fontsize=13)
    if j == 0:
        ax.legend(fontsize=8)

plt.suptitle("Mispricing Autocorrelation (mean reversion evidence)", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig("figs/fig_42_mispricing_acf.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ---
# ## 3 — Latent Space Regime Detection
#
# We cluster historical latent codes to identify distinct yield curve regimes.
# Using KMeans with 4 clusters:
# - Regime A: Low rates, steep curve (QE era)
# - Regime B: Rising rates, flattening (tightening)
# - Regime C: High rates, flat/inverted (restrictive)
# - Regime D: Falling rates, re-steepening (easing)

# %%
N_REGIMES = 4

# Collect all latent codes with currency label and date
regime_data: Dict[str, pd.DataFrame] = {}

for ccy in CURRENCIES:
    df = swap_aligned[ccy]
    rates_norm = normalize_rates(df.values.astype(np.float32))
    z = multi_vae.get_latent(torch.from_numpy(rates_norm))
    
    # Also compute level and slope for interpretation
    level = df.values.mean(axis=1) * 100
    slope = (df.values[:, -1] - df.values[:, 0]) * 10000
    
    regime_df = pd.DataFrame({
        "z1": z[:, 0],
        "z2": z[:, 1],
        "level_pct": level,
        "slope_bp": slope,
    }, index=df.index)
    
    regime_data[ccy] = regime_df

# %% [markdown]
# ### 3.1 — KMeans clustering per currency

# %%
cluster_models: Dict[str, KMeans] = {}
cluster_labels: Dict[str, np.ndarray] = {}

for ccy in CURRENCIES:
    rd = regime_data[ccy]
    X_cluster = rd[["z1", "z2"]].values
    
    km = KMeans(n_clusters=N_REGIMES, n_init=20, random_state=42)
    labels = km.fit_predict(X_cluster)
    
    cluster_models[ccy] = km
    cluster_labels[ccy] = labels
    regime_data[ccy]["regime"] = labels

# Sort regimes by average level (ascending)
for ccy in CURRENCIES:
    rd = regime_data[ccy]
    regime_order = rd.groupby("regime")["level_pct"].mean().sort_values().index.tolist()
    rename_map = {old: new for new, old in enumerate(regime_order)}
    regime_data[ccy]["regime"] = rd["regime"].map(rename_map)
    cluster_labels[ccy] = regime_data[ccy]["regime"].values

# %% [markdown]
# ### 3.2 — Visualise regimes in latent space and over time

# %%
regime_colors = ["tab:blue", "tab:green", "tab:orange", "tab:red"]
regime_names = ["Low/Steep", "Medium", "High/Flat", "Very High"]

fig, axes = plt.subplots(len(CURRENCIES), 2, figsize=(18, 6 * len(CURRENCIES)), squeeze=False)

for row, ccy in enumerate(CURRENCIES):
    rd = regime_data[ccy]
    
    # Left: latent space coloured by regime
    ax = axes[row, 0]
    for r in range(N_REGIMES):
        mask = rd["regime"] == r
        ax.scatter(rd.loc[mask, "z1"], rd.loc[mask, "z2"],
                   s=5, alpha=0.4, color=regime_colors[r],
                   label=f"R{r}: {regime_names[r]}")
    ax.set_xlabel("$z_1$")
    ax.set_ylabel("$z_2$")
    ax.set_title(f"{ccy} — Regimes in latent space")
    ax.legend(fontsize=9, markerscale=3)
    
    # Right: regime timeline
    ax = axes[row, 1]
    for r in range(N_REGIMES):
        mask = rd["regime"] == r
        dates = rd.index[mask]
        ax.scatter(dates, [r] * mask.sum(), s=2, alpha=0.5, color=regime_colors[r])
    ax.set_yticks(range(N_REGIMES))
    ax.set_yticklabels([f"R{r}" for r in range(N_REGIMES)])
    ax.set_title(f"{ccy} — Regime timeline")
    ax.set_xlabel("Date")

plt.suptitle("Yield Curve Regimes from VAE Latent Space", fontsize=15, y=1.01)
plt.tight_layout()
plt.savefig("figs/fig_43_regime_detection.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ### 3.3 — Regime characteristics

# %%
for ccy in CURRENCIES:
    rd = regime_data[ccy]
    print(f"\n{'─'*50}")
    print(f"  {ccy} — Regime Characteristics")
    print(f"{'─'*50}")
    print(f"{'Regime':<12} {'Count':>6} {'Level(%)':>10} {'Slope(bp)':>10} {'z₁':>8} {'z₂':>8}")
    print("─" * 58)
    for r in range(N_REGIMES):
        mask = rd["regime"] == r
        sub = rd.loc[mask]
        print(f"R{r} {regime_names[r]:<9} {mask.sum():>6} "
              f"{sub['level_pct'].mean():>10.2f} {sub['slope_bp'].mean():>10.0f} "
              f"{sub['z1'].mean():>8.3f} {sub['z2'].mean():>8.3f}")

# %% [markdown]
# ---
# ## 4 — Carry & Roll-Down Analysis
#
# **Carry** = income from holding the position (coupon − financing)
# **Roll-down** = capital gain from maturity shortening along the curve
#
# For a receiver swap at tenor T:
# - carry ≈ S(T) − S(shortest) per year
# - roll_down ≈ S(T) − S(T−1) per year (approximately)
#
# We decompose daily PnL into carry + roll + alpha (residual) components.

# %%
def compute_carry_rolldown(
    swap_data: pd.DataFrame,
    tenors: np.ndarray,
    short_rate_idx: int = 0,
) -> pd.DataFrame:
    """
    Compute daily carry and roll-down for each tenor.
    
    carry_n = (S_n - S_short) / 252  (daily accrual of positive carry)
    rolldown_n = (S_n - S_{n-1}) * (1/Δτ) / 252  (from curve steepness)
    
    Returns DataFrame with columns: carry_{tenor}, roll_{tenor}
    """
    rates = swap_data.values
    n_obs, n_tenors = rates.shape
    
    carry = pd.DataFrame(index=swap_data.index)
    rolldown = pd.DataFrame(index=swap_data.index)
    
    short_rate = rates[:, short_rate_idx]
    
    for i, tenor in enumerate(tenors):
        col = swap_data.columns[i]
        
        # Carry: rate differential to short tenor (annualised, daily accrual)
        carry[f"carry_{col}"] = (rates[:, i] - short_rate) * BP_PER_UNIT / 252.0
        
        # Roll-down: if we hold for 1 day, the bond moves Δτ down the curve
        # Approximate as (rate at current tenor - rate at next shorter tenor) * speed
        if i > 0:
            delta_tau = tenors[i] - tenors[i - 1]
            rolldown[f"roll_{col}"] = -(rates[:, i] - rates[:, i - 1]) / delta_tau * BP_PER_UNIT / 252.0
        else:
            rolldown[f"roll_{col}"] = 0.0
    
    return pd.concat([carry, rolldown], axis=1)

# %%
carry_roll_data: Dict[str, pd.DataFrame] = {}
for ccy in CURRENCIES:
    carry_roll_data[ccy] = compute_carry_rolldown(swap_aligned[ccy], TENORS)

# Show summary for USD
ccy_demo = "USD"
cr = carry_roll_data[ccy_demo]
print(f"\n{ccy_demo} — Average daily carry & roll-down (bp/day):")
carry_cols = [c for c in cr.columns if c.startswith("carry")]
roll_cols = [c for c in cr.columns if c.startswith("roll")]
print(f"  Carry:    {cr[carry_cols].mean().values}")
print(f"  Rolldown: {cr[roll_cols].mean().values}")

# %% [markdown]
# ---
# ## 5 — Trading Signal Construction
#
# **Composite signal** per currency per tenor:
#
# $$\text{signal}_n(t) = w_1 \cdot \text{mispricing\_zscore}_n(t) 
#                       + w_2 \cdot \Delta z_{\text{momentum}}(t)
#                       + w_3 \cdot \text{regime\_indicator}(t)$$
#
# Where:
# - **mispricing_zscore**: Mean-reverting signal from §2 (negative zscore → buy)
# - **z_momentum**: 5-day change in z, projected onto the tenor's sensitivity
# - **regime_indicator**: +1 for accommodative, -1 for restrictive regimes
#
# We trade a simple mean-reversion strategy: go long when signal < -threshold,
# short when signal > +threshold.

# %%
def construct_trading_signal(
    mispricing_zs: pd.DataFrame,
    regime_labels: np.ndarray,
    z_series: np.ndarray,
    dates: pd.DatetimeIndex,
    w_mispricing: float = 0.7,
    w_momentum: float = 0.2,
    w_regime: float = 0.1,
    momentum_lookback: int = 5,
) -> pd.DataFrame:
    """
    Construct composite trading signal.
    
    Signal < 0 → expect rates to fall → go long (receive fixed)
    Signal > 0 → expect rates to rise → go short (pay fixed)
    """
    # Mispricing component (main driver): negative z-score = cheap → buy
    signal_misprice = -mispricing_zs  # flip: negative z-score → positive signal
    
    # Momentum in z-space (trend-following)
    z_df = pd.DataFrame(z_series, index=dates, columns=["z1", "z2"])
    z_momentum = z_df.diff(momentum_lookback)
    
    # Project z momentum onto each tenor's sensitivity
    # Simplified: z1 momentum affects level, z2 affects shape
    # For now: uniform momentum signal across tenors
    z_mom_signal = pd.DataFrame(index=dates, columns=mispricing_zs.columns)
    for col in mispricing_zs.columns:
        z_mom_signal[col] = -z_momentum["z1"]  # negative z1 change = rates falling → long signal
    z_mom_signal = z_mom_signal.astype(float)
    
    # Standardise momentum
    z_mom_std = z_mom_signal.rolling(63, min_periods=21).std().clip(lower=1e-6)
    z_mom_signal = z_mom_signal / z_mom_std
    
    # Regime component
    regime_signal = pd.DataFrame(index=dates, columns=mispricing_zs.columns)
    # Low-rate regimes (0,1) → expect normalisation → short signal
    # High-rate regimes (2,3) → expect normalisation → long signal
    regime_map = {0: -0.5, 1: -0.25, 2: 0.25, 3: 0.5}
    for col in mispricing_zs.columns:
        regime_signal[col] = pd.Series(regime_labels, index=dates).map(regime_map)
    regime_signal = regime_signal.astype(float)
    
    # Composite
    signal = (w_mispricing * signal_misprice.fillna(0)
              + w_momentum * z_mom_signal.fillna(0)
              + w_regime * regime_signal.fillna(0))
    
    return signal

# %%
signals: Dict[str, pd.DataFrame] = {}

for ccy in CURRENCIES:
    rates_norm = normalize_rates(swap_aligned[ccy].values.astype(np.float32))
    z = multi_vae.get_latent(torch.from_numpy(rates_norm))
    
    sig = construct_trading_signal(
        mispricing_z[ccy],
        cluster_labels[ccy],
        z,
        swap_aligned[ccy].index,
    )
    signals[ccy] = sig
    print(f"{ccy}: signal computed, shape {sig.shape}")

# %% [markdown]
# ---
# ## 6 — Backtesting Framework
#
# Strategy: For each tenor, take a position proportional to the signal.
# - Position = clip(signal, -2, 2) (cap at ±2 notional units)
# - Daily PnL = position(t-1) × (−ΔS_n(t)) × duration (receiver swap gains from rate falls)
# - Transaction cost = |ΔPosition| × cost_bp
# - Evaluation: Sharpe, Calmar, max drawdown, hit rate

# %%
def backtest_strategy(
    signal: pd.DataFrame,
    swap_data: pd.DataFrame,
    tenors: np.ndarray,
    entry_threshold: float = 1.0,
    exit_threshold: float = 0.3,
    max_position: float = 2.0,
    cost_bp: float = 0.5,
    start_idx: int = 126,
) -> Dict[str, pd.DataFrame]:
    """
    Backtest mean-reversion strategy on swap rate mispricing.
    
    Parameters
    ----------
    signal : (T, N_tenors) trading signal
    swap_data : (T, N_tenors) swap rates in decimal
    entry_threshold : enter when |signal| > threshold
    exit_threshold : exit when |signal| < threshold
    max_position : max position size
    cost_bp : round-trip transaction cost in bp
    start_idx : burn-in period (for rolling stats to stabilise)
    
    Returns
    -------
    dict with pnl, positions, cumulative pnl, stats
    """
    rates = swap_data.values
    dates = swap_data.index
    T, N = rates.shape
    
    # Rate changes (in bp)
    rate_changes = np.diff(rates, axis=0) * BP_PER_UNIT  # (T-1, N)
    
    # Position sizing: proportional to signal, with threshold
    positions = np.zeros((T, N))
    
    for t in range(start_idx, T):
        for n in range(N):
            sig = signal.iloc[t, n] if not np.isnan(signal.iloc[t, n]) else 0
            
            # Simple proportional sizing with entry/exit thresholds
            if abs(sig) > entry_threshold:
                positions[t, n] = np.clip(sig, -max_position, max_position)
            elif abs(sig) < exit_threshold:
                positions[t, n] = 0.0
            else:
                positions[t, n] = positions[t - 1, n]  # hold
    
    # PnL: position at t-1 × (−rate change at t) (receiver swap: gain from rate fall)
    # Duration approximation: use tenor as proxy for DV01 scaling
    gross_pnl = np.zeros((T - 1, N))
    for n in range(N):
        duration_scale = tenors[n] / 10.0  # normalise to 10Y
        gross_pnl[:, n] = -positions[start_idx:-1, n] * rate_changes[start_idx:, n] * duration_scale
    
    # Transaction costs
    position_changes = np.abs(np.diff(positions, axis=0))
    tc = position_changes[start_idx:] * cost_bp
    
    net_pnl = gross_pnl[start_idx - start_idx:] - tc[start_idx - start_idx:]
    
    # Total PnL per day (sum across tenors)
    total_daily_pnl = np.nansum(net_pnl, axis=1)
    
    # Convert to DataFrames
    pnl_dates = dates[start_idx + 1:]
    
    pnl_df = pd.DataFrame(net_pnl, index=pnl_dates, columns=swap_data.columns)
    total_pnl_ts = pd.Series(total_daily_pnl, index=pnl_dates)
    cum_pnl = total_pnl_ts.cumsum()
    
    # Statistics
    sharpe = total_pnl_ts.mean() / total_pnl_ts.std() * np.sqrt(252) if total_pnl_ts.std() > 0 else 0
    hit_rate = (total_pnl_ts > 0).mean()
    
    running_max = cum_pnl.cummax()
    drawdown = cum_pnl - running_max
    max_dd = drawdown.min()
    calmar = (total_pnl_ts.mean() * 252) / abs(max_dd) if max_dd != 0 else 0
    
    return {
        "pnl_by_tenor": pnl_df,
        "total_daily_pnl": total_pnl_ts,
        "cumulative_pnl": cum_pnl,
        "positions": pd.DataFrame(positions, index=dates, columns=swap_data.columns),
        "stats": {
            "sharpe": sharpe,
            "hit_rate": hit_rate,
            "max_drawdown": max_dd,
            "calmar": calmar,
            "annual_pnl": total_pnl_ts.mean() * 252,
            "daily_vol": total_pnl_ts.std(),
            "n_trades": (np.abs(np.diff(positions[start_idx:], axis=0)) > 0.1).sum(),
        },
    }

# %% [markdown]
# ### 6.1 — Run backtest for each currency

# %%
backtest_results: Dict[str, dict] = {}

print(f"\n{'='*70}")
print(f"  BACKTESTING RESULTS — Mean-Reversion on VAE Mispricing")
print(f"{'='*70}")
print(f"\n{'Ccy':<6} {'Sharpe':>8} {'HitRate':>8} {'AnnPnL':>10} {'MaxDD':>10} {'Calmar':>8} {'#Trades':>8}")
print("─" * 62)

for ccy in CURRENCIES:
    bt = backtest_strategy(
        signals[ccy],
        swap_aligned[ccy],
        TENORS,
        entry_threshold=1.0,
        exit_threshold=0.3,
        cost_bp=0.5,
        start_idx=126,
    )
    backtest_results[ccy] = bt
    s = bt["stats"]
    print(f"{ccy:<6} {s['sharpe']:>8.2f} {s['hit_rate']:>8.1%} {s['annual_pnl']:>10.1f} "
          f"{s['max_drawdown']:>10.1f} {s['calmar']:>8.2f} {s['n_trades']:>8}")

# %% [markdown]
# ### 6.2 — Visualise backtest results

# %%
fig, axes = plt.subplots(len(CURRENCIES), 2, figsize=(18, 5 * len(CURRENCIES)), squeeze=False)

for row, ccy in enumerate(CURRENCIES):
    bt = backtest_results[ccy]
    
    # Left: cumulative PnL
    ax = axes[row, 0]
    cum = bt["cumulative_pnl"]
    ax.plot(cum.index, cum.values, lw=1.5, color="steelblue")
    ax.fill_between(cum.index, 0, cum.values, alpha=0.1, color="steelblue")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("Cumulative PnL (bp)")
    ax.set_title(f"{ccy} — Cumulative PnL (Sharpe={bt['stats']['sharpe']:.2f})")
    
    # Right: rolling Sharpe
    ax = axes[row, 1]
    daily = bt["total_daily_pnl"]
    rolling_sharpe = daily.rolling(126).mean() / daily.rolling(126).std() * np.sqrt(252)
    ax.plot(rolling_sharpe.index, rolling_sharpe.values, lw=1, color="tab:orange")
    ax.axhline(0, color="black", lw=0.5)
    ax.axhline(1, color="green", lw=0.5, ls="--", label="Sharpe=1")
    ax.axhline(-1, color="red", lw=0.5, ls="--", label="Sharpe=-1")
    ax.set_ylabel("Rolling Sharpe (6m)")
    ax.set_title(f"{ccy} — Rolling 6-month Sharpe")
    ax.legend()

plt.suptitle("VAE Mispricing Mean-Reversion Strategy", fontsize=15, y=1.01)
plt.tight_layout()
plt.savefig("figs/fig_44_backtest_results.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ### 6.3 — PnL decomposition by tenor

# %%
fig, axes = plt.subplots(1, len(CURRENCIES), figsize=(6 * len(CURRENCIES), 5), squeeze=False)

for j, ccy in enumerate(CURRENCIES):
    ax = axes[0, j]
    bt = backtest_results[ccy]
    tenor_pnl = bt["pnl_by_tenor"].sum()
    
    ax.bar(range(len(TENORS)), tenor_pnl.values, alpha=0.8)
    ax.set_xticks(range(len(TENORS)))
    ax.set_xticklabels([f"{t}Y" for t in TARGET_TENORS], rotation=45)
    ax.set_ylabel("Total PnL (bp)")
    ax.set_title(f"{ccy}")
    ax.axhline(0, color="black", lw=0.5)

plt.suptitle("PnL Contribution by Tenor", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig("figs/fig_45_pnl_by_tenor.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ---
# ## 7 — Cross-Currency Portfolio
#
# Combine single-currency strategies into a diversified portfolio.

# %%
# Equal-weight portfolio across currencies
portfolio_pnl = pd.DataFrame()
for ccy in CURRENCIES:
    bt = backtest_results[ccy]
    daily = bt["total_daily_pnl"]
    # Weight equally
    portfolio_pnl[ccy] = daily

# Align dates
portfolio_pnl = portfolio_pnl.dropna(how="all")
portfolio_daily = portfolio_pnl.mean(axis=1)  # equal-weight
portfolio_cum = portfolio_daily.cumsum()

# Stats
port_sharpe = portfolio_daily.mean() / portfolio_daily.std() * np.sqrt(252)
port_annual = portfolio_daily.mean() * 252
port_dd = (portfolio_cum - portfolio_cum.cummax()).min()
port_calmar = port_annual / abs(port_dd) if port_dd != 0 else 0

print(f"\n{'='*50}")
print(f"  CROSS-CURRENCY PORTFOLIO (equal-weight)")
print(f"{'='*50}")
print(f"  Sharpe Ratio:    {port_sharpe:.2f}")
print(f"  Annual PnL:      {port_annual:.1f} bp")
print(f"  Max Drawdown:    {port_dd:.1f} bp")
print(f"  Calmar Ratio:    {port_calmar:.2f}")
print(f"  Hit Rate:        {(portfolio_daily > 0).mean():.1%}")
print(f"  Daily Vol:       {portfolio_daily.std():.2f} bp")

# %%
fig, axes = plt.subplots(2, 1, figsize=(18, 10), gridspec_kw={"height_ratios": [3, 1]})

ax = axes[0]
ax.plot(portfolio_cum.index, portfolio_cum.values, lw=2, color="navy", label="Portfolio")
for ccy in CURRENCIES:
    cum = backtest_results[ccy]["cumulative_pnl"]
    ax.plot(cum.index, cum.values / len(CURRENCIES), alpha=0.4, lw=0.8, label=ccy)
ax.set_ylabel("Cumulative PnL (bp)")
ax.set_title(f"Cross-Currency Portfolio (Sharpe={port_sharpe:.2f}, Annual={port_annual:.0f}bp)")
ax.legend()
ax.axhline(0, color="black", lw=0.5)

ax = axes[1]
dd = portfolio_cum - portfolio_cum.cummax()
ax.fill_between(dd.index, 0, dd.values, alpha=0.3, color="red")
ax.set_ylabel("Drawdown (bp)")
ax.set_xlabel("Date")

plt.tight_layout()
plt.savefig("figs/fig_46_portfolio_backtest.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ---
# ## 8 — Regime-Conditioned Performance

# %%
print(f"\n{'='*60}")
print(f"  PERFORMANCE BY REGIME — {ccy_demo}")
print(f"{'='*60}")

rd = regime_data[ccy_demo]
bt = backtest_results[ccy_demo]
daily_pnl = bt["total_daily_pnl"]

# Align regime labels with PnL dates
regime_for_pnl = rd["regime"].reindex(daily_pnl.index, method="ffill")

print(f"\n{'Regime':<15} {'Days':>6} {'AvgPnL(bp)':>12} {'Sharpe':>8} {'HitRate':>8}")
print("─" * 52)
for r in range(N_REGIMES):
    mask = regime_for_pnl == r
    if mask.sum() > 10:
        pnl_r = daily_pnl[mask]
        sr = pnl_r.mean() / pnl_r.std() * np.sqrt(252) if pnl_r.std() > 0 else 0
        print(f"R{r} {regime_names[r]:<12} {mask.sum():>6} {pnl_r.mean():>12.3f} "
              f"{sr:>8.2f} {(pnl_r > 0).mean():>8.1%}")

# %% [markdown]
# ---
# ## 9 — Save trading results

# %%
trading_output = {
    "mispricing": {ccy: mispricing[ccy].to_dict() for ccy in CURRENCIES},
    "regime_labels": {ccy: cluster_labels[ccy].tolist() for ccy in CURRENCIES},
    "backtest_stats": {ccy: backtest_results[ccy]["stats"] for ccy in CURRENCIES},
    "portfolio_stats": {
        "sharpe": port_sharpe,
        "annual_pnl": port_annual,
        "max_drawdown": port_dd,
        "calmar": port_calmar,
    },
}

with open("trading_results.pkl", "wb") as f:
    pickle.dump(trading_output, f)

print("Saved to trading_results.pkl")
print("\n Trading strategy notebook complete.")
print(f"\n  Portfolio Sharpe: {port_sharpe:.2f}")
print(f"  Portfolio Annual PnL: {port_annual:.0f} bp")
