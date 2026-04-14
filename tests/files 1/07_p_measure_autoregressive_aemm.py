# %% [markdown]
# # Notebook 7 — P-Measure Autoregressive AEMM (Chapter 4.1)
#
# This notebook implements the **Autoregressive P-Measure Models** from §4.1:
#
# 1. **Dynamic Nelson-Siegel (DNS)** baseline (§4.1.1, Diebold-Li):
#    - VAR(1) on NS factors β = (β₁, β₂, β₃)
# 2. **Autoregressive AEMM** (§4.1.2):
#    - VAR(1) on VAE latent variables z = (z₁, z₂)
# 3. **Forecasting comparison**: DNS vs AEMM at horizons 1d, 1w, 1m, 3m, 6m, 1y
# 4. **P-measure simulation**: generate future yield curve scenarios
# 5. **Risk metrics**: VaR and ES on swap rate changes
#
# Key insight: P-measure AEMM replaces NS latent factors with VAE latent variables
# in the same AR(1) framework, but with only K=2 dimensions instead of 3.

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
from scipy import stats
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
# ## 1 — Load data

# %%
with open("clean_swap_data.pkl", "rb") as f:
    data = pickle.load(f)
with open("ns_results.pkl", "rb") as f:
    ns_data = pickle.load(f)
with open("vae_results.pkl", "rb") as f:
    vae_data = pickle.load(f)

swap_aligned = data["swap_aligned"]
TARGET_TENORS = data["target_tenors"]
TENORS = np.array(TARGET_TENORS, dtype=float)
ns_factors = ns_data["ns_factors"]

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
# ## 2 — Extract time series of latent variables per currency

# %%
vae_latent: Dict[str, pd.DataFrame] = {}

for ccy in CURRENCIES:
    df = swap_aligned[ccy]
    rates_norm = normalize_rates(df.values.astype(np.float32))
    z = multi_vae.get_latent(torch.from_numpy(rates_norm))
    vae_latent[ccy] = pd.DataFrame(z, index=df.index, columns=["z1", "z2"])
    print(f"{ccy}: z shape = {z.shape}, range z1=[{z[:,0].min():.3f}, {z[:,0].max():.3f}], z2=[{z[:,1].min():.3f}, {z[:,1].max():.3f}]")

# %% [markdown]
# ---
# ## 3 — VAR(1) Estimation
#
# The Dynamic Nelson-Siegel / Autoregressive AEMM uses a VAR(1) model:
#
# $$X_t = c + A \cdot X_{t-1} + \varepsilon_t, \quad \varepsilon_t \sim N(0, \Sigma)$$
#
# where $X_t$ is either:
# - NS factors $\beta_t = (\beta_1, \beta_2, \beta_3)$ for DNS
# - VAE latent $z_t = (z_1, z_2)$ for AEMM
#
# We estimate (c, A, Σ) by OLS on each equation.

# %%
class VAR1:
    """
    VAR(1) model: X_t = c + A·X_{t-1} + ε_t
    
    Estimated by OLS equation-by-equation.
    """
    
    def __init__(self):
        self.c = None       # (K,) intercept
        self.A = None       # (K, K) transition matrix
        self.Sigma = None   # (K, K) residual covariance
        self.K = None       # number of state variables
        self.residuals = None
    
    def fit(self, X: np.ndarray):
        """
        Fit VAR(1) to time series X of shape (T, K).
        """
        T, K = X.shape
        self.K = K
        
        # Y = X[1:], Z = X[:-1]
        Y = X[1:]   # (T-1, K)
        Z = X[:-1]  # (T-1, K)
        
        # Add constant: Z_aug = [1, Z]
        Z_aug = np.column_stack([np.ones(T - 1), Z])  # (T-1, K+1)
        
        # OLS: [c | A'] = (Z'Z)^{-1} Z'Y
        params = np.linalg.lstsq(Z_aug, Y, rcond=None)[0]  # (K+1, K)
        
        self.c = params[0]        # (K,)
        self.A = params[1:].T     # (K, K)
        
        # Residuals and covariance
        Y_hat = Z_aug @ params
        self.residuals = Y - Y_hat
        self.Sigma = np.cov(self.residuals.T)
        if self.Sigma.ndim == 0:
            self.Sigma = np.array([[self.Sigma]])
        
        return self
    
    def forecast(self, X_current: np.ndarray, horizon: int = 1) -> np.ndarray:
        """
        Iterative forecast: E[X_{t+h} | X_t]
        
        Returns: (horizon, K) array of forecasted means.
        """
        K = self.K
        forecasts = np.zeros((horizon, K))
        X_prev = X_current.copy()
        
        for h in range(horizon):
            X_next = self.c + self.A @ X_prev
            forecasts[h] = X_next
            X_prev = X_next
        
        return forecasts
    
    def simulate(self, X_init: np.ndarray, n_steps: int, n_paths: int = 1000,
                 seed: int = 42) -> np.ndarray:
        """
        Monte Carlo simulation from VAR(1).
        
        Returns: (n_paths, n_steps+1, K)
        """
        np.random.seed(seed)
        K = self.K
        L = np.linalg.cholesky(self.Sigma)
        
        paths = np.zeros((n_paths, n_steps + 1, K))
        paths[:, 0, :] = X_init
        
        for p in range(n_paths):
            X = X_init.copy()
            for t in range(n_steps):
                eps = L @ np.random.randn(K)
                X = self.c + self.A @ X + eps
                paths[p, t + 1] = X
        
        return paths
    
    def eigenvalues(self):
        """Eigenvalues of A (stability check: all |λ| < 1)."""
        return np.linalg.eigvals(self.A)
    
    def half_life(self):
        """Mean-reversion half-life in periods (for each eigenvalue)."""
        evals = np.abs(self.eigenvalues())
        return -np.log(2) / np.log(evals + 1e-10)
    
    def summary(self, name: str = "VAR(1)"):
        """Print estimation summary."""
        print(f"\n{'='*50}")
        print(f"  {name}")
        print(f"{'='*50}")
        print(f"  K = {self.K}")
        print(f"  c = {self.c}")
        print(f"\n  A =")
        for row in self.A:
            print(f"    [{', '.join(f'{v:+.5f}' for v in row)}]")
        print(f"\n  Σ =")
        for row in self.Sigma:
            print(f"    [{', '.join(f'{v:.6e}' for v in row)}]")
        evals = self.eigenvalues()
        print(f"\n  Eigenvalues of A: {[f'{e:.4f}' for e in evals]}")
        print(f"  |λ|: {[f'{abs(e):.4f}' for e in evals]}")
        hl = self.half_life()
        print(f"  Half-lives (days): {[f'{h:.1f}' for h in hl]}")

# %% [markdown]
# ---
# ## 4 — Fit DNS (VAR(1) on NS factors) and AEMM (VAR(1) on z)

# %%
dns_models: Dict[str, VAR1] = {}
aemm_models: Dict[str, VAR1] = {}

for ccy in CURRENCIES:
    print(f"\n{'─'*40}\n  {ccy}\n{'─'*40}")
    
    # DNS: VAR(1) on Nelson-Siegel β₁, β₂, β₃
    if ccy in ns_factors:
        X_ns = ns_factors[ccy].values  # (T, 3)
        dns = VAR1().fit(X_ns)
        dns.summary(f"DNS [{ccy}] — β₁, β₂, β₃")
        dns_models[ccy] = dns
    
    # AEMM: VAR(1) on VAE z₁, z₂
    X_z = vae_latent[ccy].values  # (T, 2)
    aemm = VAR1().fit(X_z)
    aemm.summary(f"AEMM [{ccy}] — z₁, z₂")
    aemm_models[ccy] = aemm

# %% [markdown]
# ---
# ## 5 — Forecasting Comparison: DNS vs AEMM
#
# We evaluate forecast accuracy using a **rolling window** approach:
# - At each date t, forecast swap rates at horizons h = 1, 5, 21, 63, 126, 252 days
# - Compute RMSE of forecast vs realised swap rates
# - Compare DNS (3 factors) vs AEMM (2 factors)

# %%
def ns_basis(tau, lam=0.4):
    """Nelson-Siegel basis functions."""
    lt = lam * tau
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(lt > 1e-10, (1.0 - np.exp(-lt)) / lt, 1.0)
    f2 = f1 - np.exp(-lt)
    return np.column_stack([np.ones_like(tau), f1, f2])


def ns_reconstruct(beta: np.ndarray, tenors: np.ndarray, lam: float = 0.4) -> np.ndarray:
    """Reconstruct swap rates from NS factors."""
    B = ns_basis(tenors, lam)
    return B @ beta


def vae_reconstruct(z: np.ndarray, model: VAE) -> np.ndarray:
    """Reconstruct swap rates from VAE latent code."""
    model.eval()
    z_t = torch.from_numpy(z.astype(np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        s_norm = model.decode(z_t).cpu().numpy()[0]
    return denormalize_rates(s_norm)

# %%
HORIZONS = [1, 5, 21, 63, 126, 252]
HORIZON_LABELS = ["1d", "1w", "1m", "3m", "6m", "1y"]

forecast_results = {}

for ccy in CURRENCIES:
    print(f"\nForecasting {ccy}...")
    df = swap_aligned[ccy]
    T = len(df)
    N_tenors = df.shape[1]
    
    # Actual swap rates (decimal)
    actual = df.values
    
    # NS factors and VAE latent
    X_ns = ns_factors[ccy].values if ccy in ns_factors else None
    X_z = vae_latent[ccy].values
    
    dns_model = dns_models.get(ccy)
    aemm_model = aemm_models[ccy]
    
    ccy_results = {}
    
    for h_idx, h in enumerate(HORIZONS):
        n_forecasts = T - h
        if n_forecasts < 50:
            continue
        
        dns_errors = []
        aemm_errors = []
        
        for t in range(n_forecasts):
            # Actual rates at t+h
            actual_th = actual[t + h]
            
            # DNS forecast
            if dns_model is not None and X_ns is not None:
                ns_forecast = dns_model.forecast(X_ns[t], horizon=h)
                beta_forecast = ns_forecast[-1]  # forecast at horizon h
                swap_dns = ns_reconstruct(beta_forecast, TENORS, lam=0.4)
                dns_err = np.sqrt(np.mean((swap_dns - actual_th) ** 2)) * BP_PER_UNIT
                dns_errors.append(dns_err)
            
            # AEMM forecast
            z_forecast = aemm_model.forecast(X_z[t], horizon=h)
            z_fh = z_forecast[-1]  # forecast at horizon h
            swap_aemm = vae_reconstruct(z_fh, multi_vae)
            aemm_err = np.sqrt(np.mean((swap_aemm - actual_th) ** 2)) * BP_PER_UNIT
            aemm_errors.append(aemm_err)
        
        ccy_results[HORIZON_LABELS[h_idx]] = {
            "dns_rmse": np.mean(dns_errors) if dns_errors else np.nan,
            "aemm_rmse": np.mean(aemm_errors),
            "dns_median": np.median(dns_errors) if dns_errors else np.nan,
            "aemm_median": np.median(aemm_errors),
        }
    
    forecast_results[ccy] = ccy_results

# %% [markdown]
# ### 5.1 — Forecast comparison table

# %%
print(f"\n{'Ccy':<6} {'Horizon':<8} {'DNS RMSE(bp)':>14} {'AEMM RMSE(bp)':>14} {'Winner':>8}")
print("─" * 55)

for ccy in CURRENCIES:
    for h_label in HORIZON_LABELS:
        if h_label in forecast_results[ccy]:
            r = forecast_results[ccy][h_label]
            dns_r = r["dns_rmse"]
            aemm_r = r["aemm_rmse"]
            winner = "AEMM" if aemm_r < dns_r else "DNS"
            print(f"{ccy:<6} {h_label:<8} {dns_r:>14.2f} {aemm_r:>14.2f} {winner:>8}")

# %% [markdown]
# ### 5.2 — Forecast RMSE by horizon (visualisation)

# %%
fig, axes = plt.subplots(1, len(CURRENCIES), figsize=(6 * len(CURRENCIES), 5),
                         sharey=True, squeeze=False)

for j, ccy in enumerate(CURRENCIES):
    ax = axes[0, j]
    dns_rmses = []
    aemm_rmses = []
    labels = []
    
    for h_label in HORIZON_LABELS:
        if h_label in forecast_results[ccy]:
            r = forecast_results[ccy][h_label]
            dns_rmses.append(r["dns_rmse"])
            aemm_rmses.append(r["aemm_rmse"])
            labels.append(h_label)
    
    x_pos = np.arange(len(labels))
    ax.bar(x_pos - 0.2, dns_rmses, 0.35, label="DNS (3D)", alpha=0.8)
    ax.bar(x_pos + 0.2, aemm_rmses, 0.35, label="AEMM (2D)", alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_title(f"{ccy}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Forecast horizon")
    if j == 0:
        ax.set_ylabel("Forecast RMSE (bp)")
    ax.legend(fontsize=9)

plt.suptitle("P-Measure Forecast: DNS vs AEMM", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("figs/fig_28_forecast_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ---
# ## 6 — P-Measure Simulation & Scenario Generation
#
# Generate future yield curve scenarios using AEMM's VAR(1) in latent space.

# %%
ccy_sim = "USD"
aemm = aemm_models[ccy_sim]
z_last = vae_latent[ccy_sim].iloc[-1].values

print(f"Simulating {ccy_sim} yield curves from {swap_aligned[ccy_sim].index[-1].date()}")
print(f"Starting z = {z_last}")

# Simulate 1 year of daily observations
z_paths = aemm.simulate(z_last, n_steps=252, n_paths=1000, seed=42)
print(f"Simulated z_paths shape: {z_paths.shape}")

# %% [markdown]
# ### 6.1 — Decode simulated paths to swap rate curves

# %%
# Decode terminal curves (at t=1Y)
terminal_z = z_paths[:, -1, :]  # (n_paths, K)
terminal_swaps = np.zeros((terminal_z.shape[0], len(TENORS)))

for i in range(terminal_z.shape[0]):
    terminal_swaps[i] = vae_reconstruct(terminal_z[i], multi_vae)

# Decode paths at selected horizons
horizon_steps = {"1m": 21, "3m": 63, "6m": 126, "1y": 252}
horizon_swaps = {}

for label, h in horizon_steps.items():
    z_h = z_paths[:, h, :]
    swaps_h = np.zeros((z_h.shape[0], len(TENORS)))
    for i in range(z_h.shape[0]):
        swaps_h[i] = vae_reconstruct(z_h[i], multi_vae)
    horizon_swaps[label] = swaps_h

# %%
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# (a) z-space trajectories
ax = axes[0, 0]
z_hist = vae_latent[ccy_sim].values
ax.scatter(z_hist[:, 0], z_hist[:, 1], s=1, alpha=0.1, color="gray", label="Historical")
for p in range(min(50, z_paths.shape[0])):
    ax.plot(z_paths[p, :, 0], z_paths[p, :, 1], alpha=0.1, lw=0.3, color="tab:blue")
ax.plot(z_last[0], z_last[1], "r*", ms=15, zorder=5, label="Start")
ax.set_xlabel("$z_1$")
ax.set_ylabel("$z_2$")
ax.set_title("(a) Simulated paths in latent space")
ax.legend()

# (b) 10Y swap rate fan chart
ax = axes[0, 1]
tenor_idx = 3  # 10Y
for label, h in horizon_steps.items():
    z_h = z_paths[:, h, :]
    swaps_h = np.array([vae_reconstruct(z_h[i], multi_vae)[tenor_idx] for i in range(min(500, z_h.shape[0]))])
    ax.boxplot(swaps_h * 100, positions=[h], widths=15, manage_ticks=False)

current_10y = swap_aligned[ccy_sim].iloc[-1].values[tenor_idx] * 100
ax.axhline(current_10y, color="red", ls="--", label=f"Current: {current_10y:.2f}%")
ax.set_xlabel("Horizon (business days)")
ax.set_ylabel("10Y Swap Rate (%)")
ax.set_title(f"(b) {ccy_sim} 10Y distribution at different horizons")
ax.legend()

# (c) Terminal curve distribution
ax = axes[1, 0]
for i in range(min(100, terminal_swaps.shape[0])):
    ax.plot(TENORS, terminal_swaps[i] * 100, alpha=0.1, lw=0.3, color="steelblue")
current_curve = swap_aligned[ccy_sim].iloc[-1].values * 100
ax.plot(TENORS, current_curve, "k-o", lw=2, label="Current curve")
ax.plot(TENORS, np.median(terminal_swaps, axis=0) * 100, "r--", lw=2, label="Median 1Y forecast")
ax.set_xlabel("Maturity (years)")
ax.set_ylabel("Swap Rate (%)")
ax.set_title(f"(c) Terminal curve distribution (1Y horizon)")
ax.legend()

# (d) Distribution of 10Y rate change
ax = axes[1, 1]
rate_changes = (terminal_swaps[:, tenor_idx] - swap_aligned[ccy_sim].iloc[-1].values[tenor_idx]) * 10000
ax.hist(rate_changes, bins=50, density=True, alpha=0.7, color="steelblue")
ax.axvline(np.percentile(rate_changes, 1), color="red", ls="--",
           label=f"1% VaR: {np.percentile(rate_changes, 1):.0f}bp")
ax.axvline(np.percentile(rate_changes, 99), color="red", ls="--",
           label=f"99% VaR: {np.percentile(rate_changes, 99):.0f}bp")
ax.set_xlabel("10Y rate change (bp)")
ax.set_ylabel("Density")
ax.set_title("(d) Distribution of 1Y 10Y rate change")
ax.legend()

plt.suptitle(f"P-Measure Autoregressive AEMM — {ccy_sim}", fontsize=15, y=1.01)
plt.tight_layout()
plt.savefig("figs/fig_29_pmeasure_simulation.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ---
# ## 7 — Risk Metrics: VaR and Expected Shortfall
#
# Using the P-measure simulation to compute risk metrics at various horizons.

# %%
print(f"\n{'='*60}")
print(f"  Risk Metrics — {ccy_sim}")
print(f"{'='*60}")

current_rates = swap_aligned[ccy_sim].iloc[-1].values

for label, h in horizon_steps.items():
    z_h = z_paths[:, h, :]
    swaps_h = np.array([vae_reconstruct(z_h[i], multi_vae) for i in range(z_h.shape[0])])
    
    # Rate changes in bp
    changes_bp = (swaps_h - current_rates) * BP_PER_UNIT
    
    print(f"\n  Horizon: {label}")
    print(f"  {'Tenor':<6} {'Mean':>8} {'Std':>8} {'VaR1%':>8} {'VaR99%':>8} {'ES1%':>8} {'ES99%':>8}")
    print(f"  {'─'*54}")
    
    for t_idx, tenor in enumerate(TARGET_TENORS):
        ch = changes_bp[:, t_idx]
        var_1 = np.percentile(ch, 1)
        var_99 = np.percentile(ch, 99)
        es_1 = np.mean(ch[ch <= var_1])
        es_99 = np.mean(ch[ch >= var_99])
        print(f"  {tenor}Y{'':<4} {np.mean(ch):>8.1f} {np.std(ch):>8.1f} {var_1:>8.1f} {var_99:>8.1f} {es_1:>8.1f} {es_99:>8.1f}")

# %% [markdown]
# ---
# ## 8 — Compare DNS vs AEMM simulated distributions

# %%
# Also simulate with DNS for comparison
if ccy_sim in dns_models:
    dns = dns_models[ccy_sim]
    ns_last = ns_factors[ccy_sim].iloc[-1].values
    
    ns_paths = dns.simulate(ns_last, n_steps=252, n_paths=1000, seed=42)
    
    # Decode DNS terminal curves
    ns_terminal_betas = ns_paths[:, -1, :]
    dns_terminal_swaps = np.zeros((ns_terminal_betas.shape[0], len(TENORS)))
    for i in range(ns_terminal_betas.shape[0]):
        dns_terminal_swaps[i] = ns_reconstruct(ns_terminal_betas[i], TENORS, lam=0.4)
    
    # Compare distributions
    fig, axes = plt.subplots(1, len(TENORS), figsize=(3 * len(TENORS), 5), sharey=True)
    
    for t_idx, tenor in enumerate(TARGET_TENORS):
        ax = axes[t_idx]
        dns_ch = (dns_terminal_swaps[:, t_idx] - current_rates[t_idx]) * 10000
        aemm_ch = (terminal_swaps[:, t_idx] - current_rates[t_idx]) * 10000
        
        ax.hist(dns_ch, bins=30, alpha=0.5, density=True, label="DNS", color="tab:blue")
        ax.hist(aemm_ch, bins=30, alpha=0.5, density=True, label="AEMM", color="tab:orange")
        ax.set_title(f"{tenor}Y", fontsize=11)
        ax.set_xlabel("Δ (bp)")
        if t_idx == 0:
            ax.set_ylabel("Density")
            ax.legend(fontsize=9)
    
    plt.suptitle(f"1Y Rate Change Distribution: DNS vs AEMM — {ccy_sim}", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig("figs/fig_30_dns_vs_aemm_distributions.png", dpi=150, bbox_inches="tight")
    plt.show()

# %% [markdown]
# ---
# ## 9 — Save results

# %%
p_ar_output = {
    "dns_models": {ccy: {"c": m.c, "A": m.A, "Sigma": m.Sigma}
                   for ccy, m in dns_models.items()},
    "aemm_models": {ccy: {"c": m.c, "A": m.A, "Sigma": m.Sigma}
                    for ccy, m in aemm_models.items()},
    "vae_latent": {ccy: vae_latent[ccy].values for ccy in CURRENCIES},
    "forecast_results": forecast_results,
    "simulation_z_paths": z_paths,
    "terminal_swaps": terminal_swaps,
}

with open("p_autoregressive_aemm.pkl", "wb") as f:
    pickle.dump(p_ar_output, f)

print("Saved to p_autoregressive_aemm.pkl")
