# %% [markdown]
# # Notebook 8 — Dual-Measure AEMM (Chapter 4.2)
#
# This notebook implements the **Dual-Measure Autoencoder Market Model** from §4.2:
#
# 1. **Risk premium estimation** (§4.2.1): Average excess drift Ψ(τ) between
#    forward rates and P-measure forecast of short rate (Eq. 31-32)
# 2. **Q-measure migration map** (Fig. 21): "Snip off" first τ years and re-encode
#    to visualise Q-measure drift in latent space
# 3. **Dual-measure AEMM** (§4.2.3): Combine Q-side (calibrated to market) with
#    P-side (drift = Q drift − risk premium) using shared latent variables (Fig. 20)
# 4. **Comparison**: Autoregressive vs dual-measure P-drift estimation
# 5. **Pricing under P-measure** (§4.3): PFE-style calculations
#
# Key insight: A dual-measure model consists of two sides (Q and P) sharing
# the same state variables. The risk premium creates excess drift on the Q-side.
# By estimating this risk premium in latent space, we derive P-measure dynamics
# without making the risk premium depend on model time.

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
# ## 1 — Load data & models

# %%
with open("clean_swap_data.pkl", "rb") as f:
    data = pickle.load(f)
with open("vae_results.pkl", "rb") as f:
    vae_data = pickle.load(f)
with open("p_autoregressive_aemm.pkl", "rb") as f:
    p_ar_data = pickle.load(f)

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
# ## 2 — Utilities

# %%
def swap_to_discount(swap_rates, tenors):
    max_tenor = int(tenors[-1])
    annual_tenors = np.arange(1, max_tenor + 1, dtype=float)
    swap_annual = np.interp(annual_tenors, tenors, swap_rates)
    df = np.zeros(max_tenor)
    for i in range(max_tenor):
        s = swap_annual[i]
        df[i] = (1.0 - s * np.sum(df[:i])) / (1.0 + s) if i > 0 else 1.0 / (1.0 + s)
    return np.interp(tenors, annual_tenors, df)

def discount_to_zero(df, tenors):
    return -np.log(np.maximum(df, 1e-10)) / tenors

def zero_to_forward(zero_rates, tenors):
    tau_R = tenors * zero_rates
    return np.gradient(tau_R, tenors)

def swap_to_forward(swap_rates, tenors):
    df = swap_to_discount(swap_rates, tenors)
    zr = discount_to_zero(df, tenors)
    return zero_to_forward(zr, tenors)

def vae_reconstruct(z, model):
    model.eval()
    z_t = torch.from_numpy(z.astype(np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        s_norm = model.decode(z_t).cpu().numpy()[0]
    return denormalize_rates(s_norm)

# %% [markdown]
# ---
# ## 3 — Risk Premium Estimation (§4.2.1)
#
# The risk premium Ψ(τ) is estimated from:
#
# $$\Psi(\tau) = \langle f(t_i, t_i + \tau) - E^P[r(t_i + \tau) | X(t_i)] \rangle$$
#
# where:
# - $f(t_i, t_i + \tau)$ is the instantaneous forward rate observed at $t_i$
# - $E^P[r(t_i + \tau) | X(t_i)]$ is the P-measure forecast of the short rate
# - The average ⟨·⟩ is taken over all historical observations
#
# **Stanton's observation** (Eq. 31): The historical average of an unbiased 
# P-measure forecast equals the historical average of the spot rate:
# ⟨E^P[r(t_i + τ)|X(t_i)]⟩ = ⟨r(t_i)⟩
#
# So the risk premium simplifies to:
# Ψ(τ) ≈ ⟨f(t_i, t_i + τ)⟩ − ⟨r(t_i)⟩

# %%
def estimate_risk_premium(
    swap_data: pd.DataFrame,
    tenors: np.ndarray,
    short_rate_tenor_idx: int = 0,
) -> np.ndarray:
    """
    Estimate risk premium Ψ(τ) using Stanton's method.
    
    Ψ(τ) = mean(forward rate at maturity τ) - mean(short rate proxy)
    
    Parameters
    ----------
    swap_data : DataFrame of swap rates (decimal)
    tenors : maturity grid
    short_rate_tenor_idx : index of tenor to use as short rate proxy
    
    Returns
    -------
    psi : (N_tenors,) risk premium at each maturity
    """
    N = len(swap_data)
    N_tenors = len(tenors)
    
    # Compute forward rates for all dates
    forwards = np.zeros((N, N_tenors))
    for i in range(N):
        swaps = swap_data.iloc[i].values.astype(float)
        forwards[i] = swap_to_forward(swaps, tenors)
    
    # Mean forward rate at each maturity
    mean_fwd = forwards.mean(axis=0)
    
    # Mean short rate (proxy: 2Y rate or shortest forward)
    mean_short = forwards[:, short_rate_tenor_idx].mean()
    
    # Risk premium
    psi = mean_fwd - mean_short
    
    return psi, forwards

# %%
risk_premia: Dict[str, np.ndarray] = {}
all_forwards: Dict[str, np.ndarray] = {}

for ccy in CURRENCIES:
    psi, fwds = estimate_risk_premium(swap_aligned[ccy], TENORS)
    risk_premia[ccy] = psi
    all_forwards[ccy] = fwds

# %% [markdown]
# ### 3.1 — Visualise risk premium by currency

# %%
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# (a) Risk premium Ψ(τ) per currency
ax = axes[0]
for ccy in CURRENCIES:
    ax.plot(TENORS, risk_premia[ccy] * 10000, "o-", label=ccy, lw=2)
ax.set_xlabel("Time-to-maturity τ (years)")
ax.set_ylabel("Ψ(τ) (bp/year)")
ax.set_title("(a) Risk Premium Term Structure")
ax.legend()
ax.axhline(0, color="black", lw=0.5, ls="--")

# (b) Average forward curve vs spot
ax = axes[1]
for ccy in CURRENCIES:
    mean_fwd = all_forwards[ccy].mean(axis=0) * 100
    ax.plot(TENORS, mean_fwd, "o-", label=f"{ccy} avg fwd", lw=2)
    # Mean spot (2Y rate as proxy)
    mean_spot = swap_aligned[ccy].values[:, 0].mean() * 100
    ax.axhline(mean_spot, ls=":", alpha=0.5)

ax.set_xlabel("Maturity (years)")
ax.set_ylabel("Rate (%)")
ax.set_title("(b) Average Forward Curve vs Average Short Rate")
ax.legend(fontsize=9)

plt.suptitle("Risk Premium Estimation (§4.2.1)", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("figs/fig_31_risk_premium.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ---
# ## 4 — Q-Measure Migration Map in Latent Space (Fig. 21)
#
# To visualise Q-measure drift, we "snip off" the first τ years of the term
# structure and re-encode the truncated curve. The difference between the
# original and re-encoded latent codes reveals the Q-drift direction.
#
# Procedure:
# 1. For each historical curve S at date tᵢ, encode → z(tᵢ)
# 2. Remove the first τ years: shift tenors [2,3,5,...,30] → [2-τ, 3-τ, ...]
#    The forward rate at maturity (T-τ) after τ years equals f(tᵢ, T) under 
#    the Q-measure if rates remain constant (zero drift approximation)
# 3. Re-encode the shifted curve → z'(tᵢ)
# 4. The difference z' - z represents Q-measure drift direction

# %%
def compute_migration_map(
    model: VAE,
    swap_data: pd.DataFrame,
    tenors: np.ndarray,
    tau_shift: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Q-measure migration map by snipping off the first τ years.
    
    For each observation:
    1. Compute forward rates f(tᵢ, T) for T in tenors
    2. Shift: f_shifted(T-τ) = f(tᵢ, T) for T-τ > 0
    3. Convert shifted forwards back to approximate swap rates
    4. Re-encode shifted swaps → z'
    5. Migration = z' - z
    
    Simplified approach: interpolate swap rates at shifted tenors.
    
    Parameters
    ----------
    model : trained VAE
    swap_data : DataFrame of swap rates
    tenors : original maturity grid
    tau_shift : years to snip off
    
    Returns
    -------
    z_original : (N, K) original latent codes
    z_shifted : (N, K) latent codes after shift
    migration : (N, K) drift vectors
    """
    N = len(swap_data)
    rates = swap_data.values.astype(np.float32)
    
    # Original encoding
    rates_norm = normalize_rates(rates)
    z_original = model.get_latent(torch.from_numpy(rates_norm))
    
    # Shifted encoding
    # For each observation, interpolate swap rates at (tenors - tau_shift)
    # Tenors that become < 0 are dropped, so we need tenors > tau_shift
    shifted_tenors = tenors - tau_shift
    valid_mask = shifted_tenors > 0.5  # keep tenors > 0.5Y after shift
    
    # We need to produce a 7-tenor curve from the shifted data
    # Strategy: the swap rate at maturity T-τ (after snipping τ) is approximately
    # the swap rate at maturity T (before snipping), adjusted for the rolldown
    # Simple approximation: interpolate the original N-tenor curve at shifted tenors
    # that map back to the original grid
    
    z_shifted = np.zeros_like(z_original)
    
    for i in range(N):
        original_swaps = rates[i]
        
        # Create shifted swap curve: rate for maturity m is the original rate
        # at maturity (m + tau_shift)
        # This means we read off the curve at longer maturities
        shifted_swaps = np.interp(
            tenors,                           # target maturities
            tenors - tau_shift,               # shifted source maturities
            original_swaps,                   # original values
            left=original_swaps[0],           # extrapolate flat at short end
            right=original_swaps[-1],         # extrapolate flat at long end
        )
        
        # This creates a flatter curve at the short end (Q-measure drift effect)
        shifted_norm = normalize_rates(shifted_swaps).astype(np.float32)
        with torch.no_grad():
            mu, _ = model.encode(
                torch.from_numpy(shifted_norm).unsqueeze(0).to(device)
            )
        z_shifted[i] = mu.cpu().numpy()[0]
    
    migration = z_shifted - z_original
    
    return z_original, z_shifted, migration

# %% [markdown]
# ### 4.1 — Compute migration map for each currency

# %%
TAU_SHIFT = 2.0  # snip off 2 years (as in Fig. 21)

migration_data: Dict[str, Dict] = {}

for ccy in CURRENCIES:
    print(f"Computing migration map for {ccy} (τ={TAU_SHIFT}y)...")
    z_orig, z_shift, mig = compute_migration_map(
        multi_vae, swap_aligned[ccy], TENORS, tau_shift=TAU_SHIFT
    )
    migration_data[ccy] = {
        "z_original": z_orig,
        "z_shifted": z_shift,
        "migration": mig,
    }
    
    mean_mig = mig.mean(axis=0)
    print(f"  Mean migration: Δz₁={mean_mig[0]:+.4f}, Δz₂={mean_mig[1]:+.4f}")

# %% [markdown]
# ### 4.2 — Visualise Q-measure migration map (cf. Fig. 21)

# %%
fig, axes = plt.subplots(1, len(CURRENCIES), figsize=(7 * len(CURRENCIES), 7), squeeze=False)

for j, ccy in enumerate(CURRENCIES):
    ax = axes[0, j]
    md = migration_data[ccy]
    z_orig = md["z_original"]
    mig = md["migration"]
    
    # Background scatter
    ax.scatter(z_orig[:, 0], z_orig[:, 1], s=2, alpha=0.15, color="steelblue")
    
    # Migration arrows (subsample for clarity)
    step = max(1, len(z_orig) // 80)
    scale = 5.0  # scale arrows for visibility
    for i in range(0, len(z_orig), step):
        ax.arrow(
            z_orig[i, 0], z_orig[i, 1],
            mig[i, 0] * scale, mig[i, 1] * scale,
            head_width=0.008, head_length=0.004,
            fc="red", ec="red", alpha=0.5, lw=0.5,
        )
    
    ax.set_xlabel("$z_1$", fontsize=14)
    ax.set_ylabel("$z_2$", fontsize=14)
    ax.set_title(f"{ccy} — Q-measure migration\n(τ={TAU_SHIFT}y shift)", fontsize=13)

plt.suptitle("Q-Measure Migration Map in Latent Space (cf. Fig. 21)", fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig("figs/fig_32_migration_map.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ### 4.3 — Average migration direction analysis

# %%
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# (a) Average migration vector by currency
ax = axes[0]
for ccy in CURRENCIES:
    mig = migration_data[ccy]["migration"]
    z = migration_data[ccy]["z_original"]
    mean_z = z.mean(axis=0)
    mean_mig = mig.mean(axis=0)
    
    ax.plot(mean_z[0], mean_z[1], "o", ms=10, label=ccy)
    ax.arrow(mean_z[0], mean_z[1], mean_mig[0] * 10, mean_mig[1] * 10,
             head_width=0.01, head_length=0.005, fc=ax.lines[-1].get_color(),
             ec=ax.lines[-1].get_color(), lw=2)

ax.set_xlabel("$z_1$")
ax.set_ylabel("$z_2$")
ax.set_title("(a) Average Q-measure drift direction")
ax.legend()

# (b) Migration magnitude vs z₁ (rate level proxy)
ax = axes[1]
for ccy in CURRENCIES:
    z = migration_data[ccy]["z_original"]
    mig = migration_data[ccy]["migration"]
    mig_mag = np.sqrt(mig[:, 0]**2 + mig[:, 1]**2)
    
    # Bin by z₁
    n_bins = 10
    z1_edges = np.percentile(z[:, 0], np.linspace(0, 100, n_bins + 1))
    z1_centers = 0.5 * (z1_edges[:-1] + z1_edges[1:])
    mig_binned = np.zeros(n_bins)
    for b in range(n_bins):
        mask = (z[:, 0] >= z1_edges[b]) & (z[:, 0] < z1_edges[b+1])
        if mask.sum() > 0:
            mig_binned[b] = mig_mag[mask].mean()
    
    ax.plot(z1_centers, mig_binned, "o-", label=ccy)

ax.set_xlabel("$z_1$ (level proxy)")
ax.set_ylabel("Mean migration magnitude")
ax.set_title("(b) Q-drift magnitude vs rate level")
ax.legend()

plt.tight_layout()
plt.savefig("figs/fig_33_migration_analysis.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ---
# ## 5 — Dual-Measure AEMM (§4.2.3)
#
# The dual-measure model:
# - **Q-side**: drift estimated from migration map (market-implied)
# - **P-side**: drift = Q-drift − risk premium (estimated from historical data)
# - Both sides share the same latent variables z
#
# **P-measure drift estimation (Fig. 20)**:
# - Label A: P-measure drift (what we want)
# - Label B: Q-measure drift (from migration map)
# - Label C: Excess drift due to risk premium
# - Relationship: A = B − C
#
# We estimate P-drift in two ways:
# 1. **Autoregressive** (direct): from historical z differences (NB7)
# 2. **Dual-measure**: Q-drift minus estimated risk premium

# %%
def estimate_q_drift_field(
    migration_data: dict,
    n_bins: int = 8,
) -> Dict[str, np.ndarray]:
    """
    Estimate Q-measure drift on a grid from migration data.
    
    The drift at each z is the average migration vector (z_shifted - z_original).
    We annualise by dividing by τ_shift.
    """
    z = migration_data["z_original"]
    mig = migration_data["migration"]
    
    z_mean = z.mean(axis=0)
    z_std = z.std(axis=0)
    
    z1_edges = np.linspace(z_mean[0] - 3*z_std[0], z_mean[0] + 3*z_std[0], n_bins + 1)
    z2_edges = np.linspace(z_mean[1] - 3*z_std[1], z_mean[1] + 3*z_std[1], n_bins + 1)
    
    K = z.shape[1]
    drift_q = np.full((n_bins, n_bins, K), np.nan)
    count = np.zeros((n_bins, n_bins))
    
    for i in range(len(z)):
        i1 = np.searchsorted(z1_edges[1:], z[i, 0])
        i2 = np.searchsorted(z2_edges[1:], z[i, 1])
        if 0 <= i1 < n_bins and 0 <= i2 < n_bins:
            if np.isnan(drift_q[i1, i2, 0]):
                drift_q[i1, i2] = mig[i] / TAU_SHIFT  # annualise
            else:
                n = count[i1, i2]
                drift_q[i1, i2] = (drift_q[i1, i2] * n + mig[i] / TAU_SHIFT) / (n + 1)
            count[i1, i2] += 1
    
    z1_centers = 0.5 * (z1_edges[:-1] + z1_edges[1:])
    z2_centers = 0.5 * (z2_edges[:-1] + z2_edges[1:])
    
    return {
        "drift_q": drift_q,
        "count": count,
        "z1_centers": z1_centers,
        "z2_centers": z2_centers,
    }


def estimate_p_drift_from_history(
    z_series: np.ndarray,
    dt: float = 1/252,
    n_bins: int = 8,
) -> Dict[str, np.ndarray]:
    """
    Estimate P-measure drift directly from historical z differences.
    """
    dz = np.diff(z_series, axis=0)
    z = z_series[:-1]
    
    z_mean = z.mean(axis=0)
    z_std = z.std(axis=0)
    K = z.shape[1]
    
    z1_edges = np.linspace(z_mean[0] - 3*z_std[0], z_mean[0] + 3*z_std[0], n_bins + 1)
    z2_edges = np.linspace(z_mean[1] - 3*z_std[1], z_mean[1] + 3*z_std[1], n_bins + 1)
    
    drift_p = np.full((n_bins, n_bins, K), np.nan)
    count = np.zeros((n_bins, n_bins))
    
    for i in range(len(z)):
        i1 = np.searchsorted(z1_edges[1:], z[i, 0])
        i2 = np.searchsorted(z2_edges[1:], z[i, 1])
        if 0 <= i1 < n_bins and 0 <= i2 < n_bins:
            annualised_drift = dz[i] / dt  # annualise daily change
            if np.isnan(drift_p[i1, i2, 0]):
                drift_p[i1, i2] = annualised_drift
            else:
                n = count[i1, i2]
                drift_p[i1, i2] = (drift_p[i1, i2] * n + annualised_drift) / (n + 1)
            count[i1, i2] += 1
    
    z1_centers = 0.5 * (z1_edges[:-1] + z1_edges[1:])
    z2_centers = 0.5 * (z2_edges[:-1] + z2_edges[1:])
    
    return {
        "drift_p": drift_p,
        "count": count,
        "z1_centers": z1_centers,
        "z2_centers": z2_centers,
    }

# %% [markdown]
# ### 5.1 — Compare Q-drift, P-drift, and risk premium in latent space

# %%
ccy_dual = "USD"

# Q-drift from migration map
q_drift_field = estimate_q_drift_field(migration_data[ccy_dual])

# P-drift from historical z
rates_norm = normalize_rates(swap_aligned[ccy_dual].values.astype(np.float32))
z_series = multi_vae.get_latent(torch.from_numpy(rates_norm))
p_drift_field = estimate_p_drift_from_history(z_series, dt=1/252)

# Excess drift (risk premium in latent space) = Q-drift - P-drift
# We need to align the grids
n_bins = 8
z1c_q = q_drift_field["z1_centers"]
z1c_p = p_drift_field["z1_centers"]

# %%
fig, axes = plt.subplots(1, 3, figsize=(21, 7))

def plot_drift_field(ax, drift, count, z1c, z2c, title, color):
    # Background scatter
    ax.scatter(z_series[:, 0], z_series[:, 1], s=1, alpha=0.1, color="gray")
    
    for i in range(len(z1c)):
        for j in range(len(z2c)):
            if count[i, j] > 3 and not np.any(np.isnan(drift[i, j])):
                scale = 0.002  # scaling for visibility
                ax.arrow(z1c[i], z2c[j],
                        drift[i, j, 0] * scale, drift[i, j, 1] * scale,
                        head_width=0.005, head_length=0.002,
                        fc=color, ec=color, alpha=0.7, lw=1.5)
    
    ax.set_xlabel("$z_1$", fontsize=14)
    ax.set_ylabel("$z_2$", fontsize=14)
    ax.set_title(title, fontsize=13)

# (a) P-measure drift (A in Fig. 20)
plot_drift_field(axes[0], p_drift_field["drift_p"], p_drift_field["count"],
                 p_drift_field["z1_centers"], p_drift_field["z2_centers"],
                 "(A) P-measure drift\n(from historical data)", "tab:blue")

# (b) Q-measure drift (B in Fig. 20)
plot_drift_field(axes[1], q_drift_field["drift_q"], q_drift_field["count"],
                 q_drift_field["z1_centers"], q_drift_field["z2_centers"],
                 "(B) Q-measure drift\n(from migration map)", "tab:orange")

# (c) Excess drift = risk premium (C = B - A in Fig. 20)
# Compute on aligned grid
excess_drift = np.full_like(q_drift_field["drift_q"], np.nan)
excess_count = np.minimum(q_drift_field["count"], p_drift_field["count"])
for i in range(n_bins):
    for j in range(n_bins):
        if (not np.any(np.isnan(q_drift_field["drift_q"][i, j])) and 
            not np.any(np.isnan(p_drift_field["drift_p"][i, j])) and
            excess_count[i, j] > 3):
            excess_drift[i, j] = q_drift_field["drift_q"][i, j] - p_drift_field["drift_p"][i, j]

plot_drift_field(axes[2], excess_drift, excess_count,
                 q_drift_field["z1_centers"], q_drift_field["z2_centers"],
                 "(C) Risk premium drift\n(Q - P)", "tab:red")

plt.suptitle(f"Dual-Measure Drift Decomposition — {ccy_dual} (cf. Fig. 20)",
             fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig("figs/fig_34_dual_measure_drift.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ---
# ## 6 — Dual-Measure Simulation
#
# Using the dual-measure approach:
# - P-drift = Q-drift − excess drift (risk premium)
# - Volatility from historical z changes
# - Decode z → swap curves using VAE

# %%
def simulate_dual_measure(
    model: VAE,
    z_init: np.ndarray,
    q_drift: dict,
    p_drift: dict,
    vol_z: np.ndarray,
    n_steps: int = 252,
    dt: float = 1/252,
    n_paths: int = 500,
    seed: int = 42,
    measure: str = "P",
) -> np.ndarray:
    """
    Simulate under P or Q measure using estimated drift fields.
    
    Parameters
    ----------
    measure : "P" for P-measure, "Q" for Q-measure
    
    Returns
    -------
    z_paths : (n_paths, n_steps+1, K)
    """
    np.random.seed(seed)
    K = len(z_init)
    
    drift_field = p_drift["drift_p"] if measure == "P" else q_drift["drift_q"]
    count_field = p_drift["count"] if measure == "P" else q_drift["count"]
    z1c = p_drift["z1_centers"] if measure == "P" else q_drift["z1_centers"]
    z2c = p_drift["z2_centers"] if measure == "P" else q_drift["z2_centers"]
    
    # Fallback: simple mean-reverting drift if grid lookup fails
    z_mean_global = z_init.copy()  # target of mean reversion
    a_fallback = np.array([1.0, 2.0])  # annual mean reversion speed
    
    z_paths = np.zeros((n_paths, n_steps + 1, K))
    z_paths[:, 0, :] = z_init
    
    for p in range(n_paths):
        z = z_init.copy()
        for t in range(n_steps):
            # Look up drift from grid
            i1 = np.searchsorted(z1c, z[0]) - 1
            i2 = np.searchsorted(z2c, z[1]) - 1
            i1 = np.clip(i1, 0, len(z1c) - 1)
            i2 = np.clip(i2, 0, len(z2c) - 1)
            
            if (count_field[i1, i2] > 3 and 
                not np.any(np.isnan(drift_field[i1, i2]))):
                drift = drift_field[i1, i2]
            else:
                # Fallback to OU process
                drift = -a_fallback * (z - z_mean_global)
            
            # Euler step
            xi = np.random.randn(K)
            z = z + drift * dt + vol_z * np.sqrt(dt) * xi
            z_paths[p, t + 1] = z
    
    return z_paths

# %%
# Simulate under both measures
dz_hist = np.diff(z_series, axis=0)
vol_z = np.std(dz_hist, axis=0)
z_last = z_series[-1]

print(f"Simulating dual-measure AEMM for {ccy_dual}...")

z_paths_P = simulate_dual_measure(
    multi_vae, z_last, q_drift_field, p_drift_field, vol_z,
    n_steps=252, n_paths=500, seed=42, measure="P"
)

z_paths_Q = simulate_dual_measure(
    multi_vae, z_last, q_drift_field, p_drift_field, vol_z,
    n_steps=252, n_paths=500, seed=42, measure="Q"
)

print(f"P-measure paths: {z_paths_P.shape}")
print(f"Q-measure paths: {z_paths_Q.shape}")

# %% [markdown]
# ### 6.1 — Decode and compare P vs Q distributions

# %%
# Decode terminal curves
def decode_terminal(z_terminal, model):
    swaps = np.zeros((z_terminal.shape[0], len(TENORS)))
    for i in range(z_terminal.shape[0]):
        swaps[i] = vae_reconstruct(z_terminal[i], model)
    return swaps

terminal_P = decode_terminal(z_paths_P[:, -1, :], multi_vae)
terminal_Q = decode_terminal(z_paths_Q[:, -1, :], multi_vae)

# %%
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

current_curve = swap_aligned[ccy_dual].iloc[-1].values

# (a) P-measure terminal curves
ax = axes[0, 0]
for i in range(min(100, terminal_P.shape[0])):
    ax.plot(TENORS, terminal_P[i] * 100, alpha=0.08, lw=0.3, color="tab:blue")
ax.plot(TENORS, current_curve * 100, "k-o", lw=2, label="Current")
ax.plot(TENORS, np.median(terminal_P, axis=0) * 100, "r--", lw=2, label="P-median")
ax.set_title("(a) P-measure terminal curves (1Y)")
ax.set_xlabel("Maturity (years)")
ax.set_ylabel("Swap Rate (%)")
ax.legend()

# (b) Q-measure terminal curves
ax = axes[0, 1]
for i in range(min(100, terminal_Q.shape[0])):
    ax.plot(TENORS, terminal_Q[i] * 100, alpha=0.08, lw=0.3, color="tab:orange")
ax.plot(TENORS, current_curve * 100, "k-o", lw=2, label="Current")
ax.plot(TENORS, np.median(terminal_Q, axis=0) * 100, "r--", lw=2, label="Q-median")
ax.set_title("(b) Q-measure terminal curves (1Y)")
ax.set_xlabel("Maturity (years)")
ax.set_ylabel("Swap Rate (%)")
ax.legend()

# (c) P vs Q trajectories in latent space
ax = axes[1, 0]
for p in range(min(30, z_paths_P.shape[0])):
    ax.plot(z_paths_P[p, :, 0], z_paths_P[p, :, 1], alpha=0.15, lw=0.3, color="tab:blue")
for p in range(min(30, z_paths_Q.shape[0])):
    ax.plot(z_paths_Q[p, :, 0], z_paths_Q[p, :, 1], alpha=0.15, lw=0.3, color="tab:orange")
ax.plot(z_last[0], z_last[1], "r*", ms=15, zorder=5)
ax.scatter(z_series[:, 0], z_series[:, 1], s=1, alpha=0.05, color="gray")
ax.set_xlabel("$z_1$")
ax.set_ylabel("$z_2$")
ax.set_title("(c) P-paths (blue) vs Q-paths (orange)")

# (d) 10Y rate distribution comparison
ax = axes[1, 1]
tenor_idx = 3  # 10Y
p_10y = (terminal_P[:, tenor_idx] - current_curve[tenor_idx]) * 10000
q_10y = (terminal_Q[:, tenor_idx] - current_curve[tenor_idx]) * 10000
ax.hist(p_10y, bins=40, alpha=0.5, density=True, label="P-measure", color="tab:blue")
ax.hist(q_10y, bins=40, alpha=0.5, density=True, label="Q-measure", color="tab:orange")
ax.axvline(0, color="black", lw=0.5, ls="--")
ax.set_xlabel("10Y rate change (bp)")
ax.set_ylabel("Density")
ax.set_title("(d) P vs Q: 10Y rate change distribution")
ax.legend()

plt.suptitle(f"Dual-Measure AEMM — {ccy_dual}", fontsize=15, y=1.01)
plt.tight_layout()
plt.savefig("figs/fig_35_dual_measure_simulation.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ---
# ## 7 — PFE-Style Risk Calculation (§4.3)
#
# Using the dual-measure framework for potential future exposure (PFE):
# - P-measure for generating scenarios of future state variables
# - Q-measure for pricing at each scenario
#
# Simplified example: PFE of a 10Y receiver swap

# %%
def price_receiver_swap(swap_curve: np.ndarray, tenors: np.ndarray,
                        fixed_rate: float, notional: float = 1e6,
                        swap_tenor: float = 10.0) -> float:
    """
    Simple pricing of a receiver swap (receive fixed, pay floating).
    PV = Notional * (fixed_rate - par_swap_rate) * annuity
    """
    # Interpolate par swap rate at swap tenor
    par_rate = np.interp(swap_tenor, tenors, swap_curve)
    
    # Simple annuity approximation
    df = swap_to_discount(swap_curve, tenors)
    # Annuity = sum of discount factors at annual dates
    annual_df = np.interp(np.arange(1, int(swap_tenor) + 1), tenors, df)
    annuity = annual_df.sum()
    
    pv = notional * (fixed_rate - par_rate) * annuity
    return pv

# %%
# PFE calculation
fixed_rate = current_curve[3]  # ATM 10Y rate
print(f"ATM 10Y fixed rate: {fixed_rate*100:.3f}%")

horizons = [21, 63, 126, 252]
horizon_labels = ["1m", "3m", "6m", "1y"]

pfe_results = {}

for h, label in zip(horizons, horizon_labels):
    z_h_P = z_paths_P[:, h, :]
    
    # Price swap at each P-measure scenario
    pvs = []
    for i in range(z_h_P.shape[0]):
        swap_curve = vae_reconstruct(z_h_P[i], multi_vae)
        pv = price_receiver_swap(swap_curve, TENORS, fixed_rate)
        pvs.append(max(pv, 0))  # PFE = max(MtM, 0)
    
    pvs = np.array(pvs)
    pfe_results[label] = {
        "expected_exposure": pvs.mean(),
        "pfe_95": np.percentile(pvs, 95),
        "pfe_99": np.percentile(pvs, 99),
        "pvs": pvs,
    }

# %%
print(f"\n{'='*60}")
print(f"  PFE of 10Y Receiver Swap (ATM) — {ccy_dual}")
print(f"{'='*60}")
print(f"{'Horizon':<10} {'EE ($)':>12} {'PFE 95% ($)':>14} {'PFE 99% ($)':>14}")
print("─" * 52)
for label in horizon_labels:
    r = pfe_results[label]
    print(f"{label:<10} {r['expected_exposure']:>12,.0f} {r['pfe_95']:>14,.0f} {r['pfe_99']:>14,.0f}")

# %%
fig, ax = plt.subplots(figsize=(10, 6))
ee = [pfe_results[l]["expected_exposure"] for l in horizon_labels]
pfe95 = [pfe_results[l]["pfe_95"] for l in horizon_labels]
pfe99 = [pfe_results[l]["pfe_99"] for l in horizon_labels]

ax.plot(horizons, ee, "o-", lw=2, label="Expected Exposure")
ax.plot(horizons, pfe95, "s--", lw=2, label="PFE 95%")
ax.plot(horizons, pfe99, "^:", lw=2, label="PFE 99%")
ax.set_xlabel("Horizon (business days)")
ax.set_ylabel("Exposure ($)")
ax.set_title(f"PFE Profile — 10Y ATM Receiver Swap — {ccy_dual}")
ax.legend()
ax.set_xticks(horizons)
ax.set_xticklabels(horizon_labels)
plt.tight_layout()
plt.savefig("figs/fig_36_pfe_profile.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ---
# ## 8 — Save results

# %%
dual_output = {
    "risk_premia": risk_premia,
    "migration_data": {ccy: {
        "z_original": migration_data[ccy]["z_original"],
        "z_shifted": migration_data[ccy]["z_shifted"],
        "migration": migration_data[ccy]["migration"],
    } for ccy in CURRENCIES},
    "q_drift_field": q_drift_field,
    "p_drift_field": p_drift_field,
    "excess_drift": excess_drift,
    "pfe_results": {k: {kk: vv for kk, vv in v.items() if kk != "pvs"}
                    for k, v in pfe_results.items()},
    "tau_shift": TAU_SHIFT,
}

with open("dual_measure_aemm.pkl", "wb") as f:
    pickle.dump(dual_output, f)

print("Saved to dual_measure_aemm.pkl")
print("\n Dual-Measure AEMM pipeline complete.")
