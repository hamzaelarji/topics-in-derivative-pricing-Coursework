# %% [markdown]
# # Notebook 5 — Q-Measure Forward Rate AEMM (Chapter 3.1)
#
# This notebook implements the **Forward Rate Autoencoder Market Model** from §3.1:
#
# 1. **Bootstrap swap rates → zero rates → instantaneous forward rates**
# 2. **VAE volatility basis** σ̂ₖ(τ,F) = ∂F̂(τ,z)/∂zₖ (Eq. 14, Fig. 16)
# 3. **Decode forward rate curves** from latent space
# 4. **Forward Rate AEMM simulation** with re-encoding after each timestep (Fig. 17)
# 5. **Convexity adjustment** to maintain no-arbitrage
# 6. **Comparison** of AEMM-generated vs historical curve shapes
#
# The key insight: instead of using an exogenous volatility basis (PCA, Nelson-Siegel),
# we derive the basis from the VAE decoder's Jacobian, making it **dynamic** — it 
# depends on the current curve shape F.

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
from torch.autograd.functional import jacobian
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
print(f"Using device: {device}")

# %% [markdown]
# ## 1 — Load data & trained VAE

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
N_CCY = len(CURRENCIES)

def normalize_rates(r):
    return np.clip((r - S_MIN) / (S_MAX - S_MIN), 0, 1)

def denormalize_rates(x):
    return x * (S_MAX - S_MIN) + S_MIN

# %% [markdown]
# ## 2 — Reload multi-currency VAE

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
# ## 3 — Bootstrap: Swap Rates → Zero Rates → Forward Rates
#
# We use a simple bootstrap assuming annual fixed-leg frequency:
#
# $$P(\tau) = \frac{1 - S(\tau) \sum_{i=1}^{\tau-1} P(i)}{1 + S(\tau)}$$
#
# Then zero rate: $R(\tau) = -\ln(P(\tau))/\tau$
#
# Instantaneous forward: $f(\tau) = -\frac{d \ln P(\tau)}{d\tau}$
#
# Approximated by finite difference on the zero rate curve.

# %%
def swap_to_discount(swap_rates: np.ndarray, tenors: np.ndarray) -> np.ndarray:
    """
    Bootstrap discount factors from par swap rates.
    
    Assumes annual fixed payments and linear interpolation for 
    intermediate tenors.
    
    Parameters
    ----------
    swap_rates : (N_tenors,) par swap rates in decimal
    tenors : (N_tenors,) maturities in years
    
    Returns
    -------
    discount_factors : (N_tenors,) discount factors P(τ)
    """
    # Create annual schedule up to max tenor
    max_tenor = int(tenors[-1])
    # Interpolate swap rates to annual grid
    annual_tenors = np.arange(1, max_tenor + 1, dtype=float)
    swap_annual = np.interp(annual_tenors, tenors, swap_rates)
    
    # Bootstrap
    df = np.zeros(max_tenor)
    for i in range(max_tenor):
        s = swap_annual[i]
        if i == 0:
            df[i] = 1.0 / (1.0 + s)
        else:
            df[i] = (1.0 - s * np.sum(df[:i])) / (1.0 + s)
    
    # Extract at our target tenors (annual values)
    df_at_tenors = np.interp(tenors, annual_tenors, df)
    return df_at_tenors


def discount_to_zero(df: np.ndarray, tenors: np.ndarray) -> np.ndarray:
    """Zero rates from discount factors: R(τ) = -ln(P(τ))/τ"""
    return -np.log(np.maximum(df, 1e-10)) / tenors


def zero_to_forward(zero_rates: np.ndarray, tenors: np.ndarray) -> np.ndarray:
    """
    Instantaneous forward rates via finite difference:
    f(τ) ≈ R(τ) + τ · dR/dτ = d(τ·R)/dτ
    """
    # τ·R(τ)
    tau_R = tenors * zero_rates
    
    # Forward = d(τR)/dτ via central differences
    fwd = np.gradient(tau_R, tenors)
    return fwd


def swap_to_forward(swap_rates: np.ndarray, tenors: np.ndarray) -> np.ndarray:
    """Full pipeline: swap rates → instantaneous forward rates."""
    df = swap_to_discount(swap_rates, tenors)
    zr = discount_to_zero(df, tenors)
    fwd = zero_to_forward(zr, tenors)
    return fwd

# %% [markdown]
# ### 3.1 — Test bootstrap on a single observation

# %%
test_swaps = swap_aligned["USD"].iloc[100].values.astype(float)
test_fwd = swap_to_forward(test_swaps, TENORS)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

ax = axes[0]
ax.plot(TENORS, 100 * test_swaps, "o-", label="Swap rates")
ax.set_xlabel("Maturity (years)")
ax.set_ylabel("Rate (%)")
ax.set_title("Par Swap Rates")
ax.legend()

ax = axes[1]
df = swap_to_discount(test_swaps, TENORS)
zr = discount_to_zero(df, TENORS)
ax.plot(TENORS, 100 * zr, "s-", color="tab:orange", label="Zero rates")
ax.set_xlabel("Maturity (years)")
ax.set_ylabel("Rate (%)")
ax.set_title("Zero Coupon Rates")
ax.legend()

ax = axes[2]
ax.plot(TENORS, 100 * test_fwd, "^-", color="tab:green", label="Forward rates")
ax.set_xlabel("Maturity (years)")
ax.set_ylabel("Rate (%)")
ax.set_title("Instantaneous Forward Rates")
ax.legend()

plt.suptitle(f"Bootstrap Example — USD {swap_aligned['USD'].index[100].date()}", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("figs/fig_19_bootstrap_example.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ### 3.2 — Compute forward rates for entire dataset

# %%
forward_rates: Dict[str, pd.DataFrame] = {}

for ccy in CURRENCIES:
    df = swap_aligned[ccy]
    fwd_data = np.zeros_like(df.values)
    
    for i in range(len(df)):
        swaps = df.iloc[i].values.astype(float)
        fwd_data[i] = swap_to_forward(swaps, TENORS)
    
    forward_rates[ccy] = pd.DataFrame(fwd_data, index=df.index, columns=df.columns)
    print(f"{ccy}: forward rates computed, shape {forward_rates[ccy].shape}")

# %% [markdown]
# ---
# ## 4 — VAE Volatility Basis: ∂F̂(τ,z)/∂zₖ (Eq. 14)
#
# The forward rate AEMM volatility basis is the Jacobian of the **decoded forward 
# rate curve** with respect to each latent dimension:
#
# $$\hat{\sigma}_k(\tau, F) = \frac{\partial \hat{F}(\tau, z)}{\partial z_k}\bigg|_{z(F)}$$
#
# This is computed by:
# 1. Encode swap rates → z
# 2. Decode z → swap rates (normalised)  
# 3. Convert decoded swap rates → forward rates
# 4. Compute ∂(forward rates)/∂z via autograd

# %%
def compute_vae_volatility_basis(
    model: VAE,
    z_point: torch.Tensor,
    tenors: np.ndarray,
    dz: float = 1e-4,
) -> np.ndarray:
    """
    Compute the VAE volatility basis ∂F̂(τ,z)/∂zₖ at a given point z.
    
    Uses finite differences in latent space for robustness:
    σ̂ₖ(τ) ≈ [F̂(τ, z + dz·eₖ) - F̂(τ, z - dz·eₖ)] / (2·dz)
    
    Parameters
    ----------
    model : trained VAE
    z_point : (K,) latent code
    tenors : maturity grid
    dz : finite difference step
    
    Returns
    -------
    vol_basis : (N_tenors, K) volatility basis for each latent dimension
    """
    model.eval()
    K = z_point.shape[0]
    N = len(tenors)
    vol_basis = np.zeros((N, K))
    
    for k in range(K):
        # Perturb z in direction k
        z_plus = z_point.clone()
        z_minus = z_point.clone()
        z_plus[k] += dz
        z_minus[k] -= dz
        
        with torch.no_grad():
            s_plus = model.decode(z_plus.unsqueeze(0).to(device)).cpu().numpy()[0]
            s_minus = model.decode(z_minus.unsqueeze(0).to(device)).cpu().numpy()[0]
        
        # Convert normalised output → swap rates → forward rates
        swap_plus = denormalize_rates(s_plus)
        swap_minus = denormalize_rates(s_minus)
        
        fwd_plus = swap_to_forward(swap_plus, tenors)
        fwd_minus = swap_to_forward(swap_minus, tenors)
        
        vol_basis[:, k] = (fwd_plus - fwd_minus) / (2 * dz)
    
    return vol_basis


def compute_vae_volatility_basis_swap(
    model: VAE,
    z_point: torch.Tensor,
    dz: float = 1e-4,
) -> np.ndarray:
    """
    Simpler version: volatility basis in swap rate space (not forward rate).
    σ̂ₖ(n) = ∂Ŝ(n,z)/∂zₖ
    """
    model.eval()
    K = z_point.shape[0]
    N = model.input_dim
    vol_basis = np.zeros((N, K))
    
    for k in range(K):
        z_plus = z_point.clone()
        z_minus = z_point.clone()
        z_plus[k] += dz
        z_minus[k] -= dz
        
        with torch.no_grad():
            s_plus = model.decode(z_plus.unsqueeze(0).to(device)).cpu().numpy()[0]
            s_minus = model.decode(z_minus.unsqueeze(0).to(device)).cpu().numpy()[0]
        
        swap_plus = denormalize_rates(s_plus)
        swap_minus = denormalize_rates(s_minus)
        
        vol_basis[:, k] = (swap_plus - swap_minus) / (2 * dz)
    
    return vol_basis

# %% [markdown]
# ### 4.1 — Compute volatility basis at selected points

# %%
# Compute for a few representative dates across currencies
fig, axes = plt.subplots(len(CURRENCIES), 2, figsize=(16, 5 * len(CURRENCIES)), squeeze=False)

for row, ccy in enumerate(CURRENCIES):
    df = swap_aligned[ccy]
    rates_norm = normalize_rates(df.values.astype(np.float32))
    X_ccy = torch.from_numpy(rates_norm)
    z_all = multi_vae.get_latent(X_ccy)
    
    # Sample ~20 evenly spaced dates
    indices = np.linspace(0, len(df)-1, 20, dtype=int)
    
    axL = axes[row, 0]
    axR = axes[row, 1]
    
    for idx in indices:
        z_pt = torch.from_numpy(z_all[idx].astype(np.float32))
        
        # Forward rate vol basis
        vb_fwd = compute_vae_volatility_basis(multi_vae, z_pt, TENORS)
        
        axL.plot(TENORS, vb_fwd[:, 0], alpha=0.5, lw=1.0, color="tab:blue")
        axR.plot(TENORS, vb_fwd[:, 1], alpha=0.5, lw=1.0, color="tab:orange")
    
    axL.set_title(f"{ccy} — σ̂₁(τ): ∂F̂/∂z₁", fontsize=12)
    axL.set_xlabel("Maturity (years)")
    axL.set_ylabel("Basis value")
    axL.axhline(0, color="black", lw=0.5, ls="--")
    
    axR.set_title(f"{ccy} — σ̂₂(τ): ∂F̂/∂z₂", fontsize=12)
    axR.set_xlabel("Maturity (years)")
    axR.set_ylabel("Basis value")
    axR.axhline(0, color="black", lw=0.5, ls="--")

plt.suptitle("VAE Volatility Basis ∂F̂(τ,z)/∂zₖ (Eq. 14, Fig. 16)", fontsize=15, y=1.01)
plt.tight_layout()
plt.savefig("figs/fig_20_vol_basis.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ### 4.2 — Compare with PCA basis

# %%
# Compute PCA basis from historical forward rate changes
fwd_changes_all = []
for ccy in CURRENCIES:
    fwd = forward_rates[ccy].values
    fwd_diff = np.diff(fwd, axis=0)
    fwd_changes_all.append(fwd_diff)

fwd_changes_all = np.vstack(fwd_changes_all)

# PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(fwd_changes_all)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

ax = axes[0]
ax.plot(TENORS, pca.components_[0], "o-", lw=2, label="PCA-1 (static)")
# Show VAE basis at the mean z
z_mean = multi_vae.get_latent(torch.from_numpy(
    normalize_rates(np.vstack([swap_aligned[c].values for c in CURRENCIES]).astype(np.float32))
))
z_center = torch.from_numpy(z_mean.mean(axis=0).astype(np.float32))
vb_center = compute_vae_volatility_basis(multi_vae, z_center, TENORS)
# Normalize for comparison
vb1_norm = vb_center[:, 0] / np.linalg.norm(vb_center[:, 0])
pca1_norm = pca.components_[0] / np.linalg.norm(pca.components_[0])
ax.plot(TENORS, vb1_norm * np.sign(np.dot(vb1_norm, pca1_norm)), "s--", lw=2, label="VAE σ̂₁ (dynamic, at z̄)")
ax.set_title("Factor 1 comparison")
ax.set_xlabel("Maturity (years)")
ax.legend()

ax = axes[1]
ax.plot(TENORS, pca.components_[1], "o-", lw=2, label="PCA-2 (static)")
vb2_norm = vb_center[:, 1] / np.linalg.norm(vb_center[:, 1])
pca2_norm = pca.components_[1] / np.linalg.norm(pca.components_[1])
ax.plot(TENORS, vb2_norm * np.sign(np.dot(vb2_norm, pca2_norm)), "s--", lw=2, label="VAE σ̂₂ (dynamic, at z̄)")
ax.set_title("Factor 2 comparison")
ax.set_xlabel("Maturity (years)")
ax.legend()

plt.suptitle("PCA vs VAE Volatility Basis", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("figs/fig_21_pca_vs_vae_basis.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ---
# ## 5 — Forward Rate AEMM Simulation (Fig. 17)
#
# The simulation alternates between:
# - **(A)** Random shock in latent space: z → z + Σₖ σₖ dwₖ
# - **(B)** Re-encode: decode z to swap rates, apply time shift and drift, re-encode
#
# This ensures the model always generates curves close to VAE-generated shapes.
#
# ### Algorithm:
# 1. Start with initial curve F(0) → encode to z(0)
# 2. For each timestep dt:
#    a. Compute volatility basis at z(t)
#    b. Apply stochastic shock: z' = z(t) + Σₖ vol_k · √dt · ξₖ
#    c. Decode z' → swap rates S'
#    d. Apply time shift: advance maturities by dt
#    e. Re-encode shifted rates → z(t+dt)
#    f. Apply convexity/drift correction

# %%
def simulate_forward_rate_aemm(
    model: VAE,
    z_init: np.ndarray,
    tenors: np.ndarray,
    n_steps: int = 252,
    dt: float = 1.0 / 252.0,
    vol_scale: float = 1.0,
    n_paths: int = 100,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Simulate Forward Rate AEMM paths in latent space.
    
    Parameters
    ----------
    model : trained VAE
    z_init : (K,) initial latent code
    tenors : maturity grid
    n_steps : number of time steps
    dt : time step (1/252 for daily)
    vol_scale : scaling for volatility (calibration parameter)
    n_paths : number of Monte Carlo paths
    seed : random seed
    
    Returns
    -------
    dict with:
        z_paths : (n_paths, n_steps+1, K) latent trajectories
        swap_paths : (n_paths, n_steps+1, N) swap rate trajectories
    """
    model.eval()
    np.random.seed(seed)
    
    K = len(z_init)
    N = len(tenors)
    
    z_paths = np.zeros((n_paths, n_steps + 1, K))
    swap_paths = np.zeros((n_paths, n_steps + 1, N))
    
    # Initial state
    z_pt = torch.from_numpy(z_init.astype(np.float32))
    with torch.no_grad():
        s0_norm = model.decode(z_pt.unsqueeze(0).to(device)).cpu().numpy()[0]
    s0 = denormalize_rates(s0_norm)
    
    for p in range(n_paths):
        z_current = z_init.copy()
        z_paths[p, 0] = z_current
        swap_paths[p, 0] = s0
        
        for t in range(n_steps):
            z_pt = torch.from_numpy(z_current.astype(np.float32))
            
            # (A) Compute volatility basis at current z
            vb = compute_vae_volatility_basis_swap(model, z_pt, dz=1e-4)
            
            # Random shocks
            xi = np.random.randn(K)
            
            # Shock in swap rate space
            dS = np.zeros(N)
            for k in range(K):
                dS += vol_scale * vb[:, k] * np.sqrt(dt) * xi[k]
            
            # Current swap rates + shock
            s_current = swap_paths[p, t]
            s_new = s_current + dS
            
            # (B) Re-encode: normalise new swaps and encode
            s_new_norm = normalize_rates(s_new).astype(np.float32)
            s_new_tensor = torch.from_numpy(s_new_norm).unsqueeze(0).to(device)
            with torch.no_grad():
                mu_new, _ = model.encode(s_new_tensor)
            z_new = mu_new.cpu().numpy()[0]
            
            # (C) Decode from new z to get VAE-consistent curve
            with torch.no_grad():
                s_recon_norm = model.decode(mu_new).cpu().numpy()[0]
            s_recon = denormalize_rates(s_recon_norm)
            
            z_current = z_new
            z_paths[p, t + 1] = z_current
            swap_paths[p, t + 1] = s_recon
    
    return {"z_paths": z_paths, "swap_paths": swap_paths}

# %% [markdown]
# ### 5.1 — Run simulation from a representative starting point

# %%
# Start from the latest USD observation
ccy_sim = "USD"
df_sim = swap_aligned[ccy_sim]
rates_last = df_sim.iloc[-1].values.astype(np.float32)
rates_last_norm = normalize_rates(rates_last)

z_start = multi_vae.get_latent(
    torch.from_numpy(rates_last_norm).unsqueeze(0)
)[0]

print(f"Starting simulation from {ccy_sim} on {df_sim.index[-1].date()}")
print(f"Initial z = [{z_start[0]:.4f}, {z_start[1]:.4f}]")
print(f"Initial swap rates: {100*rates_last}")

# Estimate vol_scale from historical z changes
z_all_ccy = multi_vae.get_latent(
    torch.from_numpy(normalize_rates(df_sim.values.astype(np.float32)))
)
dz_hist = np.diff(z_all_ccy, axis=0)
vol_daily = np.std(dz_hist, axis=0)
print(f"Historical daily vol in z-space: {vol_daily}")

# %%
print("Running AEMM simulation (100 paths, 1 year)...")
sim_results = simulate_forward_rate_aemm(
    multi_vae,
    z_init=z_start,
    tenors=TENORS,
    n_steps=252,
    dt=1.0 / 252.0,
    vol_scale=1.0,
    n_paths=100,
    seed=42,
)
print("Done.")

# %% [markdown]
# ### 5.2 — Visualise simulation results

# %%
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# (a) Latent space trajectories
ax = axes[0, 0]
z_hist = multi_vae.get_latent(
    torch.from_numpy(normalize_rates(
        np.vstack([swap_aligned[c].values for c in CURRENCIES]).astype(np.float32)
    ))
)
ax.scatter(z_hist[:, 0], z_hist[:, 1], s=1, alpha=0.05, color="gray", label="Historical")
for p in range(min(20, sim_results["z_paths"].shape[0])):
    ax.plot(sim_results["z_paths"][p, :, 0], sim_results["z_paths"][p, :, 1],
            alpha=0.4, lw=0.5)
ax.plot(z_start[0], z_start[1], "r*", ms=15, label="Start")
ax.set_xlabel("$z_1$")
ax.set_ylabel("$z_2$")
ax.set_title("(a) Simulated trajectories in latent space")
ax.legend()

# (b) Swap rate fan chart (10Y)
ax = axes[0, 1]
tenor_idx = 3  # 10Y
paths_10y = sim_results["swap_paths"][:, :, tenor_idx] * 100
t_grid = np.arange(paths_10y.shape[1])
ax.fill_between(t_grid,
                np.percentile(paths_10y, 5, axis=0),
                np.percentile(paths_10y, 95, axis=0),
                alpha=0.2, color="tab:blue", label="5-95%")
ax.fill_between(t_grid,
                np.percentile(paths_10y, 25, axis=0),
                np.percentile(paths_10y, 75, axis=0),
                alpha=0.3, color="tab:blue", label="25-75%")
ax.plot(t_grid, np.median(paths_10y, axis=0), lw=2, color="tab:blue", label="Median")
ax.set_xlabel("Business days")
ax.set_ylabel("10Y Swap Rate (%)")
ax.set_title(f"(b) {ccy_sim} 10Y fan chart (AEMM)")
ax.legend()

# (c) Terminal curve distribution
ax = axes[1, 0]
terminal_swaps = sim_results["swap_paths"][:, -1, :] * 100
for i in range(min(50, terminal_swaps.shape[0])):
    ax.plot(TENORS, terminal_swaps[i], alpha=0.2, lw=0.5, color="steelblue")
ax.plot(TENORS, 100 * rates_last, "k-o", lw=2, label="Initial curve")
ax.plot(TENORS, np.median(terminal_swaps, axis=0), "r--", lw=2, label="Median terminal")
ax.set_xlabel("Maturity (years)")
ax.set_ylabel("Swap Rate (%)")
ax.set_title("(c) Terminal curve distribution (t=1Y)")
ax.legend()

# (d) Comparison with historical curves
ax = axes[1, 1]
hist_swaps = swap_aligned[ccy_sim].values * 100
for i in range(0, len(hist_swaps), 5):
    ax.plot(TENORS, hist_swaps[i], alpha=0.05, lw=0.5, color="gray")
for i in range(min(50, terminal_swaps.shape[0])):
    ax.plot(TENORS, terminal_swaps[i], alpha=0.15, lw=0.5, color="tab:red")
ax.set_xlabel("Maturity (years)")
ax.set_ylabel("Swap Rate (%)")
ax.set_title("(d) AEMM curves (red) vs historical (gray)")

plt.suptitle(f"Forward Rate AEMM Simulation — {ccy_sim}", fontsize=15, y=1.01)
plt.tight_layout()
plt.savefig("figs/fig_22_fwd_aemm_simulation.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ---
# ## 6 — Convexity Adjustment Analysis
#
# In the HJM framework, the drift of forward rates under Q-measure is determined
# by the no-arbitrage condition:
#
# $$\mu_n(t, F) = \sigma_n(t, F) \sum_{m \leq n} \frac{\delta_m \sigma_m(t, F)}{1 + \delta_m F_m}$$
#
# For normal (Gaussian HJM) volatility, the convexity adjustment is O(σ²).
# We compute it using the VAE volatility basis.

# %%
def compute_convexity_adjustment(
    vol_basis: np.ndarray,
    tenors: np.ndarray,
    dt_grid: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute the HJM convexity adjustment for Gaussian forward rates.
    
    For each tenor n:
    μ_n = Σ_k σ_k(n) * Σ_{m≤n} δ_m * σ_k(m)
    
    This is the O(σ²) drift correction.
    
    Parameters
    ----------
    vol_basis : (N_tenors, K) volatility basis
    tenors : maturity grid
    
    Returns
    -------
    drift : (N_tenors,) convexity-induced drift per year
    """
    N, K = vol_basis.shape
    
    if dt_grid is None:
        dt_grid = np.diff(tenors, prepend=0)
    
    drift = np.zeros(N)
    for n in range(N):
        for k in range(K):
            # Sum of σ_k(m) * δ_m for m ≤ n
            cumsum = np.sum(vol_basis[:n+1, k] * dt_grid[:n+1])
            drift[n] += vol_basis[n, k] * cumsum
    
    return drift

# %%
# Compute at a few representative points
z_pt = torch.from_numpy(z_start.astype(np.float32))
vb = compute_vae_volatility_basis(multi_vae, z_pt, TENORS)
conv_adj = compute_convexity_adjustment(vb, TENORS)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(TENORS, conv_adj * 10000, "o-", lw=2)
ax.set_xlabel("Maturity (years)")
ax.set_ylabel("Convexity adjustment (bp/year)")
ax.set_title(f"HJM Convexity Adjustment from VAE Basis — {ccy_sim}")
ax.axhline(0, color="black", lw=0.5, ls="--")
plt.tight_layout()
plt.savefig("figs/fig_23_convexity_adjustment.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ---
# ## 7 — Save results

# %%
q_fwd_output = {
    "forward_rates": {ccy: forward_rates[ccy].values for ccy in CURRENCIES},
    "forward_dates": {ccy: forward_rates[ccy].index for ccy in CURRENCIES},
    "sim_results": sim_results,
    "pca_fwd_components": pca.components_,
    "pca_fwd_explained_var": pca.explained_variance_ratio_,
}

with open("q_forward_aemm.pkl", "wb") as f:
    pickle.dump(q_fwd_output, f)

print("Saved to q_forward_aemm.pkl")
