# %% [markdown]
# # Notebook 3b — VAE Extensions
#
# This notebook implements three key extensions missing from the base VAE training:
#
# 1. **Post-processing gradient descent** (§2.3.1): After encoding, optimise z starting
#    from μ(S) to minimise L2 reconstruction loss alone. This is the step from [11]
#    that significantly improves RMSE beyond the raw encoder output.
#
# 2. **Latent dimension study K=1, 2, 3** (§2.4.2): Train multi-currency VAEs with
#    different latent dimensions and compare RMSE distributions (reproduces Fig. 9/10).
#
# 3. **Latent space rotation via PCA**: Align axes so that z₁ ≈ level, z₂ ≈ shape,
#    matching the paper's convention for visualisation.

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
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Dict, Tuple, List
from pathlib import Path
from matplotlib.patches import Ellipse
from copy import deepcopy

Path("figs").mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.figsize": (14, 6),
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
})

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %% [markdown]
# ## 1 — Load data & existing models

# %%
with open("clean_swap_data.pkl", "rb") as f:
    data = pickle.load(f)

with open("vae_results.pkl", "rb") as f:
    vae_data = pickle.load(f)

swap_data = data["swap_data"]
swap_aligned = data["swap_aligned"]
TARGET_TENORS = data["target_tenors"]

cfg = vae_data["config"]
S_MIN = float(cfg["S_MIN"])
S_MAX = float(cfg["S_MAX"])
BP_PER_UNIT = float(cfg["bp_per_unit"])
CURRENCIES = cfg["currencies"]
N_CCY = len(CURRENCIES)
CCY_TO_IDX = {c: i for i, c in enumerate(CURRENCIES)}

print(f"Currencies: {CURRENCIES}")
print(f"Tenors: {TARGET_TENORS}")

# %%
def normalize_rates(rates: np.ndarray) -> np.ndarray:
    x = (rates - S_MIN) / (S_MAX - S_MIN)
    return np.clip(x, 0.0, 1.0)

def denormalize_rates(normed: np.ndarray) -> np.ndarray:
    return normed * (S_MAX - S_MIN) + S_MIN

# %% [markdown]
# ## 2 — Model definitions (same as NB3, reproduced for self-containment)

# %%
class VAE(nn.Module):
    """
    VAE for swap curves (Tables 1-2 of Sokol 2022).
    """
    def __init__(self, input_dim: int = 7, latent_dim: int = 2,
                 hidden_dim: int = 4, multi_currency: bool = False):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.multi_currency = multi_currency

        if not multi_currency:
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim), nn.Tanh(),
                nn.Linear(hidden_dim, 2 * latent_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim), nn.Tanh(),
                nn.Linear(hidden_dim, input_dim), nn.Sigmoid(),
            )
        else:
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, input_dim), nn.Tanh(),
                nn.Linear(input_dim, 2 * latent_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim), nn.Tanh(),
                nn.Linear(hidden_dim, input_dim), nn.Tanh(),
                nn.Linear(input_dim, input_dim), nn.Sigmoid(),
            )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mu = h[:, :self.latent_dim]
        logvar = h[:, self.latent_dim:]
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + std * torch.randn_like(std)
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def get_latent(self, x: torch.Tensor) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            mu, _ = self.encode(x.to(next(self.parameters()).device))
        return mu.cpu().numpy()


class CVAE(nn.Module):
    """Conditional VAE (Table 3 of Sokol 2022)."""
    def __init__(self, input_dim: int = 7, latent_dim: int = 2,
                 n_currencies: int = 4):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_currencies = n_currencies
        enc_in = input_dim + n_currencies
        dec_in = latent_dim + n_currencies

        self.encoder = nn.Sequential(
            nn.Linear(enc_in, enc_in - 1), nn.Tanh(),
            nn.Linear(enc_in - 1, 2 * latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(dec_in, input_dim), nn.Tanh(),
            nn.Linear(input_dim, input_dim), nn.Tanh(),
            nn.Linear(input_dim, input_dim), nn.Sigmoid(),
        )

    def encode(self, x, y):
        h = self.encoder(torch.cat([x, y], dim=-1))
        return h[:, :self.latent_dim], h[:, self.latent_dim:]

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + std * torch.randn_like(std)
        return mu

    def decode(self, z, y):
        return self.decoder(torch.cat([z, y], dim=-1))

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, y), mu, logvar

    def get_latent(self, x, y):
        self.eval()
        with torch.no_grad():
            mu, _ = self.encode(
                x.to(next(self.parameters()).device),
                y.to(next(self.parameters()).device),
            )
        return mu.cpu().numpy()

# %% [markdown]
# ## 3 — Build multi-currency dataset

# %%
multi_rates, multi_labels, multi_ccy_ids, multi_dates = [], [], [], []
for ccy in CURRENCIES:
    df = swap_aligned[ccy]
    rates = df.values.astype(np.float32)
    rates_norm = normalize_rates(rates).astype(np.float32)
    n = len(rates_norm)
    one_hot = np.zeros((n, N_CCY), dtype=np.float32)
    one_hot[:, CCY_TO_IDX[ccy]] = 1.0
    multi_rates.append(rates_norm)
    multi_labels.append(one_hot)
    multi_ccy_ids.extend([ccy] * n)
    multi_dates.extend(df.index.tolist())

multi_rates = np.vstack(multi_rates)
multi_labels = np.vstack(multi_labels)
X_multi = torch.from_numpy(multi_rates)
Y_multi = torch.from_numpy(multi_labels)

print(f"Multi-currency dataset: {X_multi.shape}")

# %% [markdown]
# ## 4 — Reload trained multi-currency VAE (K=2) from NB3

# %%
multi_vae_k2 = VAE(7, 2, 4, multi_currency=True).to(device)
multi_vae_k2.load_state_dict(torch.load("multi_vae_weights.pt", map_location=device))
multi_vae_k2.eval()
print("Loaded multi-currency VAE (K=2)")

# %% [markdown]
# ---
# ## 5 — Post-Processing Gradient Descent (§2.3.1)
#
# The paper states:
# > "A post-processing step described in [11] increases the accuracy of VAE mapping
# > by performing gradient descent minimising L2 loss starting from the centre μ(S)
# > of the distribution produced by the encoder."
#
# **Algorithm**:
# 1. Encode input S → μ(S) via encoder
# 2. Starting from z₀ = μ(S), perform gradient descent on z to minimise
#    ‖Decoder(z) − normalize(S)‖²
# 3. Use the optimised z* as the final latent representation
#
# This is done **per observation**, keeping the decoder weights frozen.

# %%
def post_process_gradient_descent(
    model: VAE,
    x_norm: torch.Tensor,
    n_steps: int = 100,
    lr: float = 0.01,
    verbose: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Post-processing gradient descent on latent codes.
    
    For each sample, optimise z to minimise L2 reconstruction loss
    starting from the encoder's μ(S).
    
    Parameters
    ----------
    model : trained VAE (frozen)
    x_norm : (N, 7) normalised swap rates
    n_steps : number of GD iterations per batch
    lr : learning rate for z optimisation
    
    Returns
    -------
    z_opt : (N, K) optimised latent codes
    x_recon_opt : (N, 7) optimised reconstructions
    """
    model.eval()
    x_dev = x_norm.to(device)
    
    # Step 1: get encoder output as starting point
    with torch.no_grad():
        mu, _ = model.encode(x_dev)
    
    # Step 2: make z a learnable parameter, initialised to μ
    z = mu.clone().detach().requires_grad_(True)
    
    optimizer = optim.Adam([z], lr=lr)
    
    for step in range(n_steps):
        optimizer.zero_grad()
        x_recon = model.decode(z)
        loss = torch.mean((x_recon - x_dev) ** 2)
        loss.backward()
        optimizer.step()
        
        if verbose and (step + 1) % 20 == 0:
            print(f"  GD step {step+1}/{n_steps} | L2 loss: {loss.item():.6e}")
    
    # Final reconstruction
    with torch.no_grad():
        x_recon_final = model.decode(z)
    
    return z.detach(), x_recon_final.detach()


def post_process_batched(
    model: VAE,
    x_norm: torch.Tensor,
    batch_size: int = 512,
    n_steps: int = 100,
    lr: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Batched version for large datasets.
    Returns (z_opt, rmse_bp) as numpy arrays.
    """
    model.eval()
    all_z, all_rmse = [], []
    n = x_norm.shape[0]
    
    for i in range(0, n, batch_size):
        x_batch = x_norm[i:i+batch_size]
        z_opt, x_recon_opt = post_process_gradient_descent(
            model, x_batch, n_steps=n_steps, lr=lr
        )
        
        # Compute RMSE in bp
        x_true_dec = denormalize_rates(x_batch.numpy())
        x_recon_dec = denormalize_rates(x_recon_opt.cpu().numpy())
        rmse_bp = np.sqrt(np.mean((x_true_dec - x_recon_dec) ** 2, axis=1)) * BP_PER_UNIT
        
        all_z.append(z_opt.cpu().numpy())
        all_rmse.append(rmse_bp)
    
    return np.vstack(all_z), np.concatenate(all_rmse)

# %% [markdown]
# ### 5.1 — Apply post-processing to multi-currency VAE

# %%
print("Post-processing gradient descent on multi-currency VAE (K=2)...")
print("This optimises z per observation to minimise decoder L2 loss.\n")

z_pp, rmse_pp = post_process_batched(
    multi_vae_k2, X_multi, batch_size=512, n_steps=150, lr=0.01
)

# Compare with raw encoder RMSE
with torch.no_grad():
    X_recon_raw, _, _ = multi_vae_k2(X_multi.to(device))
x_raw_dec = denormalize_rates(X_recon_raw.cpu().numpy())
x_true_dec = denormalize_rates(X_multi.numpy())
rmse_raw = np.sqrt(np.mean((x_true_dec - x_raw_dec) ** 2, axis=1)) * BP_PER_UNIT

print(f"\n{'Method':<30} {'Mean(bp)':>10} {'Median(bp)':>10} {'P95(bp)':>10}")
print("─" * 65)
print(f"{'Raw encoder μ(S)':<30} {rmse_raw.mean():>10.2f} {np.median(rmse_raw):>10.2f} {np.percentile(rmse_raw, 95):>10.2f}")
print(f"{'Post-processed (GD on z)':<30} {rmse_pp.mean():>10.2f} {np.median(rmse_pp):>10.2f} {np.percentile(rmse_pp, 95):>10.2f}")
print(f"\nImprovement: {(1 - rmse_pp.mean()/rmse_raw.mean())*100:.1f}%")

# %% [markdown]
# ### 5.2 — Post-processing RMSE by currency

# %%
offset = 0
print(f"\n{'Currency':<8} {'Raw μ(bp)':>12} {'Post-proc(bp)':>14} {'Improvement':>12}")
print("─" * 50)
for ccy in CURRENCIES:
    n = len(swap_aligned[ccy])
    raw_slice = rmse_raw[offset:offset+n]
    pp_slice = rmse_pp[offset:offset+n]
    imp = (1 - pp_slice.mean() / raw_slice.mean()) * 100
    print(f"{ccy:<8} {raw_slice.mean():>12.2f} {pp_slice.mean():>14.2f} {imp:>11.1f}%")
    offset += n

# %% [markdown]
# ### 5.3 — Visualise improvement

# %%
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

ax = axes[0]
ax.hist(rmse_raw, bins=60, alpha=0.5, density=True, label=f"Raw encoder (μ={rmse_raw.mean():.1f}bp)", range=(0, 60))
ax.hist(rmse_pp, bins=60, alpha=0.5, density=True, label=f"Post-processed (μ={rmse_pp.mean():.1f}bp)", range=(0, 60))
ax.set_xlabel("RMSE (bp)")
ax.set_ylabel("Density")
ax.set_title("Effect of Post-Processing Gradient Descent")
ax.legend()

ax = axes[1]
ax.scatter(rmse_raw, rmse_pp, s=2, alpha=0.3)
ax.plot([0, 60], [0, 60], "k--", lw=1, label="y=x")
ax.set_xlabel("Raw encoder RMSE (bp)")
ax.set_ylabel("Post-processed RMSE (bp)")
ax.set_title("Per-observation improvement")
ax.legend()
ax.set_xlim(0, 60)
ax.set_ylim(0, 60)
ax.set_aspect("equal")

plt.tight_layout()
plt.savefig("figs/fig_16_postprocessing.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ---
# ## 6 — Latent Dimension Study K=1, K=2, K=3 (§2.4.2)
#
# The paper compares RMSE for different latent dimensions.
# We train multi-currency VAEs for K ∈ {1, 2, 3} and compare.

# %%
def vae_loss(x_recon, x, mu, logvar, beta=1e-7, N=7):
    recon = torch.mean((x_recon - x) ** 2, dim=-1).mean() / N
    kld = (-0.5 * torch.sum(1 + logvar - mu**2 - torch.exp(logvar), dim=-1)).mean()
    return recon + beta * kld, recon, kld


def train_vae(model, X_train, Y_train=None, n_epochs=3000, batch_size=256,
              lr=1e-3, beta=1e-7, print_every=500):
    model.to(device)
    is_cvae = isinstance(model, CVAE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=150, factor=0.5
    )

    if is_cvae:
        dataset = TensorDataset(X_train, Y_train)
    else:
        dataset = TensorDataset(X_train)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    history = {"total": [], "recon": [], "kld": []}

    for epoch in range(1, n_epochs + 1):
        model.train()
        tot = rec = kldv = 0.0
        nb = 0
        for batch in loader:
            if is_cvae:
                xb, yb = batch[0].to(device), batch[1].to(device)
                xr, mu, lv = model(xb, yb)
            else:
                xb = batch[0].to(device)
                xr, mu, lv = model(xb)
            loss, r, k = vae_loss(xr, xb, mu, lv, beta=beta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tot += loss.item(); rec += r.item(); kldv += k.item(); nb += 1

        tot /= nb; rec /= nb; kldv /= nb
        history["total"].append(tot)
        history["recon"].append(rec)
        history["kld"].append(kldv)
        scheduler.step(tot)

        if epoch == 1 or epoch % print_every == 0:
            print(f"  Epoch {epoch:5d}/{n_epochs} | Loss: {tot:.6e} | Recon: {rec:.6e}")

    model.eval()
    return history


def compute_rmse_bp(model, X_norm, Y_labels=None):
    model.eval()
    with torch.no_grad():
        X_dev = X_norm.to(device)
        if isinstance(model, CVAE):
            X_recon, _, _ = model(X_dev, Y_labels.to(device))
        else:
            X_recon, _, _ = model(X_dev)
    x_recon = denormalize_rates(X_recon.cpu().numpy())
    x_true = denormalize_rates(X_norm.numpy())
    return np.sqrt(np.mean((x_recon - x_true) ** 2, axis=1)) * BP_PER_UNIT

# %% [markdown]
# ### 6.1 — Train K=1 and K=3 multi-currency VAEs

# %%
latent_dims = [1, 2, 3]
models_by_K: Dict[int, VAE] = {}
histories_by_K: Dict[int, dict] = {}
rmse_by_K: Dict[int, np.ndarray] = {}
rmse_pp_by_K: Dict[int, np.ndarray] = {}

for K in latent_dims:
    print(f"\n{'='*60}")
    print(f"Training Multi-Currency VAE with K={K}")
    print(f"{'='*60}")
    
    if K == 2:
        # Reuse already-trained model
        models_by_K[K] = multi_vae_k2
        rmse_by_K[K] = rmse_raw
        rmse_pp_by_K[K] = rmse_pp
        print("  (reusing K=2 from NB3)")
        continue
    
    model = VAE(input_dim=7, latent_dim=K, hidden_dim=4, multi_currency=True)
    hist = train_vae(model, X_multi, n_epochs=3000, batch_size=256,
                     lr=1e-3, beta=1e-7, print_every=500)
    
    models_by_K[K] = model
    histories_by_K[K] = hist
    
    # Raw RMSE
    rmse_by_K[K] = compute_rmse_bp(model, X_multi)
    
    # Post-processed RMSE
    _, rmse_pp_k = post_process_batched(model, X_multi, n_steps=150, lr=0.01)
    rmse_pp_by_K[K] = rmse_pp_k
    
    print(f"  K={K} Raw  mean RMSE: {rmse_by_K[K].mean():.2f} bp")
    print(f"  K={K} PP   mean RMSE: {rmse_pp_k.mean():.2f} bp")

# %% [markdown]
# ### 6.2 — Comparison table (reproduces Fig. 9/10 data)

# %%
print(f"\n{'K':<4} {'Raw Mean(bp)':>14} {'Raw Med(bp)':>14} {'PP Mean(bp)':>14} {'PP Med(bp)':>14}")
print("─" * 65)
for K in latent_dims:
    raw = rmse_by_K[K]
    pp = rmse_pp_by_K[K]
    print(f"{K:<4} {raw.mean():>14.2f} {np.median(raw):>14.2f} {pp.mean():>14.2f} {np.median(pp):>14.2f}")

# %% [markdown]
# ### 6.3 — RMSE distributions by K (Fig. 9/10 style)

# %%
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Raw
ax = axes[0]
for K in latent_dims:
    arr = rmse_by_K[K]
    ax.hist(arr, bins=60, alpha=0.45, density=True,
            label=f"K={K} (μ={arr.mean():.1f}bp)", range=(0, 80))
ax.set_xlabel("RMSE (bp)")
ax.set_ylabel("Density")
ax.set_title("Raw encoder RMSE by latent dimension K")
ax.legend()

# Post-processed
ax = axes[1]
for K in latent_dims:
    arr = rmse_pp_by_K[K]
    ax.hist(arr, bins=60, alpha=0.45, density=True,
            label=f"K={K} (μ={arr.mean():.1f}bp)", range=(0, 80))
ax.set_xlabel("RMSE (bp)")
ax.set_ylabel("Density")
ax.set_title("Post-processed RMSE by latent dimension K")
ax.legend()

plt.suptitle("Multi-Currency VAE — RMSE vs Latent Dimension (cf. Fig. 9/10)", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("figs/fig_17_rmse_by_K.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ### 6.4 — Per-currency breakdown by K

# %%
for K in latent_dims:
    print(f"\n  K={K}:")
    offset = 0
    for ccy in CURRENCIES:
        n = len(swap_aligned[ccy])
        raw_slice = rmse_by_K[K][offset:offset+n]
        pp_slice = rmse_pp_by_K[K][offset:offset+n]
        print(f"    {ccy}: raw={raw_slice.mean():.2f}bp  post-proc={pp_slice.mean():.2f}bp")
        offset += n

# %% [markdown]
# ---
# ## 7 — Latent Space Rotation via PCA
#
# The paper states:
# > "The coordinates in latent space were rotated such that the horizontal axis
# > predominantly encodes interest rate levels and the vertical axis curve shapes."
#
# We apply PCA to the latent codes to find the rotation matrix.

# %%
def rotate_latent_pca(z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Rotate latent codes via PCA so that:
    - z₁_rotated captures maximum variance (≈ level)
    - z₂_rotated captures residual (≈ shape)
    
    Returns
    -------
    z_rotated : rotated latent codes
    rotation_matrix : (K, K) orthogonal matrix
    explained_var : explained variance ratio per component
    """
    from sklearn.decomposition import PCA
    
    K = z.shape[1]
    pca = PCA(n_components=K)
    z_rotated = pca.fit_transform(z)
    
    return z_rotated, pca.components_, pca.explained_variance_ratio_

# %%
# Apply to K=2 model
z_raw = multi_vae_k2.get_latent(X_multi)
z_rotated, R_matrix, var_ratio = rotate_latent_pca(z_raw)

print(f"PCA rotation matrix:\n{R_matrix}")
print(f"\nExplained variance: PC1={var_ratio[0]:.3f}, PC2={var_ratio[1]:.3f}")

# %% [markdown]
# ### 7.1 — Verify that PC1 ≈ level

# %%
# Compute correlation between PC1 and mean swap rate (level proxy)
x_true_dec = denormalize_rates(X_multi.numpy())
level_proxy = x_true_dec.mean(axis=1) * 100  # percent

corr_pc1_level = np.corrcoef(z_rotated[:, 0], level_proxy)[0, 1]
corr_pc2_level = np.corrcoef(z_rotated[:, 1], level_proxy)[0, 1]

# If PC1 is anti-correlated with level, flip sign
if abs(corr_pc1_level) < abs(corr_pc2_level):
    print("WARNING: PC1 is less correlated with level than PC2. Swapping.")
    z_rotated = z_rotated[:, ::-1].copy()
    corr_pc1_level, corr_pc2_level = corr_pc2_level, corr_pc1_level

if corr_pc1_level < 0:
    z_rotated[:, 0] *= -1
    corr_pc1_level *= -1

print(f"Correlation PC1 ↔ Level: {corr_pc1_level:.3f}")
print(f"Correlation PC2 ↔ Level: {corr_pc2_level:.3f}")

# %% [markdown]
# ### 7.2 — Rotated world map (cf. Fig. 13)

# %%
default_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
colors = {ccy: default_cycle[i % len(default_cycle)] for i, ccy in enumerate(CURRENCIES)}

fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# (a) Raw latent space
ax = axes[0]
offset = 0
for ccy in CURRENCIES:
    n = len(swap_aligned[ccy])
    ax.scatter(z_raw[offset:offset+n, 0], z_raw[offset:offset+n, 1],
               s=3, alpha=0.3, label=ccy, color=colors[ccy])
    offset += n
ax.set_xlabel("$z_1$ (raw)", fontsize=14)
ax.set_ylabel("$z_2$ (raw)", fontsize=14)
ax.set_title("(a) Raw latent space", fontsize=13)
ax.legend(markerscale=5)

# (b) PCA-rotated latent space
ax = axes[1]
offset = 0
for ccy in CURRENCIES:
    n = len(swap_aligned[ccy])
    ax.scatter(z_rotated[offset:offset+n, 0], z_rotated[offset:offset+n, 1],
               s=3, alpha=0.3, label=ccy, color=colors[ccy])
    offset += n
ax.set_xlabel("$z_1^{rot}$ ≈ Level", fontsize=14)
ax.set_ylabel("$z_2^{rot}$ ≈ Shape", fontsize=14)
ax.set_title("(b) PCA-rotated latent space", fontsize=13)
ax.legend(markerscale=5)

plt.suptitle("World Map: Raw vs PCA-Rotated (cf. Fig. 13)", fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig("figs/fig_18_rotated_worldmap.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ---
# ## 8 — Save extension results

# %%
extensions_output = {
    "z_post_processed": z_pp,
    "rmse_post_processed": rmse_pp,
    "rmse_raw_encoder": rmse_raw,
    "models_by_K": {K: m.state_dict() for K, m in models_by_K.items()},
    "rmse_by_K": rmse_by_K,
    "rmse_pp_by_K": rmse_pp_by_K,
    "z_raw_k2": z_raw,
    "z_rotated_k2": z_rotated,
    "rotation_matrix": R_matrix,
    "pca_var_ratio": var_ratio,
}

with open("vae_extensions.pkl", "wb") as f:
    pickle.dump(extensions_output, f)

print("Saved to vae_extensions.pkl")
print(f"\nSummary of improvements (K=2):")
print(f"  Raw encoder:    {rmse_raw.mean():.2f} bp")
print(f"  Post-processed: {rmse_pp.mean():.2f} bp")
print(f"  Improvement:    {(1 - rmse_pp.mean()/rmse_raw.mean())*100:.1f}%")
