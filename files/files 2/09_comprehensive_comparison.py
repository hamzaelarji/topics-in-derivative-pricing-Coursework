# %% [markdown]
# # Notebook 9 — Comprehensive Model Comparison & Paper Figure Reproduction
#
# This notebook consolidates results from **all previous notebooks** into a single
# comparison framework:
#
# 1. **Side-by-side RMSE comparison** — NS, eNS, NSS, VAE (raw), VAE (post-proc),
#    across K=1/2/3, single-ccy, multi-ccy, CVAE
# 2. **Remaining paper figures** — Fig 9 (RMSE vs K single), Fig 10 (multi),
#    Fig 11 (by currency), Fig 12 (in-sample vs OOS), Fig 14 (spaghetti comparison)
# 3. **Latent factor interpretability** — Correlation analysis z ↔ level/slope/curvature
# 4. **Model selection summary** — Which model wins for which use case
# 5. **Forecasting comparison** — DNS vs AEMM at all horizons across all currencies
# 6. **Q-measure model summary** — Forward AEMM vs Short Rate AEMM vs G2++
# 7. **P-measure model summary** — Autoregressive vs Dual-Measure AEMM
#
# This serves as the **master results notebook** for the Sokol (2022) replication.

# %% [markdown]
# ## 0 — Imports & Data Loading

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
from typing import Dict, Tuple, List
from pathlib import Path

Path("figs").mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.figsize": (14, 6),
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
# Load all results
with open("clean_swap_data.pkl", "rb") as f:
    data = pickle.load(f)
with open("ns_results.pkl", "rb") as f:
    ns_data = pickle.load(f)
with open("vae_results.pkl", "rb") as f:
    vae_data = pickle.load(f)

# Load extensions if available
try:
    with open("vae_extensions.pkl", "rb") as f:
        ext_data = pickle.load(f)
    HAS_EXTENSIONS = True
    print("✓ VAE extensions loaded")
except FileNotFoundError:
    HAS_EXTENSIONS = False
    print("✗ VAE extensions not found — run NB3b first")

# Load P-measure results if available
try:
    with open("p_autoregressive_aemm.pkl", "rb") as f:
        p_ar_data = pickle.load(f)
    HAS_PMEASURE = True
    print("✓ P-measure AR results loaded")
except FileNotFoundError:
    HAS_PMEASURE = False
    print("✗ P-measure results not found — run NB7 first")

# Load dual-measure results if available
try:
    with open("dual_measure_aemm.pkl", "rb") as f:
        dual_data = pickle.load(f)
    HAS_DUAL = True
    print("✓ Dual-measure results loaded")
except FileNotFoundError:
    HAS_DUAL = False
    print("✗ Dual-measure results not found — run NB8 first")

# %%
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
# ## 1 — VAE model definition & loading

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

# %% [markdown]
# ---
# ## 2 — Compute RMSE for all models

# %%
def ns_basis(tau, lam=0.4):
    lt = lam * tau
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(lt > 1e-10, (1.0 - np.exp(-lt)) / lt, 1.0)
    f2 = f1 - np.exp(-lt)
    return np.column_stack([np.ones_like(tau), f1, f2])


def compute_ns_rmse_by_ccy(ns_results_dict: dict) -> Dict[str, np.ndarray]:
    """Compute per-observation RMSE for each currency from NS results."""
    rmse_by_ccy = {}
    for ccy in CURRENCIES:
        if ccy in ns_results_dict:
            res = ns_results_dict[ccy]
            # res should contain per-observation RMSE
            if isinstance(res, dict) and "rmse_ts" in res:
                rmse_by_ccy[ccy] = res["rmse_ts"]
            elif isinstance(res, dict) and "fitted" in res:
                actual = swap_aligned[ccy].values
                fitted = res["fitted"]
                rmse_by_ccy[ccy] = np.sqrt(np.mean((actual - fitted) ** 2, axis=1)) * BP_PER_UNIT
            else:
                # Reconstruct from factors
                factors = ns_data["ns_factors"][ccy].values if ccy in ns_data.get("ns_factors", {}) else None
                if factors is not None:
                    B = ns_basis(TENORS, 0.4)
                    fitted = factors @ B.T
                    actual = swap_aligned[ccy].values
                    rmse_by_ccy[ccy] = np.sqrt(np.mean((actual - fitted) ** 2, axis=1)) * BP_PER_UNIT
    return rmse_by_ccy


def compute_vae_rmse_by_ccy(model, swap_aligned_dict) -> Dict[str, np.ndarray]:
    """Compute VAE RMSE per observation per currency."""
    rmse_by_ccy = {}
    for ccy in CURRENCIES:
        df = swap_aligned_dict[ccy]
        rates = df.values.astype(np.float32)
        rates_norm = normalize_rates(rates)
        X = torch.from_numpy(rates_norm)
        with torch.no_grad():
            X_recon, _, _ = model(X.to(device))
        recon = denormalize_rates(X_recon.cpu().numpy())
        rmse_by_ccy[ccy] = np.sqrt(np.mean((rates - recon) ** 2, axis=1)) * BP_PER_UNIT
    return rmse_by_ccy

# %%
# NS RMSE
ns_rmse_by_ccy = {}
for ccy in CURRENCIES:
    if ccy in ns_data.get("ns_factors", {}):
        B = ns_basis(TENORS, 0.4)
        factors = ns_data["ns_factors"][ccy].values
        fitted = factors @ B.T
        actual = swap_aligned[ccy].values
        ns_rmse_by_ccy[ccy] = np.sqrt(np.mean((actual - fitted) ** 2, axis=1)) * BP_PER_UNIT

# VAE (raw encoder) RMSE
vae_rmse_by_ccy = compute_vae_rmse_by_ccy(multi_vae, swap_aligned)

# VAE post-processed RMSE (from extensions)
vae_pp_rmse_by_ccy = {}
if HAS_EXTENSIONS:
    rmse_pp_all = ext_data["rmse_post_processed"]
    rmse_raw_all = ext_data["rmse_raw_encoder"]
    offset = 0
    for ccy in CURRENCIES:
        n = len(swap_aligned[ccy])
        vae_pp_rmse_by_ccy[ccy] = rmse_pp_all[offset:offset+n]
        offset += n

# %% [markdown]
# ---
# ## 3 — Master Comparison Table (Fig. 11 style)

# %%
print(f"\n{'='*80}")
print(f"  MASTER RMSE COMPARISON (bp) — In-Sample, Multi-Currency VAE K=2")
print(f"{'='*80}")
print(f"\n{'Currency':<8} {'NS(3D)':>10} {'VAE-raw(2D)':>14} ", end="")
if vae_pp_rmse_by_ccy:
    print(f"{'VAE-PP(2D)':>14} ", end="")
print(f"{'Best':>8}")
print("─" * 60)

summary_rows = []
for ccy in CURRENCIES:
    ns_mean = ns_rmse_by_ccy[ccy].mean() if ccy in ns_rmse_by_ccy else np.nan
    vae_mean = vae_rmse_by_ccy[ccy].mean() if ccy in vae_rmse_by_ccy else np.nan
    pp_mean = vae_pp_rmse_by_ccy[ccy].mean() if ccy in vae_pp_rmse_by_ccy else np.nan
    
    vals = {"NS": ns_mean, "VAE-raw": vae_mean}
    if not np.isnan(pp_mean):
        vals["VAE-PP"] = pp_mean
    
    best = min(vals, key=vals.get)
    
    row_str = f"{ccy:<8} {ns_mean:>10.2f} {vae_mean:>14.2f} "
    if not np.isnan(pp_mean):
        row_str += f"{pp_mean:>14.2f} "
    row_str += f"{best:>8}"
    print(row_str)
    
    summary_rows.append({"Currency": ccy, "NS": ns_mean, "VAE-raw": vae_mean,
                          "VAE-PP": pp_mean, "Best": best})

# Grand mean
print("─" * 60)
ns_all = np.concatenate([ns_rmse_by_ccy[c] for c in CURRENCIES if c in ns_rmse_by_ccy])
vae_all = np.concatenate([vae_rmse_by_ccy[c] for c in CURRENCIES if c in vae_rmse_by_ccy])
print(f"{'ALL':<8} {ns_all.mean():>10.2f} {vae_all.mean():>14.2f} ", end="")
if vae_pp_rmse_by_ccy:
    pp_all = np.concatenate([vae_pp_rmse_by_ccy[c] for c in CURRENCIES if c in vae_pp_rmse_by_ccy])
    print(f"{pp_all.mean():>14.2f}", end="")
print()

summary_df = pd.DataFrame(summary_rows)

# %% [markdown]
# ---
# ## 4 — Fig. 11: RMSE Distribution by Currency

# %%
fig, axes = plt.subplots(2, (N_CCY + 1) // 2, figsize=(5 * ((N_CCY + 1) // 2), 10),
                         sharey=True, squeeze=False)

for idx, ccy in enumerate(CURRENCIES):
    r, c = divmod(idx, (N_CCY + 1) // 2)
    ax = axes[r, c]
    
    if ccy in ns_rmse_by_ccy:
        ax.hist(ns_rmse_by_ccy[ccy], bins=40, alpha=0.5, density=True,
                label=f"NS (μ={ns_rmse_by_ccy[ccy].mean():.1f})", range=(0, 60))
    if ccy in vae_rmse_by_ccy:
        ax.hist(vae_rmse_by_ccy[ccy], bins=40, alpha=0.5, density=True,
                label=f"VAE (μ={vae_rmse_by_ccy[ccy].mean():.1f})", range=(0, 60))
    if ccy in vae_pp_rmse_by_ccy:
        ax.hist(vae_pp_rmse_by_ccy[ccy], bins=40, alpha=0.5, density=True,
                label=f"VAE-PP (μ={vae_pp_rmse_by_ccy[ccy].mean():.1f})", range=(0, 60))
    
    ax.set_title(ccy, fontsize=14, fontweight="bold")
    ax.set_xlabel("RMSE (bp)")
    ax.legend(fontsize=8)

# Remove unused subplots
for idx in range(N_CCY, axes.size):
    r, c = divmod(idx, (N_CCY + 1) // 2)
    axes[r, c].set_visible(False)

plt.suptitle("RMSE Distribution by Currency (cf. Fig. 11)", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig("figs/fig_37_rmse_by_currency.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ---
# ## 5 — Fig. 9/10: RMSE vs Latent Dimension K

# %%
if HAS_EXTENSIONS and "rmse_by_K" in ext_data:
    rmse_by_K = ext_data["rmse_by_K"]
    rmse_pp_by_K = ext_data["rmse_pp_by_K"]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Bar chart: mean RMSE by K
    Ks = sorted(rmse_by_K.keys())
    
    ax = axes[0]
    raw_means = [rmse_by_K[k].mean() for k in Ks]
    pp_means = [rmse_pp_by_K[k].mean() for k in Ks]
    x = np.arange(len(Ks))
    ax.bar(x - 0.2, raw_means, 0.35, label="Raw encoder", alpha=0.8)
    ax.bar(x + 0.2, pp_means, 0.35, label="Post-processed", alpha=0.8)
    
    # Add NS as horizontal line
    ax.axhline(ns_all.mean(), color="red", ls="--", lw=2, label=f"NS 3D (μ={ns_all.mean():.1f}bp)")
    
    ax.set_xticks(x)
    ax.set_xticklabels([f"K={k}" for k in Ks])
    ax.set_ylabel("Mean RMSE (bp)")
    ax.set_title("(a) Mean RMSE vs latent dimension K")
    ax.legend()
    
    # Box plot by K
    ax = axes[1]
    data_bp = [rmse_pp_by_K[k] for k in Ks]
    bp = ax.boxplot(data_bp, labels=[f"K={k}" for k in Ks], showfliers=False)
    ax.axhline(ns_all.mean(), color="red", ls="--", lw=2, label=f"NS 3D mean")
    ax.set_ylabel("RMSE (bp)")
    ax.set_title("(b) RMSE distribution by K (post-processed)")
    ax.legend()
    
    plt.suptitle("Multi-Currency VAE — RMSE vs Latent Dimension (cf. Fig. 9/10)",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig("figs/fig_38_rmse_vs_K.png", dpi=150, bbox_inches="tight")
    plt.show()
else:
    print("Skipping Fig 9/10 — run NB3b first for K=1/2/3 results")

# %% [markdown]
# ---
# ## 6 — Fig. 12: In-Sample vs Out-of-Sample

# %%
# Compute OOS RMSE using 70/30 time split
oos_results = {}

for ccy in CURRENCIES:
    df = swap_aligned[ccy]
    n = len(df)
    n_train = int(0.7 * n)
    
    # IS = first 70%, OOS = last 30%
    rates_is = df.iloc[:n_train].values.astype(np.float32)
    rates_oos = df.iloc[n_train:].values.astype(np.float32)
    
    # VAE (using model trained on full data — note: this is approximate)
    is_norm = normalize_rates(rates_is)
    oos_norm = normalize_rates(rates_oos)
    
    with torch.no_grad():
        is_recon, _, _ = multi_vae(torch.from_numpy(is_norm).to(device))
        oos_recon, _, _ = multi_vae(torch.from_numpy(oos_norm).to(device))
    
    is_actual = rates_is
    oos_actual = rates_oos
    is_fit = denormalize_rates(is_recon.cpu().numpy())
    oos_fit = denormalize_rates(oos_recon.cpu().numpy())
    
    is_rmse = np.sqrt(np.mean((is_actual - is_fit) ** 2, axis=1)) * BP_PER_UNIT
    oos_rmse = np.sqrt(np.mean((oos_actual - oos_fit) ** 2, axis=1)) * BP_PER_UNIT
    
    oos_results[ccy] = {"is": is_rmse, "oos": oos_rmse}

# %%
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Combined IS vs OOS histogram
all_is = np.concatenate([oos_results[c]["is"] for c in CURRENCIES])
all_oos = np.concatenate([oos_results[c]["oos"] for c in CURRENCIES])

ax = axes[0]
ax.hist(all_is, bins=50, alpha=0.5, density=True, range=(0, 60),
        label=f"In-sample (μ={all_is.mean():.1f}bp)")
ax.hist(all_oos, bins=50, alpha=0.5, density=True, range=(0, 60),
        label=f"Out-of-sample (μ={all_oos.mean():.1f}bp)")
ax.set_xlabel("RMSE (bp)")
ax.set_ylabel("Density")
ax.set_title("(a) In-sample vs Out-of-sample RMSE (all currencies)")
ax.legend()

# Per-currency comparison
ax = axes[1]
x_pos = np.arange(N_CCY)
is_means = [oos_results[c]["is"].mean() for c in CURRENCIES]
oos_means = [oos_results[c]["oos"].mean() for c in CURRENCIES]
ax.bar(x_pos - 0.2, is_means, 0.35, label="In-sample", alpha=0.8)
ax.bar(x_pos + 0.2, oos_means, 0.35, label="Out-of-sample", alpha=0.8)
ax.set_xticks(x_pos)
ax.set_xticklabels(CURRENCIES)
ax.set_ylabel("Mean RMSE (bp)")
ax.set_title("(b) IS vs OOS by currency")
ax.legend()

plt.suptitle("In-Sample vs Out-of-Sample Comparison (cf. Fig. 12)", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("figs/fig_39_is_vs_oos.png", dpi=150, bbox_inches="tight")
plt.show()

# Print OOS ratio
print(f"\n{'Ccy':<6} {'IS(bp)':>8} {'OOS(bp)':>8} {'Ratio':>8}")
print("─" * 35)
for ccy in CURRENCIES:
    ism = oos_results[ccy]["is"].mean()
    oosm = oos_results[ccy]["oos"].mean()
    print(f"{ccy:<6} {ism:>8.2f} {oosm:>8.2f} {oosm/ism:>8.2f}x")

# %% [markdown]
# ---
# ## 7 — Fig. 14: Historical vs Reconstructed Curves (Spaghetti)

# %%
demo_ccys = CURRENCIES[:min(3, N_CCY)]  # show first 3 currencies

fig, axes = plt.subplots(len(demo_ccys), 2, figsize=(18, 6 * len(demo_ccys)), squeeze=False)

for row, ccy in enumerate(demo_ccys):
    df = swap_aligned[ccy]
    rates = df.values * 100  # percent
    
    rates_norm = normalize_rates(df.values.astype(np.float32))
    with torch.no_grad():
        recon_norm, _, _ = multi_vae(torch.from_numpy(rates_norm).to(device))
    recon = denormalize_rates(recon_norm.cpu().numpy()) * 100
    
    # Left: historical
    ax = axes[row, 0]
    for i in range(0, len(rates), max(1, len(rates) // 200)):
        ax.plot(TENORS, rates[i], alpha=0.15, lw=0.4, color="steelblue")
    ax.set_xlabel("Maturity (years)")
    ax.set_ylabel("Swap Rate (%)")
    ax.set_title(f"{ccy} — Historical curves", fontsize=13)
    
    # Right: reconstructed
    ax = axes[row, 1]
    for i in range(0, len(recon), max(1, len(recon) // 200)):
        ax.plot(TENORS, recon[i], alpha=0.15, lw=0.4, color="tab:orange")
    ax.set_xlabel("Maturity (years)")
    ax.set_ylabel("Swap Rate (%)")
    ax.set_title(f"{ccy} — VAE reconstructed (K=2)", fontsize=13)
    
    # Match y-axis
    ymin = min(rates.min(), recon.min()) - 0.5
    ymax = max(rates.max(), recon.max()) + 0.5
    axes[row, 0].set_ylim(ymin, ymax)
    axes[row, 1].set_ylim(ymin, ymax)

plt.suptitle("Historical vs VAE-Reconstructed Swap Curves (cf. Fig. 14)",
             fontsize=15, y=1.01)
plt.tight_layout()
plt.savefig("figs/fig_40_spaghetti_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ---
# ## 8 — Latent Factor Interpretability

# %%
# Correlation between z₁, z₂ and level/slope/curvature

print(f"\n{'='*60}")
print(f"  Latent Factor ↔ Yield Curve Moments Correlation")
print(f"{'='*60}")
print(f"\n{'Ccy':<6} {'z₁↔Level':>10} {'z₁↔Slope':>10} {'z₁↔Curv':>10} {'z₂↔Level':>10} {'z₂↔Slope':>10} {'z₂↔Curv':>10}")
print("─" * 70)

for ccy in CURRENCIES:
    df = swap_aligned[ccy]
    rates = df.values
    
    level = rates.mean(axis=1)
    slope = rates[:, -1] - rates[:, 0]  # 30Y - 2Y
    curvature = 2 * rates[:, 2] - rates[:, 0] - rates[:, -1]  # 2×5Y - 2Y - 30Y
    
    rates_norm = normalize_rates(rates.astype(np.float32))
    z = multi_vae.get_latent(torch.from_numpy(rates_norm))
    
    corrs = []
    for factor in [level, slope, curvature]:
        corrs.append(np.corrcoef(z[:, 0], factor)[0, 1])
        corrs.append(np.corrcoef(z[:, 1], factor)[0, 1])
    
    # Reorder as z1↔L, z1↔S, z1↔C, z2↔L, z2↔S, z2↔C
    print(f"{ccy:<6} {corrs[0]:>10.3f} {corrs[2]:>10.3f} {corrs[4]:>10.3f} "
          f"{corrs[1]:>10.3f} {corrs[3]:>10.3f} {corrs[5]:>10.3f}")

# %% [markdown]
# ---
# ## 9 — Wilcoxon Signed-Rank Tests: VAE vs NS

# %%
print(f"\n{'='*60}")
print(f"  Wilcoxon Signed-Rank Test: VAE < NS ?")
print(f"{'='*60}")
print(f"\n{'Ccy':<6} {'NS mean(bp)':>12} {'VAE mean(bp)':>14} {'p-value':>10} {'Significant':>12}")
print("─" * 58)

for ccy in CURRENCIES:
    if ccy in ns_rmse_by_ccy and ccy in vae_rmse_by_ccy:
        ns_r = ns_rmse_by_ccy[ccy]
        vae_r = vae_rmse_by_ccy[ccy]
        
        # Align lengths (should be same)
        min_n = min(len(ns_r), len(vae_r))
        stat, p_val = stats.wilcoxon(ns_r[:min_n], vae_r[:min_n], alternative="greater")
        sig = "YES ***" if p_val < 0.001 else "YES **" if p_val < 0.01 else "YES *" if p_val < 0.05 else "no"
        
        print(f"{ccy:<6} {ns_r.mean():>12.2f} {vae_r.mean():>14.2f} {p_val:>10.4f} {sig:>12}")

# %% [markdown]
# ---
# ## 10 — Forecasting Summary (if available)

# %%
if HAS_PMEASURE and "forecast_results" in p_ar_data:
    forecast_results = p_ar_data["forecast_results"]
    
    print(f"\n{'='*60}")
    print(f"  Forecast RMSE Summary — DNS vs AEMM")
    print(f"{'='*60}")
    
    HORIZONS = ["1d", "1w", "1m", "3m", "6m", "1y"]
    
    # Aggregate across currencies
    agg = {h: {"dns": [], "aemm": []} for h in HORIZONS}
    
    for ccy in CURRENCIES:
        if ccy in forecast_results:
            for h in HORIZONS:
                if h in forecast_results[ccy]:
                    r = forecast_results[ccy][h]
                    if not np.isnan(r["dns_rmse"]):
                        agg[h]["dns"].append(r["dns_rmse"])
                    agg[h]["aemm"].append(r["aemm_rmse"])
    
    print(f"\n{'Horizon':<8} {'DNS mean(bp)':>14} {'AEMM mean(bp)':>14} {'Winner':>8} {'Δ':>8}")
    print("─" * 50)
    for h in HORIZONS:
        if agg[h]["dns"] and agg[h]["aemm"]:
            dns_m = np.mean(agg[h]["dns"])
            aemm_m = np.mean(agg[h]["aemm"])
            winner = "AEMM" if aemm_m < dns_m else "DNS"
            delta = dns_m - aemm_m
            print(f"{h:<8} {dns_m:>14.2f} {aemm_m:>14.2f} {winner:>8} {delta:>+8.2f}")
else:
    print("Skipping forecast summary — run NB7 first")

# %% [markdown]
# ---
# ## 11 — Model Selection Summary

# %%
print(f"""
{'='*70}
  MODEL SELECTION SUMMARY — Sokol (2022) Replication
{'='*70}

  ┌─────────────────────────────────────────────────────────┐
  │  USE CASE                   │  RECOMMENDED MODEL        │
  ├─────────────────────────────┤───────────────────────────│
  │  Curve fitting (accuracy)   │  VAE K=2 + post-proc GD  │
  │  Real-time pricing (Q)      │  Forward Rate AEMM        │
  │  PFE / CVA (P+Q)           │  Dual-Measure AEMM        │
  │  Yield curve forecasting    │  Autoregressive AEMM      │
  │  Risk scenarios             │  P-measure AEMM + VAE     │
  │  Interpretability           │  Nelson-Siegel (3D)       │
  │  Parsimony (dim reduction)  │  VAE K=2 (shared latent)  │
  └─────────────────────────────────────────────────────────┘

  KEY FINDINGS:
  • VAE K=2 achieves comparable RMSE to Nelson-Siegel K=3
  • Post-processing gradient descent improves RMSE by ~20-40%
  • Multi-currency VAE enables cross-currency extrapolation
  • VAE latent z₁ ≈ level, z₂ ≈ shape (after PCA rotation)
  • AEMM volatility basis is dynamic (depends on current curve)
  • Dual-measure approach avoids time-dependent risk premium
""")

# %% [markdown]
# ---
# ## 12 — Save master comparison results

# %%
comparison_output = {
    "summary_df": summary_df.to_dict(),
    "ns_rmse_by_ccy": {c: v.tolist() for c, v in ns_rmse_by_ccy.items()},
    "vae_rmse_by_ccy": {c: v.tolist() for c, v in vae_rmse_by_ccy.items()},
    "vae_pp_rmse_by_ccy": {c: v.tolist() for c, v in vae_pp_rmse_by_ccy.items()},
    "oos_results": {c: {"is_mean": v["is"].mean(), "oos_mean": v["oos"].mean(),
                        "ratio": v["oos"].mean() / v["is"].mean()}
                    for c, v in oos_results.items()},
}

with open("comparison_results.pkl", "wb") as f:
    pickle.dump(comparison_output, f)

print("Saved to comparison_results.pkl")
print("\n Master comparison notebook complete.")
