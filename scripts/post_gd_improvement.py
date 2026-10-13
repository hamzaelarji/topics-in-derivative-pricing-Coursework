"""
Re-run NB3 §6b: apply the Sokol (2022) post-processing gradient-descent
refinement to the already-trained MultiVAE and report the before / after
reconstruction RMSE.

Starting from the encoder posterior mean μ(S), we run 100 Adam steps at
lr=0.01 directly on z to minimise ‖decode(z) - S‖².  No re-training.

Outputs to stdout and appends LaTeX-ready lines to results/report_values.tex.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "df_multi.csv"
RES = ROOT / "results"

# ── Config (must match NB3) ───────────────────────────────────────────────
EXCLUDE       = {"CHF"}
TARGET_TENORS = [2, 3, 5, 10, 15, 20, 30]
TENOR_COLS    = [str(t) for t in TARGET_TENORS]
S_MIN, S_MAX  = -0.05, 0.25
BP_PER_UNIT   = 10_000.0
N_STEPS       = 100
LR_GD         = 0.01
SEED          = 42

torch.manual_seed(SEED); np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device={device}")

def normalize_rates(r):  return np.clip((r - S_MIN) / (S_MAX - S_MIN), 0.0, 1.0)
def denormalize_rates(x): return x * (S_MAX - S_MIN) + S_MIN

# ── Architectures (verbatim from NB3, minimal subset) ─────────────────────
class MultiVAE(nn.Module):
    def __init__(self, input_dim=7, latent_dim=2, hidden_dim=4):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim), nn.Tanh(),
            nn.Linear(input_dim, 2 * latent_dim))
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, input_dim),  nn.Tanh(),
            nn.Linear(input_dim, input_dim),   nn.Sigmoid())
    def encode(self, x):
        h = self.encoder(x)
        return h[:, :self.latent_dim], h[:, self.latent_dim:]
    def decode(self, z): return self.decoder(z)
    def forward(self, x):
        mu, logvar = self.encode(x)
        return self.decode(mu), mu, logvar

class CVAE(nn.Module):
    def __init__(self, input_dim=7, latent_dim=2, n_currencies=7):
        super().__init__()
        self.latent_dim = latent_dim
        enc_in, dec_in = input_dim + n_currencies, latent_dim + n_currencies
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
    def decode(self, z, y): return self.decoder(torch.cat([z, y], dim=-1))
    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        return self.decode(mu, y), mu, logvar

# ── Post-processing GD ────────────────────────────────────────────────────
def post_gd(model, X, Y=None, n_steps=N_STEPS, lr=LR_GD):
    model.eval()
    is_cvae = isinstance(model, CVAE)
    X_dev = X.to(device)
    Y_dev = Y.to(device) if is_cvae else None
    with torch.no_grad():
        mu, _ = (model.encode(X_dev, Y_dev) if is_cvae else model.encode(X_dev))
    z = mu.clone().detach().requires_grad_(True)
    opt = optim.Adam([z], lr=lr)
    for _ in range(n_steps):
        opt.zero_grad()
        xr = model.decode(z, Y_dev) if is_cvae else model.decode(z)
        loss = torch.mean(torch.sum((xr - X_dev) ** 2, dim=-1))
        loss.backward()
        opt.step()
    with torch.no_grad():
        xr_final = model.decode(z, Y_dev) if is_cvae else model.decode(z)
    return xr_final.cpu().numpy()

def rmse_bp(x_recon_norm, x_true_norm):
    return np.sqrt(np.mean((denormalize_rates(x_recon_norm) - denormalize_rates(x_true_norm)) ** 2, axis=1)) * BP_PER_UNIT

# ── Load aligned data identically to NB3 ──────────────────────────────────
df_long = pd.read_csv(DATA_PATH, parse_dates=["Date"])
df_long = df_long[~df_long["currency"].isin(EXCLUDE)].copy()
currencies = sorted(df_long["currency"].unique())
N_CCY = len(currencies)
CCY_TO_IDX = {c: i for i, c in enumerate(currencies)}

rates_stack, labels_stack = [], []
for ccy in currencies:
    sub = (df_long[df_long["currency"] == ccy]
           .set_index("Date").sort_index()[TENOR_COLS].copy())
    sub.columns = TARGET_TENORS
    rn = normalize_rates(sub.values.astype(np.float32)).astype(np.float32)
    oh = np.zeros((len(rn), N_CCY), dtype=np.float32); oh[:, CCY_TO_IDX[ccy]] = 1.0
    rates_stack.append(rn); labels_stack.append(oh)

X_multi = torch.from_numpy(np.vstack(rates_stack))
Y_multi = torch.from_numpy(np.vstack(labels_stack))
print(f"X_multi={tuple(X_multi.shape)}  Y_multi={tuple(Y_multi.shape)}")

# ── Load saved weights ────────────────────────────────────────────────────
with open(RES / "vae_results_extended.pkl", "rb") as f:
    vae_data = pickle.load(f)

multi_vae = MultiVAE().to(device)
multi_vae.load_state_dict(vae_data["multi_vae_state"])
multi_vae.eval()

cvae = CVAE(n_currencies=N_CCY).to(device)
cvae.load_state_dict(vae_data["cvae_state"])
cvae.eval()

# ── Before / after on MultiVAE ────────────────────────────────────────────
with torch.no_grad():
    xr_before_m, _, _ = multi_vae(X_multi.to(device))
xr_before_m = xr_before_m.cpu().numpy()
xr_after_m  = post_gd(multi_vae, X_multi)

mean_before_m = rmse_bp(xr_before_m, X_multi.numpy()).mean()
mean_after_m  = rmse_bp(xr_after_m,  X_multi.numpy()).mean()
rel_m = (1 - mean_after_m / mean_before_m) * 100

# ── Before / after on CVAE (bonus diagnostic) ─────────────────────────────
with torch.no_grad():
    xr_before_c, _, _ = cvae(X_multi.to(device), Y_multi.to(device))
xr_before_c = xr_before_c.cpu().numpy()
xr_after_c  = post_gd(cvae, X_multi, Y_multi)

mean_before_c = rmse_bp(xr_before_c, X_multi.numpy()).mean()
mean_after_c  = rmse_bp(xr_after_c,  X_multi.numpy()).mean()
rel_c = (1 - mean_after_c / mean_before_c) * 100

print("\n── Post-processing GD results ────────────────────────────────────")
print(f"MultiVAE  before: {mean_before_m:.3f} bp   after: {mean_after_m:.3f} bp   Δ = {rel_m:.2f}%")
print(f"CVAE      before: {mean_before_c:.3f} bp   after: {mean_after_c:.3f} bp   Δ = {rel_c:.2f}%")

# ── Append to report_values.tex ───────────────────────────────────────────
out_path = RES / "report_values.tex"
appendix = f"""
% ───── §5.6  Post-Processing Gradient Descent  (NB3 §6b) ─────────────
% Produced by scripts/post_gd_improvement.py
% 100 Adam steps at lr=0.01 starting from encoder mean μ(S).
%
% MultiVAE:   before = {mean_before_m:.2f} bp → after = {mean_after_m:.2f} bp   ({rel_m:.1f}% relative improvement)
% CVAE:       before = {mean_before_c:.2f} bp → after = {mean_after_c:.2f} bp   ({rel_c:.1f}% relative improvement)
"""
with open(out_path, "a") as f:
    f.write(appendix)
print(f"\n✓ Appended to {out_path.relative_to(ROOT)}")
