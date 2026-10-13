"""
Re-run NB3 §13: train a fresh MultiVAE on the TRAIN window only (<=
31 March 2025) and report mean RMSE on the train and test subsets.

This is the proper IS / OOS diagnostic promised by the report's
`fig:oos` caption. It is NOT saved to the main vae_results_extended
pickle, so we run it here standalone.

Outputs (appended to results/report_values.tex):
    % \TODO{IS mean RMSE}  → xx.xx bp
    % \TODO{OOS mean RMSE} → xx.xx bp
    % OOS/IS ratio         → xx.xx
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "df_multi.csv"
RES = ROOT / "results"

# ── Configuration (must match NB3) ─────────────────────────────────────────
SEED          = 42
TRAIN_CUTOFF  = pd.Timestamp("2025-03-31")
EXCLUDE       = {"CHF"}
TARGET_TENORS = [2, 3, 5, 10, 15, 20, 30]
TENOR_COLS    = [str(t) for t in TARGET_TENORS]
S_MIN, S_MAX  = -0.05, 0.25
BP_PER_UNIT   = 10_000.0
N_EPOCHS      = 5_000
BATCH_SIZE    = 256
LR            = 1e-3
BETA          = 1e-7

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device={device}")

# ── Normalisation helpers ──────────────────────────────────────────────────
def normalize_rates(r: np.ndarray) -> np.ndarray:
    return np.clip((r - S_MIN) / (S_MAX - S_MIN), 0.0, 1.0)

def denormalize_rates(x: np.ndarray) -> np.ndarray:
    return x * (S_MAX - S_MIN) + S_MIN

# ── VAE (multi-currency, Table 2 of Sokol) ─────────────────────────────────
class VAE(nn.Module):
    def __init__(self, input_dim=7, latent_dim=2, hidden_dim=4, multi_currency=True):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim), nn.Tanh(),
            nn.Linear(input_dim, 2 * latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, input_dim),  nn.Tanh(),
            nn.Linear(input_dim, input_dim),   nn.Sigmoid(),
        )
    def encode(self, x):
        h = self.encoder(x)
        return h[:, :self.latent_dim], h[:, self.latent_dim:]
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + std * torch.randn_like(std)
        return mu
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

def vae_loss(x_recon, x, mu, logvar, beta=BETA):
    recon = torch.mean(torch.sum((x_recon - x) ** 2, dim=-1))
    kld   = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - torch.exp(logvar), dim=-1))
    return recon + beta * kld, recon, kld

def train_vae(model, X, n_epochs=N_EPOCHS, batch_size=BATCH_SIZE,
              lr=LR, beta=BETA, print_every=500, scheduler_patience=500, grad_clip=1.0):
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min",
                                                 patience=scheduler_patience,
                                                 factor=0.5, min_lr=1e-6)
    loader = DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=True, drop_last=False)
    for epoch in range(1, n_epochs + 1):
        model.train()
        tot = rec = kldv = 0.0
        nb = 0
        for (xb,) in loader:
            xb = xb.to(device)
            xr, mu, lv = model(xb)
            loss, r, k = vae_loss(xr, xb, mu, lv, beta=beta)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            tot += float(loss); rec += float(r); kldv += float(k); nb += 1
        tot /= nb; rec /= nb; kldv /= nb
        sched.step(tot)
        if epoch == 1 or epoch % print_every == 0:
            print(f"  epoch {epoch:>5d}/{n_epochs}  loss={tot:.3e}  recon={rec:.3e}  kld={kldv:.3e}  lr={opt.param_groups[0]['lr']:.1e}")
    model.eval()

def compute_rmse_bp(model, X_norm):
    model.eval()
    with torch.no_grad():
        xr, _, _ = model(X_norm.to(device))
    x_recon = denormalize_rates(xr.cpu().numpy())
    x_true  = denormalize_rates(X_norm.cpu().numpy())
    return np.sqrt(np.mean((x_recon - x_true) ** 2, axis=1)) * BP_PER_UNIT

# ── Load data and build the train/test tensors ─────────────────────────────
df_long = pd.read_csv(DATA_PATH, parse_dates=["Date"])
df_long = df_long[~df_long["currency"].isin(EXCLUDE)].copy()
currencies = sorted(df_long["currency"].unique())
print(f"currencies={currencies}")

swap_aligned: Dict[str, pd.DataFrame] = {}
for ccy in currencies:
    sub = (df_long[df_long["currency"] == ccy]
           .set_index("Date").sort_index()[TENOR_COLS].copy())
    sub.columns = TARGET_TENORS
    swap_aligned[ccy] = sub

train_stack, test_stack = [], []
for ccy in currencies:
    df = swap_aligned[ccy]
    tr = df[df.index <= TRAIN_CUTOFF].values.astype(np.float32)
    te = df[df.index >  TRAIN_CUTOFF].values.astype(np.float32)
    train_stack.append(normalize_rates(tr).astype(np.float32))
    test_stack.append(normalize_rates(te).astype(np.float32))

X_train = torch.from_numpy(np.vstack(train_stack))
X_test  = torch.from_numpy(np.vstack(test_stack))
n_train, n_test = len(X_train), len(X_test)
print(f"X_train={tuple(X_train.shape)}  X_test={tuple(X_test.shape)}")
assert n_train == 566 * len(currencies), f"expected {566*len(currencies)} train obs, got {n_train}"
assert n_test  == 208 * len(currencies), f"expected {208*len(currencies)} test  obs, got {n_test}"

# ── Train and evaluate ─────────────────────────────────────────────────────
print("\n── Training MultiVAE on TRAIN only ───────────────────────────────")
model = VAE(input_dim=7, latent_dim=2, hidden_dim=4, multi_currency=True)
train_vae(model, X_train)

rmse_is  = compute_rmse_bp(model, X_train)
rmse_oos = compute_rmse_bp(model, X_test)
mean_is, mean_oos = float(rmse_is.mean()), float(rmse_oos.mean())
ratio = mean_oos / mean_is

print("\n── Results ─────────────────────────────────────────────────────────")
print(f"IS  mean RMSE = {mean_is:.3f} bp  (n = {n_train})")
print(f"OOS mean RMSE = {mean_oos:.3f} bp  (n = {n_test})")
print(f"OOS / IS      = {ratio:.3f}")

# ── Append to report_values.tex ────────────────────────────────────────────
out_path = RES / "report_values.tex"
appendix = f"""
% ───── §5.7  Out-of-Sample Reconstruction (retrained MultiVAE) ──────
% Produced by scripts/retrain_multivae_oos.py (seed={SEED}, 5000 epochs,
% MultiVAE trained on X_train only, n = {n_train}; evaluated on X_train
% and X_test separately).
%
% \\TODO{{IS  mean RMSE}}  →  {mean_is:.2f} bp
% \\TODO{{OOS mean RMSE}}  →  {mean_oos:.2f} bp
% \\TODO{{OOS / IS ratio}} →  {ratio:.2f}
"""
with open(out_path, "a") as f:
    f.write(appendix)
print(f"\n✓ Appended to {out_path.relative_to(ROOT)}")
