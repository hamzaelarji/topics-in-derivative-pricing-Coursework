"""Shared code for the reproducibility / P&L study.

Everything below is copied verbatim (logic-wise) from NB3 / NB4 so that
results can be compared like-for-like with the notebooks.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]  # repo root
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)
# weights trained by the ORIGINAL (unaligned) NB3 run, backed up before the realigned re-run
ORIG_RESULTS = ROOT / "results" / "original_backup_2026-10-13"

TARGET_TENORS = [2, 3, 5, 10, 15, 20, 30]
TENOR_COLS = [str(t) for t in TARGET_TENORS]
S_MIN, S_MAX = -0.05, 0.25
BP = 10_000.0
TRAIN_CUTOFF = pd.Timestamp("2025-03-31")
EXCLUDE = {"CHF"}


# ── data ────────────────────────────────────────────────────────────────
import os
DATA_TAG = os.environ.get("DATA_TAG", "orig")  # "orig" = data/df_multi_unaligned.csv, "realigned" = data/df_multi_repaired_2023_2026.csv (NB1 §2b+§2c on the original export)


def load_swaps(tag: str = None) -> Dict[str, pd.DataFrame]:
    tag = tag or DATA_TAG
    path = ROOT / "data" / ("df_multi_unaligned.csv" if tag == "orig" else "df_multi_repaired_2023_2026.csv")
    df_long = pd.read_csv(path, parse_dates=["Date"])
    df_long = df_long[~df_long["currency"].isin(EXCLUDE)]
    out = {}
    for ccy in sorted(df_long["currency"].unique()):
        sub = df_long[df_long["currency"] == ccy].set_index("Date").sort_index()[TENOR_COLS].copy()
        sub.columns = TARGET_TENORS
        out[ccy] = sub
    return out


def normalize(r):
    return np.clip((r - S_MIN) / (S_MAX - S_MIN), 0.0, 1.0)


def denormalize(x):
    return x * (S_MAX - S_MIN) + S_MIN


def stack(swaps: Dict[str, pd.DataFrame], currencies, mask_fn=None):
    """Stack currencies into (X, Y one-hot). mask_fn(index)->bool mask."""
    Xs, Ys = [], []
    n_ccy = len(currencies)
    for i, ccy in enumerate(currencies):
        df = swaps[ccy]
        if mask_fn is not None:
            df = df[mask_fn(df.index)]
        x = normalize(df.values.astype(np.float32)).astype(np.float32)
        y = np.zeros((len(x), n_ccy), dtype=np.float32)
        y[:, i] = 1.0
        Xs.append(x)
        Ys.append(y)
    return torch.from_numpy(np.vstack(Xs)), torch.from_numpy(np.vstack(Ys))


# ── models (NB3, Tables 2-3 of Sokol) ───────────────────────────────────
class MultiVAE(nn.Module):
    def __init__(self, input_dim=7, latent_dim=2, hidden_dim=4):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(nn.Linear(input_dim, input_dim), nn.Tanh(),
                                     nn.Linear(input_dim, 2 * latent_dim))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.Tanh(),
                                     nn.Linear(hidden_dim, input_dim), nn.Tanh(),
                                     nn.Linear(input_dim, input_dim), nn.Sigmoid())

    def encode(self, x, y=None):
        h = self.encoder(x)
        return h[:, :self.latent_dim], h[:, self.latent_dim:]

    def decode(self, z, y=None):
        return self.decoder(z)

    def forward(self, x, y=None):
        mu, lv = self.encode(x)
        z = mu + torch.exp(0.5 * lv) * torch.randn_like(mu) if self.training else mu
        return self.decode(z), mu, lv


class CVAE(nn.Module):
    def __init__(self, input_dim=7, latent_dim=2, n_currencies=7):
        super().__init__()
        self.latent_dim = latent_dim
        enc_in, dec_in = input_dim + n_currencies, latent_dim + n_currencies
        self.encoder = nn.Sequential(nn.Linear(enc_in, enc_in - 1), nn.Tanh(),
                                     nn.Linear(enc_in - 1, 2 * latent_dim))
        self.decoder = nn.Sequential(nn.Linear(dec_in, input_dim), nn.Tanh(),
                                     nn.Linear(input_dim, input_dim), nn.Tanh(),
                                     nn.Linear(input_dim, input_dim), nn.Sigmoid())

    def encode(self, x, y):
        h = self.encoder(torch.cat([x, y], -1))
        return h[:, :self.latent_dim], h[:, self.latent_dim:]

    def decode(self, z, y):
        return self.decoder(torch.cat([z, y], -1))

    def forward(self, x, y):
        mu, lv = self.encode(x, y)
        z = mu + torch.exp(0.5 * lv) * torch.randn_like(mu) if self.training else mu
        return self.decode(z, y), mu, lv


def vae_loss(xr, x, mu, lv, beta=1e-7):
    rec = torch.mean(torch.sum((xr - x) ** 2, -1))
    kld = torch.mean(-0.5 * torch.sum(1 + lv - mu.pow(2) - torch.exp(lv), -1))
    return rec + beta * kld, rec, kld


def train(model, X, Y, n_epochs=5000, batch_size=256, lr=1e-3, beta=1e-7,
          patience=500, grad_clip=1.0, verbose=False):
    """Identical to NB3 train_vae (Adam + ReduceLROnPlateau + clip)."""
    opt = optim.Adam(model.parameters(), lr=lr)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", patience=patience,
                                                 factor=0.5, min_lr=1e-6)
    loader = DataLoader(TensorDataset(X, Y), batch_size=batch_size, shuffle=True)
    hist = []
    for ep in range(1, n_epochs + 1):
        model.train()
        tot, nb = 0.0, 0
        for xb, yb in loader:
            xr, mu, lv = model(xb, yb)
            loss, _, _ = vae_loss(xr, xb, mu, lv, beta)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            tot += float(loss)
            nb += 1
        tot /= nb
        hist.append(tot)
        sched.step(tot)
        if verbose and (ep == 1 or ep % 500 == 0):
            print(f"  ep {ep} loss {tot:.3e}")
    model.eval()
    return hist


def reconstruct(model, df: pd.DataFrame, ccy_idx: int, n_ccy: int):
    """Return (reconstructed curves DataFrame [decimal], latent mu DataFrame)."""
    x = torch.from_numpy(normalize(df.values.astype(np.float32)).astype(np.float32))
    y = torch.zeros((len(x), n_ccy))
    y[:, ccy_idx] = 1.0
    model.eval()
    with torch.no_grad():
        xr, mu, _ = model(x, y)
    rec = pd.DataFrame(denormalize(xr.numpy()), index=df.index, columns=TARGET_TENORS)
    z = pd.DataFrame(mu.numpy(), index=df.index, columns=["z1", "z2"])
    return rec, z


def rmse_bp(rec: pd.DataFrame, df: pd.DataFrame) -> np.ndarray:
    return np.sqrt(((rec.values - df.values) ** 2).mean(1)) * BP


# ── Nelson-Siegel ───────────────────────────────────────────────────────
def ns_basis(tau, lam):  # exp(-lam*tau) parametrisation (NB2)
    lt = lam * np.asarray(tau, float)
    f1 = (1 - np.exp(-lt)) / lt
    return np.column_stack([np.ones_like(lt), f1, f1 - np.exp(-lt)])


def ns_fit(df: pd.DataFrame, lam: float) -> pd.DataFrame:
    B = ns_basis(np.array(TARGET_TENORS, float), lam)
    beta = np.linalg.lstsq(B, df.values.T, rcond=None)[0]
    return pd.DataFrame((B @ beta).T, index=df.index, columns=TARGET_TENORS)


# ── backtest engine (NB4, verbatim logic) ───────────────────────────────
def zscore(s: pd.Series, lb: int) -> pd.Series:
    rm = s.rolling(lb, min_periods=lb // 2).mean()
    rs = s.rolling(lb, min_periods=lb // 2).std().clip(lower=1e-6)
    return (s - rm) / rs


def positions(z: pd.Series, entry=1.0, exit_=0.3) -> pd.Series:
    pos = np.zeros(len(z))
    prev = 0.0
    for i, zi in enumerate(z.values):
        if np.isnan(zi):
            pos[i] = prev
            continue
        if zi > entry:
            prev = -1.0
        elif zi < -entry:
            prev = 1.0
        elif abs(zi) < exit_:
            prev = 0.0
        pos[i] = prev
    return pd.Series(pos, index=z.index)


def backtest_raw(signal: pd.Series, traded_chg_bp: pd.Series, entry=1.0, exit_=0.3,
                 lookback=60, tc_bp=0.5, lag=1) -> pd.DataFrame:
    """Raw-bp backtest. lag=1 reproduces NB4 (signal at close t, P&L from t to t+1).
    lag=2 adds one extra day of execution delay."""
    z = zscore(signal, lookback)
    pos = positions(z, entry, exit_)
    pl = pos.shift(lag).fillna(0.0)
    gross = pl * traded_chg_bp
    tc = pl.diff().abs().fillna(0.0) * tc_bp
    return pd.DataFrame({"z": z, "pos": pos, "gross": gross, "tc": tc, "pnl": gross - tc})


def sharpe(pnl: pd.Series) -> float:
    pnl = pnl.dropna()
    sd = pnl.std()
    return float(pnl.mean() / sd * np.sqrt(252)) if sd > 0 else 0.0


def stats(pnl: pd.Series) -> dict:
    pnl = pnl.dropna()
    cum = pnl.cumsum()
    return {"Total": float(cum.iloc[-1]), "Sharpe": sharpe(pnl),
            "MaxDD": float((cum - cum.cummax()).min()), "N": len(pnl)}
