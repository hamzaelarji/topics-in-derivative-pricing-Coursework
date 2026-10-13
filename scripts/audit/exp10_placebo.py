"""Exp 10 — synthetic placebo: does the realignment destroy a genuine edge, and
does misalignment create a fake one?

For 7 synthetic "currencies" we simulate curves = level/slope/curvature random
walks (loadings and daily vols taken from the real AUD train-window PCA) plus
a small iid measurement noise.  Two worlds:
    EDGE : an AR(1) dislocation (phi=0.9, 3bp) is added to the 10Y point
           -> a genuine, tradable mean-reverting 2s10s residual
    NULL : no dislocation -> nothing to trade
Each world is observed (i) correctly aligned, (ii) with synthetic stepwise row
shifts per tenor (like the Bloomberg export), (iii) after realign.viterbi_shift.
Strategy: PCA-2 / PCA-3 fair-value residual on 2s10s (fit on first 566 days), NB4
fixed hyper-parameters, OOS = last 208 days, lag 1, tc 0.5bp.
"""
import numpy as np
import pandas as pd

from common import *
from realign import viterbi_shift

rng = np.random.default_rng(7)
T, N_TR = 774 + 200, 566 + 200   # 200-day burn-in buffer for the realignment
real = load_swaps("realigned")["AUD"]
tr = real[real.index <= TRAIN_CUTOFF].diff().dropna()
evals, evecs = np.linalg.eigh(np.cov(tr.values.T))
L = evecs[:, ::-1][:, :3]                       # 7 x 3 loadings
vol = np.sqrt(evals[::-1][:3])                  # daily factor vols (decimal)
i10 = TARGET_TENORS.index(10)


def simulate(edge: bool):
    f = np.cumsum(rng.standard_normal((T, 3)) * vol, axis=0)
    X = 0.04 + f @ L.T + rng.standard_normal((T, 7)) * 0.3e-4
    if edge:
        e = np.zeros(T)
        for t in range(1, T):
            e[t] = 0.9 * e[t - 1] + rng.standard_normal() * 3e-4 * np.sqrt(1 - 0.9 ** 2)
        X[:, i10] += e
    return X


def shift_paths():
    """Piecewise-constant shift per tenor, ~12 ±1 steps, starting from 0 at the end."""
    P = np.zeros((T, 7), dtype=int)
    for k in range(1, 7):
        s = np.zeros(T, dtype=int)
        for c in sorted(rng.choice(np.arange(50, T - 50), 12, replace=False)):
            s[:c] += rng.choice([-1, 1])
        P[:, k] = np.clip(s, -6, 6)
    return P


def apply_shift(X, P):
    Y = np.full_like(X, np.nan)
    for k in range(7):
        idx = np.arange(T) + P[:, k]
        ok = (idx >= 0) & (idx < T)
        Y[ok, k] = X[idx[ok], k]
    return Y


def realign(Y):
    d = np.diff(Y, axis=0, prepend=np.nan)
    z = (d - np.nanmean(d, 0)) / np.nanstd(d, 0)
    out = np.full_like(Y, np.nan); out[:, 0] = Y[:, 0]
    est = np.zeros((T, 7), dtype=int)
    for k in range(1, 7):
        s = viterbi_shift(z[:, 0], z[:, k], S=16, pen=4.0)
        est[:, k] = s
        idx = np.arange(T) + s; ok = (idx >= 0) & (idx < T)
        out[ok, k] = Y[idx[ok], k]
    return out, est


def strat_pnl(X, k=2):
    df = pd.DataFrame(X, columns=TARGET_TENORS).ffill().bfill()
    trn = df.iloc[:N_TR]; mu = trn.mean()
    V = np.linalg.svd((trn - mu).values, full_matrices=False)[2][:k].T
    fv = pd.DataFrame((df - mu).values @ V @ V.T + mu.values, columns=TARGET_TENORS)
    slope = (df[10] - df[2]) * BP
    bt = backtest_raw(slope - (fv[10] - fv[2]) * BP, slope.diff())
    return bt["pnl"].iloc[N_TR:].reset_index(drop=True)


for world in ("EDGE", "NULL"):
  for kfac in (2, 3):
    res = {"aligned (truth)": [], "misaligned": [], "realigned": []}
    acc = []
    for c in range(7):
        X = simulate(world == "EDGE")
        P = shift_paths()
        Y = apply_shift(X, P)
        Z, est = realign(Y)
        # recovery accuracy relative to the 2Y reference: true relative shift = P[:,k] - P[:,0] (P[:,0]=0)
        acc.append((est[:, 1:] == -P[:, 1:]).mean())  # out[i] = Y[i+s] = X[i+s+P] -> s = -P
        res["aligned (truth)"].append(strat_pnl(X, kfac))
        res["misaligned"].append(strat_pnl(Y, kfac))
        res["realigned"].append(strat_pnl(Z, kfac))
    print(f"\n== {world} world, PCA-{kfac} fair value  (shift-path recovery accuracy {np.mean(acc):.0%} of rows)")
    for k, v in res.items():
        port = pd.concat(v, axis=1).mean(1)
        print(f"  {k:<16} portfolio OOS Sharpe {sharpe(port):5.2f}   "
              f"per-ccy mean {np.mean([sharpe(p) for p in v]):5.2f}")
