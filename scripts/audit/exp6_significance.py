"""Exp 6 — statistical significance of the walk-forward P&L:
stationary block bootstrap CI for the Sharpe, and break-even transaction cost."""
import pickle

import numpy as np
import pandas as pd

from common import *

daily = pickle.load(open(OUT / f"exp5_daily_{DATA_TAG}.pkl", "rb"))
rng = np.random.default_rng(0)


def block_bootstrap_sharpe(x: np.ndarray, block=10, n_boot=5000):
    n = len(x)
    out = np.empty(n_boot)
    nb = int(np.ceil(n / block))
    for b in range(n_boot):
        starts = rng.integers(0, n - block + 1, nb)
        s = np.concatenate([x[i:i + block] for i in starts])[:n]
        out[b] = s.mean() / s.std() * np.sqrt(252)
    return np.percentile(out, [2.5, 97.5]), (out <= 0).mean()


print(f"{'signal':<14}{'lag':>4}{'Sharpe':>8}{'95% CI (block=10d)':>22}{'P(SR<=0)':>10}{'gross bp/yr':>12}{'turnover/yr':>12}{'breakeven tc':>13}")
for name in ["CVAE wf ens", "CVAE wf s0", "CVAE wf s1", "PCA-2 wf", "PCA-3 wf", "Naive slope", "NS λ=0.5"]:
    for lag in (1, 2):
        P = daily[(name, lag, 0.5)]
        port = P.mean(1).values
        ci, p0 = block_bootstrap_sharpe(port)
        G = daily[(name, lag, 0.0)]
        gross = G.mean(1)
        # turnover cost per bp of tc = (gross - net@0.5)/0.5
        cost_per_bp = (gross - P.mean(1)).sum() / 0.5
        years = len(gross) / 252
        be = gross.sum() / cost_per_bp if cost_per_bp > 0 else np.nan
        print(f"{name:<14}{lag:>4}{sharpe(pd.Series(port)):>8.2f}   [{ci[0]:5.2f}, {ci[1]:5.2f}]{p0:>12.3f}"
              f"{gross.sum()/years:>12.1f}{cost_per_bp/years:>12.1f}{be:>12.2f}bp")
