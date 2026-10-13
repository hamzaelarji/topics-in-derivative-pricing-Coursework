"""Exp 5 — strict walk-forward backtest (expanding window, quarterly refit).

At each refit date R_k the CVAE is trained ONLY on data <= R_k and is then used
to produce signals for dates in (R_k, R_{k+1}].  This roughly doubles the
out-of-sample window (Feb-2024 → Jan-2026, ~500 days) compared with NB4's
208-day test set, and removes the look-ahead of the full-sample CVAE.

Usage:  python exp5_walkforward.py train   # trains the models (parallel)
        python exp5_walkforward.py eval    # builds signals + backtests
"""
import pickle
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch

from common import *

REFITS = pd.to_datetime(["2024-01-31", "2024-04-30", "2024-07-31", "2024-10-31",
                         "2025-01-31", "2025-04-30", "2025-07-31", "2025-10-31"])
SEEDS = [0, 1]
WF = OUT / ("wf_models" if DATA_TAG == "orig" else f"wf_models_{DATA_TAG}")
WF.mkdir(exist_ok=True)


def job(args):
    k, seed = args
    path = WF / f"cvae_r{k}_s{seed}.pt"
    if path.exists():
        return f"skip {path.name}"
    torch.set_num_threads(1)
    swaps = load_swaps()
    ccy = list(swaps)
    X, Y = stack(swaps, ccy, lambda idx: idx <= REFITS[k])
    torch.manual_seed(seed); np.random.seed(seed)
    m = CVAE(n_currencies=len(ccy))
    hist = train(m, X, Y)
    torch.save({"state": m.state_dict(), "hist": hist}, path)
    return f"{path.name}: n={len(X)} final loss {hist[-1]:.3e}"


def pca_fv(df, cutoff, k):
    tr = df[df.index <= cutoff]
    mu = tr.mean()
    Vt = np.linalg.svd((tr - mu).values, full_matrices=False)[2]
    V = Vt[:k].T
    return pd.DataFrame((df - mu).values @ V @ V.T + mu.values, index=df.index, columns=TARGET_TENORS)


def segments(index):
    """Map each date after REFITS[0] to the latest refit k with REFITS[k] < date."""
    seg = pd.Series(-1, index=index)
    for k, r in enumerate(REFITS):
        seg[index > r] = k
    return seg


def evaluate():
    swaps = load_swaps()
    ccy = list(swaps)
    idx = swaps[ccy[0]].index
    seg = segments(idx)
    oos = seg >= 0

    def wf_signal(fv_fn, df):
        """Causal stitched residual: on dates in segment k the residual comes from
        model k (trained on data <= REFITS[k]), so every residual is out-of-sample.
        The rolling z-score is then taken over this stitched OOS residual series
        (min_periods 30 → trading starts ~6 weeks after the first refit)."""
        slope = (df[10] - df[2]) * BP
        r_out = pd.Series(np.nan, index=df.index)
        for k in range(len(REFITS)):
            fv = fv_fn(k)
            resid = slope - (fv[10] - fv[2]) * BP
            r_out[seg == k] = resid[seg == k]
        return zscore(r_out, 60)

    def run_from_z(z, dslope, lag=1, tc=0.5):
        z = z.where(oos)  # identical start state for every signal: flat until the first refit
        pos = positions(z.fillna(0.0))
        pl = pos.shift(lag).fillna(0.0)
        return pl * dslope - pl.diff().abs().fillna(0.0) * tc, pl

    models = {}
    for k in range(len(REFITS)):
        for s in SEEDS:
            m = CVAE(n_currencies=len(ccy))
            m.load_state_dict(torch.load(WF / f"cvae_r{k}_s{s}.pt")["state"]); m.eval()
            models[(k, s)] = m

    zsig = {}  # (signal_name, ccy) -> stitched z-score
    for i, c in enumerate(ccy):
        df = swaps[c]
        slope = (df[10] - df[2]) * BP
        for s in SEEDS:
            zsig[(f"CVAE wf s{s}", c)] = wf_signal(lambda k: reconstruct(models[(k, s)], df, i, len(ccy))[0], df)
        # seed-ensemble: average fair value across the 3 seeds
        zsig[("CVAE wf ens", c)] = wf_signal(
            lambda k: sum(reconstruct(models[(k, s)], df, i, len(ccy))[0] for s in SEEDS) / len(SEEDS), df)
        zsig[("PCA-2 wf", c)] = wf_signal(lambda k: pca_fv(df, REFITS[k], 2), df)
        zsig[("PCA-3 wf", c)] = wf_signal(lambda k: pca_fv(df, REFITS[k], 3), df)
        zsig[("NS λ=0.5", c)] = zscore(slope - (lambda f: (f[10] - f[2]) * BP)(ns_fit(df, 0.5)), 60)
        zsig[("Naive slope", c)] = zscore(slope, 60)

    names = sorted({n for n, _ in zsig})
    results, daily = [], {}
    for n in names:
        for lag in (1, 2):
            for tc in (0.0, 0.5, 1.0):
                pnls, turns = [], []
                for c in ccy:
                    df = swaps[c]
                    dsl = ((df[10] - df[2]) * BP).diff()
                    pnl, pl = run_from_z(zsig[(n, c)], dsl, lag, tc)
                    pnls.append(pnl[oos].rename(c))
                    turns.append(pl.diff().abs()[oos].sum())
                P = pd.concat(pnls, axis=1).fillna(0)
                port = P.mean(1)
                daily[(n, lag, tc)] = P
                h1 = port[port.index <= TRAIN_CUTOFF]
                h2 = port[port.index > TRAIN_CUTOFF]
                results.append(dict(signal=n, lag=lag, tc=tc, sharpe=sharpe(port), total=port.sum(),
                                    sharpe_H1=sharpe(h1), sharpe_H2_nb4window=sharpe(h2),
                                    turnover_per_ccy=np.mean(turns), N=len(port)))
    Rz = pd.DataFrame(results)
    Rz.to_csv(OUT / f"exp5_walkforward_{DATA_TAG}.csv", index=False)
    pickle.dump(daily, open(OUT / f"exp5_daily_{DATA_TAG}.pkl", "wb"))
    pd.set_option("display.width", 200)
    print(f"Walk-forward OOS window: {idx[oos][0].date()} → {idx[oos][-1].date()}  ({oos.sum()} days)")
    for lag in (1, 2):
        print(f"\nlag={lag}")
        print(Rz[Rz.lag == lag].pivot_table(index="signal", columns="tc", values="sharpe").round(2))
    print("\nlag=1, tc=0.5: split by sub-period (H1 = Feb-24→Mar-25, H2 = NB4 test window Apr-25→Jan-26)")
    print(Rz[(Rz.lag == 1) & (Rz.tc == 0.5)][["signal", "sharpe", "sharpe_H1", "sharpe_H2_nb4window", "total", "turnover_per_ccy"]]
          .round(2).to_string(index=False))


if __name__ == "__main__":
    if sys.argv[1] == "train":
        jobs = [(k, s) for k in range(len(REFITS)) for s in SEEDS]
        with Pool(8) as p:
            for msg in p.imap_unordered(job, jobs):
                print(msg, flush=True)
    else:
        evaluate()
