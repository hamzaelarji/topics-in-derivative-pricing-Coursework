"""Exp 4 — analyse the 8-seed retrains: reproducibility of the NB3 replication
claims and of the NB4 trading result, with and without look-ahead."""
import pickle

import numpy as np
import pandas as pd
import torch

from common import *

swaps = load_swaps()
CCY = list(swaps)
MODELS = OUT / ("models" if DATA_TAG == "orig" else f"models_{DATA_TAG}")
seeds = sorted({int(p.stem.split("_s")[1]) for p in MODELS.glob("cvae_full_s*.pt")})
is_oos = lambda idx: idx > TRAIN_CUTOFF


def load(kind, scope, seed):
    m = CVAE(n_currencies=len(CCY)) if kind == "cvae" else MultiVAE()
    ck = torch.load(MODELS / f"{kind}_{scope}_s{seed}.pt")
    m.load_state_dict(ck["state"]); m.eval()
    return m, ck["hist"]


ns_rmse = {lam: {c: rmse_bp(ns_fit(swaps[c], lam), swaps[c]) for c in CCY} for lam in (0.5, 1 / 1.5)}

rows, trade_rows, ports = [], [], {}
for s in seeds:
    for kind in ("cvae", "mvae"):
        for scope in ("full", "train"):
            m, hist = load(kind, scope, s)
            per_ccy, pnl_l1, pnl_l2 = {}, [], []
            for i, c in enumerate(CCY):
                df = swaps[c]
                rec, z = reconstruct(m, df, i, len(CCY))
                e = rmse_bp(rec, df)
                per_ccy[c] = e
                rows.append(dict(seed=s, kind=kind, scope=scope, ccy=c, rmse_all=e.mean(),
                                 rmse_is=e[~is_oos(df.index)].mean(), rmse_oos=e[is_oos(df.index)].mean(),
                                 final_loss=hist[-1]))
                slope = (df[10] - df[2]) * BP
                sig = slope - (rec[10] - rec[2]) * BP
                for lag, store in ((1, pnl_l1), (2, pnl_l2)):
                    bt = backtest_raw(sig, slope.diff(), lag=lag)
                    store.append(bt.loc[is_oos(bt.index), "pnl"].rename(c))
            p1 = pd.concat(pnl_l1, axis=1).fillna(0)
            p2 = pd.concat(pnl_l2, axis=1).fillna(0)
            ports[(s, kind, scope)] = p1
            trade_rows.append(dict(seed=s, kind=kind, scope=scope,
                                   sharpe_lag1=sharpe(p1.mean(1)), total_lag1=p1.mean(1).sum(),
                                   sharpe_lag2=sharpe(p2.mean(1)),
                                   **{f"S_{c}": sharpe(p1[c]) for c in CCY}))

R = pd.DataFrame(rows)
T = pd.DataFrame(trade_rows)
R.to_csv(OUT / f"exp4_rmse_{DATA_TAG}.csv", index=False)
T.to_csv(OUT / f"exp4_trading_{DATA_TAG}.csv", index=False)
pickle.dump(ports, open(OUT / f"exp4_ports_{DATA_TAG}.pkl", "wb"))

pd.set_option("display.width", 200)
print(f"Seeds analysed: {seeds}\n")
print("A) Full-sample models (NB3 setting): mean RMSE (bp) per currency, mean ± sd across seeds")
for kind in ("cvae", "mvae"):
    sub = R[(R.kind == kind) & (R.scope == "full")]
    g = sub.groupby("ccy")["rmse_all"]
    print(f"  {kind.upper():5} " + "  ".join(f"{c} {g.mean()[c]:.2f}±{g.std()[c]:.2f}" for c in CCY))
print("  NS λ=0.667 " + "  ".join(f"{c} {ns_rmse[1/1.5][c].mean():.2f}" for c in CCY))
print("  NS λ=0.5   " + "  ".join(f"{c} {ns_rmse[0.5][c].mean():.2f}" for c in CCY))

print("\nB) # currencies where model beats NS (mean RMSE), per seed")
for kind in ("cvae", "mvae"):
    for lam in (1 / 1.5, 0.5):
        wins = []
        for s in seeds:
            sub = R[(R.kind == kind) & (R.scope == "full") & (R.seed == s)].set_index("ccy")["rmse_all"]
            wins.append(sum(sub[c] < ns_rmse[lam][c].mean() for c in CCY))
        print(f"  {kind.upper():5} vs NS λ={lam:.3f}: {wins}")

print("\nC) Train-only models: pooled IS vs OOS reconstruction RMSE (bp)")
for kind in ("cvae", "mvae"):
    sub = R[(R.kind == kind) & (R.scope == "train")].groupby("seed")[["rmse_is", "rmse_oos"]].mean()
    sub["ratio"] = sub.rmse_oos / sub.rmse_is
    print(f"  {kind.upper()}\n" + sub.round(2).to_string())

print("\nD) OOS trading, CVAE/MVAE slope-residual signal, equal-weight portfolio, NB4 fixed HP, tc 0.5bp")
print(T[["seed", "kind", "scope", "sharpe_lag1", "sharpe_lag2", "total_lag1"]]
      .sort_values(["kind", "scope", "seed"]).round(2).to_string(index=False))
print("\n  Summary (mean / min / max Sharpe lag1 across seeds):")
print(T.groupby(["kind", "scope"])["sharpe_lag1"].agg(["mean", "min", "max"]).round(2))
print("\n  Per-currency OOS Sharpe, CVAE train-only (leak-free), mean across seeds:")
print(T[(T.kind == "cvae") & (T.scope == "train")][[f"S_{c}" for c in CCY]].mean().round(2).to_string())
