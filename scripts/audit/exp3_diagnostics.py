"""Exp 3 — model-free diagnostics of where the NB4 P&L comes from.

Uses the SAVED (full-sample) CVAE from NB3, i.e. the exact model behind the
reported Sharpe 3.99, plus classical benchmarks fitted on TRAIN data only.
"""
import pickle

import numpy as np
import pandas as pd
import torch
from scipy import stats as sst

from common import *

swaps = load_swaps()
CCY = list(swaps)
vae = pickle.load(open(ORIG_RESULTS / "vae_results_extended.pkl", "rb"))
cvae = CVAE(n_currencies=len(CCY)); cvae.load_state_dict(vae["cvae_state"]); cvae.eval()
is_oos = lambda idx: idx > TRAIN_CUTOFF

# 1. Autocorrelation of daily 2s10s changes and stale quotes ----------------
print("1) Daily 2s10s slope changes: lag-1 autocorrelation and stale-quote share")
print(f"{'ccy':<5}{'AC1 train':>10}{'AC1 OOS':>10}{'%zero d2Y':>11}{'%zero d10Y':>11}{'%zero dSlope':>13}")
for c in CCY:
    df = swaps[c]
    ds = ((df[10] - df[2]) * BP).diff()
    tr, te = ds[~is_oos(ds.index)], ds[is_oos(ds.index)]
    z2 = (df[2].diff() == 0)[is_oos(df.index)].mean()
    z10 = (df[10].diff() == 0)[is_oos(df.index)].mean()
    zs = (te == 0).mean()
    print(f"{c:<5}{tr.autocorr(1):>10.3f}{te.autocorr(1):>10.3f}{z2:>10.1%}{z10:>11.1%}{zs:>12.1%}")


# 2. Fair-value residuals: CVAE (leaky), NS (λ=0.5 and 1/1.5), PCA-k (train-fit)
def pca_fair_value(df, k):
    tr = df[df.index <= TRAIN_CUTOFF]
    mu = tr.mean()
    U, S, Vt = np.linalg.svd((tr - mu).values, full_matrices=False)
    V = Vt[:k].T
    rec = (df - mu).values @ V @ V.T + mu.values
    return pd.DataFrame(rec, index=df.index, columns=TARGET_TENORS)


signals = {}
for i, c in enumerate(CCY):
    df = swaps[c]
    slope = (df[10] - df[2]) * BP
    rec, z = reconstruct(cvae, df, i, len(CCY))
    fv = {"CVAE(leaky)": rec, "NS λ=0.667 (NB4)": ns_fit(df, 1 / 1.5), "NS λ=0.5 (NB2 λ*)": ns_fit(df, 0.5),
          "PCA-2 (train)": pca_fair_value(df, 2), "PCA-3 (train)": pca_fair_value(df, 3)}
    signals[c] = {k: slope - (v[10] - v[2]) * BP for k, v in fv.items()}
    signals[c]["Naive slope"] = slope
    signals[c]["_dslope"] = slope.diff()


def portfolio(sig_name, **kw):
    pnls, gross = [], []
    for c in CCY:
        bt = backtest_raw(signals[c][sig_name], signals[c]["_dslope"], **kw)
        bt = bt[is_oos(bt.index)]
        pnls.append(bt["pnl"].rename(c))
        gross.append(bt["gross"].rename(c))
    return pd.concat(pnls, axis=1).fillna(0).mean(1), pd.concat(gross, axis=1).fillna(0).mean(1)


names = ["CVAE(leaky)", "NS λ=0.667 (NB4)", "NS λ=0.5 (NB2 λ*)", "PCA-2 (train)", "PCA-3 (train)", "Naive slope"]
print("\n2) Equal-weight OOS portfolio Sharpe (raw bp), NB4 fixed HP (entry 1, exit .3, lb 60)")
print(f"{'signal':<20}{'lag1 tc.5':>10}{'lag1 tc0':>10}{'lag1 tc1':>10}{'lag2 tc.5':>10}{'lag3 tc.5':>10}{'Total(lag1)':>12}")
for n in names:
    row = []
    for lag, tc in [(1, .5), (1, 0), (1, 1.0), (2, .5), (3, .5)]:
        p, _ = portfolio(n, lag=lag, tc_bp=tc)
        row.append(sharpe(p))
    tot = portfolio(n)[0].sum()
    print(f"{n:<20}" + "".join(f"{v:>10.2f}" for v in row) + f"{tot:>12.1f}")

# 3. Predictive regression: does residual at t predict Δslope(t→t+1)? (pooled OOS, per-ccy z-scored)
print("\n3) Predictive regression  Δslope_{t+1} = a + b·zscore(resid)_t  (pooled across ccys, OOS)")
for n in names:
    xs, ys = [], []
    for c in CCY:
        z = zscore(signals[c][n], 60)
        y = signals[c]["_dslope"].shift(-1)
        d = pd.concat([z, y], axis=1).dropna()
        d = d[is_oos(d.index)]
        xs.append(d.iloc[:, 0].values); ys.append(d.iloc[:, 1].values)
    x, y = np.concatenate(xs), np.concatenate(ys)
    r = sst.linregress(x, y)
    # same but skipping one day (t+2)
    xs2, ys2 = [], []
    for c in CCY:
        z = zscore(signals[c][n], 60)
        y2 = signals[c]["_dslope"].shift(-2)
        d = pd.concat([z, y2], axis=1).dropna(); d = d[is_oos(d.index)]
        xs2.append(d.iloc[:, 0].values); ys2.append(d.iloc[:, 1].values)
    r2 = sst.linregress(np.concatenate(xs2), np.concatenate(ys2))
    print(f"  {n:<20} b(t+1)={r.slope:+.3f} bp  t={r.slope/r.stderr:+.2f}   |  b(t+2)={r2.slope:+.3f} bp  t={r2.slope/r2.stderr:+.2f}")

# 4. Per-currency breakdown for CVAE(leaky) and Naive
print("\n4) Per-currency OOS Sharpe, lag1 vs lag2 (tc 0.5bp)")
print(f"{'ccy':<5}" + "".join(f"{n[:11]+' L'+str(l):>16}" for n in ["CVAE(leaky)", "PCA-3 (train)", "Naive slope"] for l in (1, 2)))
for c in CCY:
    row = []
    for n in ["CVAE(leaky)", "PCA-3 (train)", "Naive slope"]:
        for l in (1, 2):
            bt = backtest_raw(signals[c][n], signals[c]["_dslope"], lag=l)
            row.append(sharpe(bt.loc[is_oos(bt.index), "pnl"]))
    print(f"{c:<5}" + "".join(f"{v:>16.2f}" for v in row))

pickle.dump(signals, open(OUT / "signals_leaky.pkl", "wb"))
