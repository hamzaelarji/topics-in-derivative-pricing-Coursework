"""Exp 8 — non-circular test on the ORIGINAL (untouched) data.
Split each currency's days by whether the 10Y column is aligned with the 2Y
column (estimated shift == 0 on both day t and t+1) or not, and compare the
P&L of the Naive / PCA-2 / saved-CVAE slope strategies on each subset.
If the edge is real it should be present on aligned days too."""
import pickle
import numpy as np, pandas as pd
from common import *

swaps = load_swaps("orig"); CCY = list(swaps)
vae = pickle.load(open(ORIG_RESULTS/"vae_results_extended.pkl","rb"))
cvae = CVAE(n_currencies=len(CCY)); cvae.load_state_dict(vae["cvae_state"]); cvae.eval()

def pca_fv(df, k=2):
    tr = df[df.index <= TRAIN_CUTOFF]; mu = tr.mean()
    V = np.linalg.svd((tr-mu).values, full_matrices=False)[2][:k].T
    return pd.DataFrame((df-mu).values @ V @ V.T + mu.values, index=df.index, columns=TARGET_TENORS)

rows = []
pnl_by = {}
for i, c in enumerate(CCY):
    df = swaps[c]; slope = (df[10]-df[2])*BP
    P = pd.read_csv(OUT/f"shift_path_{c}.csv", index_col=0, parse_dates=True).reindex(df.index).ffill(); path = P.abs().max(axis=1)
    # a day's P&L (t-1 -> t) is clean only if 10Y is aligned (ALL tenors) at both t-1 and t
    aligned = (path == 0) & (path.shift(1) == 0)
    sigs = {"Naive": slope, "PCA-2": slope-(lambda f:(f[10]-f[2])*BP)(pca_fv(df)),
            "CVAE(saved)": slope-(lambda f:(f[10]-f[2])*BP)(reconstruct(cvae, df, i, len(CCY))[0])}
    for n, s in sigs.items():
        bt = backtest_raw(s, slope.diff())
        for lab, m in [("aligned", aligned), ("misaligned", ~aligned)]:
            pnl_by.setdefault((n, lab), []).append(bt["pnl"][m])
    rows.append(dict(ccy=c, aligned_share=aligned.mean()))
print(pd.DataFrame(rows).round(2).to_string(index=False))
print("\nPooled across currencies, full sample (2023-01 → 2026-01), lag 1, tc 0.5bp, per-day P&L in bp")
print(f"{'signal':<13}{'subset':<12}{'days':>6}{'mean bp/day':>13}{'t-stat':>8}{'ann. Sharpe':>13}")
for (n, lab), lst in pnl_by.items():
    x = pd.concat(lst); x = x[x.index > x.index.min()]
    t = x.mean()/x.std()*np.sqrt(len(x))
    print(f"{n:<13}{lab:<12}{len(x):>6}{x.mean():>13.3f}{t:>8.2f}{x.mean()/x.std()*np.sqrt(252):>13.2f}")
