"""Exp 7 — cheap fair-value strategies on ORIGINAL vs RE-ALIGNED data
(no neural net needed): PCA-k fitted on train window, NS, naive."""
import numpy as np, pandas as pd
from common import *

def load(path):
    d = pd.read_csv(path, parse_dates=["Date"]); d = d[d.currency != "CHF"]
    out = {}
    for c in sorted(d.currency.unique()):
        s = d[d.currency == c].set_index("Date").sort_index()[TENOR_COLS]; s.columns = TARGET_TENORS; out[c] = s
    return out

def pca_fv(df, cutoff, k):
    tr = df[df.index <= cutoff]; mu = tr.mean()
    V = np.linalg.svd((tr - mu).values, full_matrices=False)[2][:k].T
    return pd.DataFrame((df - mu).values @ V @ V.T + mu.values, index=df.index, columns=TARGET_TENORS)

is_oos = lambda i: i > TRAIN_CUTOFF
for label, path in [("ORIGINAL", ROOT/"data"/"df_multi_unaligned.csv"), ("RE-ALIGNED", ROOT/"data"/"df_multi_repaired_2023_2026.csv")]:
    sw = load(path)
    rows = []
    for name in ["PCA-2", "PCA-3", "NS λ=0.5", "Naive"]:
        for lag in (1, 2):
            P = []
            for c, df in sw.items():
                slope = (df[10]-df[2])*BP
                if name.startswith("PCA"): fv = pca_fv(df, TRAIN_CUTOFF, int(name[-1])); sig = slope-(fv[10]-fv[2])*BP
                elif name.startswith("NS"): fv = ns_fit(df, 0.5); sig = slope-(fv[10]-fv[2])*BP
                else: sig = slope
                bt = backtest_raw(sig, slope.diff(), lag=lag)
                P.append(bt.loc[is_oos(bt.index), "pnl"].rename(c))
            P = pd.concat(P, axis=1).fillna(0)
            rows.append(dict(signal=name, lag=lag, port=sharpe(P.mean(1)), **{c: sharpe(P[c]) for c in sw}))
    print(f"\n== {label}: OOS (Apr-25→Jan-26) Sharpe, NB4 fixed HP, tc 0.5bp")
    print(pd.DataFrame(rows).round(2).to_string(index=False))
