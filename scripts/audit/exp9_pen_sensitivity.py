"""Exp 9 — sensitivity of the realignment to the Viterbi penalty (PEN), and
count of shift-change events (each one duplicates/skips one raw row)."""
import numpy as np, pandas as pd
import realign as R
from common import *

ref = pd.read_csv(ROOT/"data"/"df_multi.csv", parse_dates=["Date"])
dates = pd.DatetimeIndex(sorted(ref["Date"].unique()))
def pca_fv(df, k):
    tr = df[df.index <= TRAIN_CUTOFF]; mu = tr.mean()
    V = np.linalg.svd((tr-mu).values, full_matrices=False)[2][:k].T
    return pd.DataFrame((df-mu).values @ V @ V.T + mu.values, index=df.index, columns=TARGET_TENORS)

for pen in (2.0, 4.0, 8.0):
    sw, nchg = {}, {}
    for ccy, sh in R.SHEETS.items():
        
        raw = R.load_sheet(sh).loc[R.START:].dropna()
        d = raw.diff(); z = (d-d.mean())/d.std()
        out = pd.DataFrame(index=raw.index, columns=TARGET_TENORS, dtype=float); out[2] = raw[2].values
        chg = 0
        for k in TARGET_TENORS[1:]:
            s = R.viterbi_shift(z[2].values, z[k].values, S=16, pen=pen)
            win = pd.Series(s, index=raw.index).loc["2023-01-30":"2026-01-15"]
            chg += int((win.diff().fillna(0) != 0).sum())
            idx = np.arange(len(raw)) + s; ok = (idx >= 0) & (idx < len(raw))
            v = np.full(len(raw), np.nan); v[ok] = raw[k].values[idx[ok]]; out[k] = v
        bd = pd.bdate_range(out.index.min(), max(out.index.max(), dates.max()))
        sw[ccy] = out.reindex(bd).ffill(limit=3).loc[dates]
        nchg[ccy] = chg
    res = {}
    for name in ["PCA-2", "PCA-3", "Naive"]:
        P = []
        for c, df in sw.items():
            slope = (df[10]-df[2])*BP
            sig = slope if name == "Naive" else slope-(lambda f:(f[10]-f[2])*BP)(pca_fv(df, int(name[-1])))
            bt = backtest_raw(sig, slope.diff())
            P.append(bt.loc[bt.index > TRAIN_CUTOFF, "pnl"].rename(c))
        res[name] = sharpe(pd.concat(P, axis=1).fillna(0).mean(1))
    corr23 = np.mean([sw[c][2].diff().corr(sw[c][3].diff()) for c in sw])
    print(f"PEN={pen}: shift changes in window per ccy (sum over 6 tenors) {nchg} | mean 2v3 corr {corr23:.2f} | OOS Sharpe "
          + ", ".join(f"{k} {v:.2f}" for k, v in res.items()))
