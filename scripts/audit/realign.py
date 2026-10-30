"""Re-align the Bloomberg tenor columns.

Diagnosis: within each sheet every tenor column is offset from the shared
Date column by a row shift that changes stepwise through time (values were
pasted top-aligned from series with different missing days).  We recover, for
each tenor column k, a piecewise-constant shift path s_k(i) such that
column k at row i + s_k(i) belongs to the same date as the 2Y value at row i.

Method: Viterbi on standardised daily changes, score = Σ x_i · y_{i+s_i}
minus a penalty PEN per change of shift (changes of ±1 only).
Reference column = 2Y (all columns end up mutually consistent; any residual
offset of the 2Y column vs. the true calendar applies to the whole curve and
does not create cross-tenor look-ahead).

Output: out/df_multi_realigned.csv  (same layout as data/df_multi.csv,
same 774 dates, CHF excluded).
"""
import numpy as np
import pandas as pd

from common import ROOT, OUT, TARGET_TENORS

SHEETS = {"USD": "USSO", "EUR": "EUSWE", "GBP": "BPSWS", "JPY": "JYSO",
          "AUD": "ADSO", "CAD": "CDSO", "DKK": "DKSW"}
S = 16        # max |shift| in rows
PEN = 4.0     # penalty per shift change (in units of standardised products)
START = "2021-06-01"  # buffer before the 2023-01-30 window


def load_sheet(sh):
    df = pd.read_excel(ROOT / "data" / "data_bloomberg.xlsx", sheet_name=sh)
    df = df.rename(columns={df.columns[0]: "Date"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
    df.columns = [int(str(c).split()[0].replace(sh, "")) for c in df.columns]
    return df[TARGET_TENORS].apply(pd.to_numeric, errors="coerce") / 100.0


def viterbi_shift(x, y, S=S, pen=PEN):
    """x, y: standardised daily changes on the same row grid (NaN-safe).
    Returns shift path s (len n) maximising Σ x_i y_{i+s_i} - pen·#changes."""
    n = len(x)
    shifts = np.arange(-S, S + 1)
    m = len(shifts)
    sc = np.full((n, m), -0.5)  # out-of-range / NaN rows get a small penalty
    for j, s in enumerate(shifts):
        idx = np.arange(n) + s
        ok = (idx >= 0) & (idx < n)
        v = np.where(ok, x * y[np.clip(idx, 0, n - 1)], np.nan)
        sc[ok & ~np.isnan(v), j] = v[ok & ~np.isnan(v)]
    dp = np.zeros((n, m)); bp = np.zeros((n, m), dtype=int)
    dp[0] = sc[0]
    for i in range(1, n):
        prev = dp[i - 1]
        stay = prev
        down = np.r_[-np.inf, prev[:-1]] - pen   # came from shift j-1
        up = np.r_[prev[1:], -np.inf] - pen      # came from shift j+1
        cand = np.vstack([stay, down, up])
        arg = cand.argmax(0)
        dp[i] = cand.max(0) + sc[i]
        bp[i] = np.arange(m) + np.array([0, -1, 1])[arg]
    path = np.empty(n, dtype=int)
    path[-1] = int(dp[-1].argmax())
    for i in range(n - 1, 0, -1):
        path[i - 1] = bp[i, path[i]]
    return shifts[path]


def realign_sheet(raw):
    raw = raw.loc[START:].dropna()
    d = raw.diff()
    z = (d - d.mean()) / d.std()
    out = pd.DataFrame(index=raw.index, columns=TARGET_TENORS, dtype=float)
    out[2] = raw[2].values
    paths = {}
    x = z[2].values
    for k in TARGET_TENORS[1:]:
        s = viterbi_shift(x, z[k].values)
        idx = np.arange(len(raw)) + s
        ok = (idx >= 0) & (idx < len(raw))
        vals = np.full(len(raw), np.nan)
        vals[ok] = raw[k].values[idx[ok]]
        out[k] = vals
        paths[k] = pd.Series(s, index=raw.index)
    return out, pd.DataFrame(paths)


if __name__ == "__main__":
    ref = pd.read_csv(ROOT / "data" / "df_multi_unaligned.csv", parse_dates=["Date"])
    dates = pd.DatetimeIndex(sorted(ref["Date"].unique()))
    frames, report = [], []
    for ccy, sh in SHEETS.items():
        raw = load_sheet(sh)
        al, paths = realign_sheet(raw)
        paths.to_csv(OUT / f"shift_path_{ccy}.csv")
        # NB1 pipeline: business-day calendar, ffill <= 3, restrict to the common window
        bd = pd.bdate_range(al.index.min(), max(al.index.max(), dates.max()))  # NB1 calendar spans all sheets
        al_bd = al.reindex(bd).ffill(limit=3).loc[dates]
        assert al_bd.notna().all().all(), f"{ccy}: NaN after realignment"
        tmp = al_bd.copy(); tmp.index.name = "Date"; tmp["currency"] = ccy
        frames.append(tmp.reset_index())
        win = paths.loc["2023-01-30":"2026-01-15"]
        report.append({"ccy": ccy, **{f"{k}Y": f"{win[k].min():+d}..{win[k].max():+d} ({(win[k] != 0).mean():.0%} rows≠0)" for k in TARGET_TENORS[1:]}})
    df = pd.concat(frames).sort_values(["Date", "currency"]).reset_index(drop=True)
    df.columns = [str(c) for c in df.columns]
    df.to_csv(OUT / "df_multi_realigned.csv", index=False)
    pd.set_option("display.width", 250)
    print("Estimated row-shift range of each tenor vs 2Y inside the 774-day window:")
    print(pd.DataFrame(report).to_string(index=False))

    # validation: same-day corr of daily changes before vs after
    old = ref[ref.currency != "CHF"]
    print("\nSame-day corr of daily changes (before → after realignment) and AC1 of 2s10s changes")
    for ccy in SHEETS:
        a = old[old.currency == ccy].set_index("Date")[[str(t) for t in TARGET_TENORS]].diff()
        b = df[df.currency == ccy].set_index("Date")[[str(t) for t in TARGET_TENORS]].diff()
        s_a = (a["10"] - a["2"]); s_b = (b["10"] - b["2"])
        print(f"  {ccy}: 2v3 {a['2'].corr(a['3']):+.2f}→{b['2'].corr(b['3']):+.2f}  "
              f"2v10 {a['2'].corr(a['10']):+.2f}→{b['2'].corr(b['10']):+.2f}  "
              f"10v30 {a['10'].corr(a['30']):+.2f}→{b['10'].corr(b['30']):+.2f}  "
              f"AC1(Δ2s10s) {s_a.autocorr():+.2f}→{s_b.autocorr():+.2f}")
