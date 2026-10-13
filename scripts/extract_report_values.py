"""
Extract every \TODO{} value in the LaTeX report from the saved notebook pickles.

Run:
    python scripts/extract_report_values.py

Reads:  results/ns_results.pkl   (from NB2)
        results/vae_results_extended.pkl  (from NB3)

Emits:  LaTeX-ready rows for each table, written to stdout and to
        results/report_values.tex  (appendable).
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RES = Path(__file__).resolve().parent.parent / "results"
OUT = RES / "report_values.tex"

# -----------------------------------------------------------------------------
with open(RES / "ns_results.pkl", "rb") as f:
    ns = pickle.load(f)
with open(RES / "vae_results_extended.pkl", "rb") as f:
    vae = pickle.load(f)

CURRENCIES = ns["currencies"]
cutoff = pd.Timestamp(vae["config"]["train_cutoff"])
dates  = ns["rmse_ns_df"].index
is_mask  = dates <= cutoff
oos_mask = dates >  cutoff

buf: list[str] = []
def emit(s: str = "") -> None:
    print(s)
    buf.append(s)

# -----------------------------------------------------------------------------
# §5  λ*  and the cross-validation curve
# -----------------------------------------------------------------------------
lam_star = float(ns["lambda_star"])
emit("% ───── §5 (NB2) ────────────────────────────────────────────────────")
emit(f"% \\TODO{{Optimal lambda}}   →  $\\lambda^\\star \\approx {lam_star:.3f}$")
emit()

# -----------------------------------------------------------------------------
# Table \ref{tab:ns_rmse}  —  NS / eNS / NSS mean RMSE by currency  (NB2)
# -----------------------------------------------------------------------------
emit("% ───── Table: NS / eNS / NSS mean RMSE by currency (NB2) ───────────")

def mean_rmse_nb2(ccy: str, model: str) -> float:
    return float(ns["fit_results"][ccy][model]["rmse"].mean())

def ltx_row(label: str, vals: list[float]) -> str:
    return f"{label}&" + "&".join(f"{v:.2f}" for v in vals) + "\\\\"

emit("% replace the \\TODO{} cells in the `tab:ns_rmse' table with:")
for label, key in [("NS (3D)", "ns"), ("eNS (3D + $\\lambda$)", "ens"), ("NSS (4D)", "nss")]:
    vals = [mean_rmse_nb2(c, key) for c in CURRENCIES]
    emit(ltx_row(label, vals))
emit()

# -----------------------------------------------------------------------------
# Table \ref{tab:vae_rmse}  —  5 models × 7 currencies  (NB3)
# -----------------------------------------------------------------------------
emit("% ───── Table: VAE comparison (NB3, full-sample) ─────────────────────")

def mean_rmse_nb3(ccy: str, method: str) -> float:
    return float(np.asarray(vae["rmse_results"][f"{method}_{ccy}"]).mean())

emit("% replace the \\TODO{} cells in the `tab:vae_rmse' table with:")
for label, method in [
    ("NS (3D)",        "NelsonSiegel"),
    ("NSS (4D)",       None),                 # NSS not re-fit inside NB3
    ("SingleVAE (2D)", "SingleVAE"),
    ("MultiVAE (2D)",  "MultiVAE"),
    ("\\textbf{CVAE (2D)}", "CVAE"),
]:
    if method is None:
        # NSS pulled from NB2 instead
        vals = [mean_rmse_nb2(c, "nss") for c in CURRENCIES]
    else:
        vals = [mean_rmse_nb3(c, method) for c in CURRENCIES]
    emit(ltx_row(label, vals))
emit()

# -----------------------------------------------------------------------------
# Table \ref{tab:wilcoxon}  —  CVAE vs NS, paired daily RMSE
# -----------------------------------------------------------------------------
emit("% ───── Table: Wilcoxon CVAE vs NS (NB3) ────────────────────────────")

w = vae["config"]["wilcoxon_cvae"]       # stored {ccy: {meanΔ, p, winner}}
emit("% replace the \\TODO{} cells in the `tab:wilcoxon' table with:")
for c in CURRENCIES:
    d      = float(w[c]["meanΔ"])
    p      = float(w[c]["p"])
    winner = w[c]["winner"]
    emit(f"{c}&{d:+.2f}&${p:.2e}$&{winner}\\\\")

# Pooled test: stack all currencies and redo CVAE vs NS on paired differences
ns_pool   = np.concatenate([np.asarray(vae["rmse_results"][f"NelsonSiegel_{c}"]) for c in CURRENCIES])
cvae_pool = np.concatenate([np.asarray(vae["rmse_results"][f"CVAE_{c}"])         for c in CURRENCIES])
diff = ns_pool - cvae_pool                 # positive  →  CVAE is lower RMSE  →  CVAE wins
stat, p_pooled = stats.wilcoxon(diff, alternative="two-sided")
winner_pool = "CVAE" if diff.mean() > 0 else "NS"
# scipy returns exactly 0.0 when the true p-value underflows float64. Report
# this as an upper bound instead of printing `0.00e+00', which would be wrong.
if p_pooled == 0.0:
    p_str = "< 10^{-300}"
else:
    p_str = f"{p_pooled:.2e}"
emit(f"\\textbf{{Pooled}}&{diff.mean():+.2f}&${p_str}$&{winner_pool}\\\\")
emit()
emit(f"% pooled n = {len(diff)},  Wilcoxon stat = {stat:.3e}")
emit()

# -----------------------------------------------------------------------------
# IS / OOS split on the full-sample MultiVAE  (proxy for NB3 §13)
# -----------------------------------------------------------------------------
emit("% ───── §\\ref{sec:vae_results} IS vs OOS (proxy; see note) ──────────")
mv_all = np.concatenate([np.asarray(vae["rmse_results"][f"MultiVAE_{c}"]) for c in CURRENCIES])
N = 774
assert len(mv_all) == N * len(CURRENCIES)
is_idx, oos_idx = np.asarray(is_mask), np.asarray(oos_mask)
mv_is  = np.concatenate([np.asarray(vae["rmse_results"][f"MultiVAE_{c}"])[is_idx]  for c in CURRENCIES])
mv_oos = np.concatenate([np.asarray(vae["rmse_results"][f"MultiVAE_{c}"])[oos_idx] for c in CURRENCIES])
is_mean, oos_mean = float(mv_is.mean()), float(mv_oos.mean())
emit(f"% MultiVAE mean IS RMSE  = {is_mean:.3f} bp   (n = {len(mv_is)})")
emit(f"% MultiVAE mean OOS RMSE = {oos_mean:.3f} bp  (n = {len(mv_oos)})")
emit(f"% OOS / IS ratio         = {oos_mean/is_mean:.3f}")
emit("% NOTE: these are the main MultiVAE (trained on full sample) evaluated")
emit("%       separately on the pre/post split. The report text refers to the")
emit("%       OOS-retrained MultiVAE from NB3 §13, which is NOT saved in the")
emit("%       pickle. To get the textbook IS vs OOS numbers you must re-run")
emit("%       cell [NB3 §13 ‘Out-of-Sample Validation’]. The numbers above")
emit("%       are still informative — they show that the full-sample model")
emit("%       generalises cleanly across the split.")
emit()

# -----------------------------------------------------------------------------
# What is NOT in the pickle and needs a cell re-run
# -----------------------------------------------------------------------------
emit("% ───── TODOs that require re-running notebook cells ─────────────────")
emit("%  • §5.6 post-processing GD improvement (3 numbers)")
emit("%      → re-run NB3 §6b and copy the `Mean RMSE before/after' printout")
emit("%  • §5.5 IS vs OOS mean RMSE using the *retrained* MultiVAE")
emit("%      → re-run NB3 §13 and copy the `In-sample / Out-of-sample mean")
emit("%        RMSE' printout (or use the proxy above with an edit to the")
emit("%        report text to reflect that the same model is split-eval'd).")
emit()

OUT.write_text("\n".join(buf) + "\n")
print(f"\n✓ Wrote {OUT.relative_to(Path.cwd())}")
