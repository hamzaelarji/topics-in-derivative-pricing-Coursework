"""Exp 1 — Do the saved NB3 weights reproduce the numbers printed in NB3/NB4?"""
import pickle
import time

import numpy as np
import pandas as pd
import torch

from common import *

swaps = load_swaps()
CCY = list(swaps)
vae = pickle.load(open(ORIG_RESULTS / "vae_results_extended.pkl", "rb"))
assert vae["config"]["currencies"] == CCY

cvae = CVAE(n_currencies=len(CCY)); cvae.load_state_dict(vae["cvae_state"]); cvae.eval()
mvae = MultiVAE(); mvae.load_state_dict(vae["multi_vae_state"]); mvae.eval()

print("Reconstruction RMSE from saved weights (mean bp) vs NB3 printed")
nb3_cvae = dict(AUD=4.00, CAD=4.53, DKK=4.88, EUR=2.75, GBP=2.49, JPY=3.00, USD=4.64)
nb3_mvae = dict(AUD=5.77, CAD=6.10, DKK=6.28, EUR=3.38, GBP=6.78, JPY=4.63, USD=5.52)
for i, c in enumerate(CCY):
    r_c, _ = reconstruct(cvae, swaps[c], i, len(CCY))
    r_m, _ = reconstruct(mvae, swaps[c], i, len(CCY))
    print(f"  {c}: CVAE {rmse_bp(r_c, swaps[c]).mean():5.2f} (nb {nb3_cvae[c]:.2f})   "
          f"MultiVAE {rmse_bp(r_m, swaps[c]).mean():5.2f} (nb {nb3_mvae[c]:.2f})")

# NB4 fixed-HP portfolio, raw bp
oos = lambda s: s[s.index > TRAIN_CUTOFF]
res = {k: [] for k in ["CVAE_resid", "NS_resid", "Naive", "CVAE_z2"]}
for i, c in enumerate(CCY):
    df = swaps[c]
    rec, z = reconstruct(cvae, df, i, len(CCY))
    ns = ns_fit(df, 1 / 1.5)  # NB4 NelsonSiegel(lam=1.5) uses exp(-tau/1.5)
    slope = (df[10] - df[2]) * BP
    sigs = {"CVAE_resid": slope - (rec[10] - rec[2]) * BP,
            "NS_resid": slope - (ns[10] - ns[2]) * BP,
            "Naive": slope, "CVAE_z2": z["z2"]}
    for k, s in sigs.items():
        res[k].append(oos(backtest_raw(s, slope.diff())["pnl"]).rename(c))
print("\nNB4 fixed-HP equal-weight portfolio, raw bp (NB4 printed: CVAE 3.99, NS -1.38, Naive 2.18, z2 1.68)")
for k, v in res.items():
    port = pd.concat(v, axis=1).fillna(0).mean(1)
    print(f"  {k:<11} Sharpe {sharpe(port):5.2f}  Total {port.sum():7.2f} bp")

# timing of one training run
t = time.time()
torch.manual_seed(0)
m = CVAE(n_currencies=len(CCY))
X, Y = stack(swaps, CCY)
train(m, X, Y, n_epochs=200)
print(f"\n200 epochs took {time.time()-t:.1f}s  -> 5000 epochs ≈ {(time.time()-t)*25/60:.1f} min")
