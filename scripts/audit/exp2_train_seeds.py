"""Exp 2 — retrain CVAE / MultiVAE with several seeds, on (a) the full sample
(as NB3 does) and (b) the train window only (<= 2025-03-31, leak-free).
Saves state dicts to out/models/ for the downstream analysis.
"""
import sys
import time
from multiprocessing import Pool

import torch

from common import *

MODELS = OUT / ("models" if DATA_TAG == "orig" else f"models_{DATA_TAG}")
MODELS.mkdir(exist_ok=True)


def job(args):
    kind, scope, seed = args
    path = MODELS / f"{kind}_{scope}_s{seed}.pt"
    if path.exists():
        return f"skip {path.name}"
    torch.set_num_threads(1)
    swaps = load_swaps()
    ccy = list(swaps)
    mask = None if scope == "full" else (lambda idx: idx <= TRAIN_CUTOFF)
    X, Y = stack(swaps, ccy, mask)
    torch.manual_seed(seed)
    np.random.seed(seed)
    m = CVAE(n_currencies=len(ccy)) if kind == "cvae" else MultiVAE()
    t = time.time()
    hist = train(m, X, Y)
    torch.save({"state": m.state_dict(), "hist": hist}, path)
    return f"{path.name}: final loss {hist[-1]:.3e}  ({(time.time()-t)/60:.1f} min)"


if __name__ == "__main__":
    seeds = [int(s) for s in sys.argv[1:]] or list(range(8))
    jobs = [(k, sc, s) for s in seeds for k in ("cvae", "mvae") for sc in ("train", "full")]
    with Pool(8) as p:
        for msg in p.imap_unordered(job, jobs):
            print(msg, flush=True)
