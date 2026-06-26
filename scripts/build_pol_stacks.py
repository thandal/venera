"""Build single-polarization session stacks: session_<year>_<pol>.npz (Gm, mask),
same format/grid as the combined session_<year>.npz, for the polarization-split period
fit. Reuses stack_sessions.robust_stack (σ-clip). Usage: build_pol_stacks.py [ocp|scp|both]
"""
import os, sys, glob
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import stack_sessions as ss

CACHE = os.path.join(ROOT, "results", "look_cache")
STACKS = os.path.join(ROOT, "results", "session_stacks")
YEARS = ["1988", "2001", "2012", "2015", "2017", "2020"]

which = sys.argv[1] if len(sys.argv) > 1 else "both"
pols = ["ocp", "scp"] if which == "both" else [which]
for pol in pols:
    for y in YEARS:
        files = sorted(glob.glob(os.path.join(CACHE, f"venus_{pol}_{y}*.npz")))
        Gm, mask, rej = ss.robust_stack(files)
        out = os.path.join(STACKS, f"session_{y}_{pol}.npz")
        np.savez_compressed(out, Gm=Gm, mask=mask, n=len(files), pol=pol)
        print(f"  {pol} {y}: {len(files):3d} looks  reject={rej:.3f}  -> {os.path.basename(out)}", flush=True)
print("done")
