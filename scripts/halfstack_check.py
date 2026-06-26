"""Intra-session consistency via half-stacks: split each session's looks into two
independent halves, robust-stack each, and register the halves. Clean co-registration
=> looks are internally consistent (any softness is stacking/display). Offset or low
NCC => intra-session scatter blurring the deep stack. Also split by pointing (N vs S)
to see whether the two pointings disagree.

Usage: .conda/bin/python scripts/halfstack_check.py
"""
import os, sys, glob
import numpy as np
from scipy.ndimage import binary_erosion

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stack_sessions import robust_stack
from venera.registration import register_maps, offset_to_lonlat_deg
from venera.coherence import ncc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "results", "look_cache")
HH, WW = 2000, 4000


def nccq(A, Am, B, Bm, shift=(0, 0)):
    dr, dc = int(round(shift[0])), int(round(shift[1]))
    Bs = np.roll(np.roll(B, dr, 0), dc, 1); Ms = np.roll(np.roll(Bm, dr, 0), dc, 1)
    com = binary_erosion(Am & Ms, iterations=10)
    return ncc(A, Bs, com, com) if com.sum() > 1500 else np.nan


def reg(A, Am, B, Bm):
    dr, dc, sig = register_maps(A, B, valid_a=Am, valid_b=Bm, max_shift=40,
                               smooth_px=7.0, trend_px=55.0)
    dlat, dlon = offset_to_lonlat_deg(dr, dc, (HH, WW))
    return dlon, dlat, sig, (dr, dc)


def main():
    print("  session   split           Δlon     Δlat    sig   NCC@0  NCC@shift", flush=True)
    print("  " + "-" * 64)
    for year in ["1988", "2012", "2015", "2017", "2020"]:
        looks = sorted(glob.glob(os.path.join(CACHE, f"venus_*_{year}*.npz")))
        # even/odd split (independent halves, same geometry mix)
        A, Am, _ = robust_stack(looks[0::2]); B, Bm, _ = robust_stack(looks[1::2])
        dlon, dlat, sig, sh = reg(A, Am, B, Bm)
        print(f"  {year}     even/odd       {dlon:+.3f}  {dlat:+.3f}  {sig:.2f}  "
              f"{nccq(A,Am,B,Bm):.3f}  {nccq(A,Am,B,Bm,sh):.3f}", flush=True)
        # N vs S pointing split (do the two pointings land together?)
        N = [f for f in looks if str(np.load(f, allow_pickle=True)["pointing"]) == "N"]
        S = [f for f in looks if str(np.load(f, allow_pickle=True)["pointing"]) == "S"]
        if len(N) >= 4 and len(S) >= 4:
            GN, MN, _ = robust_stack(N); GS, MS, _ = robust_stack(S)
            dlon, dlat, sig, sh = reg(GN, MN, GS, MS)
            print(f"  {year}     N-vs-S         {dlon:+.3f}  {dlat:+.3f}  {sig:.2f}  "
                  f"{nccq(GN,MN,GS,MS):.3f}  {nccq(GN,MN,GS,MS,sh):.3f}", flush=True)


if __name__ == "__main__":
    main()
