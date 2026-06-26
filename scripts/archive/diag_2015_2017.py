"""Diagnose the poor 2015<->2017 registration: is it driven by mixing pols /
pointings (a stacking choice) or by region, vs a fundamental projection issue?

Stacks 2015 and 2017 from the per-look cache under several filters and reports the
boundary-free interior NCC (global and in a well-covered northern box).

Usage: .conda/bin/python scripts/diag_2015_2017.py
"""
import os, sys, glob
import numpy as np
from scipy.ndimage import binary_erosion

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from venera.coherence import ncc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "results", "look_cache")
H, W, DS = 4000, 8000, 2
HH, WW = H // DS, W // DS


def files_for(year, pol=None, pointing=None):
    out = []
    for f in sorted(glob.glob(os.path.join(CACHE, f"venus_*_{year}*.npz"))):
        d = np.load(f, allow_pickle=True)
        if pol and str(d["pol"]) != pol:
            continue
        if pointing and str(d["pointing"]) != pointing:
            continue
        out.append(f)
    return out


def stack(files):
    Gsum = np.zeros((HH, WW), np.float64); Gcnt = np.zeros((HH, WW), np.int32)
    for f in files:
        d = np.load(f, allow_pickle=True)
        gm, m = d["gm"], d["mask"]; r0, r1, c0, c1 = d["bbox"]
        hr, hc = r0 // DS, c0 // DS; h, w = gm.shape
        sub = Gsum[hr:hr + h, hc:hc + w]; cub = Gcnt[hr:hr + h, hc:hc + w]
        sub[m] += gm[m]; cub[m] += 1
    Gm = np.divide(Gsum, Gcnt, out=np.zeros_like(Gsum), where=Gcnt > 0)
    return Gm.astype(np.float32), Gcnt > 0


def box_mask(lat0, lat1, lon0, lon1):
    m = np.zeros((HH, WW), bool)
    r0 = int((lat0 + 90) / 180 * HH); r1 = int((lat1 + 90) / 180 * HH)
    c0 = int((lon0 + 180) / 360 * WW); c1 = int((lon1 + 180) / 360 * WW)
    m[r0:r1, c0:c1] = True
    return m


def nccq(Ga, Ma, Gb, Mb, box=None):
    com = binary_erosion(Ma & Mb, iterations=12)
    if box is not None:
        com &= box
    if com.sum() < 1000:
        return np.nan, int(com.sum())
    return ncc(Ga, Gb, com, com), int(com.sum())


def main():
    NBOX = box_mask(20, 70, -80, 20)   # well-covered northern feature region
    filters = [
        ("all (both pol, N+S)", dict()),
        ("N only", dict(pointing="N")),
        ("S only", dict(pointing="S")),
        ("SCP only", dict(pol="scp")),
        ("OCP only", dict(pol="ocp")),
        ("N + SCP", dict(pol="scp", pointing="N")),
        ("N + OCP", dict(pol="ocp", pointing="N")),
    ]
    print(f"  {'filter':22s}  n15  n17   NCC(global)  NCC(N-box)  box_px")
    print("  " + "-" * 70)
    for name, kw in filters:
        f15 = files_for("2015", **kw); f17 = files_for("2017", **kw)
        if not f15 or not f17:
            print(f"  {name:22s}  {len(f15):3d}  {len(f17):3d}   (empty)"); continue
        G15, M15 = stack(f15); G17, M17 = stack(f17)
        ng, _ = nccq(G15, M15, G17, M17)
        nb, bpx = nccq(G15, M15, G17, M17, box=NBOX)
        print(f"  {name:22s}  {len(f15):3d}  {len(f17):3d}   {ng:8.3f}    "
              f"{nb:8.3f}   {bpx}")


if __name__ == "__main__":
    main()
