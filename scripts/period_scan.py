"""STEP 2 of the period search: forward-project 1988 & 2020 at a grid of trial spin
periods (a period change = an exact per-look longitude rotation = a sub-pixel column
shift), and at each period compute the INTERIOR overlapping-disk correlation at
IDENTITY (no free shift — the only shift applied is the one the period dictates).
The period that maximizes interior correlation is the estimate.

CAVEAT: one pair conflates the period with any constant longitude offset between the
two sessions; this validates the method/curve. The unbiased period (step 3) uses all
sessions so per-session offsets average out.

Writes results/figures/period_scan_1988_2020.png  (curve + peak overlay)
"""
import os, sys, glob
import numpy as np
from scipy.ndimage import gaussian_filter, binary_erosion, shift as ndshift
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from venera.coherence import ncc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STACKS = os.path.join(ROOT, "results", "session_stacks")
CACHE = os.path.join(ROOT, "results", "look_cache")
FIG = os.path.join(ROOT, "results", "figures")
HH, WW = 2000, 4000
A, B = "1988", "2020"
P_ASSUMED = 243.0185
ERODE_PX, LATMIN, LATMAX = 20, 30, 75
PGRID = np.arange(243.008, 243.0325, 0.0005)
LITER = {"IAU 243.0185": 243.0185, "Campbell 243.0212": 243.0212,
         "Margot 243.0226": 243.0226}


def mean_day(y):
    return float(np.mean([float(np.load(f, allow_pickle=True)["day"])
                          for f in glob.glob(os.path.join(CACHE, f"venus_*_{y}*.npz"))]))


def load(y):
    d = np.load(os.path.join(STACKS, f"session_{y}.npz")); return d["Gm"], d["mask"]


def flatten(G, m, s=35):
    f = np.where(m, G, 0.0); w = gaussian_filter(m.astype(float), s)
    return (G - gaussian_filter(f, s) / np.maximum(w, 1e-6)) * m


def latband():
    b = np.zeros((HH, WW), bool)
    b[int((LATMIN+90)/180*HH):int((LATMAX+90)/180*HH)] = True
    return b


def main():
    GA, MA = load(A); GB, MB = load(B)
    FA = flatten(GA, MA); FB = flatten(GB, MB)
    dt = mean_day(B) - mean_day(A)
    LB = latband()
    print(f"{A}-{B}: Δt = {dt:.0f} d = {dt/365.25:.1f} yr")

    def ncc_at(P):
        wdot = (-360.0 / P) - (-360.0 / P_ASSUMED)
        rel_deg = -wdot * dt                     # shift B onto A's frame at period P
        px = rel_deg / 360.0 * WW
        FBs = ndshift(FB, (0, px), order=1, mode="constant", cval=0.0)
        MBs = ndshift(MB.astype(float), (0, px), order=1, mode="constant", cval=0.0) > 0.99
        interior = binary_erosion(MA & MBs, iterations=ERODE_PX) & LB
        return ncc(FA, FBs, interior, interior), rel_deg

    nccs, rels = [], []
    for P in PGRID:
        v, rel = ncc_at(P); nccs.append(v); rels.append(rel)
        print(f"  P={P:.4f}  rel_shift={rel:+.3f}°  interior_NCC={v:.4f}", flush=True)
    nccs = np.array(nccs)
    ip = int(np.argmax(nccs)); Ppk = PGRID[ip]
    print(f"\npeak: P = {Ppk:.4f} d  (interior NCC {nccs[ip]:.4f})")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))
    ax1.plot(PGRID, nccs, "-o", ms=3)
    ax1.axvline(Ppk, color="b", lw=1, label=f"peak {Ppk:.4f}")
    for name, P in LITER.items():
        ax1.axvline(P, ls="--", lw=0.8, alpha=0.6,
                    color={"IAU 243.0185": "k", "Campbell 243.0212": "tab:green",
                           "Margot 243.0226": "tab:orange"}[name], label=name)
    ax1.set_xlabel("trial period (days)"); ax1.set_ylabel("interior overlapping-disk NCC")
    ax1.set_title(f"{A}↔{B} period scan (forward-projected, interior, at identity)")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

    # overlay at the peak (red=A, green=B-shifted), interior region only
    wdot = (-360.0 / Ppk) - (-360.0 / P_ASSUMED); px = -wdot * dt / 360.0 * WW
    FBs = ndshift(FB, (0, px), order=1, mode="constant", cval=0.0)
    MBs = ndshift(MB.astype(float), (0, px), order=1, mode="constant", cval=0.0) > 0.99
    interior = binary_erosion(MA & MBs, iterations=ERODE_PX) & LB
    r0, r1 = int((LATMIN+90)/180*HH), int((LATMAX+90)/180*HH)
    cc = np.where(interior.any(0))[0]; c0, c1 = cc[0], cc[-1]

    def nrm(x, m):
        v = x[m]; lo, hi = np.percentile(v, 2), np.percentile(v, 98)
        return np.clip((x - lo) / (hi - lo + 1e-9), 0, 1)
    sub = interior[r0:r1, c0:c1]
    rgb = np.zeros((r1 - r0, c1 - c0, 3))
    rgb[..., 0] = nrm(FA[r0:r1, c0:c1], sub) * sub
    rgb[..., 1] = nrm(FBs[r0:r1, c0:c1], sub) * sub
    ax2.imshow(rgb, origin="lower",
               extent=[c0/WW*360-180, c1/WW*360-180, LATMIN, LATMAX], aspect="auto")
    ax2.set_title(f"overlay at peak {Ppk:.4f} d  (red={A}, green={B}; yellow=aligned)")
    ax2.set_xlabel("lon (°E)"); ax2.set_ylabel("lat (°N)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "period_scan_1988_2020.png"), dpi=120)
    print("wrote period_scan_1988_2020.png")


if __name__ == "__main__":
    main()
