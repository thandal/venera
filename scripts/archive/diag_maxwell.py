"""Look at Maxwell Montes (the brightest, highest radar feature, ~65N) in 2015 vs
2017 -- an unambiguous registration probe. Is the cross-session error a local
shift, a rotation, or a distortion?

Loads the saved 2015/2017 session stacks, crops to Maxwell, flattens, registers the
patch, and writes red/green overlays at zero shift and at the best local shift.

Usage: .conda/bin/python scripts/diag_maxwell.py
"""
import os, sys
import numpy as np
from scipy.ndimage import gaussian_filter, binary_erosion
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from venera.registration import bandpass, xcorr_offset
from venera.coherence import ncc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STACKS = os.path.join(ROOT, "results", "session_stacks")
FIG = os.path.join(ROOT, "results", "figures")
HH, WW = 2000, 4000
PXDEG = WW / 360.0                       # 11.11 px/deg
# Maxwell Montes / Ishtar region
LAT0, LAT1, LON0, LON1 = 50, 80, -40, 45


def load(y):
    d = np.load(os.path.join(STACKS, f"session_{y}.npz"))
    return d["Gm"], d["mask"]


def crop(G, M):
    r0 = int((LAT0 + 90) / 180 * HH); r1 = int((LAT1 + 90) / 180 * HH)
    c0 = int((LON0 + 180) / 360 * WW); c1 = int((LON1 + 180) / 360 * WW)
    return G[r0:r1, c0:c1], M[r0:r1, c0:c1]


def flatten(crop, cm, sigma=25):
    filled = np.where(cm, crop, 0.0)
    wsm = gaussian_filter(cm.astype(float), sigma)
    bg = gaussian_filter(filled, sigma) / np.maximum(wsm, 1e-6)
    f = crop - bg; f[~cm] = 0.0
    return f


def rgb(A, B, cm):
    def norm(x):
        v = x[cm]
        if v.size == 0:
            return np.zeros_like(x)
        lo, hi = np.percentile(v, 2), np.percentile(v, 99)
        return np.clip((x - lo) / (hi - lo + 1e-9), 0, 1)
    out = np.zeros(A.shape + (3,))
    out[..., 0] = norm(A) * cm
    out[..., 1] = norm(B) * cm
    return out


def save(img, title, path):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(img, origin="lower", extent=[LON0, LON1, LAT0, LAT1], aspect="auto")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("longitude (°E)"); ax.set_ylabel("latitude (°N)")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main():
    G15, M15 = load("2015"); G17, M17 = load("2017")
    c15, m15 = crop(G15, M15); c17, m17 = crop(G17, M17)
    cm = m15 & m17
    print(f"Maxwell box {LAT0}-{LAT1}N {LON0}-{LON1}E: common px = {cm.sum()} "
          f"({100*cm.mean():.0f}% of box)")
    f15 = flatten(c15, m15); f17 = flatten(c17, m17)

    # register the patch
    dr, dc, sig = xcorr_offset(bandpass(f15, 3, 25) * cm, bandpass(f17, 3, 25) * cm,
                               max_shift=70)
    dlat, dlon = dr / HH * 180.0, dc / WW * 360.0
    com = binary_erosion(cm, iterations=6)
    n0 = ncc(f15, f17, com, com) if com.sum() > 500 else np.nan
    f17s = np.roll(np.roll(f17, int(round(dr)), 0), int(round(dc)), 1)
    ms = np.roll(np.roll(m17, int(round(dr)), 0), int(round(dc)), 1)
    coms = binary_erosion(m15 & ms, iterations=6)
    n1 = ncc(f15, f17s, coms, coms) if coms.sum() > 500 else np.nan
    print(f"best local shift: dlat={dlat:+.2f}°  dlon={dlon:+.2f}°  (sig={sig:.2f})")
    print(f"interior NCC on Maxwell patch:  @0 = {n0:.3f}   @best-shift = {n1:.3f}")

    save(rgb(f15, f17, cm),
         f"Maxwell 2015(red) vs 2017(green) — ZERO shift   NCC={n0:.2f}",
         os.path.join(FIG, "maxwell_2015_2017_shift0.png"))
    save(rgb(f15, f17s, m15 & ms),
         f"Maxwell 2015(red) vs 2017(green) — best local shift "
         f"(Δlat={dlat:+.1f}°, Δlon={dlon:+.1f}°)   NCC={n1:.2f}",
         os.path.join(FIG, "maxwell_2015_2017_aligned.png"))
    print("wrote maxwell_2015_2017_shift0.png, maxwell_2015_2017_aligned.png")


if __name__ == "__main__":
    main()
