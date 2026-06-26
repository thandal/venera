"""Figures to evaluate the tilt result:
  (1) tilt_vs_srp.png   -- recovered rotation: tilt & polar vs |ΔSRP latitude|
  (2) tilt_beforeafter.png -- red/green overlays, identity (top) vs best rigid
      rotation (bottom), for pairs spanning the ΔSRP range. Yellow = aligned.

Scatter data are the 15-pair results from register_pairs_sphere.py; the panels
re-run the sphere search for a few representative pairs.

Usage: .conda/bin/python scripts/figs_tilt.py
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from register_on_sphere import (flatten, search_R, decompose, rotate_map,
                                STACKS, FIG, HH, WW)

# 15-pair results (pair, |ΔSRP_lat|, tilt, |polar|, NCC_id, NCC_R)
# The two pairs whose DIRECT search stuck at identity were recovered by forcing the
# per-epoch-model predicted tilt (fix_stuck_pairs.py): 1988-2015 1.17deg (0.35->0.62),
# 2015-2020 1.04deg (0.22->0.35). They now lie on the line.
RECOVERED = {"1988-2015", "2015-2020"}
DATA = [
    ("2015-2017", 17.5, 2.60, 0.02, 0.11, 0.59), ("2001-2015", 17.1, 2.34, 0.03, 0.14, 0.54),
    ("1988-2017", 9.3, 1.51, 0.01, 0.25, 0.58), ("2012-2017", 6.6, 1.44, 0.13, 0.33, 0.63),
    ("1988-2001", 8.9, 1.28, 0.03, 0.28, 0.52), ("2017-2020", 6.6, 1.27, 0.01, 0.21, 0.33),
    ("2001-2012", 6.2, 1.25, 0.17, 0.35, 0.58), ("2012-2015", 10.9, 1.19, 0.18, 0.34, 0.57),
    ("2001-2020", 6.1, 0.92, 0.22, 0.24, 0.36), ("2001-2017", 0.4, 0.21, 0.01, 0.71, 0.75),
    ("1988-2015", 8.2, 1.17, 0.01, 0.35, 0.62), ("1988-2012", 2.7, 0.01, 0.02, 0.64, 0.64),
    ("1988-2020", 2.8, 0.01, 0.01, 0.42, 0.42), ("2015-2020", 10.9, 1.04, 0.00, 0.22, 0.35),
    ("2012-2020", 0.1, 0.00, 0.01, 0.55, 0.55),
]
# representative pairs for before/after panels (span ΔSRP)
PANEL = [("2012", "2020"), ("2012", "2017"), ("1988", "2017"), ("2015", "2017")]
# display crop: northern hemisphere
LAT0, LAT1, LON0, LON1 = 25, 82, -115, 60


def scatter_fig():
    dsrp = np.array([d[1] for d in DATA]); tilt = np.array([d[2] for d in DATA])
    polar = np.array([d[3] for d in DATA])
    rec = np.array([d[0] in RECOVERED for d in DATA])
    slope = np.sum(dsrp * tilt) / np.sum(dsrp ** 2)   # all 15 pairs now valid
    r = np.corrcoef(dsrp, tilt)[0, 1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(dsrp[~rec], tilt[~rec], s=60, c="tab:blue",
               label="tilt (direct sphere search)")
    ax.scatter(dsrp[rec], tilt[rec], s=90, c="tab:orange", marker="D",
               edgecolors="k", linewidths=0.6,
               label="tilt (recovered by forcing predicted tilt)")
    ax.scatter(dsrp, polar, s=35, c="tab:green", marker="x",
               label="polar / period component")
    xx = np.linspace(0, dsrp.max(), 50)
    ax.plot(xx, slope * xx, "k--", lw=1, label=f"tilt = {slope:.2f}·|ΔSRP_lat|")
    for d in DATA:
        ax.annotate(d[0], (d[1], d[2]), fontsize=7, xytext=(3, 3),
                    textcoords="offset points")
    ax.set_xlabel("|Δ sub-radar latitude| between sessions (°)")
    ax.set_ylabel("recovered rotation component (°)")
    ax.set_title(f"Cross-session mismatch is a TILT ∝ ΔSRP-lat  "
                 f"(corr={r:.2f}, all 15 pairs; period component ~0)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "tilt_vs_srp.png"), dpi=130)
    plt.close(fig)
    print(f"wrote tilt_vs_srp.png  (corr={r:.2f}, slope={slope:.3f})")


def crop_rgb(Af, Am, Bf, Bm):
    r0 = int((LAT0 + 90) / 180 * HH); r1 = int((LAT1 + 90) / 180 * HH)
    c0 = int((LON0 + 180) / 360 * WW); c1 = int((LON1 + 180) / 360 * WW)
    A = Af[r0:r1, c0:c1]; B = Bf[r0:r1, c0:c1]
    m = (Am & Bm)[r0:r1, c0:c1]

    def norm(x):
        v = x[m]
        if v.size == 0:
            return np.zeros_like(x)
        lo, hi = np.percentile(v, 2), np.percentile(v, 98)
        return np.clip((x - lo) / (hi - lo + 1e-9), 0, 1)
    rgb = np.zeros(A.shape + (3,))
    rgb[..., 0] = norm(A) * m; rgb[..., 1] = norm(B) * m
    return rgb


def panel_fig():
    stk = {}
    for y in set([p for pr in PANEL for p in pr]):
        d = np.load(os.path.join(STACKS, f"session_{y}.npz"))
        stk[y] = (flatten(d["Gm"], d["mask"]), d["mask"])
    n = len(PANEL)
    fig, axs = plt.subplots(2, n, figsize=(4.2 * n, 7))
    for j, (ya, yb) in enumerate(PANEL):
        Af, Am = stk[ya]; Bf, Bm = stk[yb]
        rv, nfit, nid = search_R(Af, Am, Bf, Bm, coarse_step=1.5)
        total, polar, tilt = decompose(rv)
        R = Rotation.from_rotvec(rv).as_matrix()
        Br, Brm = rotate_map(Bf, Bm, R.T)
        for i, (img, sub) in enumerate([
                (crop_rgb(Af, Am, Bf, Bm), f"identity  NCC={nid:.2f}"),
                (crop_rgb(Af, Am, Br, Brm), f"best R ({tilt:.1f}° tilt)  NCC={nfit:.2f}")]):
            ax = axs[i, j]
            ax.imshow(img, origin="lower", extent=[LON0, LON1, LAT0, LAT1], aspect="auto")
            ax.set_title((f"{ya}(r) vs {yb}(g)\n" if i == 0 else "") + sub, fontsize=9)
            if j == 0:
                ax.set_ylabel(["IDENTITY", "AFTER ROTATION"][i] + "\nlat (°N)", fontsize=9)
            ax.set_xlabel("lon (°E)", fontsize=8)
        print(f"  {ya}-{yb}: tilt {tilt:.2f}°  NCC {nid:.2f}->{nfit:.2f}", flush=True)
    fig.suptitle("Does a single rigid rotation register the sessions? "
                 "(yellow = aligned)", fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "tilt_beforeafter.png"), dpi=120)
    plt.close(fig)
    print("wrote tilt_beforeafter.png")


def main():
    scatter_fig()
    if "scatter" not in sys.argv:
        panel_fig()


if __name__ == "__main__":
    main()
