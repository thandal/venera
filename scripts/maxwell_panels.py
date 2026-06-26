"""Visual registration check on an unambiguous feature (Maxwell Montes / Ishtar):
each session's stack cropped to the same body-fixed box, side by side, plus
red/green overlays at IDENTITY (no registration applied). Judge by eye whether the
feature sits at the same place across sessions.

Writes results/figures/validate_maxwell_panels.png and validate_maxwell_overlays.png
"""
import os, sys
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STACKS = os.path.join(ROOT, "results", "session_stacks")
FIG = os.path.join(ROOT, "results", "figures")
HH, WW = 2000, 4000
YEARS = ["1988", "2001", "2012", "2015", "2017", "2020"]
DATES = {"1988": "Jun 1988", "2001": "Mar 2001", "2012": "May 2012",
         "2015": "Aug 2015", "2017": "Mar 2017", "2020": "May 2020"}
# Maxwell Montes / Ishtar Terra region (body-fixed)
LAT0, LAT1, LON0, LON1 = 48, 80, -35, 40
R0 = int((LAT0 + 90) / 180 * HH); R1 = int((LAT1 + 90) / 180 * HH)
C0 = int((LON0 + 180) / 360 * WW); C1 = int((LON1 + 180) / 360 * WW)
EXT = [LON0, LON1, LAT0, LAT1]


def load(y):
    d = np.load(os.path.join(STACKS, f"session_{y}.npz")); return d["Gm"], d["mask"]


def flatten(G, m, s=18):
    f = np.where(m, G, 0.0); w = gaussian_filter(m.astype(float), s)
    bg = gaussian_filter(f, s) / np.maximum(w, 1e-6); r = G - bg; r[~m] = np.nan
    return r


def crop(y):
    G, m = load(y)
    f = flatten(G, m)[R0:R1, C0:C1]; cm = m[R0:R1, C0:C1]
    return f, cm


def norm(x, cm):
    v = x[cm & np.isfinite(x)]
    if v.size == 0:
        return np.zeros_like(np.nan_to_num(x))
    lo, hi = np.percentile(v, 3), np.percentile(v, 99)
    return np.clip((np.nan_to_num(x) - lo) / (hi - lo + 1e-9), 0, 1)


def main():
    crops = {y: crop(y) for y in YEARS}

    # 1) each session side by side
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    for ax, y in zip(axs.ravel(), YEARS):
        f, cm = crops[y]
        img = norm(f, cm); img[~cm] = np.nan
        ax.imshow(img, origin="lower", extent=EXT, cmap="gray", aspect="auto",
                  vmin=0, vmax=1)
        ax.set_title(f"{DATES[y]}", fontsize=11)
        ax.set_xlabel("lon (°E)"); ax.set_ylabel("lat (°N)")
        ax.grid(alpha=0.25, color="c", lw=0.4)
    fig.suptitle("Maxwell/Ishtar region per session (body-fixed, no registration "
                 "applied) — does the feature sit at the same place?", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "validate_maxwell_panels.png"), dpi=120)
    plt.close(fig)

    # 2) red/green overlays at identity: a cross-geometry pair and a like-geometry pair
    def overlay(ax, ya, yb):
        fa, ma = crops[ya]; fb, mb = crops[yb]; cm = ma & mb
        rgb = np.zeros(fa.shape + (3,))
        rgb[..., 0] = norm(fa, cm) * cm; rgb[..., 1] = norm(fb, cm) * cm
        ax.imshow(rgb, origin="lower", extent=EXT, aspect="auto")
        ax.set_title(f"{ya} (red) vs {yb} (green) — identity", fontsize=11)
        ax.set_xlabel("lon (°E)"); ax.set_ylabel("lat (°N)")

    fig, axs = plt.subplots(1, 2, figsize=(15, 5.5))
    overlay(axs[0], "2015", "2017")   # cross-geometry (|ΔSRP| 17.5°)
    overlay(axs[1], "2012", "2020")   # like-geometry (|ΔSRP| 0.1°)
    fig.suptitle("Maxwell overlay: cross-geometry (left) vs like-geometry (right) "
                 "— yellow = aligned", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "validate_maxwell_overlays.png"), dpi=120)
    plt.close(fig)
    print("wrote validate_maxwell_panels.png and validate_maxwell_overlays.png")


if __name__ == "__main__":
    main()
