"""Look at the actual cross-epoch misalignment. No fitting, no warping: just
high-pass each map and overlay two epochs as red/green over the Maxwell Montes
region. Aligned features -> yellow; a uniform offset -> uniform red/green split;
a rotation -> a split that rotates across the field; local warp -> incoherent.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from venera.registration import bandpass

H, W = 4000, 8000
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "results", "epoch_stacks")
FIG = os.path.join(ROOT, "results", "figures")

# Maxwell Montes region: lat 52-74 N, lon -28..28 E
LAT0, LAT1, LON0, LON1 = 52, 74, -28, 28
r0, r1 = int((LAT0 + 90) / 180 * H), int((LAT1 + 90) / 180 * H)
c0, c1 = int((LON0 + 180) / 360 * W), int((LON1 + 180) / 360 * W)
EXT = [LON0, LON1, LAT0, LAT1]


def load_crop(y):
    d = np.load(os.path.join(CACHE, f"stack_both_{y}_N20_cf.npz"))
    g = d["Gm"][r0:r1, c0:c1].astype(float)
    m = d["mask"][r0:r1, c0:c1]
    return g, m


def norm(g, m):
    f = bandpass(g, 3, 30)
    f[~m] = 0
    valid = f[m]
    lo, hi = np.percentile(valid, 5), np.percentile(valid, 99)
    return np.clip((f - lo) / (hi - lo + 1e-9), 0, 1)


PAIRS = [("2017", "2020"), ("2012", "2020"), ("1988", "2012")]
fig, axs = plt.subplots(len(PAIRS), 3, figsize=(15, 5 * len(PAIRS)))
for row, (a, b) in enumerate(PAIRS):
    ga, ma = load_crop(a); gb, mb = load_crop(b)
    na, nb = norm(ga, ma), norm(gb, mb)
    common = ma & mb
    axs[row, 0].imshow(np.where(ma, na, np.nan), origin="lower", extent=EXT, cmap="gray", aspect="auto")
    axs[row, 0].set_title(f"{a}")
    axs[row, 1].imshow(np.where(mb, nb, np.nan), origin="lower", extent=EXT, cmap="gray", aspect="auto")
    axs[row, 1].set_title(f"{b}")
    rgb = np.zeros((*na.shape, 3))
    rgb[..., 0] = np.where(common, na, 0)      # epoch a -> red
    rgb[..., 1] = np.where(common, nb, 0)      # epoch b -> green
    axs[row, 2].imshow(rgb, origin="lower", extent=EXT, aspect="auto")
    axs[row, 2].set_title(f"overlay R={a} G={b}  (yellow=aligned)")
    for k in range(3):
        axs[row, k].set_xlabel("lon (E)"); axs[row, k].set_ylabel("lat (N)")
        axs[row, k].axhline(65, color="c", lw=0.4, alpha=0.5)
fig.suptitle("Cross-epoch alignment over Maxwell Montes (raw body-frame maps, no correction)", fontsize=13)
fig.tight_layout()
out = os.path.join(FIG, "overlay_epochs.png")
fig.savefig(out, dpi=130); print(f"wrote {out}")
