"""STEP 1 of the period search: define and SHOW the interior overlapping-disk region
used for the correlation, so it can be validated (does it avoid coverage edges and the
SRP/equatorial band?) before any period number is trusted.

Region = (mask_A & mask_B) eroded by ERODE_PX, restricted to a clean latitude band
(avoids the SRP-exclusion gap / N-S boundary near the equator and the pole edge).
Shown for the long-baseline like-geometry pair 1988 & 2020.

Writes results/figures/validate_interior_region.png
"""
import os, sys
import numpy as np
from scipy.ndimage import gaussian_filter, binary_erosion
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STACKS = os.path.join(ROOT, "results", "session_stacks")
FIG = os.path.join(ROOT, "results", "figures")
HH, WW = 2000, 4000
PAIR = ("1988", "2020")
ERODE_PX = 20            # ~1.8° guard band off every coverage edge
LATMIN, LATMAX = 30, 75  # avoid equatorial SRP/N-S gap (<~20) and pole edge (>~78)
# display crop
LAT0, LAT1, LON0, LON1 = 10, 82, -120, 65
DR0 = int((LAT0+90)/180*HH); DR1 = int((LAT1+90)/180*HH)
DC0 = int((LON0+180)/360*WW); DC1 = int((LON1+180)/360*WW)


def load(y):
    d = np.load(os.path.join(STACKS, f"session_{y}.npz")); return d["Gm"], d["mask"]


def flatten(G, m, s=35):
    f = np.where(m, G, 0.0); w = gaussian_filter(m.astype(float), s)
    bg = gaussian_filter(f, s)/np.maximum(w, 1e-6); r = G-bg; r[~m] = np.nan
    return r


def main():
    GA, MA = load(PAIR[0]); GB, MB = load(PAIR[1])
    common = MA & MB
    interior = binary_erosion(common, iterations=ERODE_PX)
    latband = np.zeros((HH, WW), bool)
    r0 = int((LATMIN+90)/180*HH); r1 = int((LATMAX+90)/180*HH)
    latband[r0:r1] = True
    interior &= latband
    print(f"{PAIR[0]}&{PAIR[1]}: common px={int(common.sum())}, "
          f"interior px={int(interior.sum())} ({100*interior.sum()/common.sum():.0f}% of common)")

    # show: 2020 flattened, interior at full brightness, everything else dimmed
    F = flatten(GB, MB)
    crop = F[DR0:DR1, DC0:DC1]; inn = interior[DR0:DR1, DC0:DC1]; cm = MB[DR0:DR1, DC0:DC1]
    v = crop[np.isfinite(crop)]; vlo, vhi = np.percentile(v, 2), np.percentile(v, 99)
    base = np.clip((np.nan_to_num(crop)-vlo)/(vhi-vlo+1e-9), 0, 1)
    rgb = np.zeros(crop.shape+(3,))
    dim = np.where(cm, 0.30, 0.0)            # covered-but-excluded = dim grey
    for k in range(3):
        rgb[..., k] = base*dim
    rgb[inn, 0] = base[inn]; rgb[inn, 1] = base[inn]; rgb[inn, 2] = base[inn]  # interior bright
    # tint interior edge green so the boundary is visible
    edge = inn & ~binary_erosion(inn, iterations=2)
    rgb[edge] = [0, 1, 0]
    ext = [LON0, LON1, LAT0, LAT1]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.imshow(rgb, origin="lower", extent=ext, aspect="auto")
    ax.set_title(f"Interior overlap region for period correlation "
                 f"({PAIR[0]}∩{PAIR[1]}, eroded {ERODE_PX}px≈1.8°, lat {LATMIN}–{LATMAX}°)\n"
                 f"bright = correlation region, dim = covered-but-excluded "
                 f"(edges/SRP-gap), green = boundary", fontsize=10)
    ax.set_xlabel("longitude (°E)"); ax.set_ylabel("latitude (°N)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "validate_interior_region.png"), dpi=120)
    print("wrote validate_interior_region.png")


if __name__ == "__main__":
    main()
