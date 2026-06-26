"""Discriminate the 2012 blur: project INDIVIDUAL looks (2012 vs 2017) and show each
on the Maxwell region. Two outcomes:
  - each single look is SHARP but Maxwell sits at a SCATTERED longitude across looks
    -> per-look Doppler centering (fo) scatter is the smear (fixable).
  - each single look is INTRINSICALLY BLURRY -> data resolution; stacking can't fix.

Grid: top row = 2012 looks, bottom row = 2017 looks, same Maxwell crop + scale.
Also prints Maxwell brightest-pixel longitude per look -> position scatter per year.

Usage: .conda/bin/python scripts/single_look_compare.py
"""
import os, sys, glob
import numpy as np
from multiprocessing import Pool
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = ("/home/than/code/venera/arecibo_radar/pds-geosciences.wustl.edu/venus/"
        "arcb_nrao-v-rtls_gbt-3-delaydoppler-v1/vrm_90xx/data/")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "results", "figures")
H, W, DS = 4000, 8000, 2
HH, WW = H // DS, W // DS
# Maxwell / Ishtar crop
LAT0, LAT1, LON0, LON1 = 55, 78, -25, 35
R0 = int((LAT0+90)/180*HH); R1 = int((LAT1+90)/180*HH)
C0 = int((LON0+180)/360*WW); C1 = int((LON1+180)/360*WW)
NLOOK = 5


def pick(year):
    fs = sorted(glob.glob(os.path.join(DATA, f"venus_scp_{year}*.img")))
    from venera.data import parse_lbl
    keep = [f for f in fs if parse_lbl(f).get("GEO_POINTING") == "N"]
    return [keep[i] for i in np.linspace(0, len(keep)-1, NLOOK).astype(int)]


def _init():
    from venera import spice_setup
    spice_setup.furnsh_kernels()


def proj(task):
    year, idx, img = task
    from venera.geometry import Spin
    from venera.projection import project_file
    G = np.zeros((H, W), np.float32); Gc = np.zeros((H, W), np.int32)
    project_file(img, Spin(), G, Gc)
    Gm = np.divide(G, Gc, out=np.zeros_like(G), where=Gc > 0)
    half = np.divide(Gm[::DS, ::DS], 1)
    crop = half[R0:R1, C0:C1]; cm = (Gc[::DS, ::DS] > 0)[R0:R1, C0:C1]
    return (year, idx, crop.astype(np.float32), cm)


def main():
    looks = {y: pick(y) for y in ("2012", "2017")}
    tasks = [(y, i, f) for y in looks for i, f in enumerate(looks[y])]
    out = {}
    with Pool(10, initializer=_init) as p:
        for y, i, crop, cm in p.imap_unordered(proj, tasks):
            out[(y, i)] = (crop, cm); print(".", end="", flush=True)
    print()

    fig, axs = plt.subplots(2, NLOOK, figsize=(3*NLOOK, 6.5))
    lon_axis = np.linspace(LON0, LON1, C1-C0)
    for row, y in enumerate(("2012", "2017")):
        peaks = []
        for i in range(NLOOK):
            crop, cm = out[(y, i)]
            f = crop - gaussian_filter(np.where(cm, crop, 0.0), 20)
            f[~cm] = np.nan
            ax = axs[row, i]
            v = f[np.isfinite(f)]
            if v.size:
                lo, hi = np.percentile(v, 2), np.percentile(v, 99.5)
                ax.imshow(f, origin="lower", extent=[LON0, LON1, LAT0, LAT1],
                          cmap="gray", vmin=lo, vmax=hi, aspect="auto")
                # Maxwell longitude = column of brightest smoothed pixel
                sm = gaussian_filter(np.where(np.isfinite(f), f, 0.0), 3)
                pr, pc = np.unravel_index(np.argmax(sm), sm.shape)
                lonpk = LON0 + pc/(C1-C0)*(LON1-LON0)
                peaks.append(lonpk)
                ax.axvline(lonpk, color="r", lw=0.6)
            ax.set_title(f"{y} look {i}", fontsize=9)
            if i == 0:
                ax.set_ylabel("lat (°N)", fontsize=8)
            ax.set_xlabel("lon (°E)", fontsize=7)
        peaks = np.array(peaks)
        print(f"{y}: Maxwell peak-lon per look = {np.round(peaks,2)}  "
              f"std={peaks.std():.2f}°", flush=True)
    fig.suptitle("Single looks on Maxwell: 2012 (top) vs 2017 (bottom). "
                 "Sharp-but-scattered (red line) = centering; smeared = resolution.",
                 fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "single_look_compare.png"), dpi=120)
    print("wrote single_look_compare.png")


if __name__ == "__main__":
    main()
