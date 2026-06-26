"""Generate the figure set: one stacked map per observing season, plus a single
cross-season combined (deep) stack. Lightweight, CPU-only.

Usage: .conda/bin/python scripts/make_figures.py [cache_tag] [N]
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

H, W = 4000, 8000
YRS = ["1988", "2001", "2012", "2015", "2017", "2020"]
DATES = {"1988": "Jun 1988", "2001": "Mar 2001", "2012": "May 2012",
         "2015": "Aug 2015", "2017": "Mar 2017", "2020": "May 2020"}
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "results", "epoch_stacks")
FIG = os.path.join(ROOT, "results", "figures")
os.makedirs(FIG, exist_ok=True)
tag = sys.argv[1] if len(sys.argv) > 1 else "cf"
N = sys.argv[2] if len(sys.argv) > 2 else "20"

# figure crop: northern hemisphere, covered longitudes
LAT0, LAT1, LON0, LON1 = 0, 82, -110, 65
r0, r1 = int((LAT0 + 90) / 180 * H), int((LAT1 + 90) / 180 * H)
c0, c1 = int((LON0 + 180) / 360 * W), int((LON1 + 180) / 360 * W)
DS = 3  # downsample for display


def panel(ax, Gm, mask, title):
    crop = Gm[r0:r1, c0:c1].astype(float)
    cm = mask[r0:r1, c0:c1]
    crop[~cm] = np.nan
    crop = crop[::DS, ::DS]
    vlo, vhi = np.nanpercentile(crop, 2), np.nanpercentile(crop, 99.5)
    ax.imshow(crop, origin="lower", extent=[LON0, LON1, LAT0, LAT1], cmap="gray",
              vmin=vlo, vmax=vhi, aspect="auto")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("longitude (°E)"); ax.set_ylabel("latitude (°N)")
    ax.grid(alpha=0.25, color="c", lw=0.4)


# --- per-season figures + accumulator for the cross-season stack ---
Gsum = np.zeros((H, W)); Gcnt = np.zeros((H, W)); total_n = 0
for y in YRS:
    f = os.path.join(CACHE, f"stack_both_{y}_N{N}_{tag}.npz")
    if not os.path.exists(f):
        print(f"missing {f}; skipping {y}"); continue
    d = np.load(f); Gm, mask, n = d["Gm"], d["mask"], int(d["n"])
    total_n += n
    Gsum[mask] += Gm[mask]; Gcnt[mask] += 1
    fig, ax = plt.subplots(figsize=(9, 5))
    panel(ax, Gm, mask, f"Venus N hemisphere — {DATES[y]}  ({n} looks)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, f"season_{y}.png"), dpi=120)
    plt.close(fig)
    print(f"wrote season_{y}.png ({n} looks)")

# --- cross-season combined (deep) stack ---
Gcomb = np.divide(Gsum, Gcnt, out=np.zeros_like(Gsum), where=Gcnt > 0)
fig, ax = plt.subplots(figsize=(11, 6))
panel(ax, Gcomb, Gcnt > 0, f"Venus N hemisphere — 1988–2020 combined  ({total_n} looks, {len(YRS)} seasons)")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "combined_all_seasons.png"), dpi=130)
plt.close(fig)
print(f"wrote combined_all_seasons.png ({total_n} looks)")

# --- 2x3 contact sheet of the seasons ---
fig, axs = plt.subplots(2, 3, figsize=(15, 9))
for ax, y in zip(axs.ravel(), YRS):
    f = os.path.join(CACHE, f"stack_both_{y}_N{N}_{tag}.npz")
    if os.path.exists(f):
        d = np.load(f); panel(ax, d["Gm"], d["mask"], f"{DATES[y]} ({int(d['n'])} looks)")
fig.suptitle("Venus northern hemisphere by observing season (Arecibo delay-Doppler)", fontsize=13)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "all_seasons_contact_sheet.png"), dpi=110)
plt.close(fig)
print("wrote all_seasons_contact_sheet.png")
print(f"\nFigures in {FIG}")
