"""Validate the 'raw cross-epoch registration is already good' claim, honestly.
2012 vs 2020 stacks, NO correction. Grayscales side by side + red/green overlay,
at full overlap and zoomed on Maxwell at fine scale, with NCC over the full common
support and over the Maxwell box annotated. Same stretch for both epochs.
"""
import os, sys, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from venera.coherence import coherent_stack
from venera.registration import bandpass

CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "look_cache")
FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "figures")
H, W, DS = 4000, 8000, 2
hh, ww = H // DS, W // DS


def load(year):
    gs, ms = [], []
    for f in sorted(glob.glob(f"{CACHE}/venus_*cp_{year}*.npz")):
        d = np.load(f, allow_pickle=True)
        if str(d["pointing"]) != "N":
            continue
        g = np.zeros((hh, ww)); m = np.zeros((hh, ww), bool)
        r0, _, c0, _ = d["bbox"]; gm, mk = d["gm"], d["mask"]; rr, cc = int(r0)//DS, int(c0)//DS
        gh, gw = gm.shape
        if rr+gh > hh or cc+gw > ww:
            gh, gw = min(gh, hh-rr), min(gw, ww-cc); gm, mk = gm[:gh, :gw], mk[:gh, :gw]
        g[rr:rr+gh, cc:cc+gw] = np.where(mk, gm, 0); m[rr:rr+gh, cc:cc+gw] = mk
        gs.append(g); ms.append(m)
    g, c = coherent_stack(gs, ms); return g, c > 0


g12, m12 = load("2012"); g20, m20 = load("2020")


def box(LAT0, LAT1, LON0, LON1):
    return (slice(int((LAT0+90)/180*hh), int((LAT1+90)/180*hh)),
            slice(int((LON0+180)/360*ww), int((LON1+180)/360*ww)), [LON0, LON1, LAT0, LAT1])


def ncc_over(sl, smooth, trend, erode=0):
    a = bandpass(g12, smooth, trend)[sl[0], sl[1]]; b = bandpass(g20, smooth, trend)[sl[0], sl[1]]
    msk = m12[sl[0], sl[1]] & m20[sl[0], sl[1]]
    if erode:
        msk = binary_erosion(msk, iterations=erode)
    if msk.sum() < 500:
        return np.nan
    a, b = a[msk]-a[msk].mean(), b[msk]-b[msk].mean()
    return float(np.sum(a*b)/(np.sqrt(np.sum(a**2)*np.sum(b**2))+1e-9))


def norm(g, m, sl, smooth, trend):
    f = bandpass(g, smooth, trend)[sl[0], sl[1]].astype(float); mm = m[sl[0], sl[1]]
    f[~mm] = np.nan
    lo, hi = np.nanpercentile(f, 4), np.nanpercentile(f, 99)
    return np.clip((f-lo)/(hi-lo+1e-9), 0, 1), mm


def gray(ax, g, m, sl, smooth, trend, title):
    f, mm = norm(g, m, sl, smooth, trend); f[~mm] = np.nan
    ax.imshow(f, origin="lower", extent=sl[2], cmap="gray", vmin=0, vmax=1, aspect="auto")
    ax.set_title(title, fontsize=10)


def over(ax, sl, smooth, trend, title):
    a, ma = norm(g12, m12, sl, smooth, trend); b, mb = norm(g20, m20, sl, smooth, trend)
    com = ma & mb; rgb = np.zeros((*a.shape, 3))
    rgb[..., 0] = np.where(com, np.nan_to_num(a), 0); rgb[..., 1] = np.where(com, np.nan_to_num(b), 0)
    ax.imshow(rgb, origin="lower", extent=sl[2], aspect="auto"); ax.set_title(title, fontsize=10)


wide = box(2, 80, -105, 55)
mx = box(54, 73, -22, 22)
nccw = ncc_over(wide, 2, 30, erode=5); nccm = ncc_over(mx, 1.5, 18, erode=3)

fig, axs = plt.subplots(2, 3, figsize=(18, 11))
gray(axs[0, 0], g12, m12, wide, 2, 30, "2012 stack (wide overlap)")
gray(axs[0, 1], g20, m20, wide, 2, 30, "2020 stack")
over(axs[0, 2], wide, 2, 30, f"raw overlay R=2012 G=2020  (NCC={nccw:.3f})")
gray(axs[1, 0], g12, m12, mx, 1.5, 18, "2012  (Maxwell, fine)")
gray(axs[1, 1], g20, m20, mx, 1.5, 18, "2020  (Maxwell, fine)")
over(axs[1, 2], mx, 1.5, 18, f"raw overlay  (NCC={nccm:.3f})")
for ax in axs.ravel():
    ax.set_xlabel("lon E"); ax.set_ylabel("lat N")
fig.suptitle("Raw cross-epoch registration, NO correction (yellow=aligned). Validate 'already good'.", fontsize=13)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "raw_alignment_check.png"), dpi=120)
print(f"wide-overlap NCC={nccw:.3f}  Maxwell-fine NCC={nccm:.3f}", flush=True)
print(f"wrote {FIG}/raw_alignment_check.png", flush=True)
