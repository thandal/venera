"""Cross-epoch registration quality for ALL 15 epoch pairs (N-pointing).

For each pair: align by a boundary-free 2-D translation (the natural rigid
alignment; the spurious tilt is excluded), then measure NCC over an eroded
interior (no coverage edge) on Maxwell and over the wide overlap. Outputs:
  - ncc_matrix.png        : 6x6 registration-quality heatmaps (Maxwell + wide)
  - pairs_maxwell.png     : all 15 aligned overlays, Maxwell region
  - pairs_wide.png        : all 15 aligned overlays, wide overlap
Each panel annotated with NCC, the fitted Δlon/Δlat, and the geometry difference.
"""
import os, sys, glob, itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion, shift as ndshift
import cspyce as csp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from venera import spice_setup
from venera.geometry import Spin, sub_radar_point, doppler_angle
from venera.coherence import coherent_stack
from venera.registration import bandpass, register_maps

CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "look_cache")
FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "figures")
H, W, DS = 4000, 8000, 2
hh, ww = H // DS, W // DS
EPOCHS = ["1988", "2001", "2012", "2015", "2017", "2020"]
EPT = {"1988": "1988-06-17T15:00", "2001": "2001-03-31T15:00", "2012": "2012-05-29T16:00",
       "2015": "2015-08-13T16:00", "2017": "2017-03-26T17:00", "2020": "2020-05-30T17:00"}
spice_setup.furnsh_kernels()
spin = Spin()
GEOM = {}
for y, t in EPT.items():
    et = csp.str2et(t); _, lat, _ = sub_radar_point(et, spin)
    GEOM[y] = (np.degrees(doppler_angle(et, spin)), np.degrees(lat))


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
    g, c = coherent_stack(gs, ms)
    return g, c > 0, len(gs)


def box(LAT0, LAT1, LON0, LON1):
    return (slice(int((LAT0+90)/180*hh), int((LAT1+90)/180*hh)),
            slice(int((LON0+180)/360*ww), int((LON1+180)/360*ww)), [LON0, LON1, LAT0, LAT1])


WIDE = box(2, 80, -105, 55)
MX = box(54, 73, -22, 22)
DEG = 360.0 / ww


def ncc_in(a, b, ma, mb, sl, smooth, trend, erode):
    A = bandpass(a, smooth, trend)[sl[0], sl[1]]; B = bandpass(b, smooth, trend)[sl[0], sl[1]]
    msk = binary_erosion(ma[sl[0], sl[1]] & mb[sl[0], sl[1]], iterations=erode)
    if msk.sum() < 400:
        return np.nan
    A, B = A[msk]-A[msk].mean(), B[msk]-B[msk].mean()
    return float(np.sum(A*B)/(np.sqrt(np.sum(A**2)*np.sum(B**2))+1e-9))


def aligned(b, mb, dr, dc):
    g2 = ndshift(b, (dr, dc), order=1, mode="constant", cval=0)
    m2 = ndshift(mb.astype(np.float32), (dr, dc), order=1, cval=0) > 0.5
    return g2, m2


def nz(g, m, sl, smooth, trend):
    f = bandpass(g, smooth, trend)[sl[0], sl[1]].astype(float); mm = m[sl[0], sl[1]]
    v = f[mm]
    if v.size < 50:
        return np.zeros_like(f), mm
    lo, hi = np.percentile(v, 4), np.percentile(v, 99)
    f = np.clip((f-lo)/(hi-lo+1e-9), 0, 1); f[~mm] = 0
    return f, mm


print("loading 6 epoch stacks...", flush=True)
S = {y: load(y) for y in EPOCHS}
for y in EPOCHS:
    print(f"  {y}: {S[y][2]} looks, coverage {S[y][1].mean()*100:.0f}%", flush=True)

pairs = list(itertools.combinations(range(6), 2))
res = {}
nmat_mx = np.full((6, 6), np.nan); nmat_wd = np.full((6, 6), np.nan)
for i, j in pairs:
    a, b = EPOCHS[i], EPOCHS[j]
    ga, ma, _ = S[a]; gb, mb, _ = S[b]
    dr, dc, sig = register_maps(ga, gb, valid_a=ma, valid_b=mb, max_shift=80, smooth_px=7.0, trend_px=55.0)
    gb2, mb2 = aligned(gb, mb, dr, dc)
    nmx = ncc_in(ga, gb2, ma, mb2, MX, 1.5, 18, 3)
    nwd = ncc_in(ga, gb2, ma, mb2, WIDE, 2.0, 30, 5)
    res[(i, j)] = dict(dr=dr, dc=dc, dlon=-dc*DEG, dlat=-dr*DEG, nmx=nmx, nwd=nwd, gb2=gb2, mb2=mb2)
    nmat_mx[i, j] = nmat_mx[j, i] = nmx; nmat_wd[i, j] = nmat_wd[j, i] = nwd
    dgeo = abs(GEOM[a][0]-GEOM[b][0]) + abs(GEOM[a][1]-GEOM[b][1])
    print(f"  {a}<->{b}: NCC_mx={nmx:.3f} NCC_wide={nwd:.3f}  Δlon={-dc*DEG:+.2f} Δlat={-dr*DEG:+.2f}  "
          f"Δgeo={dgeo:.1f}", flush=True)

# ---- NCC matrix heatmap ----
fig, axs = plt.subplots(1, 2, figsize=(15, 6.5))
for ax, M, t in [(axs[0], nmat_mx, "Maxwell (fine)"), (axs[1], nmat_wd, "wide overlap")]:
    np.fill_diagonal(M, 1.0)
    im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(6)); ax.set_yticks(range(6)); ax.set_xticklabels(EPOCHS); ax.set_yticklabels(EPOCHS)
    for i in range(6):
        for j in range(6):
            ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                    color="w" if M[i, j] < 0.6 else "k", fontsize=9)
    ax.set_title(f"boundary-free NCC after 2-D alignment — {t}")
    plt.colorbar(im, ax=ax, fraction=0.046)
fig.suptitle("Cross-epoch registration quality, all pairs (1 = perfect)", fontsize=13)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "ncc_matrix.png"), dpi=120); plt.close(fig)
print("wrote ncc_matrix.png", flush=True)


def overlay_grid(sl, smooth, trend, key, fname, title):
    order = sorted(pairs, key=lambda p: res[p][key])      # worst first
    fig, axs = plt.subplots(3, 5, figsize=(22, 13))
    for ax, (i, j) in zip(axs.ravel(), order):
        a, b = EPOCHS[i], EPOCHS[j]; r = res[(i, j)]
        ga, ma, _ = S[a]
        A, mA = nz(ga, ma, sl, smooth, trend); B, mB = nz(r["gb2"], r["mb2"], sl, smooth, trend)
        com = mA & mB; rgb = np.zeros((*A.shape, 3))
        rgb[..., 0] = np.where(com, A, 0); rgb[..., 1] = np.where(com, B, 0)
        ax.imshow(rgb, origin="lower", extent=sl[2], aspect="auto")
        dgeo = abs(GEOM[a][0]-GEOM[b][0]) + abs(GEOM[a][1]-GEOM[b][1])
        ax.set_title(f"{a}(R) {b}(G)  NCC={r[key]:.2f}\nΔdopAng={abs(GEOM[a][0]-GEOM[b][0]):.1f} "
                     f"ΔSRPlat={abs(GEOM[a][1]-GEOM[b][1]):.1f}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(title + "  (worst-registering pair first; yellow = aligned)", fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, fname), dpi=110); plt.close(fig)
    print(f"wrote {fname}", flush=True)


overlay_grid(MX, 1.5, 18, "nmx", "pairs_maxwell.png", "All 15 epoch pairs, Maxwell region")
overlay_grid(WIDE, 2.0, 30, "nwd", "pairs_wide.png", "All 15 epoch pairs, wide overlap")
print("DONE", flush=True)
