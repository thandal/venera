"""HONEST TEST of the rotation registration.

Convention (reasoned, not guessed): tile_displacements(a,b) gives the displacement
of b's features relative to a, (dlat,dlon) = ω×r with ω = fit_tile_rotation(...).
So b's feature belonging at a-direction r sits at R(ω)·r in b. To warp b INTO a's
frame, sample b at src = R(ω)·r.

Output: (1) Maxwell-region overlays raw vs rotation-aligned for the pair I can SEE
is misaligned (2017 vs 2020) and a far pair (1988 vs 2020); (2) NCC before/after
for every epoch vs the reference; (3) full cross-season stack raw vs aligned with a
sharpness metric. If the rotation is real, the red/green splits go yellow, NCC
rises, and the aligned stack is sharper.
"""
import os, sys
import numpy as np
from scipy.ndimage import map_coordinates
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from venera.registration import tile_displacements, bandpass
from venera.rotation_fit import fit_tile_rotation

H, W = 4000, 8000
YRS = ["1988", "2001", "2012", "2015", "2017", "2020"]
REF = "2020"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "results", "epoch_stacks")
FIG = os.path.join(ROOT, "results", "figures")


def load(y):
    d = np.load(os.path.join(CACHE, f"stack_both_{y}_N20_cf.npz"))
    return d["Gm"].astype(np.float32), d["mask"]


def rot_matrix(omega):
    th = float(np.linalg.norm(omega))
    if th < 1e-12:
        return np.eye(3)
    k = omega / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


# half-res grid for the warp
h, w = H // 2, W // 2
_lat = np.radians((np.arange(h) + 0.5) / h * 180 - 90)
_lon = np.radians((np.arange(w) + 0.5) / w * 360 - 180)
_LON, _LAT = np.meshgrid(_lon, _lat)
R3 = np.stack([np.cos(_LAT) * np.cos(_LON), np.cos(_LAT) * np.sin(_LON),
               np.sin(_LAT)], 0).astype(np.float32)        # 3,h,w body dirs


def warp_into_ref(Gm2, mask2, omega):
    """Sample b at src = R(ω)·r so b lands in a/ref frame."""
    rs = np.einsum("ij,jhw->ihw", rot_matrix(omega).astype(np.float32), R3)
    si = (np.degrees(np.arcsin(np.clip(rs[2], -1, 1))) + 90) / 180 * h - 0.5
    sj = (np.degrees(np.arctan2(rs[1], rs[0])) + 180) / 360 * w - 0.5
    g = map_coordinates(Gm2, [si, sj], order=1, mode="constant", cval=0)
    m = map_coordinates(mask2.astype(np.float32), [si, sj], order=1, cval=0) > 0.5
    return g, m


def ncc(a, b, ma, mb):
    m = ma & mb
    if m.sum() < 2000:
        return np.nan
    fa, fb = bandpass(a, 4, 30)[m], bandpass(b, 4, 30)[m]
    fa, fb = fa - fa.mean(), fb - fb.mean()
    return float(np.sum(fa * fb) / (np.sqrt(np.sum(fa**2) * np.sum(fb**2)) + 1e-9))


# reference
Gr, Mr = load(REF)
Gr2, Mr2 = Gr[::2, ::2], Mr[::2, ::2]
omega = {REF: np.zeros(3)}
warped = {REF: (Gr2, Mr2)}
print(f"reference = {REF}")
print(f"{'epoch':6s} {'NCC raw':>8s} {'NCC aligned':>11s} {'Δ':>7s} {'ω deg (x,y,z)':>24s}")
for y in YRS:
    if y == REF:
        continue
    Gm, Mk = load(y)
    om, _ = fit_tile_rotation(tile_displacements(Gr, Mr, Gm, Mk))
    omega[y] = om
    Gm2, Mk2 = Gm[::2, ::2], Mk[::2, ::2]
    g, m = warp_into_ref(Gm2, Mk2, om)
    warped[y] = (g, m)
    c0n, c1n = ncc(Gr2, Gm2, Mr2, Mk2), ncc(Gr2, g, Mr2, m)
    print(f"{y:6s} {c0n:8.3f} {c1n:11.3f} {c1n-c0n:+7.3f} {str(np.round(np.degrees(om),2)):>24s}")

# ---- Maxwell-region overlays raw vs aligned ----
LAT0, LAT1, LON0, LON1 = 52, 74, -28, 28
rr0, rr1 = int((LAT0+90)/180*h), int((LAT1+90)/180*h)
cc0, cc1 = int((LON0+180)/360*w), int((LON1+180)/360*w)
EXT = [LON0, LON1, LAT0, LAT1]


def nz(g, m):
    f = bandpass(g, 3, 30); f[~m] = 0
    v = f[m]; lo, hi = np.percentile(v, 5), np.percentile(v, 99)
    return np.clip((f - lo)/(hi-lo+1e-9), 0, 1)


def overlay(ax, ga, ma, gb, mb, title):
    com = ma & mb
    A = nz(ga, ma); B = nz(gb, mb)
    rgb = np.zeros((rr1-rr0, cc1-cc0, 3))
    rgb[..., 0] = np.where(com, A, 0)[rr0:rr1, cc0:cc1]
    rgb[..., 1] = np.where(com, B, 0)[rr0:rr1, cc0:cc1]
    ax.imshow(rgb, origin="lower", extent=EXT, aspect="auto")
    ax.set_title(title, fontsize=9); ax.axhline(65, color="c", lw=0.3, alpha=0.5)


fig, axs = plt.subplots(2, 2, figsize=(12, 10))
for row, y in enumerate(["2017", "1988"]):
    g0, m0 = load(y); g0, m0 = g0[::2, ::2], m0[::2, ::2]
    overlay(axs[row, 0], Gr2, Mr2, g0, m0, f"RAW  R={REF} G={y}")
    gw, mw = warped[y]
    overlay(axs[row, 1], Gr2, Mr2, gw, mw, f"ROTATION-ALIGNED  R={REF} G={y}  (ω={np.round(np.degrees(omega[y]),2)})")
fig.suptitle("Does the fitted rotation align the maps? (Maxwell region, yellow=aligned)", fontsize=12)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "alignment_check.png"), dpi=130)
print(f"wrote {FIG}/alignment_check.png")

# ---- full stack raw vs aligned + sharpness ----
Ssum = np.zeros((h, w)); Scnt = np.zeros((h, w))
Asum = np.zeros((h, w)); Acnt = np.zeros((h, w))
for y in YRS:
    g0, m0 = load(y); g0, m0 = g0[::2, ::2], m0[::2, ::2]
    Ssum += np.where(m0, g0, 0); Scnt += m0
    gw, mw = warped[y]
    Asum += np.where(mw, gw, 0); Acnt += mw
raw = np.divide(Ssum, Scnt, out=np.zeros_like(Ssum), where=Scnt > 0)
ali = np.divide(Asum, Acnt, out=np.zeros_like(Asum), where=Acnt > 0)
shp = lambda im, c: float(np.std(bandpass(im, 3, 25)[c > 1]))
print(f"\nstack sharpness (hi-pass std):  raw={shp(raw,Scnt):.4f}  aligned={shp(ali,Acnt):.4f}")
fig, axs = plt.subplots(1, 2, figsize=(16, 5))
for ax, im, c, t in [(axs[0], raw, Scnt, "RAW average"), (axs[1], ali, Acnt, "ROTATION-ALIGNED")]:
    cr = im[rr0:rr1, cc0:cc1].copy(); cm = c[rr0:rr1, cc0:cc1] > 0; cr[~cm] = np.nan
    lo, hi = np.nanpercentile(cr, 2), np.nanpercentile(cr, 99.5)
    ax.imshow(cr, origin="lower", extent=EXT, cmap="gray", vmin=lo, vmax=hi, aspect="auto"); ax.set_title(t)
fig.suptitle("Maxwell region: 6-season stack, raw vs rotation-aligned", fontsize=12)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "stack_raw_vs_aligned.png"), dpi=130)
print(f"wrote {FIG}/stack_raw_vs_aligned.png")
