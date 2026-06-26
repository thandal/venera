"""Register two body-fixed session maps ON THE SPHERE: find the single 3-D rotation
R that maximizes on-sphere NCC between them.

Equirectangular maps are a planar projection of a sphere -- high-latitude longitude
is stretched by 1/cos(lat), so a 2-D translation cannot represent a sphere rotation.
The maps are already in body-fixed lon/lat, so applying R is just resampling each
direction through R (no reprojection of raw data needed). NCC is evaluated on the
sphere (area-weighted by cos lat).

Verdict:
  - one R collapses the residual (on-sphere NCC jumps to the local feature level)
    => the cross-session mismatch IS a rigid rotation (rotation-elements / a
    projection-rotation error), and R measures it; OR
  - a structured residual survives the best R => genuine non-rigid distortion.

R is decomposed into a POLAR component (about the +z spin pole -> period/W0-like)
and a TILT component (-> pole-error-like). The mean doppler-angle difference between
the sessions is printed to test whether a per-epoch projection rotation explains R.

Self-validated: rotate a map by a known R0 and confirm the search recovers it.

Usage: .conda/bin/python scripts/register_on_sphere.py [yearA] [yearB]   (default 2015 2017)
"""
import os, sys, glob
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STACKS = os.path.join(ROOT, "results", "session_stacks")
CACHE = os.path.join(ROOT, "results", "look_cache")
FIG = os.path.join(ROOT, "results", "figures")
HH, WW = 2000, 4000
DEG = np.pi / 180.0


# ---------- sphere <-> map helpers ----------
def ll2v(lon, lat):
    cl = np.cos(lat)
    return np.stack([cl * np.cos(lon), cl * np.sin(lon), np.sin(lat)], axis=-1)


def v2ll(V):
    lon = np.arctan2(V[..., 1], V[..., 0])
    lat = np.arcsin(np.clip(V[..., 2], -1, 1))
    return lon, lat


def sample(M, mask, lon, lat):
    r = (lat + np.pi / 2) / np.pi * HH
    c = (lon + np.pi) / (2 * np.pi) * WW
    vals = map_coordinates(M, [r, c], order=1, mode="nearest")
    mv = map_coordinates(mask.astype(np.float32), [r, c], order=1, mode="nearest")
    return vals, mv > 0.999


def flatten(M, mask, sigma=35):
    filled = np.where(mask, M, 0.0)
    wsm = gaussian_filter(mask.astype(float), sigma)
    bg = gaussian_filter(filled, sigma) / np.maximum(wsm, 1e-6)
    f = M - bg
    f[~mask] = 0.0
    return f


def wncc(a, b, w):
    wsum = w.sum()
    wa = a - (w * a).sum() / wsum
    wb = b - (w * b).sum() / wsum
    den = np.sqrt((w * wa * wa).sum() * (w * wb * wb).sum())
    return float((w * wa * wb).sum() / den) if den > 0 else 0.0


def rotate_map(M, mask, R):
    """Return M' with M'(p) = M(R^T p): M rotated by R on the sphere."""
    lon = (np.arange(WW) + 0.5) / WW * 2 * np.pi - np.pi
    lat = (np.arange(HH) + 0.5) / HH * np.pi - np.pi / 2
    LON, LAT = np.meshgrid(lon, lat)
    V = ll2v(LON, LAT)                       # (HH,WW,3)
    Vs = V @ R                               # R^T p  (row-vector convention)
    lo, la = v2ll(Vs)
    vals, val = sample(M, mask, lo.ravel(), la.ravel())
    return vals.reshape(HH, WW), val.reshape(HH, WW)


# ---------- the search ----------
def make_eval_points(Avalid, n_max=30000, seed=0):
    ys, xs = np.where(Avalid)
    if ys.size > n_max:
        idx = np.linspace(0, ys.size - 1, n_max).astype(int)
        ys, xs = ys[idx], xs[idx]
    lat = (ys + 0.5) / HH * np.pi - np.pi / 2
    lon = (xs + 0.5) / WW * 2 * np.pi - np.pi
    P = ll2v(lon, lat)
    w = np.cos(lat)
    return P, w


def search_R(Aflat, Amask, Bflat, Bmask, coarse_deg=6.0, coarse_step=1.0):
    P, w = make_eval_points(Amask)
    Av, Aok = sample(Aflat, Amask, *v2ll(P))

    def negncc(rotvec):
        R = Rotation.from_rotvec(rotvec).as_matrix()
        Vr = P @ R.T
        lo, la = v2ll(Vr)
        Bv, Bok = sample(Bflat, Bmask, lo, la)
        m = Aok & Bok
        if m.sum() < 1000:
            return 0.0
        return -wncc(Av[m], Bv[m], w[m])

    # coarse 3-D grid
    g = np.arange(-coarse_deg, coarse_deg + 1e-9, coarse_step) * DEG
    best, bestv = np.zeros(3), 0.0
    for rx in g:
        for ry in g:
            for rz in g:
                v = negncc(np.array([rx, ry, rz]))
                if v < bestv:
                    bestv, best = v, np.array([rx, ry, rz])
    # refine
    res = minimize(negncc, best, method="Nelder-Mead",
                   options=dict(xatol=1e-4, fatol=1e-5, maxiter=2000))
    rv = res.x if -res.fun >= -bestv else best
    return rv, -negncc(rv), -negncc(np.zeros(3))


def decompose(rotvec):
    polar = np.degrees(rotvec[2])                       # about +z = spin pole
    tilt = np.degrees(np.hypot(rotvec[0], rotvec[1]))   # pole-error direction
    total = np.degrees(np.linalg.norm(rotvec))
    return total, polar, tilt


def mean_doppler(year):
    vals = []
    for f in glob.glob(os.path.join(CACHE, f"venus_*_{year}*.npz")):
        vals.append(float(np.load(f, allow_pickle=True)["doppler_angle"]))
    return np.mean(vals) if vals else np.nan


# ---------- rendering ----------
def overlay(Aflat, Amask, Bflat, Bmask, title, path, box=None):
    cm = Amask & Bmask
    if box:
        r0, r1, c0, c1 = box
    else:
        rows = np.where(cm.any(1))[0]; cols = np.where(cm.any(0))[0]
        r0, r1, c0, c1 = rows[0], rows[-1], cols[0], cols[-1]
    A = Aflat[r0:r1, c0:c1]; B = Bflat[r0:r1, c0:c1]; m = cm[r0:r1, c0:c1]

    def norm(x):
        v = x[m]
        if v.size == 0:
            return np.zeros_like(x)
        lo, hi = np.percentile(v, 2), np.percentile(v, 98)
        return np.clip((x - lo) / (hi - lo + 1e-9), 0, 1)
    rgb = np.zeros(A.shape + (3,))
    rgb[..., 0] = norm(A) * m; rgb[..., 1] = norm(B) * m
    ext = [c0 / WW * 360 - 180, c1 / WW * 360 - 180,
           r0 / HH * 180 - 90, r1 / HH * 180 - 90]
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.imshow(rgb, origin="lower", extent=ext, aspect="auto")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("longitude (°E)"); ax.set_ylabel("latitude (°N)")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def maxwell_box():
    r0 = int((50 + 90) / 180 * HH); r1 = int((80 + 90) / 180 * HH)
    c0 = int((-40 + 180) / 360 * WW); c1 = int((45 + 180) / 360 * WW)
    return (r0, r1, c0, c1)


def main():
    yA = sys.argv[1] if len(sys.argv) > 1 else "2015"
    yB = sys.argv[2] if len(sys.argv) > 2 else "2017"
    dA = np.load(os.path.join(STACKS, f"session_{yA}.npz"))
    dB = np.load(os.path.join(STACKS, f"session_{yB}.npz"))
    Af = flatten(dA["Gm"], dA["mask"]); Am = dA["mask"]
    Bf = flatten(dB["Gm"], dB["mask"]); Bm = dB["mask"]

    # ---- ground-truth self-validation ----
    R0 = Rotation.from_rotvec(np.array([1.0, -1.5, 2.0]) * DEG)
    Bgt, Bgtm = rotate_map(Af, Am, R0.as_matrix())
    rv, nfit, nid = search_R(Af, Am, Bgt, Bgtm)
    err = np.degrees(np.linalg.norm(
        (Rotation.from_rotvec(rv) * R0.inv()).as_rotvec()))
    print("== ground-truth self-test ==")
    print(f"  injected R0 = {np.round(np.array([1.0,-1.5,2.0]),2)} deg (axis-angle)")
    print(f"  recovered   = {np.round(np.degrees(rv),2)} deg   |error| = {err:.3f} deg")
    print(f"  NCC: identity {nid:.3f} -> recovered {nfit:.3f}   "
          f"({'PASS' if err < 0.2 and nfit > 0.95 else 'FAIL'})")

    # ---- real data ----
    print(f"\n== {yA} <-> {yB} on the sphere ==")
    rv, nfit, nid = search_R(Af, Am, Bf, Bm)
    total, polar, tilt = decompose(rv)
    print(f"  on-sphere NCC: identity {nid:.3f} -> best-R {nfit:.3f}")
    print(f"  best R: total {total:.2f} deg  |  polar(about pole) {polar:+.2f} deg  "
          f"|  tilt(pole-error) {tilt:.2f} deg")
    print(f"  mean doppler angle: {yA}={mean_doppler(yA):+.1f} deg  "
          f"{yB}={mean_doppler(yB):+.1f} deg  (diff "
          f"{mean_doppler(yA)-mean_doppler(yB):+.1f} deg)")

    # Maxwell-region NCC before/after
    R = Rotation.from_rotvec(rv).as_matrix()
    Br, Brm = rotate_map(Bf, Bm, R.T)        # bring B into A frame
    box = maxwell_box()
    overlay(Af, Am, Bf, Bm, f"{yA}(red) vs {yB}(green) — IDENTITY (NCC {nid:.2f})",
            os.path.join(FIG, f"sphere_{yA}_{yB}_identity.png"))
    overlay(Af, Am, Br, Brm,
            f"{yA}(red) vs {yB}(green) — best R ({total:.1f}°: polar {polar:+.1f}°, "
            f"tilt {tilt:.1f}°)  NCC {nfit:.2f}",
            os.path.join(FIG, f"sphere_{yA}_{yB}_bestR.png"))
    overlay(Af, Am, Br, Brm,
            f"Maxwell after best R — {yA}(red) vs {yB}(green)  NCC {nfit:.2f}",
            os.path.join(FIG, f"sphere_{yA}_{yB}_bestR_maxwell.png"), box=box)
    print(f"  wrote sphere_{yA}_{yB}_identity.png, _bestR.png, _bestR_maxwell.png")


if __name__ == "__main__":
    main()
