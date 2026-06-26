"""Does adding a tilt (pole correction) recover cross-epoch fine-scale coherence?
Longitude-only best vs +tilt best, for 2012 vs 2020 (N). Subpixel warp."""
import os, sys, glob
import numpy as np
from scipy.ndimage import map_coordinates
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from venera.coherence import frc, ncc

CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "look_cache")
H, W, DS = 4000, 8000, 2
hh, ww = H // DS, W // DS
NOM = 243.0185


def load(year):
    g = np.zeros((hh, ww)); c = np.zeros((hh, ww), int); days = []
    for f in sorted(glob.glob(f"{CACHE}/venus_*cp_{year}*.npz")):
        d = np.load(f, allow_pickle=True)
        if str(d["pointing"]) != "N":
            continue
        r0, _, c0, _ = d["bbox"]; gm, mk = d["gm"], d["mask"]; rr, cc = int(r0)//DS, int(c0)//DS
        gh, gw = gm.shape
        if rr+gh > hh or cc+gw > ww:
            gh, gw = min(gh, hh-rr), min(gw, ww-cc); gm, mk = gm[:gh, :gw], mk[:gh, :gw]
        g[rr:rr+gh, cc:cc+gw] += np.where(mk, gm, 0); c[rr:rr+gh, cc:cc+gw] += mk; days.append(float(d["day"]))
    return np.divide(g, c, out=np.zeros_like(g), where=c > 0), c > 0, float(np.mean(days))


g12, m12, d12 = load("2012"); g20, m20, d20 = load("2020"); dt = d20 - d12
print(f"loaded 2012 (day {d12:.0f}) and 2020 (day {d20:.0f}), Δt={dt/365.25:.2f}yr", flush=True)
lat = np.radians((np.arange(hh)+.5)/hh*180-90); lon = np.radians((np.arange(ww)+.5)/ww*360-180)
LON, LAT = np.meshgrid(lon, lat)
R3 = np.stack([np.cos(LAT)*np.cos(LON), np.cos(LAT)*np.sin(LON), np.sin(LAT)], 0).astype(np.float32)


def rotm(om):
    th = float(np.linalg.norm(om))
    if th < 1e-12: return np.eye(3)
    k = om/th; K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3)+np.sin(th)*K+(1-np.cos(th))*(K@K)


def warp(g, m, om):
    rs = np.einsum("ij,jhw->ihw", rotm(om).astype(np.float32), R3)
    si = (np.degrees(np.arcsin(np.clip(rs[2], -1, 1)))+90)/180*hh-.5
    sj = (np.degrees(np.arctan2(rs[1], rs[0]))+180)/360*ww-.5
    return map_coordinates(g, [si, sj], order=1, cval=0), map_coordinates(m.astype(np.float32), [si, sj], order=1, cval=0) > .5


def fhi(a, b, ma, mb):
    f, c = frc(a, b, ma & mb)
    return float(np.mean(c[f >= .15])) if (f.size and (f >= .15).any()) else np.nan


def dlam(P):
    return np.radians((-360./P - (-360./NOM)) * dt)


def score(om):
    gw, mw = warp(g20, m20, om)
    return fhi(g12, gw, m12, mw)


Ps = np.arange(243.008, 243.030, 0.002)
ls = [(P, score(np.array([0, 0, dlam(P)]))) for P in Ps]
Pb = max(ls, key=lambda x: x[1])
print(f"longitude-only best: P={Pb[0]:.4f}  frc_hi={Pb[1]:.3f}", flush=True)
dz = dlam(Pb[0]); best = (-1, None)
for tx in np.radians(np.linspace(-2, 2, 9)):
    for ty in np.radians(np.linspace(-2, 2, 9)):
        s = score(np.array([tx, ty, dz]))
        if s > best[0]:
            best = (s, (np.degrees(tx), np.degrees(ty)))
print(f"with tilt at best-P: frc_hi={best[0]:.3f}  tilt={np.round(best[1],2)}deg", flush=True)
gw, mw = warp(g20, m20, np.array([np.radians(best[1][0]), np.radians(best[1][1]), dz]))
print(f"broadband ncc (coarse features) at optimum = {ncc(g12, gw, m12, mw):.3f}", flush=True)
print("VERDICT:", "TILT recovers fine coherence -> pole error" if best[0] > Pb[1]+0.08
      else "tilt does NOT recover it -> non-rigid distortion (scale/higher-order)", flush=True)
