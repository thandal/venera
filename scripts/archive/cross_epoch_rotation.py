"""Distinguish the two causes of low cross-epoch coherence: residual TILT vs
fundamental radar APPEARANCE decorrelation.

Search a full sphere rotation (longitude = period, + tilt = pole) applied to the
2020 stack to MAXIMIZE cross-epoch FRC high-band vs 2012 (subpixel warp, no pixel
quantization). If frc_hi jumps well above the longitude-only 0.108, the tilt was
the problem and this (imagery-validated) rotation search is the method. If it
stays ~0.1, cross-epoch fine-scale coherence is appearance-limited.
"""
import os, sys, glob
import numpy as np
from scipy.ndimage import map_coordinates

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from venera.coherence import frc, ncc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "results", "look_cache")
H, W, DS = 4000, 8000, 2
hh, ww = H // DS, W // DS
NOMINAL_P = 243.0185


def load_epoch(year, pointing="N"):
    g = np.zeros((hh, ww), np.float64); c = np.zeros((hh, ww), np.int32); days = []
    for f in sorted(glob.glob(os.path.join(CACHE, f"venus_*cp_{year}*.npz"))):
        d = np.load(f, allow_pickle=True)
        if str(d["pointing"]) != pointing:
            continue
        r0, _, c0, _ = d["bbox"]; gm, mk = d["gm"], d["mask"]
        rr, cc = int(r0)//DS, int(c0)//DS; gh, gw = gm.shape
        if rr+gh > hh or cc+gw > ww:
            gh, gw = min(gh, hh-rr), min(gw, ww-cc); gm, mk = gm[:gh, :gw], mk[:gh, :gw]
        g[rr:rr+gh, cc:cc+gw] += np.where(mk, gm, 0); c[rr:rr+gh, cc:cc+gw] += mk
        days.append(float(d["day"]))
    return np.divide(g, c, out=np.zeros_like(g), where=c > 0), c > 0, float(np.mean(days))


_lat = np.radians((np.arange(hh)+0.5)/hh*180-90); _lon = np.radians((np.arange(ww)+0.5)/ww*360-180)
_LON, _LAT = np.meshgrid(_lon, _lat)
R3 = np.stack([np.cos(_LAT)*np.cos(_LON), np.cos(_LAT)*np.sin(_LON), np.sin(_LAT)], 0).astype(np.float32)
def rotm(om):
    th = float(np.linalg.norm(om))
    if th < 1e-12: return np.eye(3)
    k = om/th; K = np.array([[0,-k[2],k[1]],[k[2],0,-k[0]],[-k[1],k[0],0]])
    return np.eye(3)+np.sin(th)*K+(1-np.cos(th))*(K@K)
def warp(g, m, om):
    rs = np.einsum("ij,jhw->ihw", rotm(om).astype(np.float32), R3)
    si = (np.degrees(np.arcsin(np.clip(rs[2],-1,1)))+90)/180*hh-0.5
    sj = (np.degrees(np.arctan2(rs[1],rs[0]))+180)/360*ww-0.5
    return (map_coordinates(g, [si,sj], order=1, cval=0),
            map_coordinates(m.astype(np.float32), [si,sj], order=1, cval=0) > 0.5)
def fhi(a, b, ma, mb):
    f, c = frc(a, b, ma & mb)
    return float(np.mean(c[f >= 0.15])) if (f.size and (f >= 0.15).any()) else np.nan


def main():
    g12, m12, d12 = load_epoch("2012")
    g20, m20, d20 = load_epoch("2020")
    dt = d20 - d12

    def period_to_dlam(P):
        return np.radians((-360.0/P - (-360.0/NOMINAL_P)) * dt)   # ω_z (rad)

    base = fhi(g12, g20, m12, m20)
    print(f"baseline frc_hi (no shift)          = {base:.3f}")
    # longitude-only optimum (period), subpixel
    Ps = np.arange(243.005, 243.035, 0.0015)
    lon_scores = []
    for P in Ps:
        gw, mw = warp(g20, m20, np.array([0., 0., period_to_dlam(P)]))
        lon_scores.append(fhi(g12, gw, m12, mw))
    kl = int(np.nanargmax(lon_scores))
    print(f"longitude-only best: P={Ps[kl]:.4f}  frc_hi={lon_scores[kl]:.3f}")

    # joint (period + tilt) search
    best = (-1, None)
    txs = np.radians(np.linspace(-1.5, 1.5, 7)); tys = np.radians(np.linspace(-1.5, 1.5, 7))
    for P in Ps:
        dz = period_to_dlam(P)
        for tx in txs:
            for ty in tys:
                gw, mw = warp(g20, m20, np.array([tx, ty, dz]))
                s = fhi(g12, gw, m12, mw)
                if s > best[0]:
                    best = (s, (P, np.degrees(tx), np.degrees(ty)))
    s, (P, tx, ty) = best
    print(f"joint best: P={P:.4f}  tilt=({tx:+.2f},{ty:+.2f})deg  frc_hi={s:.3f}")
    print(f"\nlongitude-only {lon_scores[kl]:.3f}  ->  +tilt {s:.3f}   "
          f"({'TILT recovers coherence -> rotation search is the method' if s > lon_scores[kl]+0.08 else 'tilt does NOT help -> appearance-limited floor'})")
    # broadband (coarse) coherence for reference
    gw, mw = warp(g20, m20, np.array([np.radians(tx), np.radians(ty), period_to_dlam(P)]))
    print(f"broadband ncc at joint optimum = {ncc(g12, gw, m12, mw):.3f}  (coarse-feature alignment)")


if __name__ == "__main__":
    main()
