"""Visual sanity-check of the cross-epoch alignment claims. Red/green overlays
(yellow = aligned), each annotated with its measured FRC high-band:
  1. within-epoch control: 2012 half-A vs half-B (same geometry -> should be yellow)
  2. cross-epoch RAW: 2012 vs 2020 (no alignment)
  3. cross-epoch + period (longitude roll only)
  4. cross-epoch + period + tilt (best rigid rotation)
If panel 4 is visibly more yellow than 2/3, the tilt claim holds; residual red/green
fringing in 4 is the non-rigid distortion.
"""
import os, sys, glob
import numpy as np
from scipy.ndimage import map_coordinates
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from venera.coherence import coherent_stack, frc
from venera.registration import bandpass

CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "look_cache")
FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "figures")
H, W, DS = 4000, 8000, 2
hh, ww = H // DS, W // DS
NOM = 243.0185


def looks(year):
    out = []
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
        out.append((g, m, float(d["day"])))
    return out


l12, l20 = looks("2012"), looks("2020")
g12, c12 = coherent_stack([x[0] for x in l12], [x[1] for x in l12]); m12 = c12 > 0
g20, c20 = coherent_stack([x[0] for x in l20], [x[1] for x in l20]); m20 = c20 > 0
ga, ca = coherent_stack([x[0] for x in l12[0::2]], [x[1] for x in l12[0::2]])
gb, cb = coherent_stack([x[0] for x in l12[1::2]], [x[1] for x in l12[1::2]])
dt = np.mean([x[2] for x in l20]) - np.mean([x[2] for x in l12])

lat = np.radians((np.arange(hh)+.5)/hh*180-90); lon = np.radians((np.arange(ww)+.5)/ww*360-180)
LON, LAT = np.meshgrid(lon, lat)
R3 = np.stack([np.cos(LAT)*np.cos(LON), np.cos(LAT)*np.sin(LON), np.sin(LAT)], 0).astype(np.float32)
def rotm(om):
    th = float(np.linalg.norm(om))
    if th < 1e-12: return np.eye(3)
    k = om/th; K = np.array([[0,-k[2],k[1]],[k[2],0,-k[0]],[-k[1],k[0],0]])
    return np.eye(3)+np.sin(th)*K+(1-np.cos(th))*(K@K)
def warp(g, m, om):
    rs = np.einsum("ij,jhw->ihw", rotm(om).astype(np.float32), R3)
    si = (np.degrees(np.arcsin(np.clip(rs[2],-1,1)))+90)/180*hh-.5; sj = (np.degrees(np.arctan2(rs[1],rs[0]))+180)/360*ww-.5
    return map_coordinates(g,[si,sj],order=1,cval=0), map_coordinates(m.astype(np.float32),[si,sj],order=1,cval=0) > .5
def fhi(a, b, ma, mb):
    f, c = frc(a, b, ma & mb)
    return float(np.mean(c[f >= .15])) if (f.size and (f >= .15).any()) else np.nan
def dlam(P):
    return np.radians((-360./P - (-360./NOM))*dt)

# best period (longitude) then best tilt at that period (wider, to converge)
def score_lon(P):
    gw, mw = warp(g20, m20, np.array([0, 0, dlam(P)])); return fhi(g12, gw, m12, mw)
Ps = np.arange(243.008, 243.030, 0.002)
Pbest = max(Ps, key=score_lon)
dz = dlam(Pbest); best = (-1, (0,0))
for tx in np.radians(np.linspace(-3,3,13)):
    for ty in np.radians(np.linspace(-3,3,13)):
        gw, mw = warp(g20, m20, np.array([tx,ty,dz])); s = fhi(g12, gw, m12, mw)
        if s > best[0]: best = (s, (np.degrees(tx), np.degrees(ty)))
tx, ty = np.radians(best[1][0]), np.radians(best[1][1])
print(f"Pbest={Pbest:.4f}  tilt={np.round(best[1],2)}  frc_hi tilt={best[0]:.3f}", flush=True)

g20_lon, m20_lon = warp(g20, m20, np.array([0,0,dz]))
g20_tilt, m20_tilt = warp(g20, m20, np.array([tx,ty,dz]))

LAT0, LAT1, LON0, LON1 = 52, 74, -28, 28
rr0, rr1 = int((LAT0+90)/180*hh), int((LAT1+90)/180*hh)
cc0, cc1 = int((LON0+180)/360*ww), int((LON1+180)/360*ww)
EXT = [LON0, LON1, LAT0, LAT1]
def nz(g, m):
    f = bandpass(g, 2, 25); f[~m] = 0; v = f[m]
    lo, hi = np.percentile(v, 5), np.percentile(v, 99); return np.clip((f-lo)/(hi-lo+1e-9), 0, 1)
def panel(ax, gr, mr, gg, mg, title, score):
    com = mr & mg; A, B = nz(gr, mr), nz(gg, mg)
    rgb = np.zeros((rr1-rr0, cc1-cc0, 3))
    rgb[..., 0] = np.where(com, A, 0)[rr0:rr1, cc0:cc1]; rgb[..., 1] = np.where(com, B, 0)[rr0:rr1, cc0:cc1]
    ax.imshow(rgb, origin="lower", extent=EXT, aspect="auto")
    ax.set_title(f"{title}\nfrc_hi={score:.3f}", fontsize=10); ax.set_xlabel("lon E"); ax.set_ylabel("lat N")

fig, axs = plt.subplots(1, 4, figsize=(20, 5.2))
panel(axs[0], ga, ca > 0, gb, cb > 0, "WITHIN 2012: halfA(R) vs halfB(G)\nsame geometry", fhi(ga, gb, ca>0, cb>0))
panel(axs[1], g12, m12, g20, m20, "CROSS raw: 2012(R) vs 2020(G)", fhi(g12, g20, m12, m20))
panel(axs[2], g12, m12, g20_lon, m20_lon, f"CROSS +period (P={Pbest:.4f})", fhi(g12, g20_lon, m12, m20_lon))
panel(axs[3], g12, m12, g20_tilt, m20_tilt, f"CROSS +period+tilt {np.round(best[1],1)}deg", best[0])
fig.suptitle("Cross-epoch alignment sanity check (yellow=aligned; number = measured FRC high-band)", fontsize=13)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "verify_alignment.png"), dpi=125)
print(f"wrote {FIG}/verify_alignment.png", flush=True)
