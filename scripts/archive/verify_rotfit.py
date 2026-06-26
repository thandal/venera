"""Is the rotation fit trustworthy? Three checks:
 1. ORDER consistency: ω(a,b) should equal -ω(b,a). If not, the fit is unstable.
 2. CONDITIONING: condition number of the LSQ design matrix; how well is ω_z
    (pole rotation = the period signal) separated from the tilts given the coverage?
 3. SEARCH: does the NCC-maximizing rotation for the visibly-misaligned 2017<->2020
    pair actually beat raw NCC, and does its ω match the tile-fit ω?
"""
import os, sys
import numpy as np
from scipy.ndimage import map_coordinates
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from venera.registration import tile_displacements, bandpass
from venera.rotation_fit import fit_tile_rotation

H, W = 4000, 8000
CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "results", "epoch_stacks")
def load(y):
    d = np.load(os.path.join(CACHE, f"stack_both_{y}_N20_cf.npz"))
    return d["Gm"].astype(np.float32), d["mask"]

def design(tiles):
    A = []
    for lat, lon, dlat, dlon in tiles:
        la, lo = np.radians(lat), np.radians(lon)
        r = np.array([np.cos(la)*np.cos(lo), np.cos(la)*np.sin(lo), np.sin(la)])
        e = np.array([-np.sin(lo), np.cos(lo), 0.0])
        n = np.array([-np.sin(la)*np.cos(lo), -np.sin(la)*np.sin(lo), np.cos(la)])
        A.append(np.cross(r, n)); A.append(np.cross(r, e))
    return np.asarray(A)

G = {y: load(y) for y in ["1988", "2012", "2017", "2020"]}

print("== 1. ORDER consistency:  ω(a,b) vs -ω(b,a) ==")
for a, b in [("2012", "2020"), ("2017", "2020"), ("1988", "2020")]:
    oab, _ = fit_tile_rotation(tile_displacements(G[a][0], G[a][1], G[b][0], G[b][1]))
    oba, _ = fit_tile_rotation(tile_displacements(G[b][0], G[b][1], G[a][0], G[a][1]))
    print(f"  {a},{b}: ω(a,b)={np.round(np.degrees(oab),3)}   -ω(b,a)={np.round(-np.degrees(oba),3)}"
          f"   mismatch={np.degrees(np.linalg.norm(oab+oba)):.3f}deg")

print("\n== 2. CONDITIONING of the ω fit (singular values; ratio = how degenerate) ==")
for a, b in [("2012", "2020"), ("2017", "2020")]:
    tl = tile_displacements(G[a][0], G[a][1], G[b][0], G[b][1])
    A = design(tl)
    sv = np.linalg.svd(A, compute_uv=False)
    # which axis does the weakest singular vector load on? (the poorly-constrained combo)
    _, _, Vt = np.linalg.svd(A, full_matrices=False)
    print(f"  {a},{b}: n_tiles={len(tl)}  singular values={np.round(sv,1)}  cond={sv[0]/sv[-1]:.1f}")
    print(f"          weakest direction (x,y,z loadings)={np.round(Vt[-1],2)}  "
          f"-> |z-load|={abs(Vt[-1][2]):.2f} (high = ω_z poorly constrained)")

print("\n== 3. SEARCH best rotation for 2017<->2020 (does ANY rotation beat raw?) ==")
h, w = H//2, W//2
_lat = np.radians((np.arange(h)+0.5)/h*180-90); _lon = np.radians((np.arange(w)+0.5)/w*360-180)
_LON, _LAT = np.meshgrid(_lon, _lat)
R3 = np.stack([np.cos(_LAT)*np.cos(_LON), np.cos(_LAT)*np.sin(_LON), np.sin(_LAT)], 0).astype(np.float32)
def rotm(om):
    th = float(np.linalg.norm(om))
    if th < 1e-12: return np.eye(3)
    k = om/th; K = np.array([[0,-k[2],k[1]],[k[2],0,-k[0]],[-k[1],k[0],0]])
    return np.eye(3)+np.sin(th)*K+(1-np.cos(th))*(K@K)
def warp(g2, m2, om):
    rs = np.einsum("ij,jhw->ihw", rotm(om).astype(np.float32), R3)
    si = (np.degrees(np.arcsin(np.clip(rs[2],-1,1)))+90)/180*h-0.5
    sj = (np.degrees(np.arctan2(rs[1],rs[0]))+180)/360*w-0.5
    return (map_coordinates(g2,[si,sj],order=1,cval=0),
            map_coordinates(m2.astype(np.float32),[si,sj],order=1,cval=0)>0.5)
def ncc(a,b,ma,mb):
    m = ma&mb
    if m.sum()<2000: return -1.0
    fa,fb = bandpass(a,4,30)[m], bandpass(b,4,30)[m]; fa-=fa.mean(); fb-=fb.mean()
    return float(np.sum(fa*fb)/(np.sqrt(np.sum(fa**2)*np.sum(fb**2))+1e-9))
gr,mr = G["2020"][0][::2,::2], G["2020"][1][::2,::2]
g0,m0 = G["2017"][0][::2,::2], G["2017"][1][::2,::2]
raw = ncc(gr,g0,mr,m0)
tlfit,_ = fit_tile_rotation(tile_displacements(G["2020"][0],G["2020"][1],G["2017"][0],G["2017"][1]))
gw,mw = warp(g0,m0,tlfit); ncc_tlfit = ncc(gr,gw,mr,mw)
best = (raw, np.zeros(3))
deg = np.radians(1.0)
for ox in np.linspace(-2,2,9):
    for oy in np.linspace(-2,2,9):
        for oz in np.linspace(-2,2,9):
            om = np.radians([ox,oy,oz])
            gw,mw = warp(g0,m0,om); c = ncc(gr,gw,mr,mw)
            if c>best[0]: best=(c,om)
print(f"  raw NCC                = {raw:.3f}")
print(f"  tile-fit ω NCC         = {ncc_tlfit:.3f}   (ω={np.round(np.degrees(tlfit),2)})")
print(f"  best searched-ω NCC    = {best[0]:.3f}   (ω={np.round(np.degrees(best[1]),2)})")
print(f"  -> {'rotation HELPS' if best[0]>raw+0.02 else 'rotation does NOT meaningfully help'};"
      f" tile-fit {'matches' if np.degrees(np.linalg.norm(best[1]-tlfit))<0.8 else 'DISAGREES with'} the search optimum")
