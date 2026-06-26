"""Test the frc_hi gate against a boundary-free metric. Warp 2020 by raw /
period / period+tilt and compare frc_hi (current gate) to NCC over a FIXED interior
Maxwell box (no coverage edge). If frc_hi rises while interior-NCC falls, the gate
is gamed by the coverage boundary and the 'tilt helps' verdict is bogus.
"""
import os, sys, glob
import numpy as np
from scipy.ndimage import map_coordinates
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from venera.coherence import coherent_stack, frc
from venera.registration import bandpass

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

# fixed INTERIOR box on Maxwell (well inside both masks; no coverage edge)
LAT0, LAT1, LON0, LON1 = 57, 71, -13, 13
ir0, ir1 = int((LAT0+90)/180*hh), int((LAT1+90)/180*hh)
ic0, ic1 = int((LON0+180)/360*ww), int((LON1+180)/360*ww)
def interior_ncc(gw, mw):
    box = (slice(ir0, ir1), slice(ic0, ic1))
    a = bandpass(g12, 2, 25)[box]; b = bandpass(gw, 2, 25)[box]
    msk = m12[box] & mw[box]
    if msk.sum() < 500: return np.nan
    a, b = a[msk]-a[msk].mean(), b[msk]-b[msk].mean()
    return float(np.sum(a*b)/(np.sqrt(np.sum(a**2)*np.sum(b**2))+1e-9))

dz = dlam(243.016)
cases = {"raw": np.array([0,0,0.]),
         "period only": np.array([0,0,dz]),
         "period+tilt[0.5,-1.5]": np.array([np.radians(0.5), np.radians(-1.5), dz])}
print(f"{'case':24s} {'frc_hi (gate)':>14s} {'interior NCC (Maxwell)':>22s}")
for name, om in cases.items():
    gw, mw = warp(g20, m20, om)
    print(f"{name:24s} {fhi(g12, gw, m12, mw):14.3f} {interior_ncc(gw, mw):22.3f}", flush=True)
print("\nIf frc_hi rises with tilt but interior NCC falls -> the gate is boundary-gamed, verdict bogus.")
