"""Quantify per-look position scatter on the bright Maxwell feature (where single
looks CAN lock), 2012 vs 2017. Project each N-pointing look, cross-correlate its
Maxwell crop against the session reference, record (dlon, dlat). Large scatter for
2012 vs 2017 => looks land inconsistently -> that is the stack blur (fixable via
centering); small/equal => blur is elsewhere.

Usage: .conda/bin/python scripts/maxwell_look_scatter.py
"""
import os, sys, glob
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = ("/home/than/code/venera/arecibo_radar/pds-geosciences.wustl.edu/venus/"
        "arcb_nrao-v-rtls_gbt-3-delaydoppler-v1/vrm_90xx/data/")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STACKS = os.path.join(ROOT, "results", "session_stacks")
H, W, DS = 4000, 8000, 2
HH, WW = H // DS, W // DS
LAT0, LAT1, LON0, LON1 = 52, 78, -30, 40
R0 = int((LAT0+90)/180*HH); R1 = int((LAT1+90)/180*HH)
C0 = int((LON0+180)/360*WW); C1 = int((LON1+180)/360*WW)
NLOOK = 25


def pick(year):
    from venera.data import parse_lbl
    fs = sorted(glob.glob(os.path.join(DATA, f"venus_scp_{year}*.img")))
    keep = [f for f in fs if parse_lbl(f).get("GEO_POINTING") == "N"]
    return [keep[i] for i in np.linspace(0, len(keep)-1, min(NLOOK, len(keep))).astype(int)]


def _init():
    from venera import spice_setup
    spice_setup.furnsh_kernels()


def proj(task):
    year, img = task
    from venera.geometry import Spin
    from venera.projection import project_file
    from venera.registration import register_maps, offset_to_lonlat_deg
    ref = _REF[year]
    G = np.zeros((H, W), np.float32); Gc = np.zeros((H, W), np.int32)
    try:
        project_file(img, Spin(), G, Gc)
    except Exception:
        return (year, None)
    half = (Gc[::DS, ::DS] > 0)
    crop = np.divide(G, Gc, out=np.zeros_like(G), where=Gc > 0)[::DS, ::DS][R0:R1, C0:C1]
    cm = half[R0:R1, C0:C1]
    if cm.mean() < 0.3:
        return (year, None)
    dr, dc, sig = register_maps(ref[0], crop, valid_a=ref[1], valid_b=cm,
                               max_shift=40, smooth_px=5.0, trend_px=40.0)
    dlat, dlon = offset_to_lonlat_deg(dr, dc, (HH, WW))
    return (year, (dlon, dlat, sig))


_REF = {}


def main():
    global _REF
    for y in ("2012", "2017"):
        d = np.load(os.path.join(STACKS, f"session_{y}.npz"))
        _REF[y] = (d["Gm"][R0:R1, C0:C1], d["mask"][R0:R1, C0:C1])
    looks = {y: pick(y) for y in ("2012", "2017")}
    tasks = [(y, f) for y in looks for f in looks[y]]
    res = {"2012": [], "2017": []}
    with Pool(10, initializer=_init) as p:
        for y, r in p.imap_unordered(proj, tasks):
            if r is not None:
                res[y].append(r)
            print(".", end="", flush=True)
    print()
    for y in ("2012", "2017"):
        a = np.array(res[y])
        ok = a[:, 2] > 1.05
        dl = a[ok, 0]; dla = a[ok, 1]
        print(f"{y}: {ok.sum()}/{len(a)} looks locked  "
              f"dlon scatter ±{dl.std():.3f}°  dlat scatter ±{dla.std():.3f}°  "
              f"(mean dlon {dl.mean():+.2f})", flush=True)


if __name__ == "__main__":
    main()
