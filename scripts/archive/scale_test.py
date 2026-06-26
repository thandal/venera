"""Test whether a DELAY-scale (radial) error or a DOPPLER-scale error causes the
2015<->2017 tilt. A wrong scale = a radial/azimuthal zoom about the sub-radar point;
differenced across off-equator epochs that is a tilt ~ Dsrp_lat.

We scale the delay axis by patching the radius used in the projection (affects both
the limb fit and the projection identically), and separately fix the doppler scale
fs, then measure the on-sphere 2015<->2017 tilt.

Usage: .conda/bin/python scripts/scale_test.py
"""
import os, sys, glob
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from register_on_sphere import flatten, search_R, decompose

DATA = ("/home/than/code/venera/arecibo_radar/pds-geosciences.wustl.edu/venus/"
        "arcb_nrao-v-rtls_gbt-3-delaydoppler-v1/vrm_90xx/data/")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "results", "look_cache")
H, W, DS = 4000, 8000, 2
HH, WW = H // DS, W // DS
# (delay_scale_factor, doppler_mode)  doppler_mode: None=fitted, or fixed fs value
CONFIGS = [
    ("delay x0.90", 0.90, None), ("delay x0.95", 0.95, None),
    ("nominal",     1.00, None), ("delay x1.05", 1.05, None),
    ("delay x1.10", 1.10, None),
]


def pick(year, n):
    cands = []
    for f in glob.glob(os.path.join(CACHE, f"venus_*_{year}*.npz")):
        d = np.load(f, allow_pickle=True)
        if str(d["pointing"]) == "N":
            cands.append((float(d["valid_frac"]), os.path.basename(f)[:-4]))
    cands.sort(reverse=True)
    return [os.path.join(DATA, b + ".img") for _, b in cands[:n]]


def _init():
    from venera import spice_setup
    spice_setup.furnsh_kernels()


def proj_one(task):
    img, scale, ep = task
    import venera.projection as proj
    from venera.geometry import Spin
    proj.VENUS_RADIUS_KM = 6051.8 * scale          # patch radial (delay) scale
    G = np.zeros((H, W), np.float32); Gc = np.zeros((H, W), np.int32)
    try:
        proj.project_file(img, Spin(), G, Gc)
    except Exception as e:
        return (scale, ep, None)
    rows = np.where(np.any(Gc > 0, axis=1))[0]; cols = np.where(np.any(Gc > 0, axis=0))[0]
    if rows.size == 0:
        return (scale, ep, None)
    r0, r1, c0, c1 = rows[0], rows[-1], cols[0], cols[-1]
    Gm = np.divide(G, Gc, out=np.zeros_like(G), where=Gc > 0)
    return (scale, ep, (r0, c0, Gm[r0:r1+1:DS, c0:c1+1:DS].astype(np.float32),
                        (Gc[r0:r1+1:DS, c0:c1+1:DS] > 0)))


def stack_into(parts):
    Gsum = np.zeros((HH, WW), np.float64); Gcnt = np.zeros((HH, WW), np.int32)
    for r0, c0, gm, m in parts:
        hr, hc = r0 // DS, c0 // DS; h, w = gm.shape
        if hr + h > HH or hc + w > WW:
            h = min(h, HH - hr); w = min(w, WW - hc); gm = gm[:h, :w]; m = m[:h, :w]
        Gsum[hr:hr+h, hc:hc+w][m] += gm[m]; Gcnt[hr:hr+h, hc:hc+w][m] += 1
    Gm = np.divide(Gsum, Gcnt, out=np.zeros_like(Gsum), where=Gcnt > 0).astype(np.float32)
    return flatten(Gm, Gcnt > 0), Gcnt > 0


def main():
    n = 12
    looks = {"2015": pick("2015", n), "2017": pick("2017", n)}
    scales = sorted(set(c[1] for c in CONFIGS))
    tasks = [(img, s, ep) for s in scales for ep in ("2015", "2017")
             for img in looks[ep]]
    parts = {}
    with Pool(12, initializer=_init) as p:
        for s, ep, crop in p.imap_unordered(proj_one, tasks):
            if crop is not None:
                parts.setdefault((s, ep), []).append(crop)
            print(".", end="", flush=True)
    print()
    print("\n  config         2015<->2017 tilt   NCC@id   NCC@bestR")
    print("  " + "-" * 50)
    for name, s, _ in CONFIGS:
        A, Am = stack_into(parts[(s, "2015")]); B, Bm = stack_into(parts[(s, "2017")])
        rv, nfit, nid = search_R(A, Am, B, Bm, coarse_step=1.5)
        print(f"  {name:12s}    {decompose(rv)[2]:5.2f}°            {nid:.3f}    {nfit:.3f}")
    print("\nIf tilt nulls at some delay scale, the radial/delay scale is the bug.")


if __name__ == "__main__":
    main()
