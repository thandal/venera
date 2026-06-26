"""Test the delay-centering hypothesis: does a uniform delay-row offset null the
2015<->2017 tilt? A constant offset between the brightness-onset delay anchor
(data.coarse_rollup) and the true geometric sub-radar delay = a constant coeff_o
shift = a radial remap about the (off-equator) sub-radar point = a tilt ∝ srp_lat.

Reproject 2015 & 2017 with fit=(0, do, 1.0) for a range of do (rows) and measure the
on-sphere tilt + NCC@identity. If a single do drives tilt->0 and NCC up, the delay
anchor is the cause.

Usage: .conda/bin/python scripts/delay_test.py
"""
import os, sys, glob
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from register_on_sphere import flatten, search_R, decompose
from venera import spice_setup
from venera.geometry import Spin
from venera.projection import project_file

DATA = ("/home/than/code/venera/arecibo_radar/pds-geosciences.wustl.edu/venus/"
        "arcb_nrao-v-rtls_gbt-3-delaydoppler-v1/vrm_90xx/data/")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "results", "look_cache")
H, W, DS = 4000, 8000, 2
HH, WW = H // DS, W // DS
DOS = [10, 0, -10, -20, -30, -40]


def pick(year, n):
    cands = []
    for f in glob.glob(os.path.join(CACHE, f"venus_*_{year}*.npz")):
        d = np.load(f, allow_pickle=True)
        if str(d["pointing"]) == "N":
            cands.append((float(d["valid_frac"]), os.path.basename(f)[:-4]))
    cands.sort(reverse=True)
    return [os.path.join(DATA, b + ".img") for _, b in cands[:n]]


def _init():
    spice_setup.furnsh_kernels()


def proj_one(task):
    img, do, ep = task
    G = np.zeros((H, W), np.float32); Gc = np.zeros((H, W), np.int32)
    try:
        project_file(img, Spin(), G, Gc, fit=(0.0, float(do), 1.0))
    except Exception:
        return (do, ep, None)
    rows = np.where(np.any(Gc > 0, axis=1))[0]; cols = np.where(np.any(Gc > 0, axis=0))[0]
    if rows.size == 0:
        return (do, ep, None)
    r0, r1, c0, c1 = rows[0], rows[-1], cols[0], cols[-1]
    Gm = np.divide(G, Gc, out=np.zeros_like(G), where=Gc > 0)
    return (do, ep, (r0, c0, Gm[r0:r1+1:DS, c0:c1+1:DS].astype(np.float32),
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
    tasks = [(img, do, ep) for do in DOS for ep in ("2015", "2017")
             for img in looks[ep]]
    parts = {}
    with Pool(12, initializer=_init) as p:
        for do, ep, crop in p.imap_unordered(proj_one, tasks):
            if crop is not None:
                parts.setdefault((do, ep), []).append(crop)
            print(".", end="", flush=True)
    print()
    print("\n  delay_offset(rows)   2015<->2017 tilt   NCC@id   NCC@bestR")
    print("  " + "-" * 56)
    for do in DOS:
        A, Am = stack_into(parts[(do, "2015")])
        B, Bm = stack_into(parts[(do, "2017")])
        rv, nfit, nid = search_R(A, Am, B, Bm, coarse_step=1.5)
        print(f"  {do:+5d}                {decompose(rv)[2]:5.2f}°            "
              f"{nid:.3f}    {nfit:.3f}")
    print("\nIf tilt crosses ~0 at some do (and NCC@id peaks there), the delay anchor "
          "is the cause.")


if __name__ == "__main__":
    main()
