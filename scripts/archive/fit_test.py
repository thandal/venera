"""Last code-side suspect: is the per-image delay/doppler centering fit
(fit_delay_doppler_curve -> fo,do,fs) injecting the 2015<->2017 tilt?

Reproject 2015 & 2017 subsets two ways and compare the on-sphere tilt:
  - 'fitted'  : the per-image fit (nominal pipeline)
  - 'nofit'   : fixed fo=0, do=0, fs=1 (no per-image centering)
Tilt persists under 'nofit' => not the fit (real/geometry-degenerate effect).
Tilt vanishes under 'nofit' => the per-image fit is the source.

Usage: .conda/bin/python scripts/fit_test.py
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
    img, mode, ep = task
    fit = None if mode == "fitted" else (0.0, 0.0, 1.0)
    G = np.zeros((H, W), np.float32); Gc = np.zeros((H, W), np.int32)
    try:
        project_file(img, Spin(), G, Gc, fit=fit)
    except Exception:
        return (mode, ep, None)
    rows = np.where(np.any(Gc > 0, axis=1))[0]; cols = np.where(np.any(Gc > 0, axis=0))[0]
    if rows.size == 0:
        return (mode, ep, None)
    r0, r1, c0, c1 = rows[0], rows[-1], cols[0], cols[-1]
    Gm = np.divide(G, Gc, out=np.zeros_like(G), where=Gc > 0)
    return (mode, ep, (r0, c0, Gm[r0:r1+1:DS, c0:c1+1:DS].astype(np.float32),
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
    tasks = [(img, mode, ep) for mode in ("fitted", "nofit")
             for ep in ("2015", "2017") for img in looks[ep]]
    parts = {}
    with Pool(12, initializer=_init) as p:
        for mode, ep, crop in p.imap_unordered(proj_one, tasks):
            if crop is not None:
                parts.setdefault((mode, ep), []).append(crop)
            print(".", end="", flush=True)
    print()
    print("\n  mode     2015<->2017 tilt   NCC@id   NCC@bestR")
    print("  " + "-" * 46)
    for mode in ("fitted", "nofit"):
        A, Am = stack_into(parts[(mode, "2015")])
        B, Bm = stack_into(parts[(mode, "2017")])
        rv, nfit, nid = search_R(A, Am, B, Bm, coarse_step=1.5)
        print(f"  {mode:8s}    {decompose(rv)[2]:.2f}°            {nid:.3f}    {nfit:.3f}")
    print("\nTilt persists under 'nofit' => not the per-image fit.")


if __name__ == "__main__":
    main()
