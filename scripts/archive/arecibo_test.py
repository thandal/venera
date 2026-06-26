"""THE test: is the cross-session tilt caused by using the geocenter ('EARTH') as
observer instead of Arecibo? The radar is at Arecibo, whose diurnal rotation
velocity (~0.46 km/s) tilts the doppler axis d_hat=do_hat/dt by a few percent
(~2 deg), epoch-dependently. The original notebook used Arecibo; the rewrite uses
the geocenter.

Reproject 2015 & 2017 two ways -- geocenter (current) vs Arecibo (topocentric,
includes Earth rotation) -- and compare the on-sphere 2015<->2017 tilt + NCC.
Tilt collapses with Arecibo  =>  FOUND IT.

Usage: .conda/bin/python scripts/arecibo_test.py
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
# Arecibo Observatory
ARE_LON = np.radians(-66.75278); ARE_LAT = np.radians(18.34417); ARE_ALT = 0.498  # km
RE, RP = 6378.1366, 6356.7519
FLAT = (RE - RP) / RE


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


def _emission_arecibo(et, observer, abcorr):
    """Venus->Arecibo unit vector (J2000) and emission epoch; topocentric observer
    including Earth rotation (the diurnal velocity enters via the finite difference
    in doppler_basis)."""
    import cspyce as csp
    ev, lt = csp.spkpos("VENUS", et, "J2000", abcorr, "EARTH")     # Earth ctr -> Venus
    ev = np.asarray(ev, float)
    are_bf = np.asarray(csp.georec(ARE_LON, ARE_LAT, ARE_ALT, RE, FLAT), float)
    M = np.asarray(csp.pxform("IAU_EARTH", "J2000", et), float)    # body-fixed -> J2000
    are_off = M @ are_bf                                           # Earth ctr -> Arecibo
    a2v = ev - are_off                                            # Arecibo -> Venus
    u = -a2v / np.linalg.norm(a2v)                               # Venus -> Arecibo
    return u, et - lt


def proj_one(task):
    img, mode, ep = task
    import venera.geometry as geo
    import venera.projection as proj
    from venera.geometry import Spin
    if mode == "arecibo":
        geo._emission_direction_j2000 = _emission_arecibo
    else:
        import importlib; importlib.reload(geo)   # restore geocenter
    G = np.zeros((H, W), np.float32); Gc = np.zeros((H, W), np.int32)
    try:
        proj.project_file(img, Spin(), G, Gc)
    except Exception as e:
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
    # run geocenter first (whole pool), then arecibo (separate pool so reload is clean)
    out = {}
    for mode in ("geocenter", "arecibo"):
        tasks = [(img, mode, ep) for ep in ("2015", "2017") for img in looks[ep]]
        parts = {}
        with Pool(8, initializer=_init) as p:
            for m, ep, crop in p.imap_unordered(proj_one, tasks):
                if crop is not None:
                    parts.setdefault(ep, []).append(crop)
                print(".", end="", flush=True)
        A, Am = stack_into(parts["2015"]); B, Bm = stack_into(parts["2017"])
        rv, nfit, nid = search_R(A, Am, B, Bm, coarse_step=1.0)
        out[mode] = (decompose(rv)[2], nid, nfit)
        print(f"\n  {mode}: 2015<->2017 tilt={decompose(rv)[2]:.2f}°  "
              f"NCC@id={nid:.3f}  NCC@bestR={nfit:.3f}", flush=True)
    print("\n=== RESULT ===")
    for mode in ("geocenter", "arecibo"):
        t, nid, nf = out[mode]
        print(f"  {mode:9s}: tilt={t:.2f}°  NCC@id={nid:.3f}  NCC@bestR={nf:.3f}")
    print("\nIf arecibo tilt << geocenter tilt and NCC@id jumps, the observer "
          "(geocenter vs Arecibo) is the bug.")


if __name__ == "__main__":
    main()
