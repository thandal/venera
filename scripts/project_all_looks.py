"""Project ALL Arecibo Venus looks into the body frame, once, and cache each as a
cropped half-res map. **No subsetting** — every epoch, both pols, N and S pointings
(916 looks total). This is the heavy step behind the per-session stacks
(scripts/stack_sessions.py).

Parallel across looks; skips looks already cached. ~24 s/look single-threaded, so
~25-35 min on ~12 workers. Reuses the validated venera.projection pipeline and the
same cache format as build_look_cache.py.

Usage: project_all_looks.py [n_workers]      (default 12)
"""
import os, sys, glob, time
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from venera import spice_setup
from venera.geometry import Spin, sub_radar_point, doppler_angle
from venera.data import parse_lbl
from venera.projection import project_file

DATA = ("/home/than/code/venera/arecibo_radar/pds-geosciences.wustl.edu/venus/"
        "arcb_nrao-v-rtls_gbt-3-delaydoppler-v1/vrm_90xx/data/")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "results", "look_cache")
os.makedirs(CACHE, exist_ok=True)
H, W = 4000, 8000
DS = 2


def _worker_init():
    """Furnish SPICE kernels once per worker process (after fork)."""
    spice_setup.furnsh_kernels()


def cache_one(f):
    base = os.path.basename(f)[:-4]
    out = os.path.join(CACHE, f"{base}.npz")
    if os.path.exists(out):
        return (base, "skip", 0.0)
    t = time.time()
    try:
        spin = Spin()
        G = np.zeros((H, W), np.float32); Gc = np.zeros((H, W), np.int32)
        info = project_file(f, spin, G, Gc)
        et = info["et_mid"]
        lon, lat, _ = sub_radar_point(et, spin)
        rows = np.where(np.any(Gc > 0, axis=1))[0]
        cols = np.where(np.any(Gc > 0, axis=0))[0]
        if rows.size == 0:
            return (base, "EMPTY", time.time() - t)
        r0, r1, c0, c1 = rows[0], rows[-1], cols[0], cols[-1]
        Gm = np.divide(G, Gc, out=np.zeros_like(G), where=Gc > 0)
        np.savez_compressed(
            out,
            gm=Gm[r0:r1+1:DS, c0:c1+1:DS].astype(np.float32),
            mask=(Gc[r0:r1+1:DS, c0:c1+1:DS] > 0),
            bbox=np.array([r0, r1, c0, c1]), ds=DS, hw=np.array([H, W]),
            et_mid=et, day=et / 86400.0,
            srp_lon=np.degrees(lon), srp_lat=np.degrees(lat),
            doppler_angle=np.degrees(doppler_angle(et, spin)),
            pol=("scp" if "_scp_" in base else "ocp"),
            pointing=parse_lbl(f).get("GEO_POINTING", "?"),
            valid_frac=info["valid_frac"])
        return (base, f"valid={info['valid_frac']:.2f}", time.time() - t)
    except Exception as e:
        return (base, f"ERR {type(e).__name__}: {e}", time.time() - t)


def main():
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    files = sorted(glob.glob(os.path.join(DATA, "venus_scp_*.img"))
                   + glob.glob(os.path.join(DATA, "venus_ocp_*.img")))
    todo = [f for f in files
            if not os.path.exists(os.path.join(CACHE, os.path.basename(f)[:-4] + ".npz"))]
    print(f"total={len(files)} cached={len(files)-len(todo)} todo={len(todo)} "
          f"workers={nproc}", flush=True)
    t0 = time.time(); done = 0; errs = 0
    with Pool(nproc, initializer=_worker_init) as p:
        for base, status, dt in p.imap_unordered(cache_one, todo):
            done += 1
            if status.startswith("ERR") or status == "EMPTY":
                errs += 1
            print(f"[{done}/{len(todo)}] {base} {status} {dt:.0f}s "
                  f"(elapsed {(time.time()-t0)/60:.1f}m, errs={errs})", flush=True)
    print(f"done: {done} processed, {errs} errors/empty, "
          f"{(time.time()-t0)/60:.1f}m total", flush=True)


if __name__ == "__main__":
    main()
