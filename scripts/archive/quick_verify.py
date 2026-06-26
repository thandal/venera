"""Quick end-to-end check that the geometry FIX (default observer = Arecibo) makes
2015<->2017 register, using the unmodified pipeline (no monkeypatch)."""
import os, sys, glob
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from register_on_sphere import flatten, search_R, decompose

DATA = ("/home/than/code/venera/arecibo_radar/pds-geosciences.wustl.edu/venus/"
        "arcb_nrao-v-rtls_gbt-3-delaydoppler-v1/vrm_90xx/data/")
CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "results", "look_cache")
H, W, DS = 4000, 8000, 2
HH, WW = H // DS, W // DS


def pick(y, n):
    c = []
    for f in glob.glob(os.path.join(CACHE, f"venus_*_{y}*.npz")):
        d = np.load(f, allow_pickle=True)
        if str(d["pointing"]) == "N":
            c.append((float(d["valid_frac"]), os.path.basename(f)[:-4]))
    c.sort(reverse=True)
    return [os.path.join(DATA, b + ".img") for _, b in c[:n]]


def _init():
    from venera import spice_setup
    spice_setup.furnsh_kernels()


def proj(task):
    img, ep = task
    from venera.geometry import Spin
    from venera.projection import project_file
    G = np.zeros((H, W), np.float32); Gc = np.zeros((H, W), np.int32)
    project_file(img, Spin(), G, Gc)         # default observer is now ARECIBO
    rows = np.where(np.any(Gc > 0, 1))[0]; cols = np.where(np.any(Gc > 0, 0))[0]
    r0, r1, c0, c1 = rows[0], rows[-1], cols[0], cols[-1]
    Gm = np.divide(G, Gc, out=np.zeros_like(G), where=Gc > 0)
    return (ep, (r0, c0, Gm[r0:r1+1:DS, c0:c1+1:DS].astype(np.float32),
                 Gc[r0:r1+1:DS, c0:c1+1:DS] > 0))


def stack(parts):
    Gs = np.zeros((HH, WW)); Gc = np.zeros((HH, WW), int)
    for r0, c0, gm, m in parts:
        hr, hc = r0 // DS, c0 // DS; h, w = gm.shape
        Gs[hr:hr+h, hc:hc+w][m] += gm[m]; Gc[hr:hr+h, hc:hc+w][m] += 1
    return flatten(np.divide(Gs, Gc, out=np.zeros_like(Gs), where=Gc > 0).astype(np.float32),
                   Gc > 0), Gc > 0


def main():
    looks = {"2015": pick("2015", 8), "2017": pick("2017", 8)}
    tasks = [(img, ep) for ep in ("2015", "2017") for img in looks[ep]]
    parts = {}
    with Pool(8, initializer=_init) as p:
        for ep, c in p.imap_unordered(proj, tasks):
            parts.setdefault(ep, []).append(c); print(".", end="", flush=True)
    A, Am = stack(parts["2015"]); B, Bm = stack(parts["2017"])
    rv, nf, ni = search_R(A, Am, B, Bm, coarse_step=1.0)
    print(f"\nFIXED pipeline 2015<->2017: tilt={decompose(rv)[2]:.2f}°  "
          f"NCC@id={ni:.3f}  NCC@bestR={nf:.3f}")


if __name__ == "__main__":
    main()
