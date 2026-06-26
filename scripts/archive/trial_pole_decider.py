"""Model-independent decider: does any POLE correction null the 2015<->2017 tilt at
the SOURCE (in the projection), or does no pole fix it (=> doppler-basis/geometry bug)?

Reproject subsets of 2015 and 2017 looks over a grid of (dRA, dDec) pole offsets,
stack each, and measure the on-sphere registration:
  - NCC@identity  (high if the pole co-registers them with NO extra rotation)
  - residual best-fit tilt (->0 if the pole nulls it)
A pole that drives NCC@identity up to the ~0.59 best-rotation level AND tilt->0 means
it was a pole effect. If the best identity-NCC stays low for every pole, no pole fixes
it -> the tilt is a projection-code term.

Usage: .conda/bin/python scripts/trial_pole_decider.py [n_looks_per_epoch]
"""
import os, sys, glob
import numpy as np
from multiprocessing import Pool
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from register_on_sphere import flatten, search_R, decompose, FIG
from venera import spice_setup
from venera.geometry import Spin
from venera.projection import project_file

DATA = ("/home/than/code/venera/arecibo_radar/pds-geosciences.wustl.edu/venus/"
        "arcb_nrao-v-rtls_gbt-3-delaydoppler-v1/vrm_90xx/data/")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "results", "look_cache")
H, W, DS = 4000, 8000, 2
HH, WW = H // DS, W // DS
RA0, DEC0 = 272.76, 67.16
DRA = [-1.5, 0.0, 1.5]
DDEC = [-2.0, -1.0, 0.0, 1.0, 2.0]


def pick(year, n):
    """n N-pointing looks for the year (highest valid_frac), return .img paths."""
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
    img, dra, ddec, epoch = task
    spin = Spin(pole_ra_deg=RA0 + dra, pole_dec_deg=DEC0 + ddec)
    G = np.zeros((H, W), np.float32); Gc = np.zeros((H, W), np.int32)
    try:
        project_file(img, spin, G, Gc)
    except Exception as e:
        return (dra, ddec, epoch, None)
    rows = np.where(np.any(Gc > 0, axis=1))[0]
    cols = np.where(np.any(Gc > 0, axis=0))[0]
    if rows.size == 0:
        return (dra, ddec, epoch, None)
    r0, r1, c0, c1 = rows[0], rows[-1], cols[0], cols[-1]
    Gm = np.divide(G, Gc, out=np.zeros_like(G), where=Gc > 0)
    return (dra, ddec, epoch, (r0, c0, Gm[r0:r1+1:DS, c0:c1+1:DS].astype(np.float32),
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
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    looks = {"2015": pick("2015", n), "2017": pick("2017", n)}
    print(f"subset: 2015 {len(looks['2015'])} looks, 2017 {len(looks['2017'])} looks")
    tasks = [(img, dra, ddec, ep)
             for dra in DRA for ddec in DDEC
             for ep in ("2015", "2017") for img in looks[ep]]
    print(f"{len(tasks)} projections ({len(DRA)*len(DDEC)} poles x {2*n} looks)",
          flush=True)
    results = {}
    with Pool(12, initializer=_init) as p:
        for dra, ddec, ep, payload in p.imap_unordered(proj_one, tasks):
            results.setdefault((dra, ddec, ep), []).append(payload)
            print(".", end="", flush=True)
    print()

    grid = np.full((len(DDEC), len(DRA)), np.nan)
    tiltgrid = np.full((len(DDEC), len(DRA)), np.nan)
    print(f"\n  dRA   dDec   NCC@id   tilt   NCC@bestR")
    print("  " + "-" * 44)
    for i, ddec in enumerate(DDEC):
        for j, dra in enumerate(DRA):
            p15 = [x for x in results[(dra, ddec, "2015")] if x]
            p17 = [x for x in results[(dra, ddec, "2017")] if x]
            if not p15 or not p17:
                continue
            A, Am = stack_into(p15); B, Bm = stack_into(p17)
            rv, nfit, nid = search_R(A, Am, B, Bm, coarse_step=1.5)
            grid[i, j] = nid; tiltgrid[i, j] = decompose(rv)[2]
            print(f"  {dra:+.1f}  {ddec:+.1f}   {nid:.3f}   {decompose(rv)[2]:.2f}   {nfit:.3f}")

    bi = np.unravel_index(np.nanargmax(grid), grid.shape)
    print(f"\nbest NCC@identity = {grid[bi]:.3f} at dRA={DRA[bi[1]]:+.1f}, "
          f"dDec={DDEC[bi[0]]:+.1f} (tilt there {tiltgrid[bi]:.2f})")
    print(f"NCC@identity at nominal pole (0,0) = {grid[DDEC.index(0.0), DRA.index(0.0)]:.3f}")
    print("VERDICT: if best NCC@id >> nominal AND its tilt ~0 -> a pole nulls it "
          "(pole-handling). If best NCC@id stays ~nominal -> NO pole fixes it "
          "(doppler-basis/geometry bug).")

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(grid, origin="lower", cmap="viridis",
                   extent=[min(DRA)-0.75, max(DRA)+0.75, min(DDEC)-0.5, max(DDEC)+0.5],
                   aspect="auto")
    for i, ddec in enumerate(DDEC):
        for j, dra in enumerate(DRA):
            if np.isfinite(grid[i, j]):
                ax.text(dra, ddec, f"{grid[i,j]:.2f}\n{tiltgrid[i,j]:.1f}°",
                        ha="center", va="center", color="w", fontsize=8)
    fig.colorbar(im, label="2015↔2017 on-sphere NCC @ identity")
    ax.set_xlabel("Δ pole RA (°)"); ax.set_ylabel("Δ pole Dec (°)")
    ax.set_title("Does a pole correction null the tilt at the source?\n"
                 "(cell: NCC@identity / residual tilt)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "trial_pole_grid.png"), dpi=120)
    print("wrote trial_pole_grid.png")


if __name__ == "__main__":
    main()
