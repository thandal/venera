"""Decisive pole test: does a SINGLE pole offset co-register ALL 15 pairs (=> a real
/ effective pole correction), or does each pair want a different pole (=> the tilt is
not a pole; it's a per-look/data effect that's only pairwise-degenerate with a pole)?

Reproject all 6 epochs (subset) over a grid of (dRA,dDec); for each pole stack each
epoch and score every pair by interior NCC at IDENTITY (no rotation). Report the
grid of mean all-pairs NCC, the best single pole, and the per-pair NCC there (esp.
the MIN -- a real pole must lift every pair, not trade one for another). Also dumps
the fitted fo/do/fs per epoch (geometry-bias check on the per-image calibration).

Usage: .conda/bin/python scripts/all_pairs_pole_grid.py [n_looks]
"""
import os, sys, glob, itertools
import numpy as np
from multiprocessing import Pool
from scipy.ndimage import binary_erosion
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from register_on_sphere import flatten, FIG
from venera import spice_setup
from venera.geometry import Spin
from venera.projection import project_file
from venera.coherence import ncc

DATA = ("/home/than/code/venera/arecibo_radar/pds-geosciences.wustl.edu/venus/"
        "arcb_nrao-v-rtls_gbt-3-delaydoppler-v1/vrm_90xx/data/")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "results", "look_cache")
H, W, DS = 4000, 8000, 2
HH, WW = H // DS, W // DS
RA0, DEC0 = 272.76, 67.16
YEARS = ["1988", "2001", "2012", "2015", "2017", "2020"]
DRA = [-3.0, -1.5, 0.0]
DDEC = [-3.0, -2.0, -1.0, 0.0]


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
    img, dra, ddec, ep = task
    spin = Spin(pole_ra_deg=RA0 + dra, pole_dec_deg=DEC0 + ddec)
    G = np.zeros((H, W), np.float32); Gc = np.zeros((H, W), np.int32)
    try:
        info = project_file(img, spin, G, Gc)
    except Exception:
        return (dra, ddec, ep, None, None)
    rows = np.where(np.any(Gc > 0, axis=1))[0]; cols = np.where(np.any(Gc > 0, axis=0))[0]
    if rows.size == 0:
        return (dra, ddec, ep, None, None)
    r0, r1, c0, c1 = rows[0], rows[-1], cols[0], cols[-1]
    Gm = np.divide(G, Gc, out=np.zeros_like(G), where=Gc > 0)
    crop = (r0, c0, Gm[r0:r1+1:DS, c0:c1+1:DS].astype(np.float32),
            (Gc[r0:r1+1:DS, c0:c1+1:DS] > 0))
    return (dra, ddec, ep, crop, info["fit"])


def stack_into(parts):
    Gsum = np.zeros((HH, WW), np.float64); Gcnt = np.zeros((HH, WW), np.int32)
    for r0, c0, gm, m in parts:
        hr, hc = r0 // DS, c0 // DS; h, w = gm.shape
        if hr + h > HH or hc + w > WW:
            h = min(h, HH - hr); w = min(w, WW - hc); gm = gm[:h, :w]; m = m[:h, :w]
        Gsum[hr:hr+h, hc:hc+w][m] += gm[m]; Gcnt[hr:hr+h, hc:hc+w][m] += 1
    Gm = np.divide(Gsum, Gcnt, out=np.zeros_like(Gsum), where=Gcnt > 0).astype(np.float32)
    return flatten(Gm, Gcnt > 0), Gcnt > 0


def ncc_id(A, Am, B, Bm):
    com = binary_erosion(Am & Bm, iterations=12)
    return ncc(A, B, com, com) if com.sum() > 1500 else np.nan


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    looks = {y: pick(y, n) for y in YEARS}
    tasks = [(img, dra, ddec, ep) for dra in DRA for ddec in DDEC
             for ep in YEARS for img in looks[ep]]
    print(f"{len(tasks)} projections ({len(DRA)*len(DDEC)} poles x {len(YEARS)}x{n})",
          flush=True)
    parts = {}; fits = {}
    with Pool(12, initializer=_init) as p:
        for dra, ddec, ep, crop, fit in p.imap_unordered(proj_one, tasks):
            if crop is not None:
                parts.setdefault((dra, ddec, ep), []).append(crop)
                if (dra, ddec) == (0.0, 0.0):
                    fits.setdefault(ep, []).append(fit)
            print(".", end="", flush=True)
    print()

    pairs = list(itertools.combinations(YEARS, 2))
    grid = np.full((len(DDEC), len(DRA)), np.nan)
    percell = {}
    for i, ddec in enumerate(DDEC):
        for j, dra in enumerate(DRA):
            stacks = {ep: stack_into(parts[(dra, ddec, ep)]) for ep in YEARS
                      if (dra, ddec, ep) in parts}
            nccs = {}
            for a, b in pairs:
                if a in stacks and b in stacks:
                    nccs[(a, b)] = ncc_id(*stacks[a], *stacks[b])
            grid[i, j] = np.nanmean(list(nccs.values()))
            percell[(dra, ddec)] = nccs

    bi = np.unravel_index(np.nanargmax(grid), grid.shape)
    bpole = (DRA[bi[1]], DDEC[bi[0]])
    print(f"\nmean all-pairs NCC@id grid (rows dDec {DDEC}, cols dRA {DRA}):")
    print(np.round(grid, 3))
    print(f"\nnominal (0,0) mean NCC@id = {grid[DDEC.index(0.0), DRA.index(0.0)]:.3f}")
    print(f"best pole {bpole} mean NCC@id = {grid[bi]:.3f}")
    print(f"\nper-pair NCC@id  (nominal -> best pole {bpole}):")
    nom = percell[(0.0, 0.0)]; best = percell[bpole]
    for a, b in pairs:
        print(f"  {a}-{b}: {nom[(a,b)]:.2f} -> {best[(a,b)]:.2f}")
    bvals = [v for v in best.values() if np.isfinite(v)]
    print(f"\nat best pole: min per-pair NCC = {min(bvals):.2f}, mean = {np.mean(bvals):.2f}")
    print("VERDICT: one pole works only if it lifts ALL pairs (high MIN), not if it "
          "trades some for others.")

    print("\nper-image fit (fo, do, fs) at nominal pole, mean+/-std per epoch:")
    for ep in YEARS:
        if ep in fits:
            arr = np.array(fits[ep], float)
            print(f"  {ep}: fo={arr[:,0].mean():+.1f}+/-{arr[:,0].std():.1f}  "
                  f"do={arr[:,1].mean():+.1f}+/-{arr[:,1].std():.1f}  "
                  f"fs={arr[:,2].mean():.3f}+/-{arr[:,2].std():.3f}")

    fig, ax = plt.subplots(figsize=(6.5, 5))
    im = ax.imshow(grid, origin="lower", cmap="viridis",
                   extent=[min(DRA)-0.75, max(DRA)+0.75, min(DDEC)-0.5, max(DDEC)+0.5],
                   aspect="auto")
    for i, ddec in enumerate(DDEC):
        for j, dra in enumerate(DRA):
            ax.text(DRA[j], DDEC[i], f"{grid[i,j]:.2f}", ha="center", va="center",
                    color="w", fontsize=9)
    fig.colorbar(im, label="mean all-pairs NCC@identity")
    ax.set_xlabel("Δ pole RA (°)"); ax.set_ylabel("Δ pole Dec (°)")
    ax.set_title("One pole for ALL pairs? mean cross-epoch NCC@identity vs pole")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "all_pairs_pole_grid.png"), dpi=120)
    print("\nwrote all_pairs_pole_grid.png")


if __name__ == "__main__":
    main()
