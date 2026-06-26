"""Subset verification of the geometric-freq_scale change (recalibrated K), focused
on 2020 (whose per-image fs fit is noisy). Reproject 2015/2017/2020 two ways —
per-image fs fit vs geometric fs — register the three pairs on the sphere, and
render red/green overlays for visual validation.

Acceptance: geometric fs should improve the 2020 pairs (NCC up, residual tilt down)
with no regression on 2015<->2017.

Writes results/figures/freqscale_subset_overlays.png and a numeric table.

Usage: .conda/bin/python scripts/subset_freqscale_test.py
"""
import os, sys, glob, itertools
import numpy as np
from multiprocessing import Pool
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from register_on_sphere import flatten, search_R, decompose

DATA = ("/home/than/code/venera/arecibo_radar/pds-geosciences.wustl.edu/venus/"
        "arcb_nrao-v-rtls_gbt-3-delaydoppler-v1/vrm_90xx/data/")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "results", "look_cache")
FIG = os.path.join(ROOT, "results", "figures")
H, W, DS = 4000, 8000, 2
HH, WW = H // DS, W // DS
EPOCHS = ["2015", "2017", "2020"]
PAIRS = [("2015", "2017"), ("2015", "2020"), ("2017", "2020")]
MODES = ["perfit", "geomfs"]
LAT0, LAT1, LON0, LON1 = 25, 82, -115, 60


def pick(year, n=8):
    c = []
    for f in glob.glob(os.path.join(CACHE, f"venus_*_{year}*.npz")):
        d = np.load(f, allow_pickle=True)
        if str(d["pointing"]) == "N":
            c.append((float(d["valid_frac"]), os.path.basename(f)[:-4]))
    c.sort(reverse=True)
    return [os.path.join(DATA, b + ".img") for _, b in c[:n]]


def _init():
    from venera import spice_setup
    spice_setup.furnsh_kernels()


def proj(task):
    img, mode, ep = task
    from venera.geometry import Spin
    from venera.projection import project_file
    G = np.zeros((H, W), np.float32); Gc = np.zeros((H, W), np.int32)
    try:
        project_file(img, Spin(), G, Gc, geom_fs=(mode == "geomfs"))
    except Exception:
        return (mode, ep, None)
    rows = np.where(np.any(Gc > 0, 1))[0]; cols = np.where(np.any(Gc > 0, 0))[0]
    if rows.size == 0:
        return (mode, ep, None)
    r0, r1, c0, c1 = rows[0], rows[-1], cols[0], cols[-1]
    Gm = np.divide(G, Gc, out=np.zeros_like(G), where=Gc > 0)
    return (mode, ep, (r0, c0, Gm[r0:r1+1:DS, c0:c1+1:DS].astype(np.float32),
                       Gc[r0:r1+1:DS, c0:c1+1:DS] > 0))


def stack(parts):
    Gs = np.zeros((HH, WW)); Gc = np.zeros((HH, WW), int)
    for r0, c0, gm, m in parts:
        hr, hc = r0 // DS, c0 // DS; h, w = gm.shape
        if hr + h > HH or hc + w > WW:
            h = min(h, HH - hr); w = min(w, WW - hc); gm = gm[:h, :w]; m = m[:h, :w]
        Gs[hr:hr+h, hc:hc+w][m] += gm[m]; Gc[hr:hr+h, hc:hc+w][m] += 1
    return flatten(np.divide(Gs, Gc, out=np.zeros_like(Gs), where=Gc > 0).astype(np.float32),
                   Gc > 0), Gc > 0


def crop_rgb(Af, Am, Bf, Bm):
    r0 = int((LAT0 + 90) / 180 * HH); r1 = int((LAT1 + 90) / 180 * HH)
    c0 = int((LON0 + 180) / 360 * WW); c1 = int((LON1 + 180) / 360 * WW)
    A = Af[r0:r1, c0:c1]; B = Bf[r0:r1, c0:c1]; m = (Am & Bm)[r0:r1, c0:c1]

    def norm(x):
        v = x[m]
        if v.size == 0:
            return np.zeros_like(x)
        lo, hi = np.percentile(v, 2), np.percentile(v, 98)
        return np.clip((x - lo) / (hi - lo + 1e-9), 0, 1)
    rgb = np.zeros(A.shape + (3,)); rgb[..., 0] = norm(A) * m; rgb[..., 1] = norm(B) * m
    return rgb


def main():
    looks = {y: pick(y) for y in EPOCHS}
    tasks = [(img, m, ep) for m in MODES for ep in EPOCHS for img in looks[ep]]
    parts = {}
    with Pool(12, initializer=_init) as p:
        for m, ep, c in p.imap_unordered(proj, tasks):
            if c is not None:
                parts.setdefault((m, ep), []).append(c)
            print(".", end="", flush=True)
    print()
    stacks = {(m, ep): stack(parts[(m, ep)]) for m in MODES for ep in EPOCHS}

    print(f"\n  pair        mode     tilt    NCC@id   NCC@bestR")
    print("  " + "-" * 48)
    res = {}
    for (a, b) in PAIRS:
        for m in MODES:
            A, Am = stacks[(m, a)]; B, Bm = stacks[(m, b)]
            rv, nf, ni = search_R(A, Am, B, Bm, coarse_step=1.0)
            res[(a, b, m)] = (decompose(rv)[2], ni, nf)
            print(f"  {a}-{b}    {m:7s}   {decompose(rv)[2]:.2f}°   {ni:.3f}    {nf:.3f}")

    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    for j, (a, b) in enumerate(PAIRS):
        for i, m in enumerate(MODES):
            A, Am = stacks[(m, a)]; B, Bm = stacks[(m, b)]
            axs[i, j].imshow(crop_rgb(A, Am, B, Bm), origin="lower",
                             extent=[LON0, LON1, LAT0, LAT1], aspect="auto")
            t, ni, nf = res[(a, b, m)]
            axs[i, j].set_title(f"{a}(r) vs {b}(g)  [{m}]  NCC@id={ni:.2f}", fontsize=10)
            if j == 0:
                axs[i, j].set_ylabel(("per-image fs" if m == "perfit"
                                      else "geometric fs") + "\nlat (°N)", fontsize=9)
            axs[i, j].set_xlabel("lon (°E)", fontsize=8)
    fig.suptitle("freq_scale: per-image fit (top) vs geometric (bottom) — "
                 "at identity, yellow = aligned", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "freqscale_subset_overlays.png"), dpi=120)
    print("\nwrote freqscale_subset_overlays.png")


if __name__ == "__main__":
    main()
