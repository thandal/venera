"""Prove the coherence harness on REAL looks.

Same-day, same-pointing looks share geometry, so they must co-register under the
projection. Expectation: half-split FRC high-band rises with N and resolution
sharpens; a negative control (random per-look longitude roll) must drop it. If the
projection is self-consistent this passes; if not, it exposes that too.
"""
import os, sys, glob, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from venera import spice_setup
from venera.geometry import Spin
from venera.data import parse_lbl
from venera.projection import project_file
from venera.coherence import coherent_stack, half_split, stack_sharpness

DATA = ("/home/than/code/venera/arecibo_radar/pds-geosciences.wustl.edu/venus/"
        "arcb_nrao-v-rtls_gbt-3-delaydoppler-v1/vrm_90xx/data/")
FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "figures")
H, W = 4000, 8000
DAY, POINTING, NMAX = "20120526", "N", 16


def project_look(f, spin):
    G = np.zeros((H, W), np.float32); Gc = np.zeros((H, W), np.int32)
    info = project_file(f, spin, G, Gc)
    Gm = np.divide(G, Gc, out=np.zeros_like(G), where=Gc > 0)
    return Gm[::2, ::2].copy(), (Gc[::2, ::2] > 0), info   # half-res for the harness


def main():
    spice_setup.furnsh_kernels()
    spin = Spin()
    files = sorted(glob.glob(os.path.join(DATA, f"venus_scp_{DAY}_*.img"))
                   + glob.glob(os.path.join(DATA, f"venus_ocp_{DAY}_*.img")))
    looks = [f for f in files if parse_lbl(f).get("GEO_POINTING") == POINTING][:NMAX]
    print(f"{len(looks)} {POINTING}-pointing looks on {DAY}")
    maps, masks = [], []
    for f in looks:
        t = time.time()
        g, m, info = project_look(f, spin)
        maps.append(g); masks.append(m)
        print(f"  {os.path.basename(f)}  valid={info['valid_frac']:.2f}  {time.time()-t:.0f}s")

    print("\n-- half-split coherence vs N (same geometry; should rise/sharpen) --")
    for n in (4, 8, 12, len(maps)):
        if n > len(maps):
            continue
        r = half_split(maps[:n], masks[:n])
        print(f"  N={n:2d}:  frc_hi={r['frc_hi']:.3f}  resolution={r['resolution_px']:.1f}px  ncc={r['ncc']:.3f}")

    print("\n-- NEGATIVE CONTROL: random per-look longitude roll (misregistration) --")
    rng = np.random.default_rng(0)
    rolled = [np.roll(g, int(rng.normal(0, 12)), axis=1) for g in maps]   # ~±12 half-res px ≈ ±0.5deg
    rmask = [np.roll(m, int(s), axis=1) for m, s in
             zip(masks, [int(rng.normal(0, 12)) for _ in masks])]
    r0 = half_split(maps, masks); rc = half_split(rolled, masks)
    print(f"  aligned   : frc_hi={r0['frc_hi']:.3f}  res={r0['resolution_px']:.1f}px")
    print(f"  misaligned: frc_hi={rc['frc_hi']:.3f}  res={rc['resolution_px']:.1f}px")
    print(f"  -> gate {'WORKS on real data' if r0['frc_hi'] > rc['frc_hi'] + 0.1 else 'INCONCLUSIVE'}")

    # visual: single look vs full stack (Maxwell region, half-res coords)
    h, w = H // 2, W // 2
    rr0, rr1 = int((52+90)/180*h), int((74+90)/180*h)
    cc0, cc1 = int((-28+180)/360*w), int((28+180)/360*w)
    stack, cnt = coherent_stack(maps, masks)
    print(f"\n  stack_sharpness: 1 look={stack_sharpness(maps[0], masks[0].astype(int)+1):.4f}  "
          f"{len(maps)}-stack={stack_sharpness(stack, cnt):.4f}")
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    for ax, im, m, t in [(axs[0], maps[0], masks[0], "single look"),
                         (axs[1], stack, cnt > 0, f"{len(maps)}-look stack")]:
        cr = im[rr0:rr1, cc0:cc1].astype(float); cm = m[rr0:rr1, cc0:cc1]
        cr[~cm] = np.nan
        lo, hi = np.nanpercentile(cr, 2), np.nanpercentile(cr, 99.5)
        ax.imshow(cr, origin="lower", cmap="gray", vmin=lo, vmax=hi, aspect="auto"); ax.set_title(t); ax.axis("off")
    fig.suptitle(f"Same-geometry stacking proof ({DAY}, {POINTING}): single look vs stack", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "proof_same_geometry.png"), dpi=130)
    print(f"  wrote {FIG}/proof_same_geometry.png")


if __name__ == "__main__":
    main()
