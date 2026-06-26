"""Rank cross-session registration quality and surface the WORST pairs as
diagnostic targets for chasing the projection error.

Metric: **boundary-free interior NCC** between two session stacks over their eroded
common support (the coverage-edge-gamed FRC is NOT used). A small global shift is
removed first; if the best shift runs away (>10 px) the pair is decorrelated (no
lock). Tile-consensus significance is ~1 on these feature-poor stacks, so per-tile
locking does not work — interior NCC is the honest quality number.

Outputs:
  - ranked table (worst first) of pairwise NCC vs |ΔSRP_lat|
  - results/figures/registration_ncc_matrix.png   (6x6 quality heatmap)
  - results/figures/registration_ncc_vs_dsrp.png  (collapse vs geometry)
  - results/figures/overlay_<best|worst>_<pair>.png (red=A, green=B)

Usage: .conda/bin/python scripts/assess_registration.py
"""
import os, sys, glob, itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from venera.registration import register_maps, bandpass
from venera.coherence import ncc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STACKS = os.path.join(ROOT, "results", "session_stacks")
FIG = os.path.join(ROOT, "results", "figures")
os.makedirs(FIG, exist_ok=True)
YEARS = ["1988", "2001", "2012", "2015", "2017", "2020"]
HH, WW = 2000, 4000
PXDEG = WW / 360.0
ERODE = 12
RUNAWAY_PX = 10        # best shift beyond this => no real lock (decorrelated)


def load():
    out = {}
    for y in YEARS:
        f = os.path.join(STACKS, f"session_{y}.npz")
        if os.path.exists(f):
            d = np.load(f)
            out[y] = (d["Gm"], d["mask"], float(d["srp_lat_mean"]))
    return out


def interior_ncc(a, b, ma, mb, shift=(0, 0)):
    dr, dc = int(round(shift[0])), int(round(shift[1]))
    bs = np.roll(np.roll(b, dr, 0), dc, 1)
    ms = np.roll(np.roll(mb, dr, 0), dc, 1)
    com = binary_erosion(ma & ms, iterations=ERODE)
    if com.sum() < 2000:
        return np.nan
    return ncc(a, bs, com, com)


def assess_pair(Ga, Ma, Gb, Mb):
    """Return (ncc_quality, shift_used, locked) for b onto a."""
    dr, dc, sig = register_maps(Ga, Gb, Ma, Mb, max_shift=45)
    locked = (abs(dr) <= RUNAWAY_PX and abs(dc) <= RUNAWAY_PX)
    shift = (dr, dc) if locked else (0, 0)
    return interior_ncc(Ga, Gb, Ma, Mb, shift), shift, locked


def overlay(a, ma, b, mb, shift, title, path):
    dr, dc = int(round(shift[0])), int(round(shift[1]))
    fa = bandpass(a, 3, 30)
    bs = np.roll(np.roll(bandpass(b, 3, 30), dr, 0), dc, 1)
    common = ma & np.roll(np.roll(mb, dr, 0), dc, 1)
    rows = np.where(common.any(1))[0]; cols = np.where(common.any(0))[0]
    if rows.size == 0:
        return
    r0, r1, c0, c1 = rows[0], rows[-1], cols[0], cols[-1]
    A = fa[r0:r1, c0:c1]; B = bs[r0:r1, c0:c1]; cm = common[r0:r1, c0:c1]

    def norm(x):
        v = x[cm]
        lo, hi = np.percentile(v, 2), np.percentile(v, 98)
        return np.clip((x - lo) / (hi - lo + 1e-9), 0, 1)
    rgb = np.zeros(A.shape + (3,))
    rgb[..., 0] = norm(A) * cm
    rgb[..., 1] = norm(B) * cm
    ext = [c0 / PXDEG - 180, c1 / PXDEG - 180,
           r0 / (HH / 180.0) - 90, r1 / (HH / 180.0) - 90]
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.imshow(rgb, origin="lower", extent=ext, aspect="auto")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("longitude (°E)"); ax.set_ylabel("latitude (°N)")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def main():
    S = load()
    ys = [y for y in YEARS if y in S]
    print(f"loaded sessions: {ys}")
    rows = []
    M = np.full((len(ys), len(ys)), np.nan)
    for i, j in itertools.combinations(range(len(ys)), 2):
        ya, yb = ys[i], ys[j]
        Ga, Ma, la = S[ya]; Gb, Mb, lb = S[yb]
        q, shift, locked = assess_pair(Ga, Ma, Gb, Mb)
        M[i, j] = M[j, i] = q
        rows.append(dict(pair=f"{ya}-{yb}", dsrp=abs(la - lb), ncc=q,
                         locked=locked, shift=shift, ya=ya, yb=yb))
    for i in range(len(ys)):
        M[i, i] = 1.0

    rows.sort(key=lambda d: (d["ncc"] if np.isfinite(d["ncc"]) else -1))
    print("\n  pair        |ΔSRP|   interior-NCC   lock?")
    print("  " + "-" * 44)
    for d in rows:
        print(f"  {d['pair']:10s}  {d['dsrp']:5.1f}°     {d['ncc']:.3f}       "
              f"{'yes' if d['locked'] else 'NO (decorrelated)'}")

    xs = np.array([d["dsrp"] for d in rows])
    ncs = np.array([d["ncc"] for d in rows])
    ok = np.isfinite(ncs)
    if ok.sum() > 2:
        print(f"\ncorr(|ΔSRP_lat|, NCC) = {np.corrcoef(xs[ok], ncs[ok])[0,1]:+.2f}")

    # NCC quality matrix
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(ys))); ax.set_yticks(range(len(ys)))
    ax.set_xticklabels(ys); ax.set_yticklabels(ys)
    for i in range(len(ys)):
        for j in range(len(ys)):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                        color="w" if M[i, j] < 0.6 else "k", fontsize=9)
    fig.colorbar(im, label="interior NCC")
    ax.set_title("Cross-session registration quality (interior NCC)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "registration_ncc_matrix.png"), dpi=120)
    plt.close(fig)

    # NCC vs |ΔSRP_lat|
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(xs[ok], ncs[ok], s=45)
    for d in rows:
        if np.isfinite(d["ncc"]):
            ax.annotate(d["pair"], (d["dsrp"], d["ncc"]), fontsize=7,
                        xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("|ΔSRP latitude| between sessions (°)")
    ax.set_ylabel("interior NCC (registration quality)")
    ax.set_title("Registration collapses with viewing-geometry difference")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "registration_ncc_vs_dsrp.png"), dpi=120)
    plt.close(fig)

    # overlay for EVERY pair, named by NCC so files sort worst->best
    odir = os.path.join(FIG, "overlays")
    os.makedirs(odir, exist_ok=True)
    for d in rows:
        if not np.isfinite(d["ncc"]):
            continue
        Ga, Ma, _ = S[d["ya"]]; Gb, Mb, _ = S[d["yb"]]
        overlay(Ga, Ma, Gb, Mb, d["shift"],
                f"{d['pair']}  |ΔSRP|={d['dsrp']:.0f}°  NCC={d['ncc']:.2f}  "
                f"(red={d['ya']}, green={d['yb']})",
                os.path.join(odir, f"ncc{d['ncc']:.2f}_{d['pair']}.png"))
    print(f"wrote {len(rows)} pair overlays in {odir}/ (named by NCC, worst first)")
    print("wrote registration_ncc_matrix.png, registration_ncc_vs_dsrp.png")


if __name__ == "__main__":
    main()
