"""Why MM 'jumps' when flipping the per-session PNGs: those are raw projections at
the ASSUMED period (243.0185) with NO cross-session alignment. A fixed feature drifts
in longitude across sessions because the true period differs (the rotation signal),
plus a small latitude residual.

Proof: 3-color overlay of the Maxwell region for 1988(R)/2012(G)/2020(B) spanning the
32-yr baseline, (left) as-projected = identity, (right) after applying each session's
measured shift to the 2012 reference. If the chromatic spread collapses to grey, the
'jump' is exactly the (known, measured) per-session offset, dominated by period drift.

Writes results/figures/validate_maxwell_drift.png
"""
import os, sys
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from venera.registration import register_maps

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STACKS = os.path.join(ROOT, "results", "session_stacks")
FIG = os.path.join(ROOT, "results", "figures")
HH, WW = 2000, 4000
LAT0, LAT1, LON0, LON1 = 48, 80, -35, 40
R0 = int((LAT0 + 90) / 180 * HH); R1 = int((LAT1 + 90) / 180 * HH)
C0 = int((LON0 + 180) / 360 * WW); C1 = int((LON1 + 180) / 360 * WW)
EXT = [LON0, LON1, LAT0, LAT1]
RGB_YEARS = ["1988", "2012", "2020"]   # R, G, B — span the baseline
REF = "2012"


def load(y):
    d = np.load(os.path.join(STACKS, f"session_{y}.npz")); return d["Gm"], d["mask"]


def flat(G, m, s=18):
    f = np.where(m, G, 0.0); w = gaussian_filter(m.astype(float), s)
    bg = gaussian_filter(f, s) / np.maximum(w, 1e-6); r = G - bg; r[~m] = 0.0
    return r


def norm(x, m):
    v = x[m]
    lo, hi = np.percentile(v, 3), np.percentile(v, 99)
    return np.clip((x - lo) / (hi - lo + 1e-9), 0, 1) * m


def main():
    full = {y: load(y) for y in RGB_YEARS}
    flats = {y: (flat(*full[y]), full[y][1]) for y in RGB_YEARS}
    Gr, Mr = load(REF); Fr = flat(Gr, Mr)

    # measure each session's shift to the 2012 reference (full-disk register_maps)
    shifts = {}
    for y in RGB_YEARS:
        if y == REF:
            shifts[y] = (0, 0, 0.0, 0.0); continue
        Gy, My = load(y)
        dr, dc, sig = register_maps(Fr, flat(Gy, My), valid_a=Mr, valid_b=My,
                                    max_shift=60, smooth_px=7.0, trend_px=55.0)
        shifts[y] = (int(round(dr)), int(round(dc)), -dr / HH * 180, -dc / WW * 360)

    def rgb(apply_shift):
        out = np.zeros((R1 - R0, C1 - C0, 3))
        for k, y in enumerate(RGB_YEARS):
            f, m = flats[y]
            if apply_shift:
                dr, dc = shifts[y][0], shifts[y][1]
                f = np.roll(np.roll(f, dr, 0), dc, 1); m = np.roll(np.roll(m, dr, 0), dc, 1)
            out[..., k] = norm(f[R0:R1, C0:C1], m[R0:R1, C0:C1])
        return out

    fig, axs = plt.subplots(1, 2, figsize=(15, 5.6))
    for ax, apply_s, ttl in [(axs[0], False, "as projected (assumed period, no alignment)"),
                             (axs[1], True, "after per-session shift to 2012")]:
        ax.imshow(rgb(apply_s), origin="lower", extent=EXT, aspect="auto")
        ax.set_title(ttl, fontsize=11)
        ax.set_xlabel("lon (°E)"); ax.set_ylabel("lat (°N)")
    txt = "  ".join(f"{y}: Δlon={shifts[y][3]:+.2f}° Δlat={shifts[y][2]:+.2f}°"
                    for y in RGB_YEARS)
    fig.suptitle("MM 'jump' = per-session offset (R=1988, G=2012, B=2020).  " + txt,
                 fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "validate_maxwell_drift.png"), dpi=120)
    plt.close(fig)
    print("wrote validate_maxwell_drift.png")
    for y in RGB_YEARS:
        print(f"  {y}: Δlon={shifts[y][3]:+.3f}°  Δlat={shifts[y][2]:+.3f}°  (vs {REF})")


if __name__ == "__main__":
    main()
