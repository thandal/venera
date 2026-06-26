"""Render per-session stacks for flip-through alignment checking, at a CHOSEN period.

A period error is a pure per-session longitude rotation (uniform column shift on the
equirectangular map), so we can re-express the assumed-period stacks at any period
analytically — no reproject. Default = our measured 243.0259 d (not the stale IAU
243.0185 that is baked into the projection). Identical crop + common grayscale so
flipping isolates geometry, not brightness.

Usage: .conda/bin/python scripts/render_session_flip.py [period_days]
"""
import os, sys, glob
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STACKS = os.path.join(ROOT, "results", "session_stacks")
CACHE = os.path.join(ROOT, "results", "look_cache")
FIG = os.path.join(ROOT, "results", "figures")
OUT = os.path.join(FIG, "flip")
os.makedirs(OUT, exist_ok=True)
HH, WW = 2000, 4000
YEARS = ["1988", "2001", "2012", "2015", "2017", "2020"]
DATES = {"1988": "Jun 1988", "2001": "Mar 2001", "2012": "May 2012",
         "2015": "Aug 2015", "2017": "Mar 2017", "2020": "May 2020"}
P_ASSUMED = 243.0185                       # IAU/Magellan, baked into the projection
P_BEST = float(sys.argv[1]) if len(sys.argv) > 1 else 243.0259   # our fit
LAT0, LAT1, LON0, LON1 = 25, 80, -110, 60
R0 = int((LAT0 + 90) / 180 * HH); R1 = int((LAT1 + 90) / 180 * HH)
C0 = int((LON0 + 180) / 360 * WW); C1 = int((LON1 + 180) / 360 * WW)
EXT = [LON0, LON1, LAT0, LAT1]


def mean_day(year):
    d = [float(np.load(f, allow_pickle=True)["day"])
         for f in glob.glob(os.path.join(CACHE, f"venus_*_{year}*.npz"))]
    return float(np.mean(d))


def flatten(G, m, s=35):
    f = np.where(m, G, 0.0); w = gaussian_filter(m.astype(float), s)
    bg = gaussian_filter(f, s) / np.maximum(w, 1e-6); r = G - bg; r[~m] = np.nan
    return r


def main():
    # period correction: lon_best = lon_assumed - (Wdot_best - Wdot_assumed)*day
    dWdot = (-360.0 / P_BEST) - (-360.0 / P_ASSUMED)     # deg/day
    print(f"rendering at period {P_BEST:.4f} d (assumed {P_ASSUMED:.4f}); "
          f"ΔWdot={dWdot:.3e} deg/day")
    crops = {}; vals = []
    for y in YEARS:
        d = np.load(os.path.join(STACKS, f"session_{y}.npz"))
        G, m, n = d["Gm"], d["mask"], int(d["n"])
        shift_deg = -dWdot * mean_day(y)
        sp = int(round(shift_deg / 360.0 * WW))
        G = np.roll(G, sp, axis=1); m = np.roll(m, sp, axis=1)
        f = flatten(G, m)[R0:R1, C0:C1]
        crops[y] = (f, n, shift_deg)
        vals.append(f[np.isfinite(f)])
        print(f"  {y}: day={mean_day(y):+.0f}  period-shift={shift_deg:+.3f}° ({sp:+d}px)")
    v = np.concatenate(vals); vlo, vhi = np.percentile(v, 2), np.percentile(v, 99)

    frames = []
    tag = f"{P_BEST:.4f}".replace(".", "p")
    for y in YEARS:
        f, n, sh = crops[y]
        fig, ax = plt.subplots(figsize=(9, 4.2))
        ax.imshow(f, origin="lower", extent=EXT, cmap="gray", vmin=vlo, vmax=vhi,
                  aspect="auto")
        ax.set_title(f"Venus — {DATES[y]}  ({n} looks)   [period {P_BEST:.4f} d]",
                     fontsize=12)
        ax.set_xlabel("longitude (°E)"); ax.set_ylabel("latitude (°N)")
        ax.grid(alpha=0.3, color="c", lw=0.4)
        p = os.path.join(OUT, f"session_{y}_P{tag}.png")
        fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig)
        frames.append(Image.open(p).convert("RGB"))
    gif = os.path.join(FIG, f"session_flip_P{tag}.gif")
    frames[0].save(gif, save_all=True, append_images=frames[1:], duration=900, loop=0)
    print(f"wrote {gif} and per-session PNGs in {OUT}")


if __name__ == "__main__":
    main()
