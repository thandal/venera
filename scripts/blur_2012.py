"""2012 blur diagnostic (objective). Two questions:
  (1) Is 2012 actually blurrier than other sessions? -> high-frequency power of each
      session stack in a COMMON interior box (same feature region for all).
  (2) Does 2012 blur as looks accumulate? -> build 2012 sub-stacks of N=10,20,40,80,
      160,all looks and measure sharpness vs N. Rising-then-falling/flat => intra-
      session drift smears the deep stack; monotonic-rising => just per-look noise.
  Control: 2017 ladder for comparison.

Sharpness metric = mean squared high-pass (gaussian-difference) amplitude over the
common interior box (speckle-insensitive: same box, bandpassed).

Writes results/figures/blur_2012.png (sharpness bar + ladder) and crops.
"""
import os, sys, glob
import numpy as np
from scipy.ndimage import gaussian_filter, binary_erosion
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stack_sessions import robust_stack

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STACKS = os.path.join(ROOT, "results", "session_stacks")
CACHE = os.path.join(ROOT, "results", "look_cache")
FIG = os.path.join(ROOT, "results", "figures")
HH, WW = 2000, 4000
YEARS = ["1988", "2001", "2012", "2015", "2017", "2020"]
# common interior feature box (Ishtar/terrain, well covered by all)
LAT0, LAT1, LON0, LON1 = 35, 72, -90, 45
R0 = int((LAT0+90)/180*HH); R1 = int((LAT1+90)/180*HH)
C0 = int((LON0+180)/360*WW); C1 = int((LON1+180)/360*WW)


def sharp(G, m):
    """Mean-squared high-pass amplitude over the interior box (eroded)."""
    box = m[R0:R1, C0:C1]
    inn = binary_erosion(box, iterations=10)
    g = G[R0:R1, C0:C1].astype(float)
    hp = g - gaussian_filter(np.where(box, g, 0.0), 4)   # high-pass ~0.35deg
    return float(np.mean(hp[inn]**2)) if inn.sum() > 500 else np.nan


def session_stack(y):
    d = np.load(os.path.join(STACKS, f"session_{y}.npz")); return d["Gm"], d["mask"]


def ladder(y, Ns):
    looks = sorted(glob.glob(os.path.join(CACHE, f"venus_*_{y}*.npz")))
    out = []
    for N in Ns:
        sub = [looks[i] for i in np.linspace(0, len(looks)-1, min(N, len(looks))).astype(int)]
        sub = list(dict.fromkeys(sub))
        G, m, _ = robust_stack(sub)
        out.append((len(sub), sharp(G, m)))
    return out


def main():
    print("per-session sharpness (common interior box):", flush=True)
    ss = {}
    for y in YEARS:
        G, m = session_stack(y); ss[y] = sharp(G, m)
        print(f"  {y}: sharpness={ss[y]:.5f}", flush=True)

    Ns = [10, 20, 40, 80, 160, 320]
    print("\nlook-count ladder (2012 vs 2017 control):", flush=True)
    L12 = ladder("2012", Ns); L17 = ladder("2017", Ns)
    for (n12, s12), (n17, s17) in zip(L12, L17):
        print(f"  N~{n12:3d}/{n17:3d}: 2012={s12:.5f}  2017={s17:.5f}", flush=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.bar(range(len(YEARS)), [ss[y] for y in YEARS],
            color=["tab:red" if y == "2012" else "tab:blue" for y in YEARS])
    ax1.set_xticks(range(len(YEARS))); ax1.set_xticklabels(YEARS)
    ax1.set_ylabel("interior high-freq power (sharpness)")
    ax1.set_title("Per-session sharpness (same interior box)")
    ax1.grid(alpha=0.3, axis="y")

    ax2.plot([n for n, _ in L12], [s for _, s in L12], "-o", color="tab:red", label="2012")
    ax2.plot([n for n, _ in L17], [s for _, s in L17], "-s", color="tab:blue", label="2017")
    ax2.set_xlabel("# looks in sub-stack"); ax2.set_ylabel("sharpness")
    ax2.set_xscale("log"); ax2.set_title("Sharpness vs look count (drift smears if it falls)")
    ax2.legend(); ax2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "blur_2012.png"), dpi=120)
    print("wrote blur_2012.png")


if __name__ == "__main__":
    main()
