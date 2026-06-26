"""STEP 3: unbiased period from ALL session pairs (forward-project + interior
correlation, no registration fitting), separating the period from per-session
constant longitude offsets.

For each of the 15 pairs: scan trial periods, compute the interior overlapping-disk
NCC at identity, parabolic-refine the peak -> the longitude shift m_ij that best
aligns j onto i. Then m_ij = c * Δt_ij + (off_i - off_j), where c is the per-day
longitude-rate correction (-> period) and off_k are per-session constant offsets.
Global least-squares solve of {c, off_k (ref=2012)} over all pairs gives the
UNBIASED period; bootstrap over pairs gives the uncertainty.

Writes results/figures/period_solve_all.png (per-pair curves + m_ij vs Δt fit).
"""
import os, sys, glob, itertools
import numpy as np
from scipy.ndimage import gaussian_filter, binary_erosion, shift as ndshift
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from venera.coherence import ncc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STACKS = os.path.join(ROOT, "results", "session_stacks")
CACHE = os.path.join(ROOT, "results", "look_cache")
FIG = os.path.join(ROOT, "results", "figures")
HH, WW = 2000, 4000
YEARS = ["1988", "2001", "2012", "2015", "2017", "2020"]
REF = "2012"
P_ASSUMED = 243.0185
ERODE_PX, LATMIN, LATMAX = 20, 30, 75
PGRID = np.arange(243.010, 243.031, 0.0010)
# crop window (speed): lat 25..80, all covered lon
RW0, RW1 = int((25+90)/180*HH), int((80+90)/180*HH)


def mean_day(y):
    return float(np.mean([float(np.load(f, allow_pickle=True)["day"])
                          for f in glob.glob(os.path.join(CACHE, f"venus_*_{y}*.npz"))]))


def load(y):
    d = np.load(os.path.join(STACKS, f"session_{y}.npz")); return d["Gm"], d["mask"]


def flatten(G, m, s=35):
    f = np.where(m, G, 0.0); w = gaussian_filter(m.astype(float), s)
    return ((G - gaussian_filter(f, s) / np.maximum(w, 1e-6)) * m)[RW0:RW1], m[RW0:RW1]


def latband(h):
    b = np.zeros((h, WW), bool)
    lo = int((LATMIN+90)/180*HH) - RW0; hi = int((LATMAX+90)/180*HH) - RW0
    b[max(lo, 0):hi] = True
    return b


def main():
    F, M, day = {}, {}, {}
    for y in YEARS:
        G, m = load(y); F[y], M[y] = flatten(G, m); day[y] = mean_day(y)
    h = F[YEARS[0]].shape[0]; LB = latband(h)

    def curve(a, b):
        dt = day[b] - day[a]
        out = []
        for P in PGRID:
            wdot = (-360.0/P) - (-360.0/P_ASSUMED)
            px = (-wdot*dt) / 360.0 * WW
            Fbs = ndshift(F[b], (0, px), order=1, mode="constant", cval=0.0)
            Mbs = ndshift(M[b].astype(float), (0, px), order=1, mode="constant", cval=0) > 0.99
            inn = binary_erosion(M[a] & Mbs, iterations=ERODE_PX) & LB
            out.append(ncc(F[a], Fbs, inn, inn) if inn.sum() > 1500 else np.nan)
        return np.array(out), dt

    pairs = list(itertools.combinations(YEARS, 2))
    curves, peakP, mij, dts = {}, {}, [], []
    A_rows = []   # design matrix rows for [c, off_2001, off_1988, off_2015, off_2017, off_2020] (ref 2012=0)
    others = [y for y in YEARS if y != REF]
    rhs = []
    print("  pair        Δt(yr)  peakP(d)  peakNCC", flush=True)
    for a, b in pairs:
        c, dt = curve(a, b)
        curves[(a, b)] = c
        i = int(np.nanargmax(c))
        # parabolic refine in period
        if 0 < i < len(PGRID)-1 and np.all(np.isfinite(c[i-1:i+2])):
            y0, y1, y2 = c[i-1], c[i], c[i+1]
            denom = (y0 - 2*y1 + y2)
            di = 0.5*(y0 - y2)/denom if denom != 0 else 0.0
        else:
            di = 0.0
        Ppk = PGRID[i] + di*(PGRID[1]-PGRID[0])
        peakP[(a, b)] = (Ppk, np.nanmax(c))
        # m_ij = longitude shift at peak = -((-360/Ppk)-(-360/Passumed))*dt
        m = -((-360.0/Ppk) - (-360.0/P_ASSUMED)) * dt
        mij.append(m); dts.append(dt)
        # design: m = c_rate*dt + off_a - off_b
        row = [dt] + [0.0]*len(others)
        if a in others: row[1+others.index(a)] += 1.0
        if b in others: row[1+others.index(b)] -= 1.0
        A_rows.append(row); rhs.append(m)
        print(f"  {a}-{b}   {dt/365.25:5.1f}   {Ppk:.4f}   {np.nanmax(c):.3f}", flush=True)

    A_rows = np.array(A_rows); rhs = np.array(rhs)
    sol, *_ = np.linalg.lstsq(A_rows, rhs, rcond=None)
    c_rate = sol[0]
    # m = c_rate*dt with c_rate = -((-360/P)-(-360/Passumed)) => solve P
    # c_rate = -(-360/P + 360/Passumed) = 360/P - 360/Passumed
    P_fit = 360.0 / (c_rate + 360.0/P_ASSUMED)
    # bootstrap over pairs
    rng = np.random.default_rng(0); Ps = []
    n = len(rhs)
    for _ in range(2000):
        idx = rng.integers(0, n, n)
        s, *_ = np.linalg.lstsq(A_rows[idx], rhs[idx], rcond=None)
        Ps.append(360.0/(s[0] + 360.0/P_ASSUMED))
    Ps = np.array(Ps); P_lo, P_hi = np.percentile(Ps, [16, 84])
    sigma = 0.5*(P_hi - P_lo)
    resid = A_rows @ sol - rhs
    print(f"\nUNBIASED period = {P_fit:.4f} ± {sigma:.4f} d  "
          f"(bootstrap [{P_lo:.4f}, {P_hi:.4f}])")
    print(f"fit residual RMS = {np.std(resid):.3f}°   per-session offsets (deg): "
          + ", ".join(f"{y}={v:+.2f}" for y, v in zip(others, sol[1:])))
    print("literature: Campbell 243.0212±0.0006 | Margot 243.0226±0.0013 | IAU 243.0185")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.6))
    for (a, b), c in curves.items():
        ax1.plot(PGRID, c, lw=0.8, alpha=0.6)
    ax1.axvline(P_fit, color="b", lw=1.5, label=f"unbiased fit {P_fit:.4f}±{sigma:.4f}")
    for nm, P, col in [("IAU", 243.0185, "k"), ("Campbell", 243.0212, "tab:green"),
                       ("Margot", 243.0226, "tab:orange")]:
        ax1.axvline(P, ls="--", lw=0.8, color=col, alpha=0.7, label=f"{nm} {P}")
    ax1.set_xlabel("trial period (d)"); ax1.set_ylabel("interior NCC")
    ax1.set_title("Per-pair interior-NCC vs period (all 15 pairs)")
    ax1.legend(fontsize=7); ax1.grid(alpha=0.3)

    dts = np.array(dts)/365.25; mij = np.array(mij)
    ax2.scatter(dts, mij, s=35, label="per-pair best shift m_ij")
    xx = np.linspace(min(dts.min(), 0), dts.max(), 50)
    ax2.plot(xx, c_rate*xx*365.25, "b-", lw=1,
             label=f"global rate fit -> P={P_fit:.4f}")
    ax2.set_xlabel("baseline Δt (yr, signed)"); ax2.set_ylabel("best longitude shift m_ij (°)")
    ax2.set_title(f"m_ij vs Δt (slope=rate; scatter=offsets, RMS {np.std(resid):.2f}°)")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "period_solve_all.png"), dpi=120)
    print("wrote period_solve_all.png")


if __name__ == "__main__":
    main()
