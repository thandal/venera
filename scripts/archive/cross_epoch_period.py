"""Cross-epoch period via FRC imagery gate. [CAUTION — gate is boundary-gamed, §12]

CAUTION: this uses frc_hi (cross-epoch FRC high-band) as the gate, but FRC over a
common-support mask is boundary-gamed under warping (REPORT.md §12 / §9) — it can
rise while true interior alignment falls. Use a boundary-free interior NCC and check
against the eye before trusting any peak from this script.

A period error shifts one epoch's longitude relative to another by
Δλ = (Ẇ(P) − Ẇ_nominal)·Δt (a uniform roll). Stack each epoch from its cached
looks, roll one epoch over a range of trial periods, and measure cross-epoch FRC
high-band. The period that maximizes frc_hi is the trial estimate.

This is a 2-epoch proof of method; the full estimate uses all epochs jointly.
"""
import os, sys, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from venera.coherence import frc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "results", "look_cache")
FIG = os.path.join(ROOT, "results", "figures")
H, W, DS = 4000, 8000, 2
hh, ww = H // DS, W // DS
DEG_PER_PX = 360.0 / ww
NOMINAL_P = 243.0185


def load_epoch(year, pointing="N"):
    g = np.zeros((hh, ww), np.float64); c = np.zeros((hh, ww), np.int32); days = []
    for f in sorted(glob.glob(os.path.join(CACHE, f"venus_*cp_{year}*.npz"))):
        d = np.load(f, allow_pickle=True)
        if pointing != "both" and str(d["pointing"]) != pointing:
            continue
        r0, _, c0, _ = d["bbox"]; gm, mk = d["gm"], d["mask"]
        rr, cc = int(r0) // DS, int(c0) // DS
        gh, gw = gm.shape
        if rr + gh > hh or cc + gw > ww:
            gh, gw = min(gh, hh - rr), min(gw, ww - cc); gm, mk = gm[:gh, :gw], mk[:gh, :gw]
        g[rr:rr+gh, cc:cc+gw] += np.where(mk, gm, 0)
        c[rr:rr+gh, cc:cc+gw] += mk
        days.append(float(d["day"]))
    gm = np.divide(g, c, out=np.zeros_like(g), where=c > 0)
    return gm, c > 0, float(np.mean(days)), len(days)


def frc_hi(a, b, ma, mb, fhi=0.15):
    f, cc = frc(a, b, ma & mb)
    return float(np.mean(cc[f >= fhi])) if (f.size and (f >= fhi).any()) else np.nan


def main():
    g12, m12, d12, n12 = load_epoch("2012")
    g20, m20, d20, n20 = load_epoch("2020")
    dt = d20 - d12
    print(f"2012: {n12} looks (day {d12:.0f})   2020: {n20} looks (day {d20:.0f})   Δt={dt/365.25:.2f} yr")

    periods = np.arange(243.000, 243.045, 0.001)
    scores = []
    for P in periods:
        dlam = (-360.0 / P - (-360.0 / NOMINAL_P)) * dt           # deg longitude offset
        shift = int(round(dlam / DEG_PER_PX))
        g20r = np.roll(g20, shift, axis=1); m20r = np.roll(m20, shift, axis=1)
        scores.append(frc_hi(g12, g20r, m12, m20r))
    scores = np.array(scores)
    k = int(np.nanargmax(scores))
    # parabolic refine around the peak
    if 0 < k < len(periods) - 1 and np.all(np.isfinite(scores[k-1:k+2])):
        y0, y1, y2 = scores[k-1], scores[k], scores[k+1]
        denom = (y0 - 2*y1 + y2)
        off = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
    else:
        off = 0.0
    P_peak = periods[k] + off * (periods[1] - periods[0])
    print(f"\nfrc_hi peak at P = {P_peak:.4f} d   (frc_hi={scores[k]:.3f})")
    print(f"nominal {NOMINAL_P} | Campbell 243.0212 | Margot 243.0226")
    print(f"frc_hi at nominal = {scores[np.argmin(abs(periods-NOMINAL_P))]:.3f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(periods, scores, "-o", ms=3)
    ax.axvline(P_peak, color="r", label=f"peak {P_peak:.4f} d")
    for v, n, col in [(NOMINAL_P, "IAU/Magellan", "gray"), (243.0212, "Campbell", "g"), (243.0226, "Margot", "b")]:
        ax.axvline(v, color=col, ls="--", alpha=0.6, label=n)
    ax.set_xlabel("trial sidereal period (days)"); ax.set_ylabel("cross-epoch FRC high-band")
    ax.set_title("2012↔2020 period via imagery coherence (peak = co-registration)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "cross_epoch_period.png"), dpi=130)
    print(f"wrote {FIG}/cross_epoch_period.png")


if __name__ == "__main__":
    main()
