"""All-pairs on-sphere rotation registration: confirm whether the cross-session
mismatch is a per-epoch TILT tied to the doppler angle (a projection-convention bug)
rather than a period error or a per-pair coincidence.

For every session pair, find the best rigid rotation R (register_on_sphere.search_R),
decompose into polar (about pole) vs tilt, and tabulate against the doppler-angle
difference. Tests:
  - polar ~ 0 for all pairs  -> period/W0 is fine
  - tilt proportional to |Δ doppler-angle|  -> apparent-north convention is the cause

Writes results/figures/tilt_vs_doppler.png

Usage: .conda/bin/python scripts/register_pairs_sphere.py
"""
import os, sys, glob, itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from register_on_sphere import (flatten, search_R, decompose, mean_doppler,
                                STACKS, FIG)

YEARS = ["1988", "2001", "2012", "2015", "2017", "2020"]


def main():
    stacks, dop = {}, {}
    for y in YEARS:
        f = os.path.join(STACKS, f"session_{y}.npz")
        if not os.path.exists(f):
            continue
        d = np.load(f)
        stacks[y] = (flatten(d["Gm"], d["mask"]), d["mask"], float(d["srp_lat_mean"]))
        dop[y] = mean_doppler(y)
    ys = list(stacks)
    rows = []
    for ya, yb in itertools.combinations(ys, 2):
        Af, Am, la = stacks[ya]; Bf, Bm, lb = stacks[yb]
        rv, nfit, nid = search_R(Af, Am, Bf, Bm, coarse_step=1.5)
        total, polar, tilt = decompose(rv)
        rows.append(dict(pair=f"{ya}-{yb}", ddop=dop[ya] - dop[yb],
                         dsrp=la - lb, total=total, polar=polar, tilt=tilt,
                         nid=nid, nfit=nfit))
        print(f"  {ya}-{yb}: Δdop={dop[ya]-dop[yb]:+5.1f}  tilt={tilt:5.2f}  "
              f"polar={polar:+5.2f}  NCC {nid:.2f}->{nfit:.2f}", flush=True)

    print("\n  pair        Δdop   Δsrp   tilt   polar   NCC_id  NCC_R")
    print("  " + "-" * 60)
    for d in sorted(rows, key=lambda r: -r["tilt"]):
        print(f"  {d['pair']:10s} {d['ddop']:+5.1f} {d['dsrp']:+6.1f}  "
              f"{d['tilt']:5.2f}  {d['polar']:+5.2f}   {d['nid']:.2f}    {d['nfit']:.2f}")

    ddop = np.array([abs(r["ddop"]) for r in rows])
    tilt = np.array([r["tilt"] for r in rows])
    polar = np.array([abs(r["polar"]) for r in rows])
    print(f"\n  mean |polar| = {polar.mean():.2f}° (should be ~0 if period is fine)")
    if len(ddop) > 2:
        print(f"  corr(|Δdoppler|, tilt) = {np.corrcoef(ddop, tilt)[0,1]:+.2f}")
        # slope through origin
        slope = np.sum(ddop * tilt) / np.sum(ddop * ddop)
        print(f"  tilt/|Δdoppler| slope (through origin) = {slope:.2f}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(ddop, tilt, s=45, label="tilt")
    ax.scatter(ddop, polar, s=30, c="r", marker="x", label="|polar| (period)")
    for r in rows:
        ax.annotate(r["pair"], (abs(r["ddop"]), r["tilt"]), fontsize=7,
                    xytext=(3, 3), textcoords="offset points")
    lim = max(ddop.max(), tilt.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=0.7, label="slope 1")
    ax.set_xlabel("|Δ doppler-angle| between sessions (°)")
    ax.set_ylabel("recovered rotation component (°)")
    ax.set_title("On-sphere rotation: tilt tracks the doppler-angle difference?")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "tilt_vs_doppler.png"), dpi=120)
    print("\nwrote tilt_vs_doppler.png")


if __name__ == "__main__":
    main()
