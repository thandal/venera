"""Rotation-elements sanity check off the FIXED (Arecibo) co-registered stacks.

Uses the deep per-session stacks (scripts/stack_sessions.py, all looks per session,
topocentric-Arecibo geometry). Registers each session to a reference, measures the
longitude offset vs time, and fits the sidereal period from the drift
(venera.rotation_fit). Pole is adopted from IAU (REPORT.md §10). Reports the
cross-session latitude residual too (should be ~0 now that the observer bug is fixed).

Usage: .conda/bin/python scripts/rotation_elements.py
"""
import os, sys, glob
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from venera.registration import register_maps, offset_to_lonlat_deg
from venera.rotation_fit import (fit_rotation_from_offsets, bootstrap_period,
                                 period_to_wdot)
from venera.geometry import Spin

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STACKS = os.path.join(ROOT, "results", "session_stacks")
CACHE = os.path.join(ROOT, "results", "look_cache")
HH, WW = 2000, 4000
YEARS = ["1988", "2001", "2012", "2015", "2017", "2020"]
REF = "2012"          # deepest session (314 looks), near-equator


def mean_day(year):
    days = [float(np.load(f, allow_pickle=True)["day"])
            for f in glob.glob(os.path.join(CACHE, f"venus_*_{year}*.npz"))]
    return float(np.mean(days))


def main():
    S, day = {}, {}
    for y in YEARS:
        f = os.path.join(STACKS, f"session_{y}.npz")
        if not os.path.exists(f):
            continue
        d = np.load(f)
        S[y] = (d["Gm"], d["mask"]); day[y] = mean_day(y)
    ref = REF if REF in S else sorted(S)[len(S) // 2]
    print(f"reference session: {ref}")
    print(f"\n  year  day(J2000)   Δlon(deg)  Δlat(deg)   sig")
    print("  " + "-" * 50)
    rows = []
    for y in sorted(S):
        if y == ref:
            dlon, dlat, sig = 0.0, 0.0, np.inf
        else:
            dr, dc, sig = register_maps(S[ref][0], S[y][0], valid_a=S[ref][1],
                                        valid_b=S[y][1], max_shift=60,
                                        smooth_px=7.0, trend_px=55.0)
            dlat, dlon = offset_to_lonlat_deg(-dr, -dc, (HH, WW))
        rows.append((y, day[y], dlon, dlat, sig))
        print(f"  {y}  {day[y]:9.1f}  {dlon:+8.3f}  {dlat:+8.3f}  {sig:.2f}")

    days = np.array([r[1] for r in rows]); dlon = np.array([r[2] for r in rows])
    sig = np.array([r[4] for r in rows])
    w = np.where(np.isfinite(sig), sig, np.nanmax(sig[np.isfinite(sig)])) ** 2
    assumed = Spin().period_days
    fit = fit_rotation_from_offsets(days, dlon, period_to_wdot(assumed), weights=w)
    boot = bootstrap_period(days, dlon, period_to_wdot(assumed), weights=w)

    print("\n=== ROTATION ELEMENTS (sanity check, fixed/Arecibo stacks) ===")
    print(f"  assumed (IAU) period : {assumed:.5f} d")
    print(f"  venera fit period    : {fit['period_days']:.5f} ± {fit['sigma_period_days']:.5f} d")
    print(f"  bootstrap period     : {boot['period_median']:.5f} "
          f"[{boot['period_p16']:.5f}, {boot['period_p84']:.5f}] d")
    print(f"  pole (adopted, IAU)  : RA 272.76°, Dec 67.16°  (venera floor ±0.15°/±1.2°)")
    print(f"  RMS lon residual     : {fit['rms_resid_deg']*1000:.1f} mdeg over {fit['n']} epochs")
    print(f"  max |Δlat| (QA)      : {max(abs(r[3]) for r in rows):.3f}° (should be ~0 post-fix)")
    print(f"  baseline             : {(days.max()-days.min())/365.25:.1f} yr")
    print("  literature: Campbell 2019 243.0212±0.0006 | Margot 2020 243.0226±0.0013 "
          "| Magellan/IAU 243.0185")


if __name__ == "__main__":
    main()
