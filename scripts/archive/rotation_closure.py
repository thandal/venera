"""Reference-independent rotation period via a global closure solve.

Register ALL epoch pairs, solve for self-consistent per-epoch longitudes by
weighted least-squares (closure), then fit the period to longitude vs time. This
replaces the single-reference fit (which inherited the reference's degeneracies)
and reports a closure residual that exposes unreliable pairs (moon-radar B3).

Run: .conda/bin/python scripts/rotation_closure.py
"""
import os, sys, itertools
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from venera.registration import register_maps
from venera.rotation_fit import (solve_relative_longitudes, fit_rotation_from_offsets,
                                 bootstrap_period, period_to_wdot)

H, W = 4000, 8000
YEARS = ["1988", "2001", "2012", "2015", "2017", "2020"]
CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "results", "epoch_stacks")


def load(y):
    d = np.load(os.path.join(CACHE, f"stack_both_{y}_N30_sci.npz"))
    return d["Gm"], d["mask"], float(d["mean_day"])


def main():
    E = {y: load(y) for y in YEARS}
    days = np.array([E[y][2] for y in YEARS])

    # Register all pairs.
    raw = []  # (i, j, dlon_deg, dlat_deg, sig)
    print("=== all-pairs registration ===")
    for i, j in itertools.combinations(range(len(YEARS)), 2):
        gi, mi, _ = E[YEARS[i]]
        gj, mj, _ = E[YEARS[j]]
        dr, dc, sig = register_maps(gi, gj, valid_a=mi, valid_b=mj, max_shift=70,
                                    smooth_px=7.0, trend_px=55.0)
        dlon = -dc / W * 360.0
        dlat = -dr / H * 180.0
        raw.append((i, j, dlon, dlat, sig))
        print(f"  {YEARS[i]}<->{YEARS[j]}: dlon={dlon:+.3f} dlat={dlat:+.3f} sig={sig:.2f}")

    def run(label, pairs):
        lon, crms, resid = solve_relative_longitudes(len(YEARS), pairs)
        # period fit: solved longitude vs time (uniform weights — closure already used sig)
        fit = fit_rotation_from_offsets(days, lon, period_to_wdot(243.0185))
        boot = bootstrap_period(days, lon, period_to_wdot(243.0185))
        print(f"\n=== {label} ({len(pairs)} pairs) ===")
        print(f"  closure RMS = {crms*1000:.1f} mdeg")
        print(f"  P = {fit['period_days']:.5f} ± {fit['sigma_period_days']:.5f} d  "
              f"(bootstrap [{boot['period_p16']:.5f}, {boot['period_p84']:.5f}])")
        print(f"  fit RMS resid = {fit['rms_resid_deg']*1000:.0f} mdeg")
        worst = np.argsort(-np.abs(resid))[:3]
        for k in worst:
            i, j, *_ = pairs[k]
            print(f"    worst closure pair {YEARS[i]}<->{YEARS[j]}: {resid[k]*1000:+.0f} mdeg")
        return fit

    # (a) all pairs, sig^2 weighted
    run("all pairs, sig^2 weighted",
        [(i, j, dlon, sig ** 2) for i, j, dlon, dlat, sig in raw])

    # (b) drop degenerate pairs (large |dlat| = lat-lon ridge), sig^2 weighted
    clean = [(i, j, dlon, sig ** 2) for i, j, dlon, dlat, sig in raw if abs(dlat) < 0.7]
    run("clean pairs only (|dlat|<0.7deg), sig^2 weighted", clean)


if __name__ == "__main__":
    main()
