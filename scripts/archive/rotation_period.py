"""Venus rotation period via ROTATION registration. [RETRACTED — see REPORT.md §12]

RETRACTED: the rigid-rotation real-data period (243.0174 ± 0.0016 d) from this
script FAILED the imagery test — applying the fitted ω makes cross-epoch NCC worse,
and the per-pair ω is order-inconsistent by ~2°. Real cross-epoch differences are a
non-rigid warp (REPORT.md §9), not a rigid rotation. Kept as history only; not an
estimator.

For each epoch pair, fit a rigid sphere rotation ω to the tile displacement field
and take ω_z (rotation about the pole) as the longitude/spin shift — separating
the spin from the doppler-angle-driven tilt that a translation matcher mis-reads
as longitude. Then closure-solve the per-epoch longitudes and fit the period.

Usage: .conda/bin/python scripts/rotation_period.py [cache_tag] [N]
"""
import os, sys, itertools
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from venera.registration import bandpass, tile_displacements
from venera.rotation_fit import (fit_tile_rotation, longitude_shift_from_rotation,
                                 solve_relative_longitudes, fit_rotation_from_offsets,
                                 bootstrap_period, fit_rotation_rate_change,
                                 period_to_wdot)

H, W = 4000, 8000
YRS = ["1988", "2001", "2012", "2015", "2017", "2020"]
CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "results", "epoch_stacks")
tag = sys.argv[1] if len(sys.argv) > 1 else "cf"
N = sys.argv[2] if len(sys.argv) > 2 else "20"


def load(y):
    d = np.load(os.path.join(CACHE, f"stack_both_{y}_N{N}_{tag}.npz"))
    return d["Gm"], d["mask"], float(d["mean_day"])


E = {y: load(y) for y in YRS}
days = np.array([E[y][2] for y in YRS])

print(f"=== rotation-registration period (tag {tag}, N={N}) ===")
print(f"{'pair':14s} {'ω_z=Δlon':>9s} {'tilt(ωx,ωy)':>14s} {'rot_resid':>9s} {'ntile':>6s}")
raw = []
for i, j in itertools.combinations(range(len(YRS)), 2):
    tl = tile_displacements(E[YRS[i]][0], E[YRS[j]][0], E[YRS[i]][1], E[YRS[j]][1])
    if len(tl) < 8:
        continue
    om, rr = fit_tile_rotation(tl)
    raw.append((i, j, longitude_shift_from_rotation(om), 1.0))
    print(f"{YRS[i]}<->{YRS[j]} {longitude_shift_from_rotation(om):+9.3f} "
          f"{np.degrees(om[0]):+6.2f},{np.degrees(om[1]):+6.2f} {rr*1000:9.0f} {len(tl):6d}")

lon, crms, _ = solve_relative_longitudes(len(YRS), raw)
fit = fit_rotation_from_offsets(days, lon, period_to_wdot(243.0185))
boot = bootstrap_period(days, lon, period_to_wdot(243.0185))
rc = fit_rotation_rate_change(days, lon, period_to_wdot(243.0185))
print(f"\nclosure={crms*1000:.0f} mdeg   line-resid={fit['rms_resid_deg']*1000:.0f} mdeg")
for y, r in zip(YRS, fit["residuals_deg"]):
    print(f"   {y}: {r*1000:+.0f} mdeg")
print(f"\nP = {fit['period_days']:.5f} ± {fit['sigma_period_days']:.5f} d   "
      f"bootstrap[{boot['period_p16']:.5f}, {boot['period_p84']:.5f}]")
print(f"dP/dt = {rc['dP_dt_days_per_year']*1000:+.2f} ± "
      f"{rc['sigma_dP_dt_days_per_year']*1000:.2f} mday/yr ({rc['rate_change_significance']:.1f}σ)")
print("Campbell 243.0212±0.0006 | Margot 243.0226±0.0013 | Magellan 243.0185")
