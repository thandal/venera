"""Validate the rotation-period fit: math/units in isolation, then integrated with
registration on synthetic shifted+speckled maps across a 1988-2020 epoch spread.

Run: .conda/bin/python test/test_rotation_fit.py
"""
import os, sys
import numpy as np
from scipy.ndimage import gaussian_filter, fourier_shift

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from venera.rotation_fit import (period_to_wdot, fit_rotation_from_offsets,
                                 bootstrap_period, solve_relative_longitudes,
                                 fit_tile_rotation, longitude_shift_from_rotation)
from venera.registration import register_maps

_fail = 0
def check(name, cond, detail=""):
    global _fail
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {detail}")
    if not cond:
        _fail += 1

# Approximate days past J2000 for the six observing campaigns.
DAYS = np.array([-4216., 455., 4531., 5703., 6294., 7455.])  # 1988..2020
ASSUMED_P = 243.0185
TRUE_P = 243.0210
ASSUMED_WDOT = period_to_wdot(ASSUMED_P)
TRUE_WDOT = period_to_wdot(TRUE_P)
SLOPE_TRUE = TRUE_WDOT - ASSUMED_WDOT


print("== 1. Fit math: recover injected period from clean offsets ==")
dlon_clean = SLOPE_TRUE * (DAYS - DAYS[3])  # relative to 2015 reference
r = fit_rotation_from_offsets(DAYS, dlon_clean, ASSUMED_WDOT)
print(f"    injected P={TRUE_P}  recovered P={r['period_days']:.6f}  "
      f"slope {r['slope_deg_per_day']:.3e} vs {SLOPE_TRUE:.3e}")
check("clean recovery to 1e-5 day", abs(r["period_days"] - TRUE_P) < 1e-5,
      f"|ΔP|={abs(r['period_days']-TRUE_P):.2e}")


print("== 2. Fit math: noisy offsets -> error bar covers truth ==")
rng = np.random.default_rng(1)
sigma_lon = 0.02  # deg per-epoch registration noise (~moon-radar sub-pixel scale)
dlon_noisy = dlon_clean + rng.normal(0, sigma_lon, len(DAYS))
r2 = fit_rotation_from_offsets(DAYS, dlon_noisy, ASSUMED_WDOT)
within = abs(r2["period_days"] - TRUE_P) < 3 * r2["sigma_period_days"]
print(f"    recovered P={r2['period_days']:.5f} ± {r2['sigma_period_days']:.5f}  "
      f"(truth {TRUE_P})")
check("3-sigma covers truth", within)
check("sigma in the Campbell ballpark (< 2 mday)", r2["sigma_period_days"] < 2e-3,
      f"σ={r2['sigma_period_days']*1000:.3f} mday")


print("== 3. Integrated fit+registration on synthetic shifted+speckled maps ==")
# Full 360-deg equirectangular grid; feature band in the north.
H, W = 300, 4000
deg_per_px = 360.0 / W
true = np.zeros((H, W), float)
rng2 = np.random.default_rng(2)
true[60:240, :] = gaussian_filter(rng2.standard_normal((180, W)), 2.5)  # stable features
valid = np.zeros((H, W), bool)
valid[60:240, 200:3800] = True

ref_idx = 3
maps = []
for d in DAYS:
    dlam_deg = SLOPE_TRUE * d                  # absolute assigned-lon shift for this epoch
    shift_px = dlam_deg / deg_per_px
    shifted = np.real(np.fft.ifft2(fourier_shift(np.fft.fft2(true), (0, shift_px))))
    speckle = gaussian_filter(rng2.standard_normal((H, W)), 2.5) * 0.6  # decorrelated
    maps.append(shifted + speckle)

dlon_meas, weights = [], []
for k, d in enumerate(DAYS):
    drow, dcol, sig = register_maps(maps[ref_idx], maps[k], valid_a=valid,
                                    valid_b=valid, max_shift=40)
    # measured assigned-lon offset of epoch k relative to ref (see module docstring)
    dlon_meas.append(-dcol * deg_per_px)
    weights.append(sig ** 2)
    print(f"    epoch d={d:7.0f}: dcol={dcol:+.3f}px -> Δlon={-dcol*deg_per_px:+.4f}deg "
          f"(true {SLOPE_TRUE*(d-DAYS[ref_idx]):+.4f})  sig={sig:.2f}")

r3 = fit_rotation_from_offsets(DAYS, dlon_meas, ASSUMED_WDOT, weights=weights)
boot = bootstrap_period(DAYS, dlon_meas, ASSUMED_WDOT, weights=weights)
print(f"    recovered P={r3['period_days']:.5f} ± {r3['sigma_period_days']:.5f} day  "
      f"(truth {TRUE_P})")
print(f"    bootstrap  P={boot['period_median']:.5f} [{boot['period_p16']:.5f}, "
      f"{boot['period_p84']:.5f}]  rms_resid={r3['rms_resid_deg']*1000:.2f} mdeg")
check("integrated recovery within 3 mday of truth",
      abs(r3["period_days"] - TRUE_P) < 3e-3, f"|ΔP|={abs(r3['period_days']-TRUE_P)*1000:.2f} mday")
check("integrated 3-sigma covers truth",
      abs(r3["period_days"] - TRUE_P) < 3 * r3["sigma_period_days"])


print("== 4. Closure solve recovers consistent longitudes + flags bad pairs ==")
import itertools
true_lon = np.array([0.0, 0.3, -0.2, 0.5, -0.4, 0.1])
rng4 = np.random.default_rng(7)
pairs = []
for i, j in itertools.combinations(range(len(true_lon)), 2):
    pairs.append((i, j, (true_lon[i] - true_lon[j]) + rng4.normal(0, 0.01), 1.0))
# inject one badly-degenerate pair (wrong offset, but full weight) -> should surface
bad_k = 3
i, j, _, _ = pairs[bad_k]
pairs[bad_k] = (i, j, true_lon[i] - true_lon[j] + 0.8, 1.0)
lon, crms, resid = solve_relative_longitudes(len(true_lon), pairs)
check("bad pair flagged by largest closure residual",
      int(np.argmax(np.abs(resid))) == bad_k, f"argmax={np.argmax(np.abs(resid))}, bad={bad_k}")
# real workflow: down-weight the flagged pair, re-solve -> clean recovery
pairs[bad_k] = (pairs[bad_k][0], pairs[bad_k][1], pairs[bad_k][2], 0.01)
lon2, crms2, _ = solve_relative_longitudes(len(true_lon), pairs)
lon2 -= lon2.mean(); tl = true_lon - true_lon.mean()
err = np.max(np.abs(lon2 - tl))
check("down-weighting the flagged pair recovers longitudes < 0.05deg", err < 0.05,
      f"max err={err:.3f}deg, closure_rms {crms*1000:.0f}->{crms2*1000:.0f} mdeg")

print("== 5. Tile-rotation fit recovers an injected rotation (spin separated from tilt) ==")
omega_true = np.array([0.0040, -0.0025, 0.0060])   # rad; ω_z=spin/longitude, ω_x,ω_y=tilt
rng5 = np.random.default_rng(11)
tl = []
for lat in range(5, 70, 8):
    for lon in range(-90, 60, 12):
        la, lo = np.radians(lat), np.radians(lon)
        r = np.array([np.cos(la)*np.cos(lo), np.cos(la)*np.sin(lo), np.sin(la)])
        e = np.array([-np.sin(lo), np.cos(lo), 0.0])
        n = np.array([-np.sin(la)*np.cos(lo), -np.sin(la)*np.sin(lo), np.cos(la)])
        dr = np.cross(omega_true, r)
        dlat = np.degrees(np.dot(dr, n)) + rng5.normal(0, 0.05)     # add 0.05deg tile noise
        dlon = np.degrees(np.dot(dr, e) / np.cos(la)) + rng5.normal(0, 0.05)
        tl.append((lat, lon, dlat, dlon))
om, rr = fit_tile_rotation(tl)
err = np.degrees(np.max(np.abs(om - omega_true)))
lon_err = abs(longitude_shift_from_rotation(om) - np.degrees(omega_true[2]))
print(f"    injected ω(deg)={np.round(np.degrees(omega_true),3)} -> recovered {np.round(np.degrees(om),3)}  "
      f"resid={rr*1000:.0f}mdeg")
check("rotation recovered < 0.05deg (incl. ω_z = longitude)", err < 0.05 and lon_err < 0.05,
      f"max err={err:.3f}deg, ω_z err={lon_err:.3f}deg")
# a pure translation in the data (uniform dlon, no rotation) -> ω_z ~ that shift, tilt ~ 0
tl2 = [(lat, lon, 0.0, 0.30) for lat in range(10, 60, 10) for lon in range(-80, 40, 15)]
om2, _ = fit_tile_rotation(tl2)
check("uniform longitude shift -> ω_z≈shift, tilt≈0",
      abs(longitude_shift_from_rotation(om2) - 0.30) < 0.05
      and abs(np.degrees(om2[0])) < 0.05 and abs(np.degrees(om2[1])) < 0.05,
      f"ω_z={longitude_shift_from_rotation(om2):.3f}, tilt=({np.degrees(om2[0]):.3f},{np.degrees(om2[1]):.3f})")

print()
if _fail:
    print(f"FAILED ({_fail} check(s))")
    sys.exit(1)
print("ALL ROTATION-FIT CHECKS PASSED")
