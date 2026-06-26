"""How well can venera constrain Venus' pole? (geometry-only sensitivity).

The pole enters the projection through the sub-radar latitude and the Doppler
angle (which rotates features about the SRP). We perturb the pole by 0.1 deg and
measure the change in those observables per epoch, then convert the Doppler-angle
change to a feature shift and compare against the ~0.16 deg registration noise.

Conclusion (see output): Dec is constrainable to ~0.15 deg, RA only to ~1 deg —
both consistent with, but far coarser than, the IAU/Magellan pole (+-0.01 deg).
venera should adopt IAU 272.76 / 67.16; the 2023 writeup's 0.01-0.03 deg "offsets"
are below this noise floor and are not detections.

Run: .conda/bin/python scripts/pole_sensitivity.py
"""
import os, sys
import numpy as np
import cspyce as csp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from venera import spice_setup
from venera.geometry import Spin, sub_radar_point, doppler_angle

spice_setup.furnsh_kernels()

EPOCHS = {"1988": "1988-06-17T15:00", "2001": "2001-03-31T15:00",
          "2012": "2012-05-29T16:00", "2015": "2015-08-13T16:00",
          "2017": "2017-03-26T17:00", "2020": "2020-05-30T17:00"}

REG_NOISE_DEG = 0.16        # measured per-epoch registration scatter
FEATURE_DIST_DEG = 45.0     # typical distance of a registration feature from the SRP

base = Spin()                                   # IAU
dRA = Spin(pole_ra_deg=272.76 + 0.1)
dDEC = Spin(pole_dec_deg=67.16 + 0.1)

print(f"{'epoch':6s} {'SRPlat':>7s} {'dDopAng/dRA':>12s} {'dDopAng/dDec':>13s}  (deg per +0.1deg pole)")
da_RA, da_DEC = {}, {}
for y, t in EPOCHS.items():
    et = csp.str2et(t)
    lat0 = np.degrees(sub_radar_point(et, base)[1])
    da0 = np.degrees(doppler_angle(et, base))
    da_RA[y] = np.degrees(doppler_angle(et, dRA)) - da0
    da_DEC[y] = np.degrees(doppler_angle(et, dDEC)) - da0
    print(f"{y:6s} {lat0:7.2f} {da_RA[y]:12.3f} {da_DEC[y]:13.3f}")

# Best leverage = most SRP-diverse pair (2015 vs 2017).
diff_RA = abs(da_RA["2017"] - da_RA["2015"]) / 0.1     # deg DopAng per deg RA
diff_DEC = abs(da_DEC["2017"] - da_DEC["2015"]) / 0.1
rho = np.radians(FEATURE_DIST_DEG)
# feature shift (deg) per deg pole = rho(rad) * dDopAng(deg)
shift_per_RA = rho * diff_RA
shift_per_DEC = rho * diff_DEC
print(f"\n2015<->2017 leverage: {diff_RA:.2f} deg DopAng/deg RA, {diff_DEC:.2f} deg DopAng/deg Dec")
print(f"feature shift @ {FEATURE_DIST_DEG} deg: {shift_per_RA:.3f} deg/deg RA, "
      f"{shift_per_DEC:.3f} deg/deg Dec")
print(f"=> venera pole precision ~ {REG_NOISE_DEG/shift_per_RA:.2f} deg (RA), "
      f"{REG_NOISE_DEG/shift_per_DEC:.2f} deg (Dec)")
print(f"   IAU/Magellan pole precision ~0.01 deg  =>  venera cannot improve on it.")
print(f"   2023 writeup claimed offsets: RA 0.01 deg, Dec 0.03 deg -> below venera's floor.")
