"""Investigate the per-image freq_scale (the ~10% early/late step + residual error),
now that the observer is fixed (Arecibo apparent-rotation rate is correct).

For a few looks per session: the per-image FITTED freq_scale (the limb-curve fit in
projection) vs the GEOMETRIC prediction (geometry.predicted_freq_scale, from the
apparent-rotation bandwidth). If the geometric prediction (with the correct Arecibo
wapp) now tracks the fit, freq_scale can be set from geometry (one calibration
constant) instead of fit per image — removing per-image scatter and the early/late
step if it is a fit artifact. If they disagree, the step is a real instrument/
processing difference between campaigns.

Usage: .conda/bin/python scripts/freq_scale_check.py
"""
import os, sys, glob
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from venera import spice_setup
from venera.geometry import Spin, predicted_freq_scale, apparent_rotation_rate
from venera.data import parse_lbl
from venera.projection import project_file
import cspyce as csp

spice_setup.furnsh_kernels()
DATA = ("/home/than/code/venera/arecibo_radar/pds-geosciences.wustl.edu/venus/"
        "arcb_nrao-v-rtls_gbt-3-delaydoppler-v1/vrm_90xx/data/")
H, W = 4000, 8000
YEARS = ["1988", "2001", "2012", "2015", "2017", "2020"]
K = 0.968   # geometry.CALIBRATED_FREQ_SCALE_K (calibrated pre-fix; re-check here)


def pick(year, n=3):
    fs = sorted(glob.glob(os.path.join(DATA, f"venus_scp_{year}*.img")))
    keep = [f for f in fs if parse_lbl(f).get("GEO_POINTING") == "N"]
    return keep[:n]


def main():
    spin = Spin()
    print(f"  year   baud  code_len   fitted_fs        geom_fs(K={K})   ratio fit/geom")
    print("  " + "-" * 66)
    for y in YEARS:
        looks = pick(y)
        fits, geoms = [], []
        for f in looks:
            lbl = parse_lbl(f)
            et0 = csp.str2et(lbl["START_TIME"]); et1 = csp.str2et(lbl["STOP_TIME"])
            et_mid = 0.5 * (et0 + et1)
            G = np.zeros((H, W), np.float32); Gc = np.zeros((H, W), np.int32)
            info = project_file(f, spin, G, Gc)
            fitted = info["fit"][2]
            geom = predicted_freq_scale(et_mid, spin, lbl["GEO_BAUD"],
                                        lbl.get("GEO_CODE_LENGTH", 1), K=K)
            fits.append(fitted); geoms.append(geom)
        fits = np.array(fits); geoms = np.array(geoms)
        cl = parse_lbl(looks[0]).get("GEO_CODE_LENGTH", "?")
        bd = parse_lbl(looks[0])["GEO_BAUD"]
        print(f"  {y}   {bd:.1f}   {cl!s:>6}    {fits.mean():.3f}±{fits.std():.3f}    "
              f"{geoms.mean():.3f}±{geoms.std():.3f}    {(fits/geoms).mean():.3f}")
    print("\nIf ratio fit/geom is ~constant across epochs, geometric freq_scale (with a "
          "single recalibrated K) reproduces the fit -> set it from geometry, drop the "
          "per-image fs fit. If the early/late step survives in the ratio, it is a real "
          "campaign calibration difference.")


if __name__ == "__main__":
    main()
