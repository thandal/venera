"""Tile-consensus longitude over well-overlapping epochs -> per-epoch line-fit
residual (the absolute-longitude scatter that limits the period) + period.

Usage: .conda/bin/python scripts/analyze_lineresidual.py <cache_tag> <N>
  e.g. analyze_lineresidual.py cf 20    -> stack_both_<year>_N20_cf.npz
"""
import os, sys, itertools
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from venera.registration import bandpass, xcorr_offset
from venera.rotation_fit import (solve_relative_longitudes, fit_rotation_from_offsets,
                                 bootstrap_period, period_to_wdot)

H, W = 4000, 8000
DPP = 360.0 / W
YRS = ["1988", "2001", "2012", "2017", "2020"]      # well-overlapping (drop far-north 2015)
CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "results", "epoch_stacks")

tag = sys.argv[1] if len(sys.argv) > 1 else "cf"
N = sys.argv[2] if len(sys.argv) > 2 else "20"


def load(y):
    d = np.load(os.path.join(CACHE, f"stack_both_{y}_N{N}_{tag}.npz"))
    return d["Gm"], d["mask"], float(d["mean_day"])


E = {y: load(y) for y in YRS}
F = {y: bandpass(E[y][0], 7, 55) for y in YRS}


def consensus(a, b, tile=384, step=240, minfrac=0.5, min_sig=0.95):
    fa, ma = F[a], E[a][1]
    fb, mb = F[b], E[b][1]
    common = ma & mb
    dcs = []
    for r0 in range(0, H - tile + 1, step):
        for c0 in range(0, W - tile + 1, step):
            tc = common[r0:r0 + tile, c0:c0 + tile]
            if tc.mean() < minfrac:
                continue
            dr, dc, sig = xcorr_offset(fa[r0:r0 + tile, c0:c0 + tile] * tc,
                                       fb[r0:r0 + tile, c0:c0 + tile] * tc, max_shift=45)
            if sig < min_sig:
                continue
            dcs.append(dc)
    dcs = np.array(dcs)
    if len(dcs) < 6:
        return None
    med = np.median(dcs)
    mad = 1.4826 * np.median(np.abs(dcs - med)) + 1e-9
    dcs = dcs[np.abs(dcs - med) < 2.5 * mad]
    return -np.median(dcs) * DPP


raw = []
for i, j in itertools.combinations(range(len(YRS)), 2):
    d = consensus(YRS[i], YRS[j])
    if d is not None:
        raw.append((i, j, d, 1.0))
days = np.array([E[y][2] for y in YRS])
lon, crms, _ = solve_relative_longitudes(len(YRS), raw)
fit = fit_rotation_from_offsets(days, lon, period_to_wdot(243.0185))
boot = bootstrap_period(days, lon, period_to_wdot(243.0185))

print(f"=== centering tag '{tag}', N={N} ===")
print(f"closure RMS (relative registration) = {crms*1000:.1f} mdeg")
print(f"line-fit RMS residual (ABSOLUTE per-epoch scatter) = {fit['rms_resid_deg']*1000:.0f} mdeg"
      f"   (self-cal centering was 391)")
for y, r in zip(YRS, fit["residuals_deg"]):
    print(f"   {y}: residual {r*1000:+.0f} mdeg")
print(f"P = {fit['period_days']:.5f} ± {fit['sigma_period_days']:.5f} d   "
      f"bootstrap[{boot['period_p16']:.5f},{boot['period_p84']:.5f}]")
print("Campbell 243.0212±0.0006 | Margot 243.0226±0.0013")
