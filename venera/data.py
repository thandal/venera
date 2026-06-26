"""Loading and coarse preprocessing of Arecibo Venus delay-Doppler ``.img`` files.

Ported from the working ``process_radar_images.ipynb`` pipeline, cleaned up. The
coarse-centering steps here (symmetry roll keyed on baud) are the ones moon-radar
lesson B1 says to replace with per-look echo self-calibration (task #7); kept
faithful for now so the projection can be validated against the existing outputs.
"""

import os
import numpy as np
from scipy.ndimage import uniform_filter1d

from . import selfcal

# Raw file layout: 8191 x 8192 complex64 (little-endian, real/imag float32 pairs).
IMG_SHAPE = (8191, 8192)
IMG_DTYPE = "<F"

# Baud (microseconds) -> delay-axis symmetry window (rows), from the notebook.
# Different baud/range-rate spreads the echo differently; these windows avoid the
# Doppler-wrap region when measuring left/right symmetry.
_BAUD_SYMMETRY_WINDOW = {
    4.2: (2800, 3800),   # 2001
    4.0: (3000, 4000),   # 1988
    3.9: (4000, 5000),   # 2017
    3.8: (4000, 5000),   # 2015 (and default); 2012/2020 overridden below
}


def parse_lbl(img_path):
    """Parse the sibling ``.lbl`` for the geometry/timing fields we use."""
    lbl_path = img_path[:-4] + ".lbl" if img_path.endswith(".img") else img_path
    d = {}
    for line in open(lbl_path):
        if "START_TIME" in line:
            d["START_TIME"] = line.split("=")[-1].strip()
        elif "STOP_TIME" in line:
            d["STOP_TIME"] = line.split("=")[-1].strip()
        elif "GEO:BAUD" in line:
            d["GEO_BAUD"] = float(line.split("=")[-1].split()[0])
        elif "GEO:CODE_LENGTH" in line:
            d["GEO_CODE_LENGTH"] = int(line.split("=")[1].strip())
        elif "GEO:CENTROID_LOCATION" in line:
            d["GEO_CENTROID_LOCATION"] = int(line.split("=")[1].strip())
        elif "GEO:DELAY_OFFSET" in line:
            d["GEO_DELAY_OFFSET"] = int(line.split("=")[1].strip())
        elif "GEO:POINTING" in line:
            d["GEO_POINTING"] = line.split('"')[-2]
        elif "GEO:MODE" in line:
            d["GEO_MODE"] = line.split('"')[-2]
        elif "INSTRUMENT_ID" in line:
            d["INSTRUMENT_ID"] = line.split("=", 1)[-1].strip()
    # Receiver station: bistatic looks (GEO:MODE "B", instrument set includes GBT)
    # are transmitted from Arecibo but RECEIVED at the Green Bank Telescope. The
    # receiver sets the Doppler axis, so the projector must know it (see geometry).
    is_bistatic = d.get("GEO_MODE") == "B" or "GBT" in d.get("INSTRUMENT_ID", "")
    d["RX_STATION"] = "GBT" if is_bistatic else "ARECIBO"
    return d


def load_raw(img_path):
    """Memmap the raw complex image (rows=round-trip delay, cols=Doppler)."""
    img = np.memmap(img_path, dtype=IMG_DTYPE, shape=IMG_SHAPE, mode="r")
    # 1988 and 2020 data have a flipped Doppler axis (documented hack).
    base = os.path.basename(img_path)
    if base[10:14] in ("1988", "2020"):
        img = np.fliplr(img)
    return img


def coarse_preprocess(img):
    """Complex -> normalized magnitude; roll Doppler so zero-Doppler is centered."""
    c2 = img.shape[1] // 2
    a = np.abs(img).astype(np.float32)
    a -= a.min()
    a /= a.max()
    return np.roll(a, c2, axis=1)


def coarse_rollup(a):
    """Row shift so the sub-radar-point (SRP) onset sits at row 0."""
    c2 = a.shape[1] // 2
    c = np.sum(a[:, 3500:-3500], axis=1)
    max_c = np.argmax(c)
    c = np.roll(c, c2 - max_c)
    w = 200
    d = np.diff(c[c2 - w:c2 + 1])
    d_pre_std = np.std(d[: c2 - w // 2])
    first_i = np.argwhere(d > d_pre_std * 2)[0][0]
    return -(max_c - w + first_i) - 1


def normalize_ocp_by_range(a):
    echo_power = 10 / np.linspace(1, 1000, a.shape[0]) + 0.04
    return (a.T / echo_power).T


def normalize_incidence(a, smooth=41, floor_frac=0.05):
    """Divide out the empirical per-delay-row brightness law (moon-radar B9).

    Delay row ~ 1-cos(incidence), so the median intensity per row *is* the
    incidence/scattering law. Dividing it out flattens the bright quasi-specular
    sub-radar falloff while preserving per-pixel feature contrast (deviations from
    the row median), making looks — and SCP/OCP — comparable for stacking and
    lifting feature contrast away from the disk center. Applies to both pols.
    """
    c2 = a.shape[1] // 2
    prof = np.median(a[:, c2 - 2500:c2 + 2500], axis=1).astype(np.float32)
    prof = uniform_filter1d(prof, smooth)
    prof = np.maximum(prof, floor_frac * prof.max())
    return a / prof[:, None]


def clip_percentile(a, percentile=99):
    thresh = np.percentile(a.ravel(), percentile)
    return np.where(a > thresh, thresh, a)


def coarse_roll_symmetry(a, baud, year=None):
    """Fine Doppler centering by maximizing left/right mirror symmetry."""
    if (year in ("2012", "2020")) and baud == 3.8:
        lo, hi = 3500, 4500
    else:
        lo, hi = _BAUD_SYMMETRY_WINDOW.get(baud, (4000, 5000))
    h = a[lo:hi, :]
    rng = range(-100, 101) if year == "1988" else range(-50, 51)
    best_off, best_sum = 0, -np.inf
    for off in rng:
        t = np.roll(h, off, axis=1)
        s = np.sum(t[:, :1000] * np.fliplr(t[:, -1000:]))
        if s > best_sum:
            best_sum, best_off = s, off
    return best_off


def preprocess(img_path):
    """Full coarse preprocessing -> (processed float32 image, lbl dict, cal dict).

    Doppler centering uses per-look self-calibration (limb-edge fit, consistent
    across years) when its quality is good; otherwise it falls back to the
    baud-keyed symmetry roll. ``cal`` carries the centering method and the
    sub-pixel Doppler residual for the projector.

    NOTE: per the PDS archive description (venus_radar.pdf), the label's
    GEO:CENTROID_LOCATION is only the *expected* zero-Doppler column; the true
    one (the GEO:DOPPLER_CENTROID_OFFSET) must be recovered from the echo, which
    is what self-calibration does.
    """
    lbl = parse_lbl(img_path)
    base = os.path.basename(img_path)
    year = base[10:14]
    is_ocp = "ocp" in base
    raw = load_raw(img_path)
    a = coarse_preprocess(raw)
    # Delay (row) centering: prefer the label's GEO:DELAY_OFFSET, which is the
    # ephemeris-derived round-trip-delay reference for the sub-radar point — EXACT,
    # set by the (rock-solid) station clocks + ephemeris. The brightness-onset
    # heuristic coarse_rollup re-discovers this and is stable on monostatic data
    # (matches the constant DELAY_OFFSET=25 for 2017) but FAILS on the bistatic 2012
    # campaign, where the Arecibo->Venus->GBT delay shifts by thousands of rows
    # look-to-look (DELAY_OFFSET 863..6091) — coarse_rollup scatters ±1334 rows
    # there, dumping mis-placed looks as noise that blurs the stack. Trust the label.
    do_lbl = lbl.get("GEO_DELAY_OFFSET")
    if do_lbl is not None:
        delay_roll = -int(do_lbl)
        centering = "label_delay"
    else:
        delay_roll = coarse_rollup(a)
        centering = "rollup"
    a = np.roll(a, delay_roll, axis=0)
    a = normalize_incidence(a)   # empirical incidence-law flattening, both pols
    a = clip_percentile(a)
    # Absolute Doppler centering is done in projection.project_file via a WIDE
    # limb-model curve fit (fit_delay_doppler_curve).
    return a, lbl, {"centering": centering, "delay_roll": delay_roll}
