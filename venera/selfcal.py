"""Per-look echo self-calibration (moon-radar lesson B1).

Replaces the baud-keyed left/right symmetry roll (the "baud isn't the right thing"
hack, which measures centering over a *different delay window per year* and so
injects a per-year longitude bias) with a single, consistent, sub-pixel Doppler
centering derived from the echo's own limb.

The zero-Doppler locus is a **tilted line** in the (delay, Doppler) image because
the apparent Doppler axis is tilted (the 7-14 deg Doppler angle). So we:
  1. find the two half-power Doppler limb edges at each delay row,
  2. robustly fit a line to their midpoints vs row (IRLS, outlier-rejecting),
  3. take the intercept at the sub-radar row as the Doppler center.

The line's slope independently cross-checks the geometric Doppler angle; the limb
half-width gives a freq_scale diagnostic. A quality flag (fit scatter, #rows)
gates noisy looks (the 1988 autofocus and 2020 low-quality data) so callers can
fall back to the coarse symmetry roll.
"""

import numpy as np
from scipy.ndimage import uniform_filter1d


def _edge_crossings(profile, mode="limb", frac=0.35, noise_k=4.0, bg_cols=300):
    """Sub-pixel left/right Doppler edges of a (smoothed) profile.

    mode="limb": threshold is **noise-relative** (bg + noise_k*sigma) — finds the
    geometric disk boundary where the echo meets the background, *independent of
    interior brightness* (a bright feature like Maxwell does not move the limb).
    mode="halfpower": threshold is **peak-relative** (bg + frac*(peak-bg)) — the
    old interior-symmetry measure, biased by bright features.
    """
    p = profile
    bg = np.median(np.r_[p[:bg_cols], p[-bg_cols:]])
    noise = np.std(p[:bg_cols]) + 1e-9
    m = p.max()
    if m - bg < 5 * noise:
        return None
    thr = (bg + noise_k * noise) if mode == "limb" else (bg + frac * (m - bg))
    if mode == "limb":
        # Edges of the connected above-threshold run that contains the disk peak.
        # The Doppler aliasing/folding sits in DISCONNECTED runs at the FFT wrap
        # (far columns) and is excluded by walking out only while contiguous.
        pk = int(np.argmax(p))
        if p[pk] <= thr:
            return None
        iL = pk
        while iL > 0 and p[iL - 1] > thr:
            iL -= 1
        iR = pk
        while iR < len(p) - 1 and p[iR + 1] > thr:
            iR += 1
    else:
        idx = np.where(p > thr)[0]
        if len(idx) == 0:
            return None
        iL, iR = int(idx.min()), int(idx.max())
    if iL > 0 and p[iL] != p[iL - 1]:
        L = iL - 1 + (thr - p[iL - 1]) / (p[iL] - p[iL - 1])
    else:
        L = float(iL)
    if iR < len(p) - 1 and p[iR] != p[iR + 1]:
        R = iR + (p[iR] - thr) / (p[iR] - p[iR + 1])
    else:
        R = float(iR)
    return L, R


def doppler_axis(img, r_lo=250, r_hi=3200, step=15, frac=0.35, smooth=81,
                 min_width=300, mode="halfpower", noise_k=4.0):
    """Robust tilt-aware Doppler axis from the echo limb.

    mode="limb" (noise-relative, connected-run) was tried to get a
    feature-independent absolute Doppler, but Venus' dim, Doppler-aliased
    monostatic limb makes it unmeasurable (unlike moon-radar's bright bistatic
    rim); the stable interior "halfpower" measure is the default.

    Returns a dict (or None if too few usable rows):
      freq_offset : Doppler center at the sub-radar row minus c2 (px)
      tilt        : zero-Doppler line slope (px Doppler per row delay)
      freq_scale  : limb half-width / c2 (diagnostic)
      scatter     : robust RMS of midpoints about the fitted line (px)
      n           : number of usable rows
    """
    c2 = img.shape[1] / 2.0
    rows, mids, hws = [], [], []
    for row in range(r_lo, min(r_hi, img.shape[0]), step):
        p = uniform_filter1d(img[row].astype(float), smooth)
        e = _edge_crossings(p, mode=mode, frac=frac, noise_k=noise_k)
        if e is None:
            continue
        L, R = e
        if R - L < min_width:
            continue
        rows.append(row)
        mids.append(0.5 * (L + R))
        hws.append(0.5 * (R - L))
    if len(rows) < 10:
        return None
    rows = np.array(rows, float)
    mids = np.array(mids)
    hws = np.array(hws)
    A = np.c_[rows, np.ones_like(rows)]
    w = np.ones_like(rows)
    coef = np.array([0.0, c2])
    for _ in range(6):  # IRLS, Cauchy weights -> reject limb outliers
        coef = np.linalg.lstsq(A * w[:, None], mids * w, rcond=None)[0]
        res = mids - A @ coef
        s = 1.4826 * np.median(np.abs(res - np.median(res))) + 1e-6
        w = 1.0 / (1.0 + (res / (2 * s)) ** 2)
    slope, intercept = coef
    scatter = float(np.sqrt(np.average((mids - A @ coef) ** 2, weights=w)))
    return {
        "freq_offset": float(intercept - c2),
        "tilt": float(slope),
        "freq_scale": float(np.percentile(hws, 90) / c2),
        "scatter": scatter,
        "n": len(rows),
    }


def self_calibrate(img, max_scatter=12.0, min_rows=40):
    """Self-cal with a quality flag. Returns dict with added ``quality`` in
    {'good','poor'}, or None if unusable."""
    r = doppler_axis(img)
    if r is None:
        return None
    r["quality"] = ("good" if (r["scatter"] <= max_scatter and r["n"] >= min_rows)
                    else "poor")
    return r
