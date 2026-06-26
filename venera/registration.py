"""Band-pass + masked sub-pixel cross-correlation for registering projected maps.

Mirrors moon-radar's ``registration_analysis.py`` recipe (lessons B3/B4):
  - **band-pass**: smooth at ~speckle scale, subtract a coarse trend, so neither
    radar speckle nor the bright quasi-specular envelope dominates the correlation;
  - **masked FFT cross-correlation** with parabolic sub-pixel peak refinement;
  - a **significance** metric = main-peak / strongest-sidelobe, to tell a real
    feature lock from the speckle floor (~1 = noise, >~1.5 = genuine).

Maps are equirectangular (row=lat, col=lon), so a measured pixel offset converts
directly to (Δlat, Δlon) in degrees — the observable the rotation fit consumes.
"""

import numpy as np
from scipy.ndimage import gaussian_filter


def bandpass(img, smooth_px=3.0, trend_px=25.0):
    """Speckle-averaging low-pass minus coarse-trend (the bright-envelope killer)."""
    img = np.asarray(img, float)
    return gaussian_filter(img, smooth_px) - gaussian_filter(img, trend_px)


def _hann2d(shape):
    wy = np.hanning(shape[0])
    wx = np.hanning(shape[1])
    return np.outer(wy, wx)


def _parabolic_subpixel(c, i):
    """Sub-sample peak offset along one axis from 3 samples around index ``i``."""
    n = len(c)
    if i <= 0 or i >= n - 1:
        return 0.0
    a, b, d = c[i - 1], c[i], c[i + 1]
    denom = (a - 2 * b + d)
    if denom == 0:
        return 0.0
    return 0.5 * (a - d) / denom


def xcorr_offset(a, b, max_shift=None, exclude_radius=3, window=True):
    """Cross-correlation offset that best maps ``b`` onto ``a`` (sub-pixel).

    Returns ``(drow, dcol, significance)``. Applying the shift ``(drow, dcol)`` to
    ``b`` aligns it with ``a``.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a = a - a.mean()
    b = b - b.mean()
    if window:
        w = _hann2d(a.shape)
        a = a * w
        b = b * w
    A = np.fft.rfft2(a)
    B = np.fft.rfft2(b)
    cc = np.fft.irfft2(A * np.conj(B), s=a.shape)
    cc = np.fft.fftshift(cc)
    cy, cx = a.shape[0] // 2, a.shape[1] // 2

    if max_shift is not None:
        # restrict the search to a central window
        mask = np.zeros_like(cc, bool)
        r = max_shift
        mask[cy - r:cy + r + 1, cx - r:cx + r + 1] = True
        cc_search = np.where(mask, cc, -np.inf)
    else:
        cc_search = cc
    pk = np.unravel_index(np.argmax(cc_search), cc.shape)
    peak_val = cc[pk]

    # sub-pixel parabolic refinement along each axis through the peak
    drow = (pk[0] - cy) + _parabolic_subpixel(cc[:, pk[1]], pk[0])
    dcol = (pk[1] - cx) + _parabolic_subpixel(cc[pk[0], :], pk[1])

    # significance: peak / strongest sidelobe outside an exclusion radius
    cc2 = cc.copy()
    y0, y1 = max(0, pk[0] - exclude_radius), min(cc.shape[0], pk[0] + exclude_radius + 1)
    x0, x1 = max(0, pk[1] - exclude_radius), min(cc.shape[1], pk[1] + exclude_radius + 1)
    cc2[y0:y1, x0:x1] = -np.inf
    sidelobe = cc2[np.isfinite(cc2)].max()
    significance = float(peak_val / sidelobe) if sidelobe > 0 else np.inf
    return float(drow), float(dcol), significance


def _corr(x, y):
    """FFT cross-correlation, fftshifted, same convention as :func:`xcorr_offset`
    (peak offset from center maps the second arg onto the first)."""
    return np.fft.fftshift(np.fft.irfft2(np.fft.rfft2(x) * np.conj(np.fft.rfft2(y)),
                                         s=x.shape))


def masked_ncc_offset(f, g, mf, mg, max_shift=None, exclude_radius=3,
                      overlap_frac=0.3):
    """Padfield masked normalized cross-correlation offset (sub-pixel).

    Handles partial overlap / irregular masks without edge artifacts. ``mf``/``mg``
    are boolean validity masks. Returns ``(drow, dcol, significance)`` mapping
    ``g`` onto ``f``.
    """
    f = np.asarray(f, float); g = np.asarray(g, float)
    mf = np.asarray(mf, float); mg = np.asarray(mg, float)
    F = f * mf; G = g * mg
    M = _corr(mf, mg)                     # overlapping-pixel count per shift
    eps = 1e-9
    Msafe = np.where(M > eps, M, np.inf)  # avoid div-by-zero; those shifts killed below
    Sfg = _corr(F, G)
    Sf = _corr(F, mg);  Sg = _corr(mf, G)
    Sf2 = _corr(F * F, mg); Sg2 = _corr(mf, G * G)
    num = Sfg - Sf * Sg / Msafe
    den_f = Sf2 - Sf * Sf / Msafe
    den_g = Sg2 - Sg * Sg / Msafe
    den = np.sqrt(np.clip(den_f, 0, None) * np.clip(den_g, 0, None))
    ncc = np.where(den > eps, num / np.where(den > eps, den, 1.0), 0.0)
    # require meaningful overlap and limit search
    ok = M > overlap_frac * M.max()
    cy, cx = f.shape[0] // 2, f.shape[1] // 2
    if max_shift is not None:
        win = np.zeros_like(ok)
        win[cy - max_shift:cy + max_shift + 1, cx - max_shift:cx + max_shift + 1] = True
        ok &= win
    cc = np.where(ok, ncc, -np.inf)
    pk = np.unravel_index(np.argmax(cc), cc.shape)
    peak_val = ncc[pk]
    drow = (pk[0] - cy) + _parabolic_subpixel(np.where(ok, ncc, -np.inf)[:, pk[1]], pk[0])
    dcol = (pk[1] - cx) + _parabolic_subpixel(np.where(ok, ncc, -np.inf)[pk[0], :], pk[1])
    cc2 = cc.copy()
    y0, y1 = max(0, pk[0] - exclude_radius), min(cc.shape[0], pk[0] + exclude_radius + 1)
    x0, x1 = max(0, pk[1] - exclude_radius), min(cc.shape[1], pk[1] + exclude_radius + 1)
    cc2[y0:y1, x0:x1] = -np.inf
    finite = cc2[np.isfinite(cc2)]
    sidelobe = finite.max() if finite.size else 0.0
    significance = float(peak_val / sidelobe) if sidelobe > 0 else np.inf
    return float(drow), float(dcol), significance


def _apodized_weight(common_mask, taper_px):
    """Smooth taper of a boolean common-support mask -> 0 at the edges, ~1 inside."""
    w = gaussian_filter(common_mask.astype(float), taper_px)
    # zero out anything originally invalid so we never sample outside support
    w = np.where(common_mask, w, 0.0)
    return w


def register_maps(a, b, valid_a=None, valid_b=None, smooth_px=3.0, trend_px=25.0,
                  max_shift=None, exclude_radius=3, taper_px=8.0):
    """Band-pass both maps and cross-register, returning (drow, dcol, significance)
    mapping ``b`` onto ``a``.

    With validity masks, the common support is **apodized** (smoothly tapered to
    zero at the coverage boundary) before correlation — this removes the hard-edge
    artifact of mask-by-zeroing without the noise amplification of full Padfield
    normalization. Both band-passed maps are weighted-mean-subtracted then multiplied
    by the taper.
    """
    fa = bandpass(a, smooth_px, trend_px)
    fb = bandpass(b, smooth_px, trend_px)
    if valid_a is None and valid_b is None:
        return xcorr_offset(fa, fb, max_shift=max_shift, exclude_radius=exclude_radius)
    va = np.ones(a.shape, bool) if valid_a is None else np.asarray(valid_a, bool)
    vb = np.ones(b.shape, bool) if valid_b is None else np.asarray(valid_b, bool)
    w = _apodized_weight(va & vb, taper_px)
    wsum = w.sum()
    if wsum <= 0:
        return 0.0, 0.0, 0.0
    aw = w * (fa - (w * fa).sum() / wsum)
    bw = w * (fb - (w * fb).sum() / wsum)
    return xcorr_offset(aw, bw, max_shift=max_shift, exclude_radius=exclude_radius,
                        window=False)


def register_consensus(a, b, valid_a, valid_b, tile_px=384, step_px=256,
                       smooth_px=7.0, trend_px=55.0, max_shift=60, min_sig=1.3,
                       min_valid_frac=0.7):
    """Robust registration by local-tile consensus.

    One global correlation locks onto large-scale brightness / a few dominant
    features and yields a broad, distortion-sensitive ridge. Instead we band-pass
    once, tile the common overlap, register each near-fully-covered tile
    independently (sharp local-feature locks), and take the **median** shift over
    tiles that clear ``min_sig``. The inter-tile scatter (MAD) is the *honest*
    registration error — and large scatter directly exposes genuine map distortion
    (vs a clean rigid shift).

    Returns dict(drow, dcol, drow_err, dcol_err, dcol_scatter_px, n_tiles,
    n_used) mapping b onto a, or None if too few tiles lock.
    """
    fa = bandpass(a, smooth_px, trend_px)
    fb = bandpass(b, smooth_px, trend_px)
    common = np.asarray(valid_a, bool) & np.asarray(valid_b, bool)
    H, W = a.shape
    drs, dcs, sigs = [], [], []
    for r0 in range(0, H - tile_px + 1, step_px):
        for c0 in range(0, W - tile_px + 1, step_px):
            tc = common[r0:r0 + tile_px, c0:c0 + tile_px]
            if tc.mean() < min_valid_frac:
                continue
            ta = fa[r0:r0 + tile_px, c0:c0 + tile_px] * tc
            tb = fb[r0:r0 + tile_px, c0:c0 + tile_px] * tc
            dr, dc, sig = xcorr_offset(ta, tb, max_shift=max_shift)
            if sig < min_sig:
                continue
            drs.append(dr); dcs.append(dc); sigs.append(sig)
    if len(drs) < 4:
        return None
    drs = np.array(drs); dcs = np.array(dcs)
    dr_med, dc_med = np.median(drs), np.median(dcs)
    dr_mad = 1.4826 * np.median(np.abs(drs - dr_med))
    dc_mad = 1.4826 * np.median(np.abs(dcs - dc_med))
    n = len(drs)
    return {
        "drow": float(dr_med), "dcol": float(dc_med),
        "drow_err": float(dr_mad / np.sqrt(n)), "dcol_err": float(dc_mad / np.sqrt(n)),
        "dcol_scatter_px": float(dc_mad), "drow_scatter_px": float(dr_mad),
        "n_used": n,
    }


def tile_displacements(a, b, valid_a, valid_b, tile=384, step=240, smooth_px=7.0,
                       trend_px=55.0, minfrac=0.5, min_sig=0.95, max_shift=45):
    """Per-tile displacement field mapping ``b`` onto ``a``: a list of
    ``(lat_deg, lon_deg, dlat_deg, dlon_deg)`` over tiles that lock (sig>min_sig).

    This is the input to a rigid-rotation fit (rotation_fit.fit_tile_rotation):
    two equirectangular maps of the sphere differ by a 3-D rotation (the spin plus
    residual tilt), NOT a translation, so the displacement field is `ω × r` — a
    rotation must be fit, not a single (drow, dcol)."""
    fa = bandpass(a, smooth_px, trend_px)
    fb = bandpass(b, smooth_px, trend_px)
    common = np.asarray(valid_a, bool) & np.asarray(valid_b, bool)
    H, W = a.shape
    out = []
    for r0 in range(0, H - tile + 1, step):
        for c0 in range(0, W - tile + 1, step):
            tc = common[r0:r0 + tile, c0:c0 + tile]
            if tc.mean() < minfrac:
                continue
            dr, dc, sig = xcorr_offset(fa[r0:r0 + tile, c0:c0 + tile] * tc,
                                       fb[r0:r0 + tile, c0:c0 + tile] * tc,
                                       max_shift=max_shift)
            if sig < min_sig:
                continue
            lat = (r0 + tile / 2) / H * 180.0 - 90.0
            lon = (c0 + tile / 2) / W * 360.0 - 180.0
            out.append((lat, lon, -dr / H * 180.0, -dc / W * 360.0))
    return out


def offset_to_lonlat_deg(drow, dcol, map_shape):
    """Convert a (drow, dcol) pixel offset on an equirectangular map to
    (dlat_deg, dlon_deg)."""
    H, W = map_shape
    return drow / H * 180.0, dcol / W * 360.0
