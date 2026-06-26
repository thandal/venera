"""Delay-Doppler image -> Venus global lon/lat projection.

Clean reformulation of the notebook's projection. The geometry is carried by two
body-fixed vectors from :func:`venera.geometry.doppler_basis`:

    o_hat : observer (sub-radar) direction
    d_hat : Doppler-gradient direction = normalize(do_hat/dt)

A surface point ``p`` (unit, body-fixed) parameterized by apparent (dlon, dlat)
from the sub-radar point is::

    p = cos(dlat)cos(dlon) o_hat + cos(dlat)sin(dlon) d_hat + sin(dlat) n_hat
    n_hat = o_hat x d_hat                      (apparent north)
    row (delay)   = (1 - p·o_hat) * R/row_dist + delay_offset
    col (Doppler) = (c2*freq_scale) * (p·d_hat) + (c2 + freq_offset)

No S/D rotation matrices, no doppler-angle scalar, no sign hack. Image sampling is
**bilinear** (moon-radar B7: avoids the nearest-neighbour iso-ring artifact).
"""

import numpy as np

from .geometry import (VENUS_RADIUS_KM, Spin, doppler_basis, predicted_freq_scale,
                       CALIBRATED_FREQ_SCALE_K)
from . import data as vdata
import cspyce as csp

SPEED_OF_LIGHT_KM_S = 299792.458


def row_dist_km(baud_us):
    """Km of one-way range per delay row (row is round-trip time)."""
    return SPEED_OF_LIGHT_KM_S * baud_us * 1e-6 / 2.0


def bilinear_sample(img, rs, cs):
    """Bilinear sample ``img`` at fractional (row, col). Returns (values, valid)."""
    H, W = img.shape
    r0 = np.floor(rs).astype(np.int64)
    c0 = np.floor(cs).astype(np.int64)
    fr = rs - r0
    fc = cs - c0
    valid = (r0 >= 0) & (c0 >= 0) & (r0 + 1 < H) & (c0 + 1 < W)
    r0c = np.clip(r0, 0, H - 2)
    c0c = np.clip(c0, 0, W - 2)
    v00 = img[r0c, c0c]
    v01 = img[r0c, c0c + 1]
    v10 = img[r0c + 1, c0c]
    v11 = img[r0c + 1, c0c + 1]
    out = (v00 * (1 - fr) * (1 - fc) + v01 * (1 - fr) * fc
           + v10 * fr * (1 - fc) + v11 * fr * fc)
    return out, valid


def fit_delay_doppler_curve(img, baud, fo_range=range(-3, 4), do_range=range(-3, 4),
                            fs_values=None):
    """Fit (freq_offset, delay_offset, freq_scale) by matching the limb edge along
    the dlat=0 meridian. Ported from the notebook ``fitDopplerDelayCurve``."""
    if fs_values is None:
        fs_values = np.linspace(0.9, 1.4, 301)
    c2 = img.shape[1] / 2.0
    rdist = row_dist_km(baud)
    dlon = np.linspace(-np.pi / 2, np.pi / 2, 8000)
    row_curve = VENUS_RADIUS_KM * (1 - np.cos(dlon)) / rdist
    sin_dlon = np.sin(dlon)
    H, W = img.shape
    row_start, row_end = 500, 8000
    best_score, best = -np.inf, None
    for fo in fo_range:
        cs_base = (c2) * sin_dlon + (c2 + fo)
        for do in do_range:
            rs = (row_curve + do).astype(int)
            for fs in fs_values:
                cs = ((c2 * fs) * sin_dlon + (c2 + fo)).astype(int)
                # orthogonal-offset comparison curve (edge detector)
                drs = np.diff(rs, append=rs[-1])
                dcs = np.diff(cs, append=cs[-1])
                cs2 = cs + np.clip(drs, -1, 1)
                rs2 = rs - np.clip(dcs, -1, 1)
                v = (cs >= 0) & (cs < W) & (rs >= row_start) & (rs < row_end)
                v2 = (cs2 >= 0) & (cs2 < W) & (rs2 >= row_start) & (rs2 < row_end)
                score = img[rs[v], cs[v]].sum() - img[rs2[v2], cs2[v2]].sum()
                if score > best_score:
                    best_score, best = score, (fo, do, float(fs))
    return best_score, best


def project_image_to_map(img, o_hat, d_hat, G, Gc, *, baud, pointing,
                         freq_offset=0.0, delay_offset=0.0, freq_scale=1.0,
                         n_lon=6000, n_lat=3000, lon_halfwidth_deg=85.0,
                         lat_min_deg=5.0, lat_max_deg=85.0, srp_exclude_deg=7.0,
                         lat_chunk=400):
    """Project one preprocessed image into the global map (G sum, Gc count).

    G/Gc are equirectangular: row = (lat+pi/2)/pi*H, col = (lon+pi)/(2pi)*W.
    Returns the fraction of mesh points that landed on valid image pixels.
    """
    o_hat = np.asarray(o_hat, float)
    d_hat = np.asarray(d_hat, float)
    n_hat = np.cross(o_hat, d_hat)
    n_hat /= np.linalg.norm(n_hat)
    # Apparent north should have a positive body-north (z) component; flip if not.
    if n_hat[2] < 0:
        n_hat = -n_hat

    dlon = np.radians(np.linspace(-lon_halfwidth_deg, lon_halfwidth_deg, n_lon))
    if pointing == "N":
        dlat = np.radians(np.linspace(lat_min_deg, lat_max_deg, n_lat))
    else:
        dlat = np.radians(np.linspace(-lat_max_deg, -lat_min_deg, n_lat))
    cos_lon = np.cos(dlon); sin_lon = np.sin(dlon)
    c2 = img.shape[1] / 2.0
    rdist = row_dist_km(baud)
    cos_excl = np.cos(np.radians(srp_exclude_deg))
    H, W = G.shape
    # Process the latitude mesh in STRIPS so peak memory is ~n_lon*lat_chunk,
    # independent of total resolution — lets n_lon/n_lat scale to the data's native
    # (~1-2 km) resolution without allocating the whole 2-D mesh at once.
    total = 0; kept = 0
    for s in range(0, n_lat, lat_chunk):
        dl = dlat[s:s + lat_chunk]
        cdl = np.cos(dl)[:, None]; sdl = np.sin(dl)[:, None]
        coeff_o = (cdl * cos_lon[None, :]).ravel()        # = p·o_hat
        coeff_d = (cdl * sin_lon[None, :]).ravel()        # = p·d_hat
        coeff_n = np.repeat(np.sin(dl), n_lon)            # = p·n_hat
        keep = coeff_o < cos_excl                         # SRP exclusion + within-disk
        coeff_o = coeff_o[keep]; coeff_d = coeff_d[keep]; coeff_n = coeff_n[keep]
        if coeff_o.size == 0:
            total += dl.size * n_lon; continue
        rs = (1 - coeff_o) * (VENUS_RADIUS_KM / rdist) + delay_offset
        cs = (c2 * freq_scale) * coeff_d + (c2 + freq_offset)
        vals, valid = bilinear_sample(img, rs, cs)
        px = coeff_o * o_hat[0] + coeff_d * d_hat[0] + coeff_n * n_hat[0]
        py = coeff_o * o_hat[1] + coeff_d * d_hat[1] + coeff_n * n_hat[1]
        pz = coeff_o * o_hat[2] + coeff_d * d_hat[2] + coeff_n * n_hat[2]
        lon = np.arctan2(py, px); lat = np.arcsin(np.clip(pz, -1, 1))
        gr = ((lat + np.pi / 2) / np.pi * H).astype(np.int64)
        gc = ((lon + np.pi) / (2 * np.pi) * W).astype(np.int64)
        m = valid & (gr >= 0) & (gr < H) & (gc >= 0) & (gc < W)
        np.add.at(G, (gr[m], gc[m]), vals[m])
        np.add.at(Gc, (gr[m], gc[m]), 1)
        total += dl.size * n_lon; kept += int(m.sum())
    return float(kept / max(total, 1))


def fo_consensus(ets, fos, deg=2, reject=12.0, iters=5):
    """Robust per-session Doppler-centering (freq_offset) trend, returning a callable
    ``et -> fo_center`` (cols). The bistatic 2012 per-look ``fo`` fit is bimodal (the
    correct majority cluster plus an alias ~25-30 cols away); we mode-seek the
    MAJORITY cluster, then fit an outlier-rejected low-order polynomial to it. For
    monostatic sessions ``fo`` is already unimodal, so the trend just tracks the
    (small) real drift. Used to constrain ``project_file(..., fo_center=...)``."""
    ets = np.asarray(ets, float); fos = np.asarray(fos, float)
    t0 = ets.mean(); t = ets - t0
    # mode-seek the majority cluster center (NOT the global median, which can fall in
    # the sparse gap between the two bistatic clusters): densest window of width
    # 2*reject. This is the prior the polynomial then refines.
    order = np.sort(fos)
    best_c, best_n = float(np.median(fos)), -1
    for x in order:
        n = int(np.sum(np.abs(fos - x) <= reject))
        if n > best_n:
            best_n, best_c = n, x
    keep = np.abs(fos - best_c) <= reject
    c = np.array([best_c]) if keep.sum() < 2 else np.polyfit(
        t[keep], fos[keep], min(deg, max(1, int(keep.sum()) - 1)))
    for _ in range(iters):                       # refine within the majority cluster
        pred = np.polyval(c, t) if c.size > 1 else np.full_like(t, c[0])
        keep = np.abs(fos - pred) < reject
        if keep.sum() < deg + 1:
            break
        c = np.polyfit(t[keep], fos[keep], min(deg, max(1, int(keep.sum()) - 1)))
    cc = c
    return lambda et: float(np.polyval(cc, np.asarray(et, float) - t0))


def fit_diurnal_fo(hours, fos, reject=10.0, iters=6):
    """Robust per-session **diurnal** Doppler-centering (freq_offset) model
    ``fo(hour) = a + b*(hour - h0)``, returned as a callable ``hour -> fo`` (cols).

    The bistatic-2012 ``fo`` residual is a small, systematic function of **hour of
    day** (a leftover in the providers' time-varying topocentric Doppler tuning that
    sweeps with Earth-rotation phase) plus a constant — confirmed by corr(fo, hour) =
    -0.92 and the magnitude check (gross station/geocentric mis-tuning would be
    ~10^5 cols, not ~10). It is NOT per-look random, so fitting ``fo`` per look from
    the noisy/contaminated bistatic limb just injects scatter (±13 cols ≈ ±0.25° lon)
    that smears the stack. This model captures the systematic and rejects the
    wrong-edge limb-fit outliers (the N/S-ambiguity second cluster), so every look can
    be projected with ``fo_center = model(hour)`` instead of a per-look fit —
    recovering near-monostatic per-look registration (Maxwell scatter ±0.27°→±0.09°).
    Diurnal (hour-of-day), not polynomial-in-absolute-time, because it repeats daily.
    """
    hours = np.asarray(hours, float); fos = np.asarray(fos, float)
    h0 = hours.mean()
    A = np.c_[np.ones_like(hours), hours - h0]
    keep = np.ones(len(hours), bool)
    coef = np.linalg.lstsq(A, fos, rcond=None)[0]
    for _ in range(iters):
        coef = np.linalg.lstsq(A[keep], fos[keep], rcond=None)[0]
        keep = np.abs(fos - A @ coef) < reject
        if keep.sum() < 3:
            break
    a, b = float(coef[0]), float(coef[1])
    return lambda hour: a + b * (np.asarray(hour, float) - h0), (a, b, h0)


def project_file(img_path, spin: Spin, G, Gc, observer="ARECIBO", abcorr="CN+S",
                 fit=None, geom_fs=True, fo_center=None, fo_window=14, **proj_kw):
    """End-to-end: load+preprocess a file, compute geometry for ``spin``, fit the
    curve (unless ``fit`` params given), and project into G/Gc.

    ``geom_fs=True`` sets ``freq_scale`` from geometry
    (:func:`geometry.predicted_freq_scale`, the apparent-rotation bandwidth) and fits
    only the centering offsets (fo, do). With the topocentric-Arecibo apparent
    rotation correct, the geometric prediction reproduces the per-image fit at a
    constant ratio across epochs (recalibrated ``CALIBRATED_FREQ_SCALE_K``), so this
    drops a noisy per-image free parameter — notably for the low-quality 2020 looks.

    ``fo_center`` (cols), if given, restricts the Doppler-centering (freq_offset) fit
    to ``fo_center ± fo_window``. The bistatic 2012 limb fit is **bimodal** — two
    Doppler-limb solutions ~25-30 cols (~0.5 deg) apart, and noise picks which wins
    per look, scattering looks and blurring the stack. Passing the per-session
    consensus ``fo`` trend (see :func:`session_fo_consensus`) keeps every look on the
    correct (majority) cluster, recovering monostatic-level sharpness.

    Returns dict with the geometry + fit used.
    """
    a, lbl, cal = vdata.preprocess(img_path)
    et0 = csp.str2et(lbl["START_TIME"])
    et1 = csp.str2et(lbl["STOP_TIME"])
    et_mid = 0.5 * (et0 + et1)
    # Receiver from the label: GBT for the bistatic 2012 campaign, else Arecibo.
    # The receiver sets the Doppler axis (see geometry.doppler_basis).
    rx = lbl.get("RX_STATION", "ARECIBO")
    o_hat, d_hat = doppler_basis(et_mid, spin, observer, abcorr,
                                 dt=max(1.0, et1 - et0), rx_station=rx)
    if fo_center is None:
        fo_range = range(-60, 61, 2)
    else:
        c = int(round(fo_center))
        fo_range = range(c - fo_window, c + fo_window + 1)
    # Fine delay (row) fit: a few rows, to absorb the small constant between the
    # label GEO:DELAY_OFFSET reference and the true sub-radar onset (~3 rows on 2017).
    do_range = range(-8, 9)
    if fit is None:
        if geom_fs:
            # freq_scale from geometry (apparent-rotation bandwidth); fit fo, do only.
            fs_g = predicted_freq_scale(et_mid, spin, lbl["GEO_BAUD"],
                                        lbl.get("GEO_CODE_LENGTH", 1),
                                        K=CALIBRATED_FREQ_SCALE_K,
                                        observer=observer, abcorr=abcorr, rx_station=rx)
            _, fit = fit_delay_doppler_curve(
                a, lbl["GEO_BAUD"], fo_range=fo_range, do_range=do_range,
                fs_values=[fs_g])
        else:
            # Wide limb-model curve fit -> absolute (freq_offset, delay_offset,
            # freq_scale), feature-independent (immune to limb Doppler aliasing).
            _, fit = fit_delay_doppler_curve(
                a, lbl["GEO_BAUD"], fo_range=fo_range, do_range=do_range,
                fs_values=np.linspace(0.9, 1.4, 71))
    fo, do, fs = fit
    frac = project_image_to_map(a, o_hat, d_hat, G, Gc, baud=lbl["GEO_BAUD"],
                                pointing=lbl["GEO_POINTING"], freq_offset=fo,
                                delay_offset=do, freq_scale=fs, **proj_kw)
    return {"fit": fit, "pointing": lbl["GEO_POINTING"], "valid_frac": frac,
            "centering": cal.get("centering"), "o_hat": o_hat, "d_hat": d_hat,
            "et_mid": et_mid}
