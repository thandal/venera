"""Estimate Venus' sidereal rotation period from cross-epoch longitude drift.

Physical basis (validated in ``test/test_geometry.py`` to 3e-11): when an epoch
observed at time ``t`` is projected to body-fixed longitude using an *assumed*
spin rate ``Wdot_assumed``, a fixed surface feature is assigned a longitude

    lambda_assigned(t) = lambda_true + (Wdot_true - Wdot_assumed) * d(t)   (+const)

where ``d`` is days. Registering epoch *i* against epoch *j* therefore measures

    dlambda_ij = (Wdot_true - Wdot_assumed) * (d_i - d_j).

So a **weighted linear regression of measured longitude offset vs time** has slope
``= Wdot_true - Wdot_assumed`` — the correction to the assumed rotation rate. The
long 1988-2020 baseline is the precision lever (moon-radar's geometry-diversity
lesson in the time domain): sigma_P scales as sigma_lambda / baseline.

This replaces the old image-sharpness grid search with a physical observable that
carries a propagated covariance.
"""

import numpy as np


def period_to_wdot(period_days, retrograde=True):
    rate = 360.0 / period_days
    return -rate if retrograde else rate


def wdot_to_period(wdot_deg_per_day):
    return 360.0 / abs(wdot_deg_per_day)


def fit_rotation_from_offsets(days, dlon_deg, assumed_wdot, weights=None):
    """Weighted linear fit of longitude offset vs time -> rotation rate & period.

    Parameters
    ----------
    days : array of epoch times (TDB days; only differences matter)
    dlon_deg : measured assigned-longitude offset of each epoch (deg), under the
        assumed spin, relative to a common reference (the reference may be one of
        the epochs, with offset 0).
    assumed_wdot : the signed Wdot (deg/day) used when projecting the maps.
    weights : optional per-epoch weights (e.g. registration significance**2).

    Returns dict with period_days, sigma_period_days, wdot, slope, intercept,
    residuals_deg, and chi2/dof diagnostics.
    """
    days = np.asarray(days, float)
    y = np.asarray(dlon_deg, float)
    n = len(days)
    if n < 2:
        raise ValueError("need >= 2 epochs")
    w = np.ones(n) if weights is None else np.asarray(weights, float)
    # center time for numerical conditioning
    t0 = np.average(days, weights=w)
    x = days - t0
    # weighted least squares for y = slope*x + b
    W = np.diag(w)
    A = np.column_stack([x, np.ones(n)])
    AtW = A.T @ W
    cov = np.linalg.inv(AtW @ A)
    coef = cov @ (AtW @ y)
    slope, intercept = coef
    resid = y - A @ coef
    dof = max(1, n - 2)
    # scale covariance by reduced chi-square (so error bars reflect actual scatter)
    chi2 = float((w * resid ** 2).sum())
    s2 = chi2 / dof
    cov_scaled = cov * s2
    sigma_slope = float(np.sqrt(cov_scaled[0, 0]))

    wdot_true = assumed_wdot + slope
    period = wdot_to_period(wdot_true)
    # dP/dWdot = -360/Wdot^2  -> sigma_P = (360/Wdot^2) sigma_slope
    sigma_period = (360.0 / wdot_true ** 2) * sigma_slope
    return {
        "period_days": period,
        "sigma_period_days": sigma_period,
        "wdot": wdot_true,
        "slope_deg_per_day": slope,
        "sigma_slope": sigma_slope,
        "intercept_deg": intercept,
        "residuals_deg": resid,
        "rms_resid_deg": float(np.sqrt(np.mean(resid ** 2))),
        "chi2": chi2,
        "dof": dof,
        "t0_days": t0,
        "n": n,
    }


def fit_tile_rotation(tiles):
    """Fit a rigid sphere rotation ω (rad) to a tile displacement field.

    ``tiles`` = list of ``(lat_deg, lon_deg, dlat_deg, dlon_deg)``. A point at unit
    vector r displaces by ``δr = ω × r`` under rotation ω, so
    ``δlat = ω·(r×n̂)`` and ``δlon·cos(lat) = ω·(r×ê)`` (local north/east). Weighted
    least-squares for ω (3 components). **``ω[2]`` — rotation about the body z/pole
    — is the longitude/spin shift** (the period observable); ``ω[0],ω[1]`` absorb
    the doppler-angle-driven residual tilt that a translation matcher mis-reads as
    longitude. Returns ``(omega[3] rad, rms_residual_deg)``.
    """
    A, y = [], []
    for lat, lon, dlat, dlon in tiles:
        la = np.radians(lat)
        lo = np.radians(lon)
        r = np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])
        e = np.array([-np.sin(lo), np.cos(lo), 0.0])
        n = np.array([-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)])
        A.append(np.cross(r, n)); y.append(np.radians(dlat))
        A.append(np.cross(r, e)); y.append(np.radians(dlon) * np.cos(la))
    A = np.asarray(A)
    y = np.asarray(y)
    omega, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ omega
    return omega, float(np.degrees(np.sqrt(np.mean(resid ** 2))))


def longitude_shift_from_rotation(omega):
    """Longitude (spin) shift in degrees = rotation about the body pole = ω_z."""
    return float(np.degrees(omega[2]))


def solve_relative_longitudes(n, pairs):
    """Globally-consistent per-epoch longitudes from pairwise offsets (closure solve).

    ``pairs``: list of ``(i, j, dlon_ij_deg, weight)`` meaning the registered
    longitude offset of epoch i relative to j. Minimizes
    ``sum_w (lon_i - lon_j - dlon_ij)^2`` with gauge ``lon_0 = 0`` (weighted LSQ).
    Returns ``(lon[n], closure_rms_deg, per_pair_residual_deg)``. The closure RMS
    measures how self-consistent the pairwise registrations are — high values flag
    a degenerate/unreliable graph (moon-radar B3).
    """
    rows, obs, wts = [], [], []
    for i, j, d, w in pairs:
        r = np.zeros(n)
        r[i] += 1.0
        r[j] -= 1.0
        rows.append(r)
        obs.append(d)
        wts.append(w)
    A = np.array(rows)
    y = np.array(obs, float)
    W = np.array(wts, float)
    A2 = A[:, 1:]                                  # drop col 0 -> gauge lon_0 = 0
    cov = np.linalg.pinv((A2 * W[:, None]).T @ A2)
    sol = cov @ ((A2 * W[:, None]).T @ y)
    lon = np.concatenate([[0.0], sol])
    resid = y - A @ lon
    closure_rms = float(np.sqrt(np.average(resid ** 2, weights=W)))
    return lon, closure_rms, resid


def fit_rotation_rate_change(days, dlon_deg, assumed_wdot, weights=None):
    """Fit a *changing* rotation rate: dlon = a + b*x + c*x^2 (x = days - mean).

    A linearly-varying spin rate Wdot(t) = Wdot0 + Wdotdot*(t-t0) accumulates a
    longitude offset (Wdot0 - Wdot_assumed)*x + 0.5*Wdotdot*x^2, so the quadratic
    coefficient c = 0.5*Wdotdot. Returns the mean period (from b), the rate change
    dP/dt, and its significance. Needs >= 4 epochs.
    """
    days = np.asarray(days, float)
    y = np.asarray(dlon_deg, float)
    n = len(days)
    if n < 4:
        raise ValueError("need >= 4 epochs for a rate-change fit")
    w = np.ones(n) if weights is None else np.asarray(weights, float)
    t0 = np.average(days, weights=w)
    x = days - t0
    A = np.column_stack([np.ones(n), x, x * x])
    W = np.diag(w)
    cov = np.linalg.inv(A.T @ W @ A)
    coef = cov @ (A.T @ W @ y)
    resid = y - A @ coef
    dof = max(1, n - 3)
    s2 = float((w * resid ** 2).sum()) / dof
    cov *= s2
    a, b, c = coef
    sigma_c = float(np.sqrt(cov[2, 2]))
    wdot_mean = assumed_wdot + b              # rate at t0
    period_mean = wdot_to_period(wdot_mean)
    wdotdot = 2.0 * c                         # deg/day^2
    # dP/dt = -360/Wdot^2 * Wdotdot  (days of period per day of time)
    dP_dt = -360.0 / wdot_mean ** 2 * wdotdot
    sigma_dP_dt = 360.0 / wdot_mean ** 2 * 2.0 * sigma_c
    dP_dt_per_year = dP_dt * 365.25
    sigma_dP_dt_per_year = sigma_dP_dt * 365.25
    return {
        "period_mean_days": period_mean,
        "dP_dt_days_per_year": dP_dt_per_year,
        "sigma_dP_dt_days_per_year": sigma_dP_dt_per_year,
        "rate_change_significance": abs(c) / sigma_c if sigma_c > 0 else np.inf,
        "quad_coef": c,
        "sigma_quad_coef": sigma_c,
        "residuals_deg": resid,
        "n": n,
    }


def bootstrap_period(days, dlon_deg, assumed_wdot, weights=None, n_boot=2000,
                     seed=0):
    """Bootstrap the period over epochs for a model-free error bar (moon-radar B4)."""
    days = np.asarray(days, float)
    y = np.asarray(dlon_deg, float)
    n = len(days)
    w = np.ones(n) if weights is None else np.asarray(weights, float)
    rng = np.random.default_rng(seed)
    periods = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(days[idx])) < 2:
            continue
        try:
            r = fit_rotation_from_offsets(days[idx], y[idx], assumed_wdot, w[idx])
            periods.append(r["period_days"])
        except Exception:
            continue
    periods = np.array(periods)
    return {
        "period_median": float(np.median(periods)),
        "period_std": float(np.std(periods)),
        "period_p16": float(np.percentile(periods, 16)),
        "period_p84": float(np.percentile(periods, 84)),
        "samples": periods,
    }
