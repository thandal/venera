"""Parameterized Venus delay-Doppler observing geometry (cspyce + DE440s).

The whole point of this module: make the **spin state we are fitting** an explicit
parameter, cleanly separated from the (fixed, excellent) ephemeris.

- Venus/Earth/observer positions and velocities come from SPICE (DE440s).
- The Venus body-fixed frame is built **here** from a :class:`Spin`
  ``(period, pole_ra, pole_dec, w0)`` via the standard IAU 3-1-3 construction —
  *not* from the PCK's nominal ``IAU_VENUS`` frame and *not* by monkeypatching a
  library. Validated against ``IAU_VENUS`` at the nominal constants
  (see ``test/test_geometry.py``).

Conventions
-----------
- ``et`` is TDB seconds past J2000 (SPICE). Days past J2000 = ``et / 86400``.
- Longitudes/latitudes are **planetocentric**, radians, from :func:`cspyce.reclat`
  (lon in (-pi, pi], lat in [-pi/2, pi/2]).
- Venus rotates **retrograde**: prime-meridian rate ``W_dot < 0``.
- ``et`` passed to the SRP/Doppler functions is the **reception epoch at the
  observer**; the body-fixed orientation is evaluated at the **emission epoch**
  ``et - lighttime`` (the moment the echo left Venus).
"""

from dataclasses import dataclass
import numpy as np
import cspyce as csp

from . import spice_setup

# Venus mean radius (km), DE440/IAU. Venus is essentially a sphere (flattening ~0).
VENUS_RADIUS_KM = 6051.8
J2000_EPOCH_DEG = None  # (placeholder; W0 is referenced to et=0 = J2000 TDB)

# Ground stations (the radar transmitter/receiver sites). These MUST be topocentric
# sites, not the Earth geocenter: a station's diurnal rotation velocity (~0.4 km/s)
# is part of the apparent rotation that sets the Doppler axis d_hat = d(o_hat)/dt.
# Using the geocenter omits it and tilts d_hat (and the apparent north
# n_hat = o_hat x d_hat) by ~2-3 deg, epoch-dependently. (The original notebook used
# Arecibo via astropy EarthLocation.of_site('arecibo'); the geocenter was a
# regression.)
#
# venera transmits from ARECIBO always; the RECEIVER is Arecibo for monostatic
# (GEO:MODE "M") looks but the Green Bank Telescope (GBT) for the bistatic 2012
# campaign (GEO:MODE "B"). A bistatic look's Doppler axis depends on the RECEIVER's
# velocity, so GBT-received looks must use the GBT receiver (see doppler_basis).
STATIONS = {                         # (lon_deg, lat_deg, alt_km)
    "ARECIBO": (-66.75278, 18.34417, 0.498),
    "GBT":     (-79.83983, 38.43312, 0.807),   # Green Bank Telescope, WV
}
_EARTH_A_KM = 6378.1366                 # WGS84-ish equatorial radius
_EARTH_F = (6378.1366 - 6356.7519) / 6378.1366   # flattening


def _topo_offset_j2000(et, station):
    """Vector (km, J2000) from Earth's center to ``station`` at epoch ``et``,
    including Earth's rotation (so a finite difference across ``et`` recovers the
    station's diurnal velocity)."""
    lon, lat, alt = STATIONS[station]
    site_bf = csp.georec(np.radians(lon), np.radians(lat), alt, _EARTH_A_KM, _EARTH_F)
    m = np.asarray(csp.pxform("IAU_EARTH", "J2000", et), float)  # body-fixed -> J2000
    return m @ np.asarray(site_bf, float)


@dataclass(frozen=True)
class Spin:
    """Venus spin state. ``period_days`` is the sidereal rotation period (magnitude).

    Defaults are the IAU 2009/2015 nominal constants (from pck00011); the fit
    typically varies ``period_days`` (and optionally the pole).
    """
    period_days: float = 243.0185      # IAU nominal (= 360 / 1.4813688)
    pole_ra_deg: float = 272.76        # IAU nominal
    pole_dec_deg: float = 67.16        # IAU nominal
    w0_deg: float = 160.20             # prime meridian at J2000 (IAU nominal)
    retrograde: bool = True

    @property
    def w_dot_deg_per_day(self) -> float:
        rate = 360.0 / self.period_days
        return -rate if self.retrograde else rate

    @classmethod
    def iau_nominal(cls) -> "Spin":
        """Spin built from the furnished PCK's Venus constants (for validation)."""
        ra, dec, w0, wdot = spice_setup.nominal_venus_spin_constants()
        period = abs(360.0 / wdot)
        return cls(period_days=period, pole_ra_deg=ra, pole_dec_deg=dec,
                   w0_deg=w0, retrograde=(wdot < 0))


def bodyfixed_matrix(et, spin: Spin) -> np.ndarray:
    """3x3 rotation J2000 -> Venus body-fixed, for the given spin at epoch ``et``.

    Standard IAU 3-1-3 Euler construction:
        M = Rz(W) . Rx(90deg - dec) . Rz(90deg + ra)
    identical to how SPICE builds ``IAU_VENUS`` from the PCK constants — but here
    with *our* spin parameters.
    """
    d = et / 86400.0
    W = np.radians(spin.w0_deg + spin.w_dot_deg_per_day * d)
    ra = np.radians(spin.pole_ra_deg)
    dec = np.radians(spin.pole_dec_deg)
    return csp.eul2m(W, np.pi / 2.0 - dec, np.pi / 2.0 + ra, 3, 1, 3)


def _emission_direction_j2000(et, observer, abcorr):
    """Apparent Venus->observer unit vector (J2000) and emission epoch et_b.

    The sub-radar point lies along Venus->observer. ``observer`` is a topocentric
    station name in :data:`STATIONS` (e.g. "ARECIBO", "GBT") — Earth center + site
    offset rotating with Earth, essential for the Doppler axis (see module note).
    "EARTH" (geocenter) is accepted for validation only.
    """
    pos, lt = csp.spkpos("VENUS", et, "J2000", abcorr, "EARTH")   # Earth ctr -> Venus
    pos = np.asarray(pos, dtype=float)
    if observer in STATIONS:
        pos = pos - _topo_offset_j2000(et, observer)             # station -> Venus
    elif observer != "EARTH":
        pos, lt = csp.spkpos("VENUS", et, "J2000", abcorr, observer)
        pos = np.asarray(pos, dtype=float)
    u = -pos / np.linalg.norm(pos)
    return u, et - lt


def observer_direction_bodyfixed(et, spin: Spin, observer="ARECIBO", abcorr="CN+S"):
    """Unit vector toward the observer in the parameterized body-fixed frame
    (i.e. the sub-radar direction ô), and the emission epoch ``et_b``."""
    u, et_b = _emission_direction_j2000(et, observer, abcorr)
    M = bodyfixed_matrix(et_b, spin)
    o_hat = M @ u
    o_hat /= np.linalg.norm(o_hat)
    return o_hat, et_b


def sub_radar_point(et, spin: Spin, observer="ARECIBO", abcorr="CN+S"):
    """Sub-radar (sub-observer) point in the parameterized body-fixed frame.

    Returns ``(lon, lat, et_emit)`` with lon, lat in radians (planetocentric).
    """
    o_hat, et_b = observer_direction_bodyfixed(et, spin, observer, abcorr)
    _, lon, lat = csp.reclat(o_hat)
    return lon, lat, et_b


def doppler_basis(et, spin: Spin, observer="ARECIBO", abcorr="CN+S", dt=30.0,
                  rx_station=None):
    """Body-fixed delay-Doppler basis vectors at reception epoch ``et``.

    Returns ``(o_hat, d_hat)`` where
      - ``o_hat`` is the (effective sub-radar) direction, and
      - ``d_hat`` = normalize(dô_eff/dt) is the Doppler-gradient direction.

    The delay-Doppler image maps surface point ``p`` (unit, body-fixed) by::

        delay  ∝ 1 - p·o_hat        (depth below the sub-radar tangent plane)
        doppler ∝ p·d_hat           (line-of-sight velocity from apparent rotation)

    **Bistatic geometry.** ``observer`` is the TRANSMITTER (always Arecibo here);
    ``rx_station`` is the RECEIVER. For a monostatic look (``rx_station`` None or ==
    transmitter) this reduces to the single-station ô and its time derivative. For a
    bistatic look (transmit Arecibo, receive GBT — the 2012 campaign) the round-trip
    path is ``|p−TX| + |p−RX|``, so the effective direction is
    ``ô_eff = normalize(ô_tx + ô_rx)`` and the Doppler axis is ``d(ô_eff)/dt``. The
    receiver's distinct diurnal velocity (GBT vs Arecibo) therefore enters d_hat —
    which is the whole point: ignoring it (projecting 2012 as monostatic-Arecibo)
    tilts d_hat per-look and blurs the 2012 stack. ``dt`` is a centered window (s).
    """
    tx = observer
    rx = tx if (rx_station is None or rx_station == tx) else rx_station

    def o_eff(t):
        o_tx, _ = observer_direction_bodyfixed(t, spin, tx, abcorr)
        if rx == tx:
            return o_tx
        o_rx, _ = observer_direction_bodyfixed(t, spin, rx, abcorr)
        v = o_tx + o_rx
        return v / np.linalg.norm(v)

    o1 = o_eff(et - dt / 2.0)
    o2 = o_eff(et + dt / 2.0)
    o_hat = 0.5 * (o1 + o2)
    o_hat /= np.linalg.norm(o_hat)
    d_vec = (o2 - o1)
    d_hat = d_vec / np.linalg.norm(d_vec)
    return o_hat, d_hat


def _wrap_pi(x):
    """Wrap angle(s) to (-pi, pi]."""
    return np.arctan2(np.sin(x), np.cos(x))


def doppler_angle(et, spin: Spin, observer="ARECIBO", abcorr="CN+S", dt=30.0):
    """Apparent-rotation ("Doppler") angle: azimuth of the sub-radar point's
    apparent track on the body, measured from the local east direction.

    This is the orientation by which the projected image must be rotated about the
    SRP. Because the body-fixed frame already rotates with the (parameterized)
    spin, the track automatically combines the orbital relative motion and the
    body spin — no explicit ``- angular_velocity`` term is needed (cf. the old
    poliastro code).

    NOTE: the absolute sign/zero of this angle is provisional and is pinned by the
    projection convention test (task #3, resolving the old ``S.T*D`` vs ``D.T``
    hack). The *magnitude* and epoch-to-epoch behaviour are physical now.

    ``dt`` is a centered finite-difference window (s); a wide window follows
    moon-radar's window-average guidance for robust SPICE derivatives.
    """
    lon1, lat1, _ = sub_radar_point(et - dt / 2.0, spin, observer, abcorr)
    lon2, lat2, _ = sub_radar_point(et + dt / 2.0, spin, observer, abcorr)
    lat = 0.5 * (lat1 + lat2)
    dlat = (lat2 - lat1) / dt
    dlon = _wrap_pi(lon2 - lon1) / dt
    return float(np.arctan2(dlat, dlon * np.cos(lat)))


VENUS_WAVELENGTH_M = 299792458.0 / 2380e6   # 12.6 cm, 2380 MHz

# Single global calibration constant for predicted_freq_scale, absorbing the
# common pre-processing / instrument scale. Recalibrated AFTER the topocentric
# Arecibo observer fix (§9): with the correct apparent-rotation rate, the geometric
# prediction reproduces the per-image fitted freq_scale at a constant ratio
# fitted/geom = 1.034 ± 0.001 across 5 of 6 epochs (2020's per-image fit is noisy),
# so K = 0.968 × 1.034 ≈ 1.001. See scripts/archive/freq_scale_check (or rerun).
CALIBRATED_FREQ_SCALE_K = 1.001


def apparent_rotation_rate(et, spin: Spin, observer="ARECIBO", abcorr="CN+S", dt=30.0,
                           rx_station=None):
    """Apparent rotation rate |dô_eff/dt| (rad/s): how fast the effective sub-radar
    direction sweeps the body (orbital drift + body spin). Sets the limb-to-limb
    Doppler bandwidth, hence ``freq_scale``.

    **Must use the same effective direction as the Doppler axis** (`doppler_basis`):
    for a bistatic look (transmit ``observer``, receive ``rx_station``) the Doppler
    bandwidth is set by ``ô_eff = normalize(ô_tx + ô_rx)``, not by the transmitter
    alone. For 2012 (Arecibo→GBT) the bistatic rate is ~0.6% higher than monostatic;
    using the monostatic rate makes the Doppler *scale* inconsistent with the Doppler
    *axis* and stretches longitude by ~0.5° at the limb (a 2012-specific warp)."""
    tx = observer
    rx = tx if (rx_station is None or rx_station == tx) else rx_station

    def o_eff(t):
        o_tx, _ = observer_direction_bodyfixed(t, spin, tx, abcorr)
        if rx == tx:
            return o_tx
        o_rx, _ = observer_direction_bodyfixed(t, spin, rx, abcorr)
        v = o_tx + o_rx
        return v / np.linalg.norm(v)

    return float(np.linalg.norm(o_eff(et + dt / 2.0) - o_eff(et - dt / 2.0)) / dt)


def predicted_freq_scale(et, spin: Spin, baud_us, code_length, K=1.0,
                         observer="ARECIBO", abcorr="CN+S", rx_station=None):
    """Geometric freq_scale prediction (moon-radar A2): the Doppler bandwidth is
    physical (∝ apparent rotation), so predict the per-epoch freq_scale from
    geometry instead of fitting it. ``K`` is a single calibration constant that
    absorbs the common pre-processing / instrument scale (exact bandwidth-to-column
    factors, provider resampling) and is fit once across epochs. ``rx_station`` gives
    the bistatic apparent rotation (must match the Doppler axis — see above).

        B_limbtolimb (Hz) = 4 * wapp * R / lambda      (monostatic, two limbs)
        freq_scale        = K * B * code_length * baud_s
    """
    wapp = apparent_rotation_rate(et, spin, observer, abcorr, rx_station=rx_station)
    B = 4.0 * wapp * (VENUS_RADIUS_KM * 1e3) / VENUS_WAVELENGTH_M
    return K * B * code_length * (baud_us * 1e-6)


def relative_state(et, observer="ARECIBO", abcorr="CN+S"):
    """Apparent Venus state relative to observer (J2000), SI: returns dict with
    range_m, range_rate_mps (radial), ortho_mps (transverse speed of the center),
    and the apparent angular rate of the LOS (rad/s)."""
    # range / LOS-rate context only; the topocentric vs geocentric difference is
    # negligible at Venus distance, and spkezr has no Arecibo body.
    if observer == "ARECIBO":
        observer = "EARTH"
    state, lt = csp.spkezr("VENUS", et, "J2000", abcorr, observer)
    pos = np.asarray(state[:3], float) * 1e3   # km -> m
    vel = np.asarray(state[3:], float) * 1e3   # km/s -> m/s
    r = np.linalg.norm(pos)
    rhat = pos / r
    v_radial = float(np.dot(vel, rhat))
    v_perp_vec = vel - v_radial * rhat
    v_ortho = float(np.linalg.norm(v_perp_vec))
    return {
        "range_m": float(r),
        "range_rate_mps": v_radial,
        "ortho_mps": v_ortho,
        "los_rate_rad_s": v_ortho / r,
    }
