"""Validation of the parameterized geometry core against SPICE references.

Run: .conda/bin/python test/test_geometry.py
"""
import os, sys
import numpy as np
import cspyce as csp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from venera import spice_setup
from venera.geometry import Spin, bodyfixed_matrix, sub_radar_point, doppler_angle

spice_setup.furnsh_kernels()

EPOCHS = ["1988-06-17T15:00:00", "2001-03-31T15:00:00", "2012-05-29T16:00:00",
          "2015-08-13T16:00:00", "2017-03-26T17:00:00", "2020-05-30T17:00:00"]
ETS = [csp.str2et(e) for e in EPOCHS]

_fail = 0
def check(name, cond, detail=""):
    global _fail
    status = "PASS" if cond else "FAIL"
    if not cond:
        _fail += 1
    print(f"  [{status}] {name}  {detail}")


print("== 1. Parameterized body-fixed frame vs SPICE IAU_VENUS (nominal constants) ==")
nom = Spin.iau_nominal()
print(f"  nominal spin: P={nom.period_days:.4f} d, RA={nom.pole_ra_deg}, "
      f"DEC={nom.pole_dec_deg}, W0={nom.w0_deg}, Wdot={nom.w_dot_deg_per_day:.7f} deg/day")
maxdiff = 0.0
for et in ETS:
    M_mine = bodyfixed_matrix(et, nom)
    M_spice = csp.pxform("J2000", "IAU_VENUS", et)
    maxdiff = max(maxdiff, np.max(np.abs(np.asarray(M_mine) - np.asarray(M_spice))))
check("body-fixed matrix matches IAU_VENUS", maxdiff < 1e-9, f"max|Δ|={maxdiff:.2e}")


print("== 2. Sub-radar point vs SPICE subpnt (nominal constants) ==")
max_sep_deg = 0.0
for e, et in zip(EPOCHS, ETS):
    lon, lat, et_b = sub_radar_point(et, nom, observer="EARTH", abcorr="CN+S")
    spoint, trgepc, srfvec = csp.subpnt("Near point: ellipsoid", "VENUS", et,
                                        "IAU_VENUS", "CN+S", "EARTH")
    _, slon, slat = csp.reclat(spoint)
    # angular separation between the two sub-radar directions
    u1 = np.array([np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)])
    u2 = np.array([np.cos(slat)*np.cos(slon), np.cos(slat)*np.sin(slon), np.sin(slat)])
    sep_deg = np.degrees(np.arccos(np.clip(np.dot(u1, u2), -1, 1)))
    max_sep_deg = max(max_sep_deg, sep_deg)
    print(f"    {e}: lon={np.degrees(lon):8.3f}  lat={np.degrees(lat):7.3f}  "
          f"sep_vs_subpnt={sep_deg*3600:7.2f} arcsec  lt_emit_match={abs(et_b-trgepc):.1e}s")
check("sub-radar point matches subpnt", max_sep_deg < 1e-3, f"max sep={max_sep_deg*3600:.2f} arcsec")


print("== 3. Physical sanity ==")
lats = [np.degrees(sub_radar_point(et, nom)[1]) for et in ETS]
check("SRP stays near Venus equator (|lat|<10deg)", max(abs(l) for l in lats) < 10,
      f"|lat|max={max(abs(l) for l in lats):.2f} deg")
das = [np.degrees(doppler_angle(et, nom)) for et in ETS]
check("Doppler angle small (|da|<15deg, cf Campbell ~10)", max(abs(d) for d in das) < 15,
      f"da={[round(d,2) for d in das]}")


print("== 4. Rotation-rate premise: Δlon = -δω·Δt (the basis of the fit) ==")
# Perturb the period; the resulting differential longitude shift between two
# epochs must equal -δWdot·Δd to high accuracy (pins sign + linearity).
et0 = csp.str2et("1988-06-17T15:00:00")
et1 = csp.str2et("2017-03-26T17:00:00")
dP = 0.001  # days
P = nom.period_days
nom2 = Spin(period_days=P + dP, pole_ra_deg=nom.pole_ra_deg,
            pole_dec_deg=nom.pole_dec_deg, w0_deg=nom.w0_deg)
lon0_a, _, etb0 = sub_radar_point(et0, nom)
lon1_a, _, etb1 = sub_radar_point(et1, nom)
lon0_b, _, _ = sub_radar_point(et0, nom2)
lon1_b, _, _ = sub_radar_point(et1, nom2)
def wrap(x): return np.arctan2(np.sin(x), np.cos(x))
d_dlon = np.degrees(wrap((lon1_b - lon0_b) - (lon1_a - lon0_a)))  # observed differential shift
d_days = (etb1 - etb0) / 86400.0
dWdot = (360.0 / (P + dP)) - (360.0 / P)   # change in |W rate|; retrograde -> sign below
predicted = -(-dWdot) * d_days   # δWdot_signed = -dWdot (retrograde); δlon = -δWdot·Δd
# i.e. predicted differential shift in degrees
predicted_deg = -(-(dWdot)) * d_days
# Cleaner: W_dot = -360/P, δW_dot = -360/(P+dP)+360/P = -dWdot... compute directly:
dWdot_signed = (-360.0 / (P + dP)) - (-360.0 / P)
predicted_deg = -dWdot_signed * d_days
rel_err = abs(d_dlon - predicted_deg) / abs(predicted_deg)
print(f"    baseline={d_days/365.25:.1f} yr, dP={dP} d -> observed Δlon={d_dlon:.5f} deg, "
      f"predicted={predicted_deg:.5f} deg")
check("longitude shift matches -δω·Δt to <1%", rel_err < 0.01, f"rel_err={rel_err:.2e}")
# Also report the inverse sensitivity: deg of registration per mday of period error
sens_deg_per_mday = abs(predicted_deg / (dP * 1000.0))
print(f"    sensitivity over this baseline: {sens_deg_per_mday:.4f} deg longitude per "
      f"milli-day of period error  (=> {1/sens_deg_per_mday:.3f} mday per deg)")


print("== 5. Observer is topocentric Arecibo, not the geocenter (regression guard) ==")
# The radar is at Arecibo; its diurnal rotation velocity (~0.46 km/s) is part of the
# apparent rotation that sets the Doppler axis d_hat. Using the geocenter omits it and
# tilts d_hat (-> the apparent north) by a few degrees, epoch-dependently, which
# mis-registers the cross-session stack. Default observer MUST be topocentric Arecibo.
from venera.geometry import doppler_basis, observer_direction_bodyfixed
et = ETS[3]  # 2015: off-equator sub-radar point, most sensitive
o_are, _ = observer_direction_bodyfixed(et, nom)                    # default = ARECIBO
o_geo, _ = observer_direction_bodyfixed(et, nom, observer="EARTH")
srp_par = np.degrees(np.arccos(np.clip(np.dot(o_are, o_geo), -1, 1)))
_, d_are = doppler_basis(et, nom)
_, d_geo = doppler_basis(et, nom, observer="EARTH")
d_ang = np.degrees(np.arccos(np.clip(np.dot(d_are, d_geo), -1, 1)))
print(f"    SRP parallax (Arecibo-geocenter) = {srp_par*3600:.1f} arcsec   "
      f"Doppler-axis tilt = {d_ang:.2f} deg")
check("default observer is topocentric (SRP differs from geocenter)",
      srp_par * 3600 > 1.0, f"{srp_par*3600:.1f} arcsec")
check("Arecibo Doppler axis differs from geocenter (the registration bug magnitude)",
      d_ang > 0.5, f"{d_ang:.2f} deg")


print()
if _fail:
    print(f"FAILED ({_fail} check(s))")
    sys.exit(1)
print("ALL GEOMETRY CHECKS PASSED")
