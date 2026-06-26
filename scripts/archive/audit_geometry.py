"""Careful audit of the projection geometry: does the code's (o_hat, d_hat) basis
match the TRUE delay-Doppler geometry computed INDEPENDENTLY from SPICE?

The projection assumes  delay ∝ 1 - p·o_hat,  doppler ∝ p·d_hat  with
d_hat = normalize(do_hat/dt). We check this against an independent ground truth:
for many body-fixed surface points p, compute the actual round-trip range and its
time-derivative (doppler) directly from the ephemeris (Earth->Venus vector) and the
body matrix M(t) -- NOT reusing the code's o_hat/d_hat. Then:
  - true delay gradient  -> compare its direction to the code's o_hat
  - true doppler gradient-> compare its direction to the code's d_hat
A mismatch in d_hat (tilting n_hat = o_hat x d_hat) that scales with sub-radar
latitude would be the cross-session tilt's source.

Also runs a projection round-trip self-consistency check.

Usage: .conda/bin/python scripts/audit_geometry.py
"""
import os, sys, glob
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from venera import spice_setup
from venera.geometry import (Spin, bodyfixed_matrix, doppler_basis,
                             observer_direction_bodyfixed, VENUS_RADIUS_KM, sub_radar_point)
import cspyce as csp

spice_setup.furnsh_kernels()
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "results", "look_cache")
ABCORR = "CN+S"


def mean_et(year):
    ets = [float(np.load(f, allow_pickle=True)["et_mid"])
           for f in glob.glob(os.path.join(CACHE, f"venus_*_{year}*.npz"))]
    return float(np.mean(ets))


def range_of(p, et, spin):
    """True Earth->point round-trip range proxy |ev + R*(M^T p)| (km), p body-fixed."""
    ev, _ = csp.spkpos("VENUS", et, "J2000", ABCORR, "EARTH")
    ev = np.asarray(ev, float)
    M = bodyfixed_matrix(et, spin)            # J2000 -> body
    pj = M.T @ p                              # body -> J2000
    return np.linalg.norm(ev + VENUS_RADIUS_KM * pj, axis=0)


def true_basis(et, spin, n=4000, cap_deg=80.0, dt=60.0):
    """Independent (o_hat, d_hat) from delay & doppler gradients over the cap."""
    o_code, d_code = doppler_basis(et, spin, dt=dt)
    # sample body points within cap of the sub-radar point
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(20000, 3)); pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    keep = pts @ o_code > np.cos(np.radians(cap_deg))
    P = pts[keep][:n]                          # (n,3) body-fixed unit points
    # delay proxy: range at et ; doppler: -d(range)/dt via finite diff
    r0 = np.array([range_of(p, et, spin) for p in P])
    rm = np.array([range_of(p, et - dt / 2, spin) for p in P])
    rp = np.array([range_of(p, et + dt / 2, spin) for p in P])
    delay = r0
    doppler = -(rp - rm) / dt
    # linear fit  q ~ a + g.p  -> gradient g
    A = np.hstack([np.ones((len(P), 1)), P])
    g_delay = np.linalg.lstsq(A, delay, rcond=None)[0][1:]
    g_dopp = np.linalg.lstsq(A, doppler, rcond=None)[0][1:]
    o_true = -g_delay / np.linalg.norm(g_delay)   # range decreases toward observer
    d_true = g_dopp / np.linalg.norm(g_dopp)
    return o_code, d_code, o_true, d_true


def ang(a, b):
    return np.degrees(np.arccos(np.clip(abs(np.dot(a, b)), -1, 1)))


def signed_ang(a, b):
    return np.degrees(np.arccos(np.clip(np.dot(a, b), -1, 1)))


def roundtrip_check(et, spin):
    """Forward (body point -> image coeffs) then inverse (coeffs -> body point)."""
    o, d = doppler_basis(et, spin, dt=60.0)
    n = np.cross(o, d); n /= np.linalg.norm(n)
    if n[2] < 0:
        n = -n
    rng = np.random.default_rng(1)
    pts = rng.normal(size=(5000, 3)); pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    keep = (pts @ o > np.cos(np.radians(80))) & (pts @ n > 0)   # visible N apparent cap
    P = pts[keep][:500]
    co = P @ o; cd = P @ d; cn = P @ n
    rec = co[:, None] * o + cd[:, None] * d + np.sqrt(np.clip(1 - co**2 - cd**2, 0, 1))[:, None] * n
    err = np.degrees(np.arccos(np.clip(np.sum(rec * P, axis=1), -1, 1)))
    return err.max()


def main():
    spin = Spin()
    print("== projection round-trip self-consistency (should be ~0) ==")
    for y in ("2015", "2017"):
        e = mean_et(y)
        print(f"  {y}: max round-trip error = {roundtrip_check(e, spin):.4f} deg")

    print("\n== code basis vs INDEPENDENT SPICE delay/doppler geometry ==")
    print(f"  {'epoch':6s} {'SRP_lat':>8s} {'ang(o_code,o_true)':>18s} "
          f"{'ang(d_code,d_true)':>18s}")
    res = {}
    for y in ("1988", "2001", "2012", "2015", "2017", "2020"):
        e = mean_et(y)
        _, lat, _ = sub_radar_point(e, spin)
        oc, dc, ot, dt_ = true_basis(e, spin)
        a_o = ang(oc, ot); a_d = ang(dc, dt_)
        res[y] = (np.degrees(lat), a_o, a_d)
        print(f"  {y:6s} {np.degrees(lat):+8.1f} {a_o:18.4f} {a_d:18.4f}")
    print("\nInterpretation: ang(o)~0 (ephemeris ok). If ang(d_code,d_true) is "
          "nonzero and grows with |SRP_lat|, the doppler basis is the tilt source.")


if __name__ == "__main__":
    main()
