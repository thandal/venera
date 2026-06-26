"""Discriminate the per-epoch tilt: is it a fixed-axis (pole-like) rotation error,
or a per-look projection error whose axis tracks the sub-radar longitude?

Steps:
  1. 15 pairwise on-sphere rotations (search_R), cached to disk.
  2. fit per-epoch tilt vectors f_e (rotvec, ref 2012=0).
  3. test which frame the f_e directions cluster in:
       - INERTIAL  (f rotated by M_e^T)         -> fixed inertial axis = pole-like
       - SRP-LON   (f rotated by Rz(-srp_lon))  -> axis tied to sub-radar meridian
       - BODY      (f as-is)                    -> fixed body axis
     tightest cluster (mean resultant length, |f|-weighted) wins.
  4. fit a single inertial axis to the f_e and report residual + magnitude.

Usage: .conda/bin/python scripts/pole_discriminator.py
"""
import os, sys, glob, itertools
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)                          # for register_on_sphere
sys.path.insert(0, os.path.dirname(_HERE))         # repo root, for `venera`
from register_on_sphere import flatten, search_R, decompose, STACKS
from venera import spice_setup
from venera.geometry import Spin, bodyfixed_matrix

spice_setup.furnsh_kernels()
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "results", "look_cache")
RVFILE = os.path.join(ROOT, "results", "pairwise_rotvecs.npz")
YEARS = ["1988", "2001", "2012", "2015", "2017", "2020"]
REF = "2012"
DEG = np.pi / 180.0


def epoch_geometry():
    et, slon, slat = {}, {}, {}
    for y in YEARS:
        ets, lons, lats = [], [], []
        for f in glob.glob(os.path.join(CACHE, f"venus_*_{y}*.npz")):
            d = np.load(f, allow_pickle=True)
            ets.append(float(d["et_mid"])); lats.append(float(d["srp_lat"]))
            lons.append(float(d["srp_lon"]))
        et[y] = np.mean(ets)
        slat[y] = np.mean(lats)
        slon[y] = np.degrees(np.arctan2(np.mean(np.sin(np.radians(lons))),
                                        np.mean(np.cos(np.radians(lons)))))
    return et, slon, slat


def Rz(deg):
    a = np.radians(deg); c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def resultant(vecs, weights):
    """|f|-weighted mean resultant length of unit directions (1=aligned, 0=scattered).
    Uses axial (sign-free) statistics since a tilt axis is a line, not a ray."""
    u = np.array([v / (np.linalg.norm(v) + 1e-12) for v in vecs])
    # axial: align signs to the weighted dominant direction
    ref = np.sum(weights[:, None] * u, axis=0)
    ref /= np.linalg.norm(ref) + 1e-12
    s = np.sign(u @ ref); s[s == 0] = 1
    u = u * s[:, None]
    R = np.linalg.norm(np.sum(weights[:, None] * u, axis=0)) / weights.sum()
    return R


def main():
    pairs = list(itertools.combinations(YEARS, 2))
    if os.path.exists(RVFILE):
        z = np.load(RVFILE)
        rv = {tuple(k.split("|")): z[k] for k in z.files}
        print("loaded cached pairwise rotvecs")
    else:
        stk = {}
        for y in YEARS:
            d = np.load(os.path.join(STACKS, f"session_{y}.npz"))
            stk[y] = (flatten(d["Gm"], d["mask"]), d["mask"])
        rv = {}
        for a, b in pairs:
            r, nf, ni = search_R(*stk[a], *stk[b], coarse_step=1.5)
            rv[(a, b)] = r
            print(f"  {a}-{b}: tilt {decompose(r)[2]:.2f}  NCC {ni:.2f}->{nf:.2f}",
                  flush=True)
        np.savez(RVFILE, **{f"{a}|{b}": rv[(a, b)] for a, b in pairs})

    et, slon, slat = epoch_geometry()
    # exclude pairs whose direct search stuck (large predicted tilt but ~0 found)
    reliable = [(a, b) for (a, b) in pairs
                if not (abs(slat[a] - slat[b]) > 5.0
                        and np.degrees(np.linalg.norm(rv[(a, b)])) < 0.3)]
    stuck = [p for p in pairs if p not in reliable]
    print(f"excluded stuck pairs from fit: {stuck}")

    # fit per-epoch f_e (ref=0), per component, on reliable pairs only
    idx = {y: i for i, y in enumerate([y for y in YEARS if y != REF])}
    A = np.zeros((len(reliable), len(idx)))
    for i, (a, b) in enumerate(reliable):
        if a in idx:
            A[i, idx[a]] += 1
        if b in idx:
            A[i, idx[b]] -= 1
    f = {y: np.zeros(3) for y in YEARS}
    for k in range(3):
        rhs = np.array([rv[p][k] for p in reliable])
        sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)
        for y, i in idx.items():
            f[y][k] = sol[i]
    resid = np.sqrt(np.mean([np.sum((rv[p] - (f[p[0]] - f[p[1]])) ** 2) for p in reliable]))
    print(f"\nper-epoch model fit RMS residual (reliable): {np.degrees(resid):.3f} deg")
    s0 = Spin()
    W = {y: (s0.w0_deg + s0.w_dot_deg_per_day * et[y] / 86400.0) % 360 for y in YEARS}
    big = [y for y in YEARS if np.degrees(np.linalg.norm(f[y])) > 0.3]
    print("\nepoch  SRP_lat  SRP_lon    W(spin)   |f|(deg)")
    for y in YEARS:
        print(f"  {y}  {slat[y]:+6.1f}  {slon[y]:+7.1f}   {W[y]:6.1f}    "
              f"{np.degrees(np.linalg.norm(f[y])):.2f}")
    lon_rng = max(slon.values()) - min(slon.values())
    w_rng = max(W.values()) - min(W.values())
    print(f"  --> SRP_lon spans {lon_rng:.0f}deg, spin-phase W spans {w_rng:.0f}deg "
          f"(same-face resonance: small spans => pole vs body-frame DEGENERATE here)")

    # frame tests (use epochs with significant tilt)
    w = np.array([np.linalg.norm(f[y]) for y in big])
    body = [f[y] for y in big]
    inertial = [bodyfixed_matrix(et[y], Spin()).T @ f[y] for y in big]
    srprel = [Rz(-slon[y]) @ f[y] for y in big]
    print(f"\nframe clustering of tilt axes (1=fixed axis, 0=scattered), epochs {big}:")
    print(f"  BODY-fixed         : {resultant(body, w):.3f}")
    print(f"  INERTIAL (pole)    : {resultant(inertial, w):.3f}")
    print(f"  SRP-longitude rel  : {resultant(srprel, w):.3f}")

    # single inertial-axis fit: f_e ~ (Rz(W_e)-Rz(W_ref)) delta  (small angle)
    s = Spin()
    W = {y: (s.w0_deg + s.w_dot_deg_per_day * et[y] / 86400.0) % 360 for y in YEARS}
    M = np.vstack([(Rz(W[y]) - Rz(W[REF])) for y in YEARS if y != REF])
    rhs = np.concatenate([f[y] for y in YEARS if y != REF])
    delta, *_ = np.linalg.lstsq(M, rhs, rcond=None)
    pred = {y: (Rz(W[y]) - Rz(W[REF])) @ delta for y in YEARS}
    polefit_resid = np.sqrt(np.mean([np.sum((f[y] - pred[y]) ** 2)
                                     for y in YEARS if y != REF]))
    print(f"\nsingle inertial-pole-offset fit:")
    print(f"  |delta| = {np.degrees(np.linalg.norm(delta)):.2f} deg")
    print(f"  fit RMS residual = {np.degrees(polefit_resid):.3f} deg "
          f"(vs per-epoch model {np.degrees(resid):.3f}); "
          f"low => pole-like, high => not a single pole")


if __name__ == "__main__":
    main()
