"""Test the tilt law on the pairs whose direct search stuck at identity
(1988<->2015, 2015<->2020). Predict their tilt from a per-epoch frame model fit to
the pairs that DID lock, force it on, and see if the imagery snaps into registration.

Model: each epoch e has a small frame-tilt vector f_e; a pair's rotation
rv(a,b) ~ f_a - f_b (small angle). Fit f_e (ref 2012=0) by least squares over the
reliable pairs, then PREDICT rv for the stuck pairs and evaluate on-sphere NCC at
identity, at the predicted tilt, and after a local refine seeded at the prediction.

Writes results/figures/stuck_beforeafter.png

Usage: .conda/bin/python scripts/fix_stuck_pairs.py
"""
import os, sys, itertools
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from register_on_sphere import (flatten, search_R, decompose, rotate_map,
                                make_eval_points, sample, v2ll, wncc,
                                STACKS, FIG, HH, WW, DEG)

YEARS = ["1988", "2001", "2012", "2015", "2017", "2020"]
REF = "2012"
LAT0, LAT1, LON0, LON1 = 25, 82, -115, 60


def ncc_at(Af, Am, Bf, Bm, rotvec, P=None, w=None, Av=None, Aok=None):
    if P is None:
        P, w = make_eval_points(Am)
        Av, Aok = sample(Af, Am, *v2ll(P))
    R = Rotation.from_rotvec(rotvec).as_matrix()
    lo, la = v2ll(P @ R.T)
    Bv, Bok = sample(Bf, Bm, lo, la)
    m = Aok & Bok
    return wncc(Av[m], Bv[m], w[m]) if m.sum() > 1000 else 0.0


def refine(Af, Am, Bf, Bm, seed):
    P, w = make_eval_points(Am)
    Av, Aok = sample(Af, Am, *v2ll(P))
    f = lambda rv: -ncc_at(Af, Am, Bf, Bm, rv, P, w, Av, Aok)
    r = minimize(f, seed, method="Nelder-Mead",
                 options=dict(xatol=1e-4, fatol=1e-5, maxiter=2000))
    return r.x, -r.fun


def crop_rgb(Af, Am, Bf, Bm):
    r0 = int((LAT0 + 90) / 180 * HH); r1 = int((LAT1 + 90) / 180 * HH)
    c0 = int((LON0 + 180) / 360 * WW); c1 = int((LON1 + 180) / 360 * WW)
    A = Af[r0:r1, c0:c1]; B = Bf[r0:r1, c0:c1]; m = (Am & Bm)[r0:r1, c0:c1]

    def norm(x):
        v = x[m]
        if v.size == 0:
            return np.zeros_like(x)
        lo, hi = np.percentile(v, 2), np.percentile(v, 98)
        return np.clip((x - lo) / (hi - lo + 1e-9), 0, 1)
    rgb = np.zeros(A.shape + (3,))
    rgb[..., 0] = norm(A) * m; rgb[..., 1] = norm(B) * m
    return rgb


def main():
    stk, srp = {}, {}
    for y in YEARS:
        d = np.load(os.path.join(STACKS, f"session_{y}.npz"))
        stk[y] = (flatten(d["Gm"], d["mask"]), d["mask"])
        srp[y] = float(d["srp_lat_mean"])

    # 1. direct search on every pair
    pairs = list(itertools.combinations(YEARS, 2))
    rec = {}
    for ya, yb in pairs:
        Af, Am = stk[ya]; Bf, Bm = stk[yb]
        rv, nfit, nid = search_R(Af, Am, Bf, Bm, coarse_step=1.5)
        rec[(ya, yb)] = dict(rv=rv, nid=nid, nfit=nfit)
        print(f"  search {ya}-{yb}: tilt {decompose(rv)[2]:.2f} NCC {nid:.2f}->{nfit:.2f}",
              flush=True)

    # 2. classify stuck vs reliable
    stuck = [(a, b) for (a, b) in pairs
             if rec[(a, b)]["nfit"] - rec[(a, b)]["nid"] < 0.03
             and abs(srp[a] - srp[b]) > 5.0]
    reliable = [p for p in pairs if p not in stuck]
    print(f"\nstuck pairs: {stuck}")

    # 3. fit per-epoch frame vectors f_e (ref REF=0) from reliable pairs, per component
    idx = {y: i for i, y in enumerate([y for y in YEARS if y != REF])}
    A = np.zeros((len(reliable), len(idx)))
    for i, (a, b) in enumerate(reliable):
        if a in idx:
            A[i, idx[a]] += 1
        if b in idx:
            A[i, idx[b]] -= 1
    f = np.zeros((len(YEARS), 3))
    fmap = {y: np.zeros(3) for y in YEARS}
    for k in range(3):
        rhs = np.array([rec[p]["rv"][k] for p in reliable])
        sol, res, *_ = np.linalg.lstsq(A, rhs, rcond=None)
        for y, i in idx.items():
            fmap[y][k] = sol[i]
    fit_resid = np.sqrt(np.mean([np.sum((rec[p]["rv"] -
                 (fmap[p[0]] - fmap[p[1]])) ** 2) for p in reliable]))
    print(f"\nper-epoch model fit RMS residual: {np.degrees(fit_resid):.3f} deg")
    print("per-epoch tilt magnitude vs SRP_lat:")
    for y in YEARS:
        print(f"  {y}: |f|={np.degrees(np.linalg.norm(fmap[y])):.2f} deg  "
              f"SRP_lat={srp[y]:+.1f}")

    # 4. force predicted tilt on stuck pairs
    n = len(stuck)
    fig, axs = plt.subplots(n, 2, figsize=(9, 4.3 * n))
    if n == 1:
        axs = axs[None, :]
    for r, (ya, yb) in enumerate(stuck):
        Af, Am = stk[ya]; Bf, Bm = stk[yb]
        pred = fmap[ya] - fmap[yb]
        nid = rec[(ya, yb)]["nid"]
        npred = ncc_at(Af, Am, Bf, Bm, pred)
        rvref, nref = refine(Af, Am, Bf, Bm, pred)
        tp = decompose(pred)[2]; tr = decompose(rvref)[2]
        dsrp = srp[ya] - srp[yb]
        print(f"\n{ya}-{yb} (ΔSRP {dsrp:+.1f}):")
        print(f"  predicted tilt {tp:.2f} deg (law ~{0.15*abs(dsrp):.2f})")
        print(f"  NCC: identity {nid:.3f} -> predicted {npred:.3f} -> "
              f"refined {nref:.3f} (refined tilt {tr:.2f})")
        R = Rotation.from_rotvec(pred).as_matrix()
        Br, Brm = rotate_map(Bf, Bm, R.T)
        axs[r, 0].imshow(crop_rgb(Af, Am, Bf, Bm), origin="lower",
                         extent=[LON0, LON1, LAT0, LAT1], aspect="auto")
        axs[r, 0].set_title(f"{ya}(r) vs {yb}(g) — IDENTITY  NCC={nid:.2f}", fontsize=10)
        axs[r, 1].imshow(crop_rgb(Af, Am, Br, Brm), origin="lower",
                         extent=[LON0, LON1, LAT0, LAT1], aspect="auto")
        axs[r, 1].set_title(f"FORCED predicted tilt {tp:.1f}°  NCC={npred:.2f}", fontsize=10)
        for c in (0, 1):
            axs[r, c].set_xlabel("lon (°E)"); axs[r, c].set_ylabel("lat (°N)")
    fig.suptitle("Stuck pairs: forcing the per-epoch-model predicted tilt "
                 "(yellow = aligned)", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "stuck_beforeafter.png"), dpi=120)
    print("\nwrote stuck_beforeafter.png")


if __name__ == "__main__":
    main()
