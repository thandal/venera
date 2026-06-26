"""Is Venus' spin rate changing across 1988-2020? Test for an acceleration term.

A constant rate makes the per-pair longitude shift linear in baseline: m_ij = c·Δt_ij.
A changing rate adds a quadratic-in-epoch term: with absolute epoch t_k (yr),
  m_ij = L_i - L_j,  L_k = c·t_k + b·t_k² + o_k,
so m_ij = c·(t_i - t_j) + b·(t_i² - t_j²) + (o_i - o_j). The coefficient b is the
acceleration (b<0 => period lengthening => spin slowing). o_k are the per-session
longitude offsets.

THE CATCH (stated up front): with 6 epochs each carrying an unknown offset o_k, the
quadratic b is degenerate with a quadratic-in-epoch pattern of offsets exactly as the
period (linear c) is degenerate with a linear one — set o_k = c·t_k + b·t_k² and any
(c,b) vanishes. So b is only identifiable UNDER THE ASSUMPTION that the offsets are
random, not systematically quadratic in time (the §4 centering premise). We fit b under
that assumption (offsets folded into the σ_o intrinsic-scatter floor) and report it with
honest errors; we also give a model-free early-vs-late split. Reuses the cached NCC
curves from period_joint_coherence.py (no image recompute).

Writes results/figures/period_acceleration.png.
"""
import os, sys, glob, itertools
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STACKS = os.path.join(ROOT, "results", "session_stacks")
CACHE = os.path.join(ROOT, "results", "look_cache")
FIG = os.path.join(ROOT, "results", "figures")
YEARS = ["1988", "2001", "2012", "2015", "2017", "2020"]
P_ASSUMED = 243.0185
T0 = 2004.0                       # epoch zero-point (yr), mid-baseline


def mean_day(y):
    return float(np.mean([float(np.load(f, allow_pickle=True)["day"])
                          for f in glob.glob(os.path.join(CACHE, f"venus_*_{y}*.npz"))]))


def peak_from_curve(P, c):
    """Per-pair peak period and σ_P from an NCC-vs-period curve (curvature method)."""
    g = np.isfinite(c)
    if g.sum() < 8:
        return None
    i = int(np.nanargmax(c)); lo, hi = max(i - 6, 0), min(i + 7, len(P))
    co = np.polyfit(P[lo:hi], c[lo:hi], 2)
    if co[0] >= 0:
        return None
    Ppk = -co[1] / (2 * co[0]); k = -2 * co[0]
    smooth = np.polyval(np.polyfit(P[g], c[g], 4), P[g])
    sigN = np.nanstd(c[g] - smooth); sigP = np.sqrt(2 * sigN / k) if k > 0 else np.nan
    return Ppk, sigP, np.nanmax(c)


def main():
    cz = np.load(os.path.join(STACKS, "period_joint_cache.npz"), allow_pickle=True)
    PGRID = cz["pgrid"]; ncc_mat = cz["ncc_mat"]
    pairs = list(itertools.combinations(YEARS, 2))
    t = {y: mean_day(y) / 365.25 + 2000.0 - T0 for y in YEARS}  # yr relative to T0
    yr = {y: mean_day(y) / 365.25 + 2000.0 for y in YEARS}
    DEG_PER_DAY_PER_P = 360.0 / P_ASSUMED**2

    rows = []   # (a,b, X1=Δt_yr, X2=t_i²-t_j², m_deg, sigm_deg, midyr)
    for qi, (a, b) in enumerate(pairs):
        pk = peak_from_curve(PGRID, ncc_mat[qi])
        if pk is None:
            continue
        Ppk, sigP, _ = pk
        dt_days = (mean_day(b) - mean_day(a))
        m = -((-360.0 / Ppk) - (-360.0 / P_ASSUMED)) * dt_days        # deg
        sigm = DEG_PER_DAY_PER_P * abs(dt_days) * sigP
        X1 = t[b] - t[a]; X2 = t[b]**2 - t[a]**2
        rows.append((a, b, X1, X2, m, sigm, 0.5 * (yr[a] + yr[b])))

    X1 = np.array([r[2] for r in rows]); X2 = np.array([r[3] for r in rows])
    Mv = np.array([r[4] for r in rows]); S = np.array([r[5] for r in rows])
    mid = np.array([r[6] for r in rows]); n = len(rows)

    # offset-jitter floor: σ_o s.t. the constant-rate (linear-only) fit has χ²/dof=1
    def lin_fit(idx, var):
        c = np.sum(X1[idx] * Mv[idx] / var[idx]) / np.sum(X1[idx]**2 / var[idx])
        return c
    def lin_chi2(var):
        c = lin_fit(np.arange(n), var); return np.sum((Mv - c * X1)**2 / var) / (n - 1)
    so = 0.0
    if lin_chi2(S**2) > 1:
        loS, hiS = 0.0, 1.0
        for _ in range(60):
            so = 0.5 * (loS + hiS)
            if lin_chi2(S**2 + so**2) > 1: loS = so
            else: hiS = so
    var = S**2 + so**2

    # joint linear+quadratic weighted LSQ:  m = c1·X1 + c2·X2
    G = np.vstack([X1, X2]).T
    W = np.diag(1.0 / var)
    cov = np.linalg.inv(G.T @ W @ G)
    coef = cov @ (G.T @ W @ Mv)
    c1, c2 = coef; s_c1, s_c2 = np.sqrt(np.diag(cov))
    chi2_qd = np.sum((Mv - G @ coef)**2 / var) / (n - 2)

    # convert to period(t) and dP/dt.  δω(t)=c1+2 c2 t (deg/yr); P=360/(δω/365.25+360/Passumed)
    def P_of(tt): return 360.0 / ((c1 + 2 * c2 * tt) / 365.25 + 360.0 / P_ASSUMED)
    t88, t20 = (1988 - 2004.0), (2020 - 2004.0)
    P88, P20 = P_of(t88), P_of(t20)
    dPdt = (P20 - P88) / 32.0
    # propagate σ on dP/dt (dominated by c2): dP/dt ≈ (P²/360)·(2 c2)/365.25 in d/yr
    fac = (((P88 + P20) / 2)**2 / 360.0) * (2.0 / 365.25)
    s_dPdt = fac * s_c2

    print(f"epochs (yr): " + ", ".join(f"{y}={yr[y]:.2f}" for y in YEARS), flush=True)
    print(f"offset-jitter floor σ_o = {so:.3f}°  (linear-rate χ²/dof forced to 1)\n", flush=True)
    print(f"linear+quadratic fit (m = c1·Δt + c2·(t_i²−t_j²)):", flush=True)
    print(f"  c1 = {c1:+.5f} ± {s_c1:.5f} °/yr      (the rate -> period)", flush=True)
    print(f"  c2 = {c2:+.6f} ± {s_c2:.6f} °/yr²    (acceleration; <0 = spin slowing)", flush=True)
    print(f"  -> c2 significance = {c2/s_c2:+.2f} σ   (χ²/dof of the 2-term fit = {chi2_qd:.2f})\n", flush=True)
    print(f"implied period drift: P(1988)={P88:.4f}, P(2020)={P20:.4f} d  "
          f"-> dP/dt = {dPdt*1000:+.3f} ± {s_dPdt*1000:.3f} mday/yr", flush=True)
    print(f"  significance of dP/dt = {dPdt/s_dPdt:+.2f} σ", flush=True)

    # robustness: is c2 a stable signal or an artifact of one epoch's offset?
    print("\nSession-jackknife on the acceleration c2 (drop one epoch, refit):", flush=True)
    for s in YEARS:
        idx = np.array([i for i, r in enumerate(rows) if s not in (r[0], r[1])])
        Gj = np.vstack([X1[idx], X2[idx]]).T; Wj = np.diag(1.0 / var[idx])
        covj = np.linalg.inv(Gj.T @ Wj @ Gj); cj = covj @ (Gj.T @ Wj @ Mv[idx])
        sj = np.sqrt(np.diag(covj))[1]
        print(f"  without {s}: c2 = {cj[1]:+.6f} ± {sj:.6f} °/yr²  ({cj[1]/sj:+.1f}σ)", flush=True)

    # model-free cross-check: split pairs by midpoint, fit a constant period to each half
    for lab, sel in [("early (mid<2008)", mid < 2008), ("late  (mid>=2008)", mid >= 2008)]:
        idx = np.where(sel)[0]
        c = lin_fit(idx, var); P = 360.0 / (c / 365.25 + 360.0 / P_ASSUMED)
        # crude σ from bootstrap
        rng = np.random.default_rng(0)
        Ps = [360.0 / (lin_fit(idx[rng.integers(0, len(idx), len(idx))], var) / 365.25
                       + 360.0 / P_ASSUMED) for _ in range(3000)]
        print(f"  {lab}: P = {P:.4f} ± {0.5*(np.percentile(Ps,84)-np.percentile(Ps,16)):.4f} d "
              f"({len(idx)} pairs)", flush=True)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    sc = ax.scatter([r[6] for r in rows], [(r[4] / r[2]) for r in rows],
                    c=np.abs(X1), s=40, cmap="viridis")
    ax.axhline(c1, color="b", lw=1.2, label=f"constant rate c1={c1:+.4f}°/yr")
    tt = np.linspace(-17, 17, 50)
    ax.plot(tt + T0, c1 + 2 * c2 * tt, "r--", lw=1.2,
            label=f"+ accel c2={c2:+.5f}°/yr² ({c2/s_c2:+.1f}σ)")
    ax.set_xlabel("pair midpoint epoch (yr)"); ax.set_ylabel("apparent rate m_ij/Δt (°/yr)")
    ax.set_title("Rate vs epoch — is there an acceleration?"); ax.legend(fontsize=9)
    ax.grid(alpha=0.3); fig.colorbar(sc, label="|Δt| (yr)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "period_acceleration.png"), dpi=120)
    print("\nwrote period_acceleration.png", flush=True)


if __name__ == "__main__":
    main()
