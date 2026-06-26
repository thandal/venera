"""Joint coherence period fit: the single sidereal period that maximizes
cross-session registration coherence, using ALL 15 session pairs / ALL baselines at
once, with honest uncertainties. This is PLAN.md item 1.

WHY THE EXISTING FITS ARE BIASED / ILL-POSED
  - period_solve_all.py fits a rate c PLUS a free constant longitude offset per
    session {o_k}. That is RANK-DEFICIENT: setting o_k = c*day_k (offsets linear in
    epoch) reproduces any rate, so the period (slope) is unidentifiable from the
    offsets. numpy lstsq returns the min-norm solution -> an arbitrary c with a
    meaninglessly tight bootstrap.
  - period_errorbars.py averages per-PAIR periods (inverse-variance). A per-session
    centering offset o_k biases a pair's apparent period by ~o_k/Δt, so SHORT-baseline
    pairs are strongly biased; IVW then over-weights those tight-but-wrong short pairs.

THE FIX — FIT THE LONGITUDE SLOPE, NO PER-SESSION OFFSETS
  For each pair (i,j): the interior-NCC-vs-period peak gives the longitude shift m_ij
  that best aligns the pair (deg), with σ_m from the peak curvature. A per-session
  offset adds a baseline-INDEPENDENT scatter to m_ij, so a weighted line
  m_ij = c·Δt_ij THROUGH THE ORIGIN (one free parameter) recovers the rate c -> period,
  dominated by the long baselines and robust to the offsets. χ²-inflation absorbs any
  real offset scatter; pair-bootstrap and session-jackknife show stability. The
  combined σ-clip image at the optimum vs at IAU is the imagery validation (the
  non-negotiable rule). Single free parameter -> well-posed, real uncertainty.

Writes results/figures/period_joint_coherence.png. Print stdout to a log via
redirection (never hand-edit the log).
"""
import os, sys, glob, itertools
import numpy as np
from scipy.ndimage import gaussian_filter, binary_erosion, shift as ndshift
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from venera.coherence import ncc, coherent_stack, stack_sharpness

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STACKS = os.path.join(ROOT, "results", "session_stacks")
CACHE = os.path.join(ROOT, "results", "look_cache")
FIG = os.path.join(ROOT, "results", "figures")
SUFFIX = sys.argv[1] if len(sys.argv) > 1 else ""
TAG = f"_{SUFFIX}" if SUFFIX else ""
HH, WW = 2000, 4000
YEARS = ["1988", "2001", "2012", "2015", "2017", "2020"]
P_ASSUMED = 243.0185
ERODE_PX, LATMIN, LATMAX = 20, 30, 75
RW0, RW1 = int((25 + 90) / 180 * HH), int((80 + 90) / 180 * HH)   # crop (speed)
PGRID = np.arange(243.0110, 243.0312, 0.0005)                     # 41 trial periods
DEG_PER_DAY_PER_P = 360.0 / P_ASSUMED**2                          # |dm/dP|/Δt at P_assumed


def mean_day(y):
    return float(np.mean([float(np.load(f, allow_pickle=True)["day"])
                          for f in glob.glob(os.path.join(CACHE, f"venus_*_{y}*.npz"))]))


def n_looks(y):
    pats = ([f"venus_{SUFFIX}_{y}*.npz"] if SUFFIX in ("ocp", "scp") else []) + [f"venus_*_{y}*.npz"]
    for pat in pats:
        n = len(glob.glob(os.path.join(CACHE, pat)))
        if n:
            return n
    return 0


def load(y):
    d = np.load(os.path.join(STACKS, f"session_{y}{TAG}.npz")); return d["Gm"], d["mask"]


def flatten(G, m, s=35):
    f = np.where(m, G, 0.0); w = gaussian_filter(m.astype(float), s)
    return ((G - gaussian_filter(f, s) / np.maximum(w, 1e-6)) * m)[RW0:RW1], m[RW0:RW1]


def latband(h):
    b = np.zeros((h, WW), bool)
    lo = int((LATMIN + 90) / 180 * HH) - RW0; hi = int((LATMAX + 90) / 180 * HH) - RW0
    b[max(lo, 0):hi] = True
    return b


def px_for(P, dday):
    """Longitude pixel shift to move a session at +dday past the reference to period P."""
    wdot = (-360.0 / P) - (-360.0 / P_ASSUMED)
    return (-wdot * dday) / 360.0 * WW


def main():
    print(f"=== polarization: {SUFFIX.upper() or 'COMBINED'} ===", flush=True)
    F, M, day, nlk = {}, {}, {}, {}
    for y in YEARS:
        G, m = load(y); F[y], M[y] = flatten(G, m); day[y] = mean_day(y); nlk[y] = n_looks(y)
    h = F[YEARS[0]].shape[0]; LB = latband(h)
    d_ref = np.mean([day[y] for y in YEARS])
    pairs = list(itertools.combinations(YEARS, 2))

    # --- single (cached) pass over the period grid: shift every session to the common
    #     ref, then (a) combined-stack sharpness and (b) all 15 pairwise interior-NCCs.
    #     The grid is the only expensive part; cache it so the fit can be iterated. ---
    CACHE_NPZ = os.path.join(STACKS, f"period_joint_cache{TAG}.npz")
    cache_ok = False
    if os.path.exists(CACHE_NPZ):
        cz = np.load(CACHE_NPZ, allow_pickle=True)
        if cz["pgrid"].shape == PGRID.shape and np.allclose(cz["pgrid"], PGRID):
            ncc_mat = cz["ncc_mat"]; sharp_curve = cz["sharp_curve"]; cache_ok = True
            print(f"(loaded cached curves from {os.path.basename(CACHE_NPZ)})", flush=True)
    if not cache_ok:
        ncc_mat = np.full((len(pairs), len(PGRID)), np.nan)
        sharp_curve = np.full(len(PGRID), np.nan)
        for pi, P in enumerate(PGRID):
            FS, MS = {}, {}
            for y in YEARS:
                px = px_for(P, day[y] - d_ref)
                FS[y] = ndshift(F[y], (0, px), order=1, mode="constant", cval=0.0)
                MS[y] = ndshift(M[y].astype(float), (0, px), order=1, mode="constant", cval=0) > 0.99
            stk, cnt = coherent_stack([FS[y] for y in YEARS], [MS[y] for y in YEARS])
            sharp_curve[pi] = stack_sharpness(stk, cnt, min_count=2)
            for qi, (a, b) in enumerate(pairs):
                inn = binary_erosion(MS[a] & MS[b], iterations=ERODE_PX) & LB
                if inn.sum() > 1500:
                    ncc_mat[qi, pi] = ncc(FS[a], FS[b], inn, inn)
        np.savez_compressed(CACHE_NPZ, pgrid=PGRID, ncc_mat=ncc_mat, sharp_curve=sharp_curve,
                            pairs=np.array([f"{a}-{b}" for a, b in pairs]))
    ncc_curve = {p: ncc_mat[i] for i, p in enumerate(pairs)}

    # --- per-pair peak shift m_ij and its σ_m (from NCC-vs-period curvature) ---
    rows = []   # (a,b,dt_days, m_deg, sigm_deg, peakP, sigP, peakNCC)
    for a, b in pairs:
        c = ncc_curve[(a, b)]; g = np.isfinite(c)
        if g.sum() < 8:
            continue
        i = int(np.nanargmax(c))
        lo, hi = max(i - 6, 0), min(i + 7, len(PGRID))
        xx, yy = PGRID[lo:hi], c[lo:hi]
        co = np.polyfit(xx, yy, 2)
        if co[0] >= 0:           # not concave -> no usable peak (flat/ill-defined)
            continue
        Ppk = -co[1] / (2 * co[0]); k = -2 * co[0]
        smooth = np.polyval(np.polyfit(PGRID[g], c[g], 4), PGRID)
        sigN = np.nanstd((c - smooth)[g])
        sigP = np.sqrt(2 * sigN / k) if k > 0 else np.nan
        dt = day[b] - day[a]                                   # days (signed)
        m = -((-360.0 / Ppk) - (-360.0 / P_ASSUMED)) * dt      # deg longitude shift
        sigm = DEG_PER_DAY_PER_P * abs(dt) * sigP              # propagate σ_P -> σ_m
        rows.append((a, b, dt, m, sigm, Ppk, sigP, np.nanmax(c)))

    A = np.array([r[2] for r in rows])          # Δt (days, signed)
    Mv = np.array([r[3] for r in rows])         # m (deg)
    S = np.array([r[4] for r in rows])          # σ_m (deg, curve-noise only)
    dt_yr = A / 365.25
    n = len(rows)

    def fit(idx, var):
        c = np.sum(A[idx] * Mv[idx] / var[idx]) / np.sum(A[idx]**2 / var[idx])
        return 360.0 / (c + 360.0 / P_ASSUMED), c

    def chi2_over_dof(var):
        _, c = fit(np.arange(n), var)
        return np.sum((Mv - c * A)**2 / var) / (n - 1)

    print("Per-pair interior-NCC peak shifts (the slope-fit data):", flush=True)
    print("  pair        Δt(yr)   m(deg)   σ_m(deg)  peakP(d)   σ_P     peakNCC", flush=True)
    for (a, b, dt, m, sigm, Ppk, sigP, pk) in rows:
        print(f"  {a}-{b}  {dt/365.25:6.1f}   {m:+7.3f}  {sigm:7.3f}   {Ppk:.4f}  {sigP:.4f}   {pk:.3f}", flush=True)

    # (1) naive: curve-noise weights only. Each pair gets ~equal slope-information, so
    #     the many short, offset-dominated pairs outvote the few long ones. χ²/dof >> 1
    #     is the data rejecting "single slope, no offsets" -> per-session offsets are real.
    var0 = S**2
    P_naive, _ = fit(np.arange(n), var0)
    print(f"\n(1) naive slope (curve-noise weights): P = {P_naive:.4f} d  "
          f"(χ²/dof = {chi2_over_dof(var0):.1f} -> model rejected; offsets real & large)", flush=True)

    # (2) ADOPTED: add a per-session offset-scatter floor σ_o (intrinsic scatter), fit so
    #     χ²/dof = 1. With a baseline-independent variance floor the weights become ∝ Δt²,
    #     so the long baselines (period signal >> offset) dominate and the short
    #     offset-dominated pairs are correctly down-weighted.
    so = 0.0
    if chi2_over_dof(var0) > 1.0:
        lo_s, hi_s = 0.0, 1.0
        for _ in range(60):
            so = 0.5 * (lo_s + hi_s)
            if chi2_over_dof(S**2 + so**2) > 1.0: lo_s = so
            else: hi_s = so
    var_adopt = S**2 + so**2
    P_joint, c_hat = fit(np.arange(n), var_adopt)
    sig_c = np.sqrt(1.0 / np.sum(A**2 / var_adopt))
    sig_P_formal = (P_joint**2 / 360.0) * sig_c
    rng = np.random.default_rng(0)
    Ps = np.array([fit(rng.integers(0, n, n), var_adopt)[0] for _ in range(5000)])
    bl, bh = np.percentile(Ps, [16, 84]); sig_P_boot = 0.5 * (bh - bl)
    sigq = max(sig_P_formal, sig_P_boot)
    print(f"(2) ADOPTED slope + offset-jitter (σ_o = {so:.3f}° per-session floor): "
          f"P = {P_joint:.4f} ± {sigq:.4f} d", flush=True)
    print(f"      formal σ_P = {sig_P_formal:.4f} | bootstrap σ_P = {sig_P_boot:.4f} "
          f"(68% CI [{bl:.4f}, {bh:.4f}])", flush=True)

    # (3) long-baseline only (Δt > 20 yr) — the pairs where period signal >> offsets
    li = np.where(dt_yr > 20)[0]
    P_long, _ = fit(li, var0)
    print(f"(3) long-baseline only (Δt>20yr, {len(li)} pairs): P = {P_long:.4f} d", flush=True)

    # session jackknife (drop all pairs touching a session, refit with adopted weights)
    print("\nSession jackknife (drop one session, refit):", flush=True)
    for s in YEARS:
        idx = np.array([i for i, r in enumerate(rows) if s not in (r[0], r[1])])
        if idx.size and np.sum(A[idx]**2 / var_adopt[idx]) > 0:
            print(f"  without {s} ({nlk[s]:3d} looks): P = {fit(idx, var_adopt)[0]:.4f} d", flush=True)

    # --- imagery validation: combined-stack sharpness at P_joint vs IAU vs 243.0216 ---
    print("\nImagery validation — combined σ-clip stack sharpness (higher = sharper):", flush=True)
    sval = {}
    for label, P in [("P_joint", P_joint), ("IAU 243.0185", 243.0185), ("adopted 243.0216", 243.0216)]:
        FS = []; MS = []
        for y in YEARS:
            px = px_for(P, day[y] - d_ref)
            FS.append(ndshift(F[y], (0, px), order=1, mode="constant", cval=0.0))
            MS.append(ndshift(M[y].astype(float), (0, px), order=1, mode="constant", cval=0) > 0.99)
        stk, cnt = coherent_stack(FS, MS)
        sval[label] = stack_sharpness(stk, cnt, min_count=2)
        print(f"  {label:18s} P={P:.4f}: sharpness = {sval[label]:.5f}", flush=True)
    print(f"  Δ(P_joint − IAU) = {sval['P_joint']-sval['IAU 243.0185']:+.5f}  "
          f"({100*(sval['P_joint']/sval['IAU 243.0185']-1):+.1f}%)", flush=True)

    print("\nliterature: Campbell 243.0212±0.0006 | Margot 243.0226±0.0013 | IAU 243.0185", flush=True)

    # --- figure: m_ij vs Δt with the slope fit; sharpness-vs-P coherence curve ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.6))
    dty = A / 365.25
    ax1.errorbar(dty, Mv, yerr=S, fmt="o", ms=5, capsize=2, alpha=0.8, label="per-pair shift m_ij ± σ_m")
    xx = np.linspace(min(dty.min(), 0), dty.max(), 50)
    ax1.plot(xx, c_hat * xx * 365.25, "b-", lw=1.5, label=f"weighted slope fit → P={P_joint:.4f}±{sigq:.4f}")
    # IAU reference slope (zero, since maps are projected at P_ASSUMED=IAU)
    ax1.axhline(0, color="k", ls=":", lw=0.8, alpha=0.6, label="IAU 243.0185 (zero slope)")
    ax1.set_xlabel("baseline Δt (yr, signed)"); ax1.set_ylabel("best longitude shift m_ij (°)")
    ax1.set_title("Joint slope fit (no per-session offsets): all 15 pairs"); ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

    ax2.plot(PGRID, sharp_curve, "-o", ms=3, lw=1, color="tab:purple", label="combined-stack sharpness")
    ax2.axvline(P_joint, color="b", lw=1.5, label=f"P_joint {P_joint:.4f}")
    for nm, P, col in [("IAU", 243.0185, "k"), ("Campbell", 243.0212, "tab:green"), ("Margot", 243.0226, "tab:orange")]:
        ax2.axvline(P, ls="--", lw=0.8, color=col, alpha=0.7, label=f"{nm} {P}")
    ax2.set_xlabel("trial period (d)"); ax2.set_ylabel("combined σ-clip stack sharpness")
    ax2.set_title("Imagery coherence vs period (all 6 sessions)"); ax2.legend(fontsize=7); ax2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, f"period_joint_coherence{TAG}.png"), dpi=120)
    print("\nwrote period_joint_coherence.png", flush=True)


if __name__ == "__main__":
    main()
