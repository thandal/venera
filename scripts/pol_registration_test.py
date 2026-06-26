"""Does OC or SC polarization register better ACROSS a viewing-geometry change?

Campbell PDS doc: OC (opposite-circular) carries the strong quasi-specular sub-radar
glint (steeply viewing-angle dependent, tracks the sub-radar point); SC (same-circular)
is the depolarized/diffuse channel (driven by wavelength-scale roughness, more
geometry-uniform) but weaker (lower SNR). So which co-registers better across a change
in sub-radar latitude is an empirical question — tested here.

Metric: best-alignment interior NCC between two session stacks built from ONE
polarization. Two pairs:
  - 2015<->2017  ΔSRP-lat ~17° (most geometry-diverse pair)
  - 2012<->2020  ΔSRP-lat ~0.3° (geometry-matched control)
SC has lower SNR, which lowers NCC for reasons unrelated to geometry, so we also report
the INTRA-session half-split NCC per polarization (same geometry, isolates SNR/repeat).
The viewing-angle sensitivity is the DROP from intra (matched geom) to cross on the
diverse pair. Reuses stack_sessions.robust_stack (σ-clip). Redirect stdout to a log.
"""
import os, sys, glob
import numpy as np
from scipy.ndimage import gaussian_filter, binary_erosion, shift as ndshift
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import stack_sessions as ss
from venera.coherence import ncc

CACHE = os.path.join(ROOT, "results", "look_cache")
HH, WW = 2000, 4000
ERODE = 20
LB0, LB1 = int((30 + 90) / 180 * HH), int((75 + 90) / 180 * HH)   # lat 30..75 band
SRP_LAT = {"1988": 1.08, "2001": -9.02, "2012": -2.63, "2015": 8.09, "2017": -9.29, "2020": -2.29}


def pol_files(y, pol):
    pat = f"venus_{pol}_{y}*.npz" if pol in ("ocp", "scp") else f"venus_*_{y}*.npz"
    return sorted(glob.glob(os.path.join(CACHE, pat)))


def flat(G, m, s=35):
    f = np.where(m, G, 0.0); w = gaussian_filter(m.astype(float), s)
    return (G - gaussian_filter(f, s) / np.maximum(w, 1e-6)) * m


def latband():
    b = np.zeros((HH, WW), bool); b[LB0:LB1] = True; return b


def best_align_ncc(Ga, Ma, Gb, Mb, LB):
    """Max interior NCC over a small longitude/lat shift search (registration quality)."""
    Fa = flat(Ga, Ma); Fb0 = flat(Gb, Mb); best = -1.0
    for dr in range(-3, 4):
        for dc in range(-5, 6):
            Fb = ndshift(Fb0, (dr, dc), order=1, mode="constant", cval=0.0)
            Mbs = ndshift(Mb.astype(float), (dr, dc), order=1, mode="constant", cval=0) > 0.99
            inn = binary_erosion(Ma & Mbs, iterations=ERODE) & LB
            if inn.sum() > 1500:
                v = ncc(Fa, Fb, inn, inn)
                if np.isfinite(v):
                    best = max(best, v)
    return best


def halfsplit_ncc(files, LB):
    """Intra-session: split looks in two, stack halves, NCC at zero shift (same geometry)."""
    rng = np.random.default_rng(0); idx = rng.permutation(len(files))
    A = [files[i] for i in idx[: len(files) // 2]]; B = [files[i] for i in idx[len(files) // 2:]]
    Ga, Ma, _ = ss.robust_stack(A); Gb, Mb, _ = ss.robust_stack(B)
    Fa = flat(Ga, Ma); Fb = flat(Gb, Mb)
    inn = binary_erosion(Ma & Mb, iterations=ERODE) & LB
    return ncc(Fa, Fb, inn, inn) if inn.sum() > 1500 else np.nan


def main():
    LB = latband()
    sessions = ["2012", "2015", "2017", "2020"]
    pairs = [("2015", "2017"), ("2012", "2020")]
    print("look counts per session/pol:", flush=True)
    for y in sessions:
        print(f"  {y}: ocp={len(pol_files(y,'ocp'))}  scp={len(pol_files(y,'scp'))}", flush=True)

    for pol in ["ocp", "scp"]:
        print(f"\n=== polarization: {pol.upper()} ===", flush=True)
        stk = {}
        for y in sessions:
            Gm, mask, rej = ss.robust_stack(pol_files(y, pol)); stk[y] = (Gm, mask)
        print("  intra-session half-split NCC (same geometry -> SNR/repeatability):", flush=True)
        intra = {}
        for y in sessions:
            intra[y] = halfsplit_ncc(pol_files(y, pol), LB)
            print(f"    {y}: {intra[y]:.3f}", flush=True)
        print("  cross-epoch best-alignment interior NCC:", flush=True)
        for a, b in pairs:
            dsrp = abs(SRP_LAT[a] - SRP_LAT[b])
            cross = best_align_ncc(stk[a][0], stk[a][1], stk[b][0], stk[b][1], LB)
            ref = np.sqrt(max(intra[a], 1e-6) * max(intra[b], 1e-6))   # geom-matched ceiling
            print(f"    {a}<->{b}  ΔSRPlat={dsrp:4.1f}°: cross-NCC={cross:.3f}  "
                  f"(intra-geomean={ref:.3f}, drop={ref-cross:+.3f})", flush=True)


if __name__ == "__main__":
    main()
