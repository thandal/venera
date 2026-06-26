"""Real-data Venus rotation-period estimate over the full 1988-2020 baseline.

For each campaign year: pick N northern-pointing SCP looks from a single day,
project them (assumed nominal spin) into a per-epoch global stack, cache it. Then
register each epoch's stack against a reference and fit the period from the
longitude drift vs time (venera/rotation_fit).

Usage:
  .conda/bin/python scripts/run_rotation_estimate.py [N_looks] [--years 1988,2015,2017]
"""
import os, sys, time, glob
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from venera import spice_setup
from venera.geometry import Spin
from venera.data import parse_lbl
from venera.projection import project_file
from venera.registration import register_maps, offset_to_lonlat_deg
from venera.rotation_fit import fit_rotation_from_offsets, bootstrap_period, period_to_wdot
import cspyce as csp

DATA = ("/home/than/code/venera/arecibo_radar/pds-geosciences.wustl.edu/venus/"
        "arcb_nrao-v-rtls_gbt-3-delaydoppler-v1/vrm_90xx/data/")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
CACHE = os.path.join(RESULTS, "epoch_stacks")
os.makedirs(CACHE, exist_ok=True)

H, W = 4000, 8000          # global equirectangular grid
POL = "both"               # SCP + OCP (incidence-normalized -> comparable)
POINTING = "N"
ALL_YEARS = ["1988", "2001", "2012", "2015", "2017", "2020"]
REF_YEAR = "2015"


def log(msg):
    print(msg, flush=True)


def pick_looks(year, max_n):
    """Up to ``max_n`` N-pointing looks spanning the year (better speckle averaging).

    Draws from BOTH polarizations (incidence normalization makes SCP/OCP
    comparable), giving more independent speckle realizations per epoch. Looks are
    projected to body-fixed coordinates individually, so spanning days is fine.
    """
    files = sorted(glob.glob(os.path.join(DATA, f"venus_scp_{year}*.img"))
                   + glob.glob(os.path.join(DATA, f"venus_ocp_{year}*.img")))
    keep = []
    for f in files:
        try:
            lbl = parse_lbl(f)
        except Exception:
            continue
        if lbl.get("GEO_POINTING") == POINTING:
            keep.append(f)
    if not keep:
        return []
    if len(keep) <= max_n:
        return keep
    # evenly subsample across the year's available looks
    idx = np.linspace(0, len(keep) - 1, max_n).round().astype(int)
    return [keep[i] for i in sorted(set(idx))]


def build_epoch_stack(year, max_n, spin):
    """Project N looks for a year into a stack; cache (Gm, mask, mean_day)."""
    cache_f = os.path.join(CACHE, f"stack_both_{year}_N{max_n}_gfs.npz")  # gfs = geometric freq_scale + curve-fit offset
    if os.path.exists(cache_f):
        d = np.load(cache_f)
        log(f"  [{year}] cached  (n={int(d['n'])}, day={float(d['mean_day']):.0f})")
        return d["Gm"], d["mask"], float(d["mean_day"]), int(d["n"])
    looks = pick_looks(year, max_n)
    if not looks:
        log(f"  [{year}] no {POINTING}-pointing {POL} looks found")
        return None
    G = np.zeros((H, W), np.float32)
    Gc = np.zeros((H, W), np.int32)
    days = []
    for f in looks:
        t0 = time.time()
        info = project_file(f, spin, G, Gc)
        days.append(info["et_mid"] / 86400.0)
        fo, do, fs = info["fit"]
        log(f"  [{year}] {os.path.basename(f)}  cen={info['centering']:8s} "
            f"fo={fo:+.2f} do={do} fs={fs:.3f}  valid={info['valid_frac']:.2f}  "
            f"{time.time()-t0:.0f}s")
    Gm = np.divide(G, Gc, out=np.zeros_like(G), where=Gc > 0)
    mask = Gc > 0
    mean_day = float(np.mean(days))
    np.savez_compressed(cache_f, Gm=Gm, mask=mask, mean_day=mean_day, n=len(looks))
    return Gm, mask, mean_day, len(looks)


def main():
    spice_setup.furnsh_kernels()
    max_n = 5
    years = ALL_YEARS
    args = sys.argv[1:]
    if args and args[0].isdigit():
        max_n = int(args[0])
    if "--years" in args:
        years = args[args.index("--years") + 1].split(",")

    spin = Spin()  # assumed nominal IAU spin
    log(f"=== Building epoch stacks (N={max_n}, pol={POL}, pointing={POINTING}) ===")
    epochs = {}
    for y in years:
        res = build_epoch_stack(y, max_n, spin)
        if res is not None:
            Gm, mask, mean_day, n = res
            epochs[y] = dict(Gm=Gm, mask=mask, day=mean_day, n=n)

    ref = REF_YEAR if REF_YEAR in epochs else sorted(epochs)[len(epochs) // 2]
    log(f"\n=== Registering stacks against reference {ref} ===")
    rows = []
    for y in sorted(epochs):
        e = epochs[y]
        if y == ref:
            drow, dcol, sig = 0.0, 0.0, np.inf
        else:
            # band-pass at moon-radar scales: smooth ~0.3deg, trend ~2.5deg
            # (grid is 360deg/W per px -> 0.3deg ~ 6.7px, 2.5deg ~ 55px).
            # max_shift kept small: after correct cspyce geometry the cross-epoch
            # residual is a few degrees at most; wider windows admit spurious peaks.
            drow, dcol, sig = register_maps(epochs[ref]["Gm"], e["Gm"],
                                            valid_a=epochs[ref]["mask"],
                                            valid_b=e["mask"], max_shift=70,
                                            smooth_px=7.0, trend_px=55.0)
        dlat, dlon = offset_to_lonlat_deg(-drow, -dcol, (H, W))  # epoch relative to ref
        rows.append((y, e["day"], dlon, dlat, sig, e["n"]))
        log(f"  [{y}] day={e['day']:8.0f}  Δlon={dlon:+.4f}  Δlat={dlat:+.4f} deg  "
            f"sig={sig:.2f}  (n={e['n']})")

    # Fit (drop the reference's infinite weight; weight others by significance^2)
    days = np.array([r[1] for r in rows])
    dlon = np.array([r[2] for r in rows])
    sig = np.array([r[4] for r in rows])
    weights = np.where(np.isfinite(sig), sig, np.nanmax(sig[np.isfinite(sig)])) ** 2
    assumed_wdot = period_to_wdot(spin.period_days)
    fit = fit_rotation_from_offsets(days, dlon, assumed_wdot, weights=weights)
    boot = bootstrap_period(days, dlon, assumed_wdot, weights=weights)

    log("\n=== RESULT ===")
    log(f"assumed (IAU) period : {spin.period_days:.5f} d")
    log(f"venera fit period    : {fit['period_days']:.5f} ± {fit['sigma_period_days']:.5f} d")
    log(f"bootstrap period     : {boot['period_median']:.5f} "
        f"[{boot['period_p16']:.5f}, {boot['period_p84']:.5f}] d")
    log(f"rms longitude resid  : {fit['rms_resid_deg']*1000:.1f} mdeg over {fit['n']} epochs")
    log(f"baseline             : {(days.max()-days.min())/365.25:.1f} yr")
    log("literature: Campbell 2019 243.0212±0.0006 | Margot 2020 243.0226±0.0013")
    for (y, d, dl, dla, s, n), res in zip(rows, fit["residuals_deg"]):
        log(f"  {y}: resid={res*1000:+.1f} mdeg")


if __name__ == "__main__":
    main()
