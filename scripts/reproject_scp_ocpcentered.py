"""Decoupled hybrid: project each SCP look with its simultaneous OCP twin's Doppler
centering, then build SCP session stacks for the period fit.

Rationale (measured): SCP registers cross-epoch better (diffuse = geometry-robust) but
its per-session centering floor σ_o is worse (0.105° vs OCP 0.060°) because the weak
per-look SCP signal makes the Doppler-centering (fo) fit noisy. The OCP twin (same
timestamp, same geometry, stronger signal) pins fo well. So: fit fo/do/fs on the OCP
look, project the SCP look with that fit -> OCP-quality centering + SC geometry-robust
features. project_file already accepts fit=(fo,do,fs), so the projection core is
unchanged.

Only the 5 MONOSTATIC sessions are re-projected here; 2012 is bistatic, where fo is the
pol-independent per-session DIURNAL model (not a per-look fit), so its existing SCP
stack already is the decoupled version (copied through).

Writes results/session_stacks/session_<year>_scpocp.npz, then run:
  .conda/bin/python scripts/period_joint_coherence.py scpocp
Usage: reproject_scp_ocpcentered.py [n_workers]   (default 12)
"""
import os, sys, glob, time, shutil
import numpy as np
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from venera import spice_setup
from venera.geometry import Spin, sub_radar_point, doppler_angle
from venera.data import parse_lbl
from venera.projection import project_file
import stack_sessions as ss

DATA = ("/home/than/code/venera/arecibo_radar/pds-geosciences.wustl.edu/venus/"
        "arcb_nrao-v-rtls_gbt-3-delaydoppler-v1/vrm_90xx/data/")
OUT_CACHE = os.path.join(ROOT, "results", "look_cache_scpocp")
STACKS = os.path.join(ROOT, "results", "session_stacks")
os.makedirs(OUT_CACHE, exist_ok=True)
H, W, DS = 4000, 8000, 2
MONO_YEARS = ["1988", "2001", "2015", "2017", "2020"]   # 2012 bistatic -> diurnal, copied


def year_of_img(f):
    return os.path.basename(f).split("_")[2][:4]


def _worker_init():
    spice_setup.furnsh_kernels()


def cache_one(scp_f):
    base = os.path.basename(scp_f)[:-4]                 # venus_scp_<date>
    out = os.path.join(OUT_CACHE, f"{base}.npz")
    if os.path.exists(out):
        return (base, "skip", 0.0)
    ocp_f = os.path.join(os.path.dirname(scp_f),
                         os.path.basename(scp_f).replace("venus_scp_", "venus_ocp_"))
    if not os.path.exists(ocp_f):
        return (base, "NO_OCP_TWIN", 0.0)
    t = time.time()
    try:
        spin = Spin()
        # 1) fit fo/do/fs on the OCP twin (the fit is independent of projection
        #    resolution, so project into a tiny throwaway grid to keep it cheap).
        Gd = np.zeros((H, W), np.float32); Gcd = np.zeros((H, W), np.int32)
        info_o = project_file(ocp_f, spin, Gd, Gcd, n_lon=40, n_lat=20)
        fit_ocp = info_o["fit"]                          # (fo, do, fs) from OCP
        # 2) project the SCP look at full resolution WITH the OCP centering.
        G = np.zeros((H, W), np.float32); Gc = np.zeros((H, W), np.int32)
        info = project_file(scp_f, spin, G, Gc, fit=fit_ocp)
        et = info["et_mid"]; lon, lat, _ = sub_radar_point(et, spin)
        rows = np.where(np.any(Gc > 0, axis=1))[0]; cols = np.where(np.any(Gc > 0, axis=0))[0]
        if rows.size == 0:
            return (base, "EMPTY", time.time() - t)
        r0, r1, c0, c1 = rows[0], rows[-1], cols[0], cols[-1]
        Gm = np.divide(G, Gc, out=np.zeros_like(G), where=Gc > 0)
        np.savez_compressed(
            out, gm=Gm[r0:r1+1:DS, c0:c1+1:DS].astype(np.float32),
            mask=(Gc[r0:r1+1:DS, c0:c1+1:DS] > 0),
            bbox=np.array([r0, r1, c0, c1]), ds=DS, hw=np.array([H, W]),
            et_mid=et, day=et / 86400.0, srp_lon=np.degrees(lon), srp_lat=np.degrees(lat),
            doppler_angle=np.degrees(doppler_angle(et, spin)), pol="scp",
            pointing=parse_lbl(scp_f).get("GEO_POINTING", "?"),
            valid_frac=info["valid_frac"], fo_ocp=float(fit_ocp[0]))
        return (base, f"fo_ocp={fit_ocp[0]:+.1f} valid={info['valid_frac']:.2f}", time.time() - t)
    except Exception as e:
        return (base, f"ERR {type(e).__name__}: {e}", time.time() - t)


def main():
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    files = sorted(f for f in glob.glob(os.path.join(DATA, "venus_scp_*.img"))
                   if year_of_img(f) in MONO_YEARS)
    todo = [f for f in files
            if not os.path.exists(os.path.join(OUT_CACHE, os.path.basename(f)[:-4] + ".npz"))]
    print(f"monostatic SCP looks: {len(files)}  todo={len(todo)}  workers={nproc}", flush=True)
    t0 = time.time(); errs = 0
    with Pool(nproc, initializer=_worker_init) as p:
        for i, (base, status, dt) in enumerate(p.imap_unordered(cache_one, todo), 1):
            if status.startswith("ERR") or status in ("EMPTY", "NO_OCP_TWIN"):
                errs += 1
            if i % 20 == 0 or status.startswith("ERR"):
                print(f"[{i}/{len(todo)}] {base} {status} ({(time.time()-t0)/60:.1f}m, errs={errs})", flush=True)
    print(f"reprojected: {len(todo)} done, {errs} errors, {(time.time()-t0)/60:.1f}m", flush=True)

    # build SCP-OCP-centered session stacks (monostatic), copy 2012 (diurnal already)
    print("\nbuilding session_<y>_scpocp.npz:", flush=True)
    for y in MONO_YEARS:
        fs = sorted(glob.glob(os.path.join(OUT_CACHE, f"venus_scp_{y}*.npz")))
        Gm, mask, rej = ss.robust_stack(fs)
        np.savez_compressed(os.path.join(STACKS, f"session_{y}_scpocp.npz"),
                            Gm=Gm, mask=mask, n=len(fs), pol="scpocp")
        print(f"  {y}: {len(fs):3d} looks  reject={rej:.3f}", flush=True)
    src = os.path.join(STACKS, "session_2012_scp.npz")
    shutil.copyfile(src, os.path.join(STACKS, "session_2012_scpocp.npz"))
    print("  2012: copied from session_2012_scp.npz (bistatic diurnal-fo, pol-independent)", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
