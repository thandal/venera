"""Decoupled hybrid for the bistatic 2012 session: fit the per-session DIURNAL
Doppler-centering model fo(hour) from the OCP looks ONLY (stronger signal -> cleaner
model), then reproject the 2012 SCP looks with that OCP-derived model. Makes the
scpocp hybrid uniform across all six sessions (the 5 monostatic ones use the per-look
OCP twin fo; 2012 uses the OCP-derived diurnal model, since 2012's fo is a per-session
function of hour, not a per-look quantity).

Rebuilds results/session_stacks/session_2012_scpocp.npz, then re-run:
  rm -f results/session_stacks/period_joint_cache_scpocp.npz
  .conda/bin/python scripts/period_joint_coherence.py scpocp
Usage: reproject_2012_scpocp.py [n_workers]   (default 12)
"""
import os, sys, glob, time
import numpy as np
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from venera import spice_setup
from venera.geometry import Spin, sub_radar_point, doppler_angle
from venera.data import parse_lbl
from venera.projection import project_file, fit_diurnal_fo
import stack_sessions as ss

DATA = ("/home/than/code/venera/arecibo_radar/pds-geosciences.wustl.edu/venus/"
        "arcb_nrao-v-rtls_gbt-3-delaydoppler-v1/vrm_90xx/data/")
OUT_CACHE = os.path.join(ROOT, "results", "look_cache_scpocp")
STACKS = os.path.join(ROOT, "results", "session_stacks")
os.makedirs(OUT_CACHE, exist_ok=True)
H, W, DS = 4000, 8000, 2


def hourof(p):
    b = os.path.basename(p).split("_")[3]
    return int(b[:2]) + int(b[2:4]) / 60.0 + int(b[4:6]) / 3600.0


def _init():
    spice_setup.furnsh_kernels()


def ocp_fo(img):
    """Per-look fo from an OCP look (small throwaway projection: fit is res-independent)."""
    Gd = np.zeros((H, W), np.float32); Gcd = np.zeros((H, W), np.int32)
    return project_file(img, Spin(), Gd, Gcd, n_lon=40, n_lat=20)["fit"][0]


def reproj_scp(args):
    """Reproject one 2012 SCP look with fo_center=foc (forced), cache it."""
    img, foc = args
    base = os.path.basename(img)[:-4]; out = os.path.join(OUT_CACHE, base + ".npz")
    try:
        G = np.zeros((H, W), np.float32); Gc = np.zeros((H, W), np.int32)
        info = project_file(img, Spin(), G, Gc, fo_center=float(foc), fo_window=0)
        et = info["et_mid"]; lon, lat, _ = sub_radar_point(et, Spin())
        rows = np.where(np.any(Gc > 0, 1))[0]; cols = np.where(np.any(Gc > 0, 0))[0]
        if rows.size == 0:
            return (base, "empty")
        r0, r1, c0, c1 = rows[0], rows[-1], cols[0], cols[-1]
        Gm = np.divide(G, Gc, out=np.zeros_like(G), where=Gc > 0)
        np.savez_compressed(
            out, gm=Gm[r0:r1+1:DS, c0:c1+1:DS].astype(np.float32),
            mask=(Gc[r0:r1+1:DS, c0:c1+1:DS] > 0), bbox=np.array([r0, r1, c0, c1]), ds=DS,
            hw=np.array([H, W]), et_mid=et, day=et / 86400.0,
            srp_lon=np.degrees(lon), srp_lat=np.degrees(lat),
            doppler_angle=np.degrees(doppler_angle(et, Spin())), pol="scp",
            pointing=info["pointing"], valid_frac=info["valid_frac"], fo_diurnal=float(foc))
        return (base, "ok")
    except Exception as e:
        return (base, f"ERR {e}")


def main():
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    spice_setup.furnsh_kernels()
    ocp = sorted(glob.glob(DATA + "venus_ocp_2012*.img"))
    scp = sorted(glob.glob(DATA + "venus_scp_2012*.img"))
    print(f"2012: {len(ocp)} OCP, {len(scp)} SCP looks; workers={nproc}", flush=True)

    # fit the diurnal model from an OCP-only sample
    samp = [ocp[i] for i in np.linspace(0, len(ocp) - 1, min(70, len(ocp))).astype(int)]
    t0 = time.time()
    with Pool(nproc, initializer=_init) as p:
        fos = p.map(ocp_fo, samp)
    hrs = np.array([hourof(f) for f in samp]); fos = np.array(fos)
    model, (a, b, h0) = fit_diurnal_fo(hrs, fos)
    print(f"OCP-derived diurnal model: fo = {a:+.2f} {b:+.2f}*(hour-{h0:.2f}) cols  "
          f"({(time.time()-t0)/60:.1f}m for {len(samp)}-look fit)", flush=True)
    print("  (current production model from mixed OCP+SCP: fo = +5.8 -8.5*(hour-17.38))", flush=True)

    # reproject all 2012 SCP looks with the OCP-derived diurnal model
    tasks = [(f, model(hourof(f))) for f in scp]; t0 = time.time(); done = err = 0
    with Pool(nproc, initializer=_init) as p:
        for base, st in p.imap_unordered(reproj_scp, tasks):
            done += 1
            if not st.startswith("ok"):
                err += 1
            if done % 40 == 0:
                print(f"  reprojected {done}/{len(tasks)} ({(time.time()-t0)/60:.1f}m, err={err})", flush=True)
    print(f"reprojected {done} 2012 SCP looks, {err} errors, {(time.time()-t0)/60:.1f}m", flush=True)

    fs = sorted(glob.glob(os.path.join(OUT_CACHE, "venus_scp_2012*.npz")))
    Gm, mask, rej = ss.robust_stack(fs)
    np.savez_compressed(os.path.join(STACKS, "session_2012_scpocp.npz"),
                        Gm=Gm, mask=mask, n=len(fs), pol="scpocp")
    print(f"rebuilt session_2012_scpocp.npz: {len(fs)} looks  reject={rej:.3f}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
