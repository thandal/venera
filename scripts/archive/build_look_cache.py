"""Project looks ONCE into the body frame and cache them (cropped, half-res), so
all downstream registration / rotation / stacking work is cheap and repeatable.

Each cached look stores the bounding-box crop of its body-frame map + count + the
geometry needed to re-register under a changed spin (et_mid, doppler angle, SRP).

Usage: build_look_cache.py <year> <pointing N|S|both> [max_n]
"""
import os, sys, glob, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from venera import spice_setup
from venera.geometry import Spin, sub_radar_point, doppler_angle
from venera.data import parse_lbl
from venera.projection import project_file
import cspyce as csp

DATA = ("/home/than/code/venera/arecibo_radar/pds-geosciences.wustl.edu/venus/"
        "arcb_nrao-v-rtls_gbt-3-delaydoppler-v1/vrm_90xx/data/")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "results", "look_cache")
os.makedirs(CACHE, exist_ok=True)
H, W = 4000, 8000
DS = 2                      # half-res storage


def pick(year, pointing):
    files = sorted(glob.glob(os.path.join(DATA, f"venus_scp_{year}*.img"))
                   + glob.glob(os.path.join(DATA, f"venus_ocp_{year}*.img")))
    out = []
    for f in files:
        try:
            lbl = parse_lbl(f)
        except Exception:
            continue
        if pointing == "both" or lbl.get("GEO_POINTING") == pointing:
            out.append(f)
    return out


def main():
    year = sys.argv[1]
    pointing = sys.argv[2] if len(sys.argv) > 2 else "both"
    max_n = int(sys.argv[3]) if len(sys.argv) > 3 else 10**9
    spice_setup.furnsh_kernels()
    spin = Spin()
    looks = pick(year, pointing)
    if len(looks) > max_n:
        looks = [looks[i] for i in np.linspace(0, len(looks)-1, max_n).round().astype(int)]
    print(f"{year} {pointing}: {len(looks)} looks", flush=True)
    for f in looks:
        base = os.path.basename(f)[:-4]
        out = os.path.join(CACHE, f"{base}.npz")
        if os.path.exists(out):
            continue
        t = time.time()
        G = np.zeros((H, W), np.float32); Gc = np.zeros((H, W), np.int32)
        info = project_file(f, spin, G, Gc)
        et = info["et_mid"]
        lon, lat, _ = sub_radar_point(et, spin)
        rows = np.where(np.any(Gc > 0, axis=1))[0]
        cols = np.where(np.any(Gc > 0, axis=0))[0]
        if rows.size == 0:
            print(f"  {base}: empty", flush=True); continue
        r0, r1, c0, c1 = rows[0], rows[-1], cols[0], cols[-1]
        Gm = np.divide(G, Gc, out=np.zeros_like(G), where=Gc > 0)
        np.savez_compressed(
            out,
            gm=Gm[r0:r1+1:DS, c0:c1+1:DS].astype(np.float32),
            mask=(Gc[r0:r1+1:DS, c0:c1+1:DS] > 0),
            bbox=np.array([r0, r1, c0, c1]), ds=DS, hw=np.array([H, W]),
            et_mid=et, day=et/86400.0,
            srp_lon=np.degrees(lon), srp_lat=np.degrees(lat),
            doppler_angle=np.degrees(doppler_angle(et, spin)),
            pol=("scp" if "_scp_" in base else "ocp"),
            pointing=parse_lbl(f).get("GEO_POINTING", "?"),
            valid_frac=info["valid_frac"])
        print(f"  {base}  valid={info['valid_frac']:.2f}  {time.time()-t:.0f}s", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
