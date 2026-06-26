"""Project a single .img into a global map and save a PNG (real-data validation).

Usage: .conda/bin/python scripts/project_one.py <img_basename_or_path> [period_days]
"""
import os, sys, time
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from venera import spice_setup
from venera.geometry import Spin, sub_radar_point
from venera.projection import project_file
import cspyce as csp

DATA = ("/home/than/code/venera/arecibo_radar/pds-geosciences.wustl.edu/venus/"
        "arcb_nrao-v-rtls_gbt-3-delaydoppler-v1/vrm_90xx/data/")
RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS, exist_ok=True)


def imwrite(path, arr):
    a = arr.astype(np.float64)
    a = a - np.nanmin(a)
    mx = np.nanmax(a)
    if mx > 0:
        a = a / mx * 255
    Image.fromarray(a.astype(np.uint8)).save(path)


def main():
    spice_setup.furnsh_kernels()
    arg = sys.argv[1] if len(sys.argv) > 1 else "venus_scp_20170326_165951.img"
    img_path = arg if os.path.isabs(arg) else os.path.join(DATA, arg)
    period = float(sys.argv[2]) if len(sys.argv) > 2 else Spin().period_days
    spin = Spin(period_days=period)

    H, W = 4000, 8000
    G = np.zeros((H, W), np.float32)
    Gc = np.zeros((H, W), np.int32)

    t0 = time.time()
    info = project_file(img_path, spin, G, Gc)
    dt = time.time() - t0

    et = info["et_mid"]
    slon, slat, _ = sub_radar_point(et, spin)
    print(f"file        : {os.path.basename(img_path)}")
    print(f"pointing    : {info['pointing']}   valid_frac: {info['valid_frac']:.3f}")
    print(f"fit (fo,do,fs): {info['fit']}")
    print(f"SRP lon,lat : {np.degrees(slon):.2f}, {np.degrees(slat):.2f} deg")
    print(f"project time: {dt:.1f}s")

    Gm = np.divide(G, Gc, out=np.zeros_like(G), where=Gc > 0)
    # report where the data landed (sanity for hemisphere assignment)
    rows, cols = np.where(Gc > 0)
    if len(rows):
        lat_lo = (rows.min() / H) * 180 - 90
        lat_hi = (rows.max() / H) * 180 - 90
        lon_lo = (cols.min() / W) * 360 - 180
        lon_hi = (cols.max() / W) * 360 - 180
        print(f"coverage    : lat [{lat_lo:.1f},{lat_hi:.1f}]  lon [{lon_lo:.1f},{lon_hi:.1f}] deg")

    out = os.path.join(RESULTS, os.path.basename(img_path)[:-4] + f"_map_P{period:.4f}.png")
    imwrite(out, np.flipud(Gm))
    print(f"wrote       : {out}")


if __name__ == "__main__":
    main()
