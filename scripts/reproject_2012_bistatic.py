"""Controlled before/after for the 2012 blur: project each 2012 look TWICE — once
monostatic (rx=Arecibo, 'before') and once bistatic (rx=GBT, 'after') — stack each,
and compare sharpness + a visual. Also records per-look centering (fo, do, fs)
scatter, which directly causes longitude/delay smear, to see if THAT (not the
bistatic d_hat) is the blur driver.

Writes results/figures/blur_2012_beforeafter.png and prints sharpness + fit scatter.
"""
import os, sys, glob
import numpy as np
from multiprocessing import Pool
from scipy.ndimage import gaussian_filter, binary_erosion
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA = ("/home/than/code/venera/arecibo_radar/pds-geosciences.wustl.edu/venus/"
        "arcb_nrao-v-rtls_gbt-3-delaydoppler-v1/vrm_90xx/data/")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "results", "figures")
H, W, DS = 4000, 8000, 2
HH, WW = H // DS, W // DS
# common interior sharpness box (matches blur_2012.py)
LAT0, LAT1, LON0, LON1 = 35, 72, -90, 45
R0 = int((LAT0+90)/180*HH); R1 = int((LAT1+90)/180*HH)
C0 = int((LON0+180)/360*WW); C1 = int((LON1+180)/360*WW)
# display crop
DLAT0, DLAT1, DLON0, DLON1 = 45, 80, -60, 45


def pick(n=60):
    fs = sorted(glob.glob(os.path.join(DATA, "venus_scp_2012*.img"))
                + glob.glob(os.path.join(DATA, "venus_ocp_2012*.img")))
    return [fs[i] for i in np.linspace(0, len(fs)-1, min(n, len(fs))).astype(int)]


def _init():
    from venera import spice_setup
    spice_setup.furnsh_kernels()


def proj(task):
    img, mode = task
    from venera.geometry import Spin, doppler_basis
    from venera.projection import (project_image_to_map, fit_delay_doppler_curve,
                                   predicted_freq_scale)
    from venera.geometry import CALIBRATED_FREQ_SCALE_K
    from venera import data as vdata
    import cspyce as csp
    a, lbl, cal = vdata.preprocess(img)
    et0 = csp.str2et(lbl["START_TIME"]); et1 = csp.str2et(lbl["STOP_TIME"])
    et_mid = 0.5*(et0+et1)
    rx = "ARECIBO" if mode == "mono" else lbl.get("RX_STATION", "ARECIBO")
    o_hat, d_hat = doppler_basis(et_mid, Spin(), "ARECIBO", "CN+S",
                                 dt=max(1.0, et1-et0), rx_station=rx)
    fs_g = predicted_freq_scale(et_mid, Spin(), lbl["GEO_BAUD"],
                                lbl.get("GEO_CODE_LENGTH", 1),
                                K=CALIBRATED_FREQ_SCALE_K, observer="ARECIBO")
    _, fit = fit_delay_doppler_curve(a, lbl["GEO_BAUD"], fo_range=range(-60, 61, 2),
                                     fs_values=[fs_g])
    G = np.zeros((H, W), np.float32); Gc = np.zeros((H, W), np.int32)
    project_image_to_map(a, o_hat, d_hat, G, Gc, baud=lbl["GEO_BAUD"],
                         pointing=lbl["GEO_POINTING"], freq_offset=fit[0],
                         delay_offset=fit[1], freq_scale=fit[2])
    rows = np.where(np.any(Gc > 0, 1))[0]; cols = np.where(np.any(Gc > 0, 0))[0]
    r0, r1, c0, c1 = rows[0], rows[-1], cols[0], cols[-1]
    Gm = np.divide(G, Gc, out=np.zeros_like(G), where=Gc > 0)
    return (mode, r0, c0, Gm[r0:r1+1:DS, c0:c1+1:DS].astype(np.float32),
            Gc[r0:r1+1:DS, c0:c1+1:DS] > 0, fit)


def stack(parts):
    Gs = np.zeros((HH, WW)); Gc = np.zeros((HH, WW), int)
    for _, r0, c0, gm, m, _ in parts:
        hr, hc = r0//DS, c0//DS; h, w = gm.shape
        if hr+h > HH or hc+w > WW:
            h = min(h, HH-hr); w = min(w, WW-hc); gm = gm[:h, :w]; m = m[:h, :w]
        Gs[hr:hr+h, hc:hc+w][m] += gm[m]; Gc[hr:hr+h, hc:hc+w][m] += 1
    return np.divide(Gs, Gc, out=np.zeros_like(Gs), where=Gc > 0).astype(np.float32), Gc > 0


def sharp(G, m):
    box = m[R0:R1, C0:C1]; inn = binary_erosion(box, iterations=10)
    g = G[R0:R1, C0:C1].astype(float)
    hp = g - gaussian_filter(np.where(box, g, 0.0), 4)
    return float(np.mean(hp[inn]**2)) if inn.sum() > 500 else np.nan


def show(ax, G, m, title):
    r0 = int((DLAT0+90)/180*HH); r1 = int((DLAT1+90)/180*HH)
    c0 = int((DLON0+180)/360*WW); c1 = int((DLON1+180)/360*WW)
    g = G[r0:r1, c0:c1].astype(float); cm = m[r0:r1, c0:c1]
    hp = g - gaussian_filter(np.where(cm, g, 0.0), 30); hp[~cm] = np.nan
    v = hp[np.isfinite(hp)]; lo, hi = np.percentile(v, 2), np.percentile(v, 98)
    ax.imshow(hp, origin="lower", extent=[DLON0, DLON1, DLAT0, DLAT1], cmap="gray",
              vmin=lo, vmax=hi, aspect="auto")
    ax.set_title(title, fontsize=11); ax.set_xlabel("lon (°E)"); ax.set_ylabel("lat (°N)")


def main():
    looks = pick(60)
    tasks = [(f, m) for m in ("mono", "bist") for f in looks]
    res = {"mono": [], "bist": []}
    with Pool(12, initializer=_init) as p:
        for out in p.imap_unordered(proj, tasks):
            res[out[0]].append(out); print(".", end="", flush=True)
    print()
    figs = {}
    for m in ("mono", "bist"):
        G, mask = stack(res[m]); figs[m] = (G, mask)
        fits = np.array([r[5] for r in res[m]])
        print(f"{m}: sharpness={sharp(G, mask):.5f}  "
              f"fo std={fits[:,0].std():.1f}  do std={fits[:,1].std():.2f}  "
              f"fs std={fits[:,2].std():.4f}  (n={len(res[m])})", flush=True)
    fig, axs = plt.subplots(1, 2, figsize=(15, 5.5))
    show(axs[0], *figs["mono"], f"2012 MONOSTATIC (before)  sharp={sharp(*figs['mono']):.4f}")
    show(axs[1], *figs["bist"], f"2012 BISTATIC=GBT (after)  sharp={sharp(*figs['bist']):.4f}")
    fig.suptitle("2012 reproject: monostatic vs bistatic(GBT) receiver geometry "
                 f"(60 looks each)", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "blur_2012_beforeafter.png"), dpi=120)
    print("wrote blur_2012_beforeafter.png")


if __name__ == "__main__":
    main()
