"""Stack ALL cached looks into one projected image per observing session (year),
and render them for visual cross-session registration assessment.

Reads every per-look cache in results/look_cache/ (built by project_all_looks.py),
groups by session year, and stacks each session on a common equirectangular grid
(half-res: 2000x4000, row=lat, col=lon). Writes:
  - results/session_stacks/session_<year>.npz   (Gm, mask, metadata)
  - results/figures/session_<year>.png          (per-session)
  - results/figures/sessions_contact_sheet.png  (2x3 all sessions)
  - results/figures/sessions_combined.png        (deep stack, all years)

Usage: .conda/bin/python scripts/stack_sessions.py
"""
import os, sys, glob
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "results", "look_cache")
STACKS = os.path.join(ROOT, "results", "session_stacks")
FIG = os.path.join(ROOT, "results", "figures")
os.makedirs(STACKS, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

H, W, DS = 4000, 8000, 2          # full-map dims and cache downsample
HH, WW = H // DS, W // DS         # common stack grid: 2000 x 4000
YEARS = ["1988", "2001", "2012", "2015", "2017", "2020"]
DATES = {"1988": "Jun 1988", "2001": "Mar 2001", "2012": "May 2012",
         "2015": "Aug 2015", "2017": "Mar 2017", "2020": "May 2020"}

# display crop (covered band): lat -10..82, lon -130..75
LAT0, LAT1, LON0, LON1 = -10, 82, -130, 75
rr0, rr1 = int((LAT0 + 90) / 180 * HH), int((LAT1 + 90) / 180 * HH)
cc0, cc1 = int((LON0 + 180) / 360 * WW), int((LON1 + 180) / 360 * WW)


def year_of(path):
    return os.path.basename(path).split("_")[2][:4]


CLIP_SIGMA = 3.0   # per-pixel sigma-clip for robust combination across looks


def _place(d):
    """Half-res canvas placement of a cached look crop: (hr, h, hc, w, gm, mask)."""
    r0, r1, c0, c1 = d["bbox"]
    hr, hc = r0 // DS, c0 // DS
    gm, m = d["gm"], d["mask"]
    h, w = gm.shape
    if hr + h > HH or hc + w > WW:            # clip to canvas (paranoia)
        h = min(h, HH - hr); w = min(w, WW - hc)
        gm, m = gm[:h, :w], m[:h, :w]
    return hr, h, hc, w, gm, m


def _accumulate(files, mu=None, sd=None, clip=CLIP_SIGMA, want_sq=False):
    """One stacking pass. Pass 1 (mu None): accumulate sum/count[/sumsq]. Pass 2
    (mu,sd given): accumulate only values within ``clip``·sd of mu (sigma-clip)."""
    S = np.zeros((HH, WW)); C = np.zeros((HH, WW), np.int64)
    S2 = np.zeros((HH, WW)) if want_sq else None
    for f in files:
        d = np.load(f, allow_pickle=True)
        hr, h, hc, w, gm, m = _place(d)
        if mu is None:
            keep = m
        else:
            keep = m & (np.abs(gm - mu[hr:hr+h, hc:hc+w])
                        <= clip * sd[hr:hr+h, hc:hc+w] + 1e-6)  # eps keeps single-look px
        S[hr:hr+h, hc:hc+w][keep] += gm[keep]
        C[hr:hr+h, hc:hc+w][keep] += 1
        if want_sq:
            S2[hr:hr+h, hc:hc+w][keep] += gm[keep] ** 2
    return (S, C, S2) if want_sq else (S, C)


def robust_stack(files, clip_sigma=CLIP_SIGMA):
    """Sigma-clipped per-pixel mean across looks (two-pass). Returns
    (Gm, mask, reject_frac). clip_sigma=None -> plain mean."""
    S, C, S2 = _accumulate(files, want_sq=True)
    mu = np.divide(S, C, out=np.zeros_like(S), where=C > 0)
    var = np.divide(S2, C, out=np.zeros_like(S2), where=C > 0) - mu ** 2
    sd = np.sqrt(np.clip(var, 0, None))
    if clip_sigma is None:
        return mu.astype(np.float32), C > 0, 0.0
    S3, C3 = _accumulate(files, mu=mu, sd=sd, clip=clip_sigma)
    Gm = np.divide(S3, C3, out=np.zeros_like(S3), where=C3 > 0).astype(np.float32)
    rej = 1.0 - C3.sum() / max(int(C.sum()), 1)
    return Gm, C3 > 0, float(rej)


def stack_year(files):
    """Robust (sigma-clipped) per-pixel stack of one session's looks + metadata."""
    Gm, mask, rej = robust_stack(files)
    srp_lats, pols, pts = [], [], []
    for f in files:
        d = np.load(f, allow_pickle=True)
        srp_lats.append(float(d["srp_lat"])); pols.append(str(d["pol"]))
        pts.append(str(d["pointing"]))
    srp_lats = np.array(srp_lats)
    meta = dict(n=len(files), srp_lat_mean=float(srp_lats.mean()),
                srp_lat_min=float(srp_lats.min()), srp_lat_max=float(srp_lats.max()),
                n_scp=pols.count("scp"), n_ocp=pols.count("ocp"),
                n_N=pts.count("N"), n_S=pts.count("S"), reject_frac=round(rej, 4))
    return Gm, mask, meta


def flatten(crop, cm, sigma=35):
    """Divide out the large-scale brightness envelope (quasi-specular point +
    limb-darkening) so the surface features are visible. Masked smooth background."""
    filled = np.where(cm, crop, 0.0)
    wsm = gaussian_filter(cm.astype(float), sigma)
    bg = gaussian_filter(filled, sigma) / np.maximum(wsm, 1e-6)
    flat = crop - bg
    flat[~cm] = np.nan
    return flat


def panel(ax, Gm, mask, title):
    crop = Gm[rr0:rr1, cc0:cc1].astype(float)
    cm = mask[rr0:rr1, cc0:cc1]
    flat = flatten(crop, cm)
    if np.isfinite(flat).any():
        vlo, vhi = np.nanpercentile(flat, 2), np.nanpercentile(flat, 98)
    else:
        vlo, vhi = 0, 1
    ax.imshow(flat, origin="lower", extent=[LON0, LON1, LAT0, LAT1], cmap="gray",
              vmin=vlo, vmax=vhi, aspect="auto")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("longitude (°E)"); ax.set_ylabel("latitude (°N)")
    ax.grid(alpha=0.25, color="c", lw=0.4)


def main():
    files = sorted(glob.glob(os.path.join(CACHE, "*.npz")))
    by_year = {y: [f for f in files if year_of(f) == y] for y in YEARS}
    print(f"cached looks: {len(files)}   (per-pixel sigma-clipped mean, "
          f"clip={CLIP_SIGMA}σ)")
    total_n = 0
    stacks = {}
    for y in YEARS:
        fs = by_year[y]
        if not fs:
            print(f"  {y}: no looks cached, skipping"); continue
        Gm, mask, meta = stack_year(fs)
        stacks[y] = (Gm, mask, meta)
        np.savez_compressed(os.path.join(STACKS, f"session_{y}.npz"),
                            Gm=Gm, mask=mask, year=y, **meta)
        total_n += meta["n"]
        print(f"  {y}: {meta['n']:3d} looks  (SCP {meta['n_scp']}, OCP {meta['n_ocp']}; "
              f"N {meta['n_N']}, S {meta['n_S']})  "
              f"SRP_lat {meta['srp_lat_mean']:+.1f}° "
              f"[{meta['srp_lat_min']:+.1f},{meta['srp_lat_max']:+.1f}]  "
              f"clip-rej {meta['reject_frac']*100:.1f}%")
        fig, ax = plt.subplots(figsize=(9, 5))
        panel(ax, Gm, mask, f"Venus — {DATES[y]}  ({meta['n']} looks, "
                            f"SRP_lat {meta['srp_lat_mean']:+.1f}°)")
        fig.tight_layout(); fig.savefig(os.path.join(FIG, f"session_{y}.png"), dpi=120)
        plt.close(fig)

    # deep combined stack: robust (sigma-clipped) per-pixel mean over ALL looks
    Gcomb, mcomb, rej = robust_stack(files)
    np.savez_compressed(os.path.join(STACKS, "combined_all.npz"),
                        Gm=Gcomb, mask=mcomb, n=total_n)
    print(f"  combined: {total_n} looks, clip-rej {rej*100:.1f}%")
    fig, ax = plt.subplots(figsize=(11, 6))
    panel(ax, Gcomb, mcomb,
          f"Venus — all looks combined ({total_n} looks, σ-clipped)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "sessions_combined.png"), dpi=130)
    plt.close(fig)

    # contact sheet
    fig, axs = plt.subplots(2, 3, figsize=(16, 9))
    for ax, y in zip(axs.ravel(), YEARS):
        if y in stacks:
            Gm, mask, meta = stacks[y]
            panel(ax, Gm, mask, f"{DATES[y]} ({meta['n']} looks, "
                               f"SRP {meta['srp_lat_mean']:+.0f}°)")
    fig.suptitle("Venus by observing session — ALL Arecibo looks "
                 "(equirectangular body frame)", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "sessions_contact_sheet.png"), dpi=110)
    plt.close(fig)
    print(f"\nwrote per-session PNGs, sessions_contact_sheet.png, sessions_combined.png "
          f"in {FIG}")
    print(f"stacks saved in {STACKS}")


if __name__ == "__main__":
    main()
