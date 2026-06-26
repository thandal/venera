"""Intra-session registration check (fast): do individual looks within a session land
at the same place, or scatter (-> the deep stack blurs)? For several sessions,
register a sample of individual cached looks against that session's stack, within each
look's own bbox region, and report the per-look offset scatter (the blur scale).

Usage: .conda/bin/python scripts/intra_session_check.py
"""
import os, sys, glob
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from venera.registration import register_maps, offset_to_lonlat_deg

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STACKS = os.path.join(ROOT, "results", "session_stacks")
CACHE = os.path.join(ROOT, "results", "look_cache")
HH, WW, DS = 2000, 4000, 2


def main():
    for year in ["1988", "2012", "2015", "2017", "2020"]:
        sd = np.load(os.path.join(STACKS, f"session_{year}.npz"))
        SG, SM = sd["Gm"], sd["mask"]
        looks = sorted(glob.glob(os.path.join(CACHE, f"venus_*_{year}*.npz")))
        idx = np.linspace(0, len(looks) - 1, min(24, len(looks))).astype(int)
        dlons, dlats, sigs, pts = [], [], [], []
        for i in idx:
            d = np.load(looks[i], allow_pickle=True)
            gm, m = d["gm"], d["mask"]; r0, r1, c0, c1 = d["bbox"]
            hr, hc = r0 // DS, c0 // DS; h, w = gm.shape
            if hr + h > HH or hc + w > WW:
                h = min(h, HH - hr); w = min(w, WW - hc); gm = gm[:h, :w]; m = m[:h, :w]
            SC = SG[hr:hr+h, hc:hc+w]; SCm = SM[hr:hr+h, hc:hc+w]   # same region of stack
            dr, dc, sig = register_maps(SC, gm, valid_a=SCm, valid_b=m,
                                        max_shift=30, smooth_px=7.0, trend_px=55.0)
            dlat, dlon = offset_to_lonlat_deg(dr, dc, (HH, WW))
            dlons.append(dlon); dlats.append(dlat); sigs.append(sig)
            pts.append(str(d["pointing"]))
        dlons = np.array(dlons); dlats = np.array(dlats); sigs = np.array(sigs)
        pts = np.array(pts); ok = sigs > 1.05
        print(f"{year}: {len(idx)} looks (N={int((pts=='N').sum())},"
              f"S={int((pts=='S').sum())}), {int(ok.sum())} locked", flush=True)
        if ok.sum() >= 3:
            print(f"   look->stack offset scatter:  dlon ±{dlons[ok].std():.3f}°   "
                  f"dlat ±{dlats[ok].std():.3f}°   (blur scale; means "
                  f"{dlons[ok].mean():+.3f}/{dlats[ok].mean():+.3f})", flush=True)
        for p in ("N", "S"):
            sel = (pts == p) & ok
            if sel.sum() >= 3:
                print(f"   {p}: scatter dlon ±{dlons[sel].std():.3f}° dlat "
                      f"±{dlats[sel].std():.3f}°  mean dlat {dlats[sel].mean():+.3f}° "
                      f"(n={int(sel.sum())})", flush=True)


if __name__ == "__main__":
    main()
