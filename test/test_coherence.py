"""Validate the coherence metrics, including the negative control: injected
misalignment MUST drop half_split_ncc. Run: .conda/bin/python test/test_coherence.py
"""
import os, sys
import numpy as np
from scipy.ndimage import gaussian_filter, fourier_shift

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from venera.coherence import coherent_stack, ncc, half_split, stack_sharpness

_fail = 0
def check(name, cond, detail=""):
    global _fail
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {detail}")
    if not cond:
        _fail += 1


H, W = 240, 600
rng = np.random.default_rng(3)
true = gaussian_filter(rng.standard_normal((H, W)), 2.0)        # stable "surface"
region = np.zeros((H, W), bool); region[40:200, 60:540] = True


def make_looks(n, shift_sigma_px=0.0, noise=1.0):
    maps, masks = [], []
    for _ in range(n):
        g = true.copy()
        if shift_sigma_px > 0:
            sh = rng.normal(0, shift_sigma_px, 2)
            g = np.real(np.fft.ifft2(fourier_shift(np.fft.fft2(g), sh)))
        g = g + gaussian_filter(rng.standard_normal((H, W)), 2.0) * noise   # speckle
        maps.append(g.astype(np.float32)); masks.append(region.copy())
    return maps, masks


print("== 1. coherent_stack averages where covered ==")
m, c = coherent_stack(*make_looks(4))
check("stack shape & coverage", m.shape == (H, W) and c.max() == 4 and (c[~region] == 0).all())

print("== 2. half-split reproducibility (ncc) rises as looks are added (aligned) ==")
maps, masks = make_looks(24, shift_sigma_px=0.0, noise=1.2)
vals = [half_split(maps[:k], masks[:k])["ncc"] for k in (4, 8, 16, 24)]
print(f"    half-split NCC at N=4,8,16,24: {[round(v,3) for v in vals]}")
check("monotone-ish increase with N", vals[-1] > vals[0] + 0.1, f"{vals[0]:.3f} -> {vals[-1]:.3f}")
check("converges high with enough looks", vals[-1] > 0.6, f"N=24 -> {vals[-1]:.3f}")

print("== 3. NEGATIVE CONTROL: injected misalignment drops the FRC high-band (the gate) ==")
amaps, amasks = make_looks(24, shift_sigma_px=0.0, noise=1.2)
bmaps, bmasks = make_looks(24, shift_sigma_px=3.0, noise=1.2)   # 3px random per-look shifts
a = half_split(amaps, amasks); b = half_split(bmaps, bmasks)
print(f"    aligned : frc_hi={a['frc_hi']:.3f}  res={a['resolution_px']:.1f}px  ncc={a['ncc']:.3f}")
print(f"    misalign: frc_hi={b['frc_hi']:.3f}  res={b['resolution_px']:.1f}px  ncc={b['ncc']:.3f}")
check("misalignment measurably lowers FRC high-band", a["frc_hi"] > b["frc_hi"] + 0.15,
      f"Δfrc_hi={a['frc_hi']-b['frc_hi']:.3f}")
check("misalignment coarsens resolution", b["resolution_px"] > a["resolution_px"],
      f"{a['resolution_px']:.1f} -> {b['resolution_px']:.1f} px")

print("== 4. stack_sharpness higher for aligned than misaligned stack ==")
sa, ca = coherent_stack(amaps, amasks); sb, cb = coherent_stack(bmaps, bmasks)
hsa, hsb = stack_sharpness(sa, ca), stack_sharpness(sb, cb)
print(f"    sharpness aligned={hsa:.4f}  misaligned={hsb:.4f}")
check("aligned stack sharper", hsa > hsb, f"{hsa:.4f} vs {hsb:.4f}")

print()
if _fail:
    print(f"FAILED ({_fail} check(s))"); sys.exit(1)
print("ALL COHERENCE CHECKS PASSED")
