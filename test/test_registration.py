"""Validate sub-pixel registration by recovering known synthetic shifts.

Run: .conda/bin/python test/test_registration.py
"""
import os, sys
import numpy as np
from scipy.ndimage import fourier_shift

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from venera.registration import xcorr_offset, register_maps, bandpass

rng = np.random.default_rng(0)

_fail = 0
def check(name, cond, detail=""):
    global _fail
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {detail}")
    if not cond:
        _fail += 1


def shifted(img, dy, dx):
    return np.real(np.fft.ifft2(fourier_shift(np.fft.fft2(img), (dy, dx))))


# A textured "surface": correlated noise (like band-passed radar speckle + features)
base = gaussian_blurred = None
field = rng.standard_normal((512, 512))
from scipy.ndimage import gaussian_filter
field = gaussian_filter(field, 2.0)

print("== 1. Recover known sub-pixel shifts ==")
truth = [(0.0, 0.0), (3.0, -5.0), (-2.4, 1.7), (0.3, -0.6), (7.2, 8.9)]
max_err = 0.0
for dy, dx in truth:
    b = shifted(field, dy, dx)
    drow, dcol, sig = xcorr_offset(field, b, max_shift=32)
    # xcorr returns shift mapping b->a, i.e. the negative of how we shifted field->b
    err = np.hypot(drow - (-dy), dcol - (-dx))
    max_err = max(max_err, err)
    print(f"    injected ({dy:+.2f},{dx:+.2f}) -> measured ({-drow:+.3f},{-dcol:+.3f})  "
          f"err={err:.3f}px  sig={sig:.2f}")
check("sub-pixel shift recovery < 0.1 px", max_err < 0.1, f"max_err={max_err:.3f}px")


print("== 2. Significance: real feature lock vs pure-noise ==")
b = shifted(field, 2.0, -3.0)
_, _, sig_real = xcorr_offset(field, b, max_shift=32)
noise2 = gaussian_filter(rng.standard_normal((512, 512)), 2.0)
_, _, sig_noise = xcorr_offset(field, noise2, max_shift=32)
print(f"    sig(real)={sig_real:.2f}  sig(noise)={sig_noise:.2f}")
check("real lock significance > 1.5", sig_real > 1.5, f"{sig_real:.2f}")
check("noise significance < real", sig_noise < sig_real, f"noise={sig_noise:.2f}")


print("== 3. Masked registration with partial overlap + residual bright envelope ==")
# A realistic *residual* quasi-specular envelope (the bulk is removed upstream by
# incidence-law normalization, task #9). Modest amplitude, slightly different per
# epoch (different sub-radar point) -> not a perfect common-mode zero-lag attractor.
yy, xx = np.mgrid[0:512, 0:512]
def envelope(cx):
    return 4.0 * np.exp(-(((yy - 256) ** 2 + (xx - cx) ** 2) / (2 * 120.0 ** 2)))
dy, dx = -1.8, 2.6
a = field + envelope(256)
b = shifted(field, dy, dx) + envelope(262)
valid = np.zeros((512, 512), bool)
valid[40:470, 40:470] = True
drow, dcol, sig = register_maps(a, b, valid_a=valid, valid_b=valid, max_shift=32)
err = np.hypot(drow - (-dy), dcol - (-dx))
print(f"    injected ({dy:+.2f},{dx:+.2f}) -> measured ({-drow:+.3f},{-dcol:+.3f})  "
      f"err={err:.3f}px  sig={sig:.2f}")
check("masked+bandpassed recovery < 0.25 px", err < 0.25, f"err={err:.3f}px")


print()
if _fail:
    print(f"FAILED ({_fail} check(s))")
    sys.exit(1)
print("ALL REGISTRATION CHECKS PASSED")
