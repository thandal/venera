"""Validate per-look self-calibration by recovering injected Doppler center/tilt
from synthetic tilted-disk echoes (with a bright off-center feature, like Maxwell).

Run: .conda/bin/python test/test_selfcal.py
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from venera.selfcal import doppler_axis

_fail = 0
def check(name, cond, detail=""):
    global _fail
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {detail}")
    if not cond:
        _fail += 1


def synth_echo(c0_col, tilt, n_rows=4000, n_cols=8192, hw_max=2800,
               bright=True, seed=0):
    """Tilted-disk echo: at delay row r the disk spans Doppler symmetric about
    (c0 + tilt*r) with half-width growing ~ sin(theta(r))."""
    rng = np.random.default_rng(seed)
    img = np.abs(rng.standard_normal((n_rows, n_cols)).astype(np.float32)) * 0.05
    rows = np.arange(n_rows)
    theta = np.clip(np.sqrt(rows / (n_rows * 0.6)), 0, 1) * (np.pi / 2)
    hw = hw_max * np.sin(theta)
    cols = np.arange(n_cols)[None, :]
    center = (c0_col + tilt * rows)[:, None]
    disk = np.abs(cols - center) < hw[:, None]
    img += disk * 0.3
    # limb brightening: extra power near the edges
    edge = (np.abs(np.abs(cols - center) - hw[:, None]) < 40) & (hw[:, None] > 40)
    img += edge * 0.2
    if bright:  # Maxwell-like bright feature, strongly off-center in Doppler
        img[700:1100, int(c0_col - 1600):int(c0_col - 1250)] += 2.0
    return img


c2 = 8192 / 2.0
print("== Recover injected Doppler center + tilt ==")
cases = [(c2 + 0.0, 0.0), (c2 - 35.0, -0.015), (c2 + 60.0, 0.02), (c2 - 12.0, 0.0)]
max_off_err = 0.0
max_tilt_err = 0.0
for c0, tilt in cases:
    img = synth_echo(c0, tilt, seed=int(abs(c0)) % 7)
    r = doppler_axis(img)
    off_true = c0 - c2
    off_err = abs(r["freq_offset"] - off_true)
    tilt_err = abs(r["tilt"] - tilt)
    max_off_err = max(max_off_err, off_err)
    max_tilt_err = max(max_tilt_err, tilt_err)
    print(f"    inj off={off_true:+6.1f} tilt={tilt:+.4f} -> "
          f"meas off={r['freq_offset']:+6.2f} tilt={r['tilt']:+.4f}  "
          f"scatter={r['scatter']:.2f} n={r['n']}  (err off={off_err:.2f}px)")
check("Doppler-center recovery < 2 px (despite bright feature)", max_off_err < 2.0,
      f"max err={max_off_err:.2f}px")
check("tilt recovery < 0.005 px/row", max_tilt_err < 0.005, f"max err={max_tilt_err:.4f}")

print("== Quality flag separates clean vs noisy ==")
from venera.selfcal import self_calibrate
clean = self_calibrate(synth_echo(c2 - 20, 0.0, seed=1))
rng = np.random.default_rng(3)
noisy_img = np.abs(rng.standard_normal((4000, 8192)).astype(np.float32))  # pure noise
noisy = self_calibrate(noisy_img)
print(f"    clean quality={clean['quality'] if clean else None}  "
      f"noise -> {None if noisy is None else noisy['quality']}")
check("clean look flagged good", clean is not None and clean["quality"] == "good")
check("pure-noise rejected or flagged poor",
      noisy is None or noisy["quality"] == "poor")

print()
if _fail:
    print(f"FAILED ({_fail} check(s))")
    sys.exit(1)
print("ALL SELF-CAL CHECKS PASSED")
