"""Stack-coherence metrics: the validation gate for registration.

The rotation elements are correct only when all looks co-register, and the only
acceptable evidence is that the *stacked imagery* reinforces at fine scale. The
gate is the Fourier ring correlation (FRC) between two independent half-stacks:
misregistration specifically destroys the HIGH-frequency correlation, and the
frequency where FRC falls off is the effective resolution — the crispness
deliverable itself.

- ``coherent_stack``   : count-normalized average of co-registered look maps.
- ``frc``              : ring correlation vs spatial frequency between two images.
- ``half_split``       : split looks in two, stack each half, FRC them. Returns
  broadband repro (``ncc``), the alignment-sensitive high-band correlation
  (``frc_hi``), and effective ``resolution_px``. A registration/rotation change is
  an improvement iff it RAISES ``frc_hi`` (finer resolution). A lower fit residual
  is not evidence.

Half-split is robust to per-look speckle (decorrelates across halves) but the
high-frequency band is sensitive to misalignment (blurs both halves, killing fine
correlation) — so ``frc_hi`` is the gate, not the broadband ``ncc``.
"""
import numpy as np
from .registration import bandpass


def coherent_stack(maps, masks):
    """Count-normalized average over a list of (map, mask). Returns (stack, count)."""
    H, W = maps[0].shape
    s = np.zeros((H, W), np.float64)
    c = np.zeros((H, W), np.int32)
    for g, m in zip(maps, masks):
        s[m] += g[m]
        c += m
    return np.divide(s, c, out=np.zeros_like(s), where=c > 0), c


def ncc(a, b, ma, mb, smooth_px=4.0, trend_px=30.0, min_overlap=2000):
    """Band-passed normalized cross-correlation over common support (broadband)."""
    m = ma & mb
    if int(m.sum()) < min_overlap:
        return np.nan
    fa = bandpass(a, smooth_px, trend_px)[m]
    fb = bandpass(b, smooth_px, trend_px)[m]
    fa = fa - fa.mean(); fb = fb - fb.mean()
    denom = np.sqrt(np.sum(fa**2) * np.sum(fb**2))
    return float(np.sum(fa * fb) / denom) if denom > 0 else np.nan


def frc(a, b, mask, smooth_px=2.0, trend_px=40.0, n_bins=30):
    """Fourier ring correlation between a and b over the common mask.

    Returns (freq_cyc_per_px, corr). Band-pass + apodize over the mask bounding box
    to suppress edge leakage, then correlate in radial frequency shells.
    """
    rows = np.where(np.any(mask, axis=1))[0]
    cols = np.where(np.any(mask, axis=0))[0]
    if rows.size < 16 or cols.size < 16:
        return np.array([]), np.array([])
    r0, r1, c0, c1 = rows[0], rows[-1], cols[0], cols[-1]
    fa = np.where(mask, bandpass(a, smooth_px, trend_px), 0)[r0:r1+1, c0:c1+1]
    fb = np.where(mask, bandpass(b, smooth_px, trend_px), 0)[r0:r1+1, c0:c1+1]
    h, w = fa.shape
    win = np.outer(np.hanning(h), np.hanning(w))
    Fa = np.fft.fftshift(np.fft.fft2(fa * win))
    Fb = np.fft.fftshift(np.fft.fft2(fb * win))
    ky = np.fft.fftshift(np.fft.fftfreq(h))[:, None]
    kx = np.fft.fftshift(np.fft.fftfreq(w))[None, :]
    kr = np.sqrt(ky**2 + kx**2)
    edges = np.linspace(0, 0.5, n_bins + 1)
    freq, corr = [], []
    for i in range(n_bins):
        sel = (kr >= edges[i]) & (kr < edges[i + 1])
        if sel.sum() < 8:
            continue
        num = np.sum(Fa[sel] * np.conj(Fb[sel])).real
        den = np.sqrt(np.sum(np.abs(Fa[sel])**2) * np.sum(np.abs(Fb[sel])**2))
        if den > 0:
            freq.append((edges[i] + edges[i + 1]) / 2)
            corr.append(num / den)
    return np.array(freq), np.array(corr)


def _resolution_px(freq, corr, thresh=0.5):
    """Effective resolution (px/cycle): first low->high frequency where FRC drops
    below ``thresh`` and the scale at which signal is still coherent (lower=sharper)."""
    if freq.size == 0:
        return np.nan
    below = np.where(corr < thresh)[0]
    fcut = freq[-1] if below.size == 0 else freq[below[0]]   # still coherent to Nyquist
    return float(1.0 / fcut) if fcut > 0 else np.inf


def half_split(maps, masks, n_splits=6, seed=0, fhi=0.15, **frc_kw):
    """Split looks in two, stack each half, FRC the halves (averaged over splits).

    Returns dict: ``ncc`` (broadband repro), ``frc_hi`` (mean FRC above ``fhi`` —
    the alignment gate), ``resolution_px`` (finest scale with FRC>0.3).
    """
    n = len(maps)
    if n < 2:
        return {"ncc": np.nan, "frc_hi": np.nan, "resolution_px": np.nan}
    rng = np.random.default_rng(seed)
    nccs, his, res = [], [], []
    for _ in range(n_splits):
        idx = rng.permutation(n)
        A, B = idx[: n // 2], idx[n // 2:]
        sa, ca = coherent_stack([maps[i] for i in A], [masks[i] for i in A])
        sb, cb = coherent_stack([maps[i] for i in B], [masks[i] for i in B])
        ma, mb = ca > 0, cb > 0
        nccs.append(ncc(sa, sb, ma, mb))
        f, c = frc(sa, sb, ma & mb, **frc_kw)
        if f.size:
            his.append(float(np.mean(c[f >= fhi])) if (f >= fhi).any() else np.nan)
            res.append(_resolution_px(f, c))
    return {"ncc": float(np.nanmean(nccs)),
            "frc_hi": float(np.nanmean(his)) if his else np.nan,
            "resolution_px": float(np.nanmedian(res)) if res else np.nan}


def stack_sharpness(stack, count, smooth_px=3.0, trend_px=25.0, min_count=2):
    """High-pass energy of the coherent stack where >=min_count looks overlap."""
    hp = bandpass(stack, smooth_px, trend_px)
    sel = count >= min_count
    return float(np.std(hp[sel])) if sel.sum() > 100 else np.nan
