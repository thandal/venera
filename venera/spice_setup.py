"""Standard SPICE kernel set for the venera (Venus radar) pipeline.

Single source of truth for the kernel list. Call ``furnsh_kernels()`` once at
script/notebook startup. Kernels live in ``spice_kernels/`` at the repo root
(currently a symlink to ``../moon-radar/spice_kernels``).

We deliberately use **cspyce** (the actively maintained CSPICE binding moon-radar
uses) rather than spiceypy.

Note: ``pck00011.tpc`` provides the IAU_VENUS body-fixed frame with the *nominal*
spin constants. The venera geometry core does **not** rely on that frame for the
spin we are fitting — it builds the body-fixed frame itself from explicit spin
parameters (see ``geometry.Spin``). The PCK is furnished only so we can validate
our parameterized frame against ``IAU_VENUS`` at the nominal constants.
"""

import os
import cspyce as csp

# Resolve relative to the repo root (parent of this package dir).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPICE_KERNEL_DIR = os.path.join(_REPO_ROOT, "spice_kernels")

# Minimal set for Venus delay-Doppler geometry.
SPICE_KERNELS = [
    "naif0012.tls",   # leapseconds (time)
    "de440s.bsp",     # planetary ephemeris 1849-2150 (Venus, Earth)
    "pck00011.tpc",   # planetary constants + nominal IAU_VENUS frame (validation only)
]

_furnished = False


def furnsh_kernels(kernel_dir=None, force=False):
    """Furnish the standard kernel set (idempotent unless ``force``)."""
    global _furnished
    if _furnished and not force:
        return
    kdir = kernel_dir if kernel_dir is not None else SPICE_KERNEL_DIR
    if force:
        csp.kclear()
    for k in SPICE_KERNELS:
        path = os.path.join(kdir, k)
        if not os.path.exists(path):
            raise FileNotFoundError(f"SPICE kernel not found: {path}")
        csp.furnsh(path)
    _furnished = True


def nominal_venus_spin_constants():
    """Read Venus' nominal IAU spin constants from the furnished PCK.

    Returns (pole_ra_deg, pole_dec_deg, w0_deg, w_dot_deg_per_day).
    """
    furnsh_kernels()
    # cspyce: gdpool(name, start=0) returns the full array.
    ra = csp.gdpool("BODY299_POLE_RA", 0)[0]
    dec = csp.gdpool("BODY299_POLE_DEC", 0)[0]
    pm = csp.gdpool("BODY299_PM", 0)  # [W0, Wdot (deg/day), ...]
    return float(ra), float(dec), float(pm[0]), float(pm[1])
