"""venera — Venus planetary radar astronomy pipeline (cspyce-based).

The live system is this ``.py`` package; the notebooks at the repo root are
archival. Run under the symlinked env: ``.conda/bin/python``.
"""
from . import spice_setup
from . import geometry

__all__ = ["spice_setup", "geometry"]
