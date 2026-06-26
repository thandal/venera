"""Before/after the Arecibo-observer fix: the cross-session tilt vs sub-radar
latitude, and the per-pair registration NCC at identity. Geocenter (bug) vs
Arecibo (fix). Data: register_pairs_sphere geocenter vs Arecibo runs."""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "results", "figures")

# pair: (|dsrp|, geocenter tilt, arecibo tilt, geocenter NCC@id, arecibo NCC@id)
D = {
    "2015-2017": (17.5, 2.60, 0.01, 0.11, 0.61), "2001-2015": (17.1, 2.34, 0.02, 0.14, 0.56),
    "2012-2015": (10.9, 1.19, 0.12, 0.34, 0.56), "2015-2020": (10.9, 1.04, 0.01, 0.22, 0.36),
    "1988-2017": (9.3, 1.51, 0.02, 0.25, 0.57), "1988-2001": (8.9, 1.28, 0.00, 0.28, 0.53),
    "1988-2015": (8.2, 1.17, 0.00, 0.35, 0.62), "2012-2017": (6.6, 1.44, 0.02, 0.33, 0.63),
    "2017-2020": (6.6, 1.27, 0.00, 0.21, 0.39), "2001-2012": (6.2, 1.25, 0.02, 0.35, 0.57),
    "2001-2020": (6.1, 0.92, 0.01, 0.24, 0.39), "1988-2020": (2.8, 0.01, 0.01, 0.42, 0.45),
    "1988-2012": (2.7, 0.01, 0.10, 0.64, 0.65), "2001-2017": (0.4, 0.21, 0.01, 0.71, 0.74),
    "2012-2020": (0.1, 0.00, 0.01, 0.55, 0.54),
}

pairs = list(D)
dsrp = np.array([D[p][0] for p in pairs])
geo_t = np.array([D[p][1] for p in pairs]); are_t = np.array([D[p][2] for p in pairs])
geo_n = np.array([D[p][3] for p in pairs]); are_n = np.array([D[p][4] for p in pairs])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# Panel A: tilt vs |dsrp|
ax1.scatter(dsrp, geo_t, s=60, c="tab:red", label="geocenter (bug)")
ax1.scatter(dsrp, are_t, s=60, c="tab:blue", label="Arecibo (fix)")
sl = np.sum(dsrp * geo_t) / np.sum(dsrp ** 2)
xx = np.linspace(0, dsrp.max(), 30)
ax1.plot(xx, sl * xx, "r--", lw=0.8, alpha=0.6)
ax1.axhline(0, color="b", ls="--", lw=0.8, alpha=0.6)
ax1.set_xlabel("|Δ sub-radar latitude| (°)")
ax1.set_ylabel("residual rigid tilt between sessions (°)")
ax1.set_title("Cross-session tilt: ∝ΔSRP-lat (geocenter) → ~0 (Arecibo)")
ax1.legend(); ax1.grid(alpha=0.3)

# Panel B: NCC@identity per pair (sorted by geocenter)
order = np.argsort(geo_n)
y = np.arange(len(pairs))
ax2.barh(y - 0.2, geo_n[order], height=0.4, color="tab:red", label="geocenter (bug)")
ax2.barh(y + 0.2, are_n[order], height=0.4, color="tab:blue", label="Arecibo (fix)")
ax2.set_yticks(y); ax2.set_yticklabels([pairs[i] for i in order], fontsize=8)
ax2.set_xlabel("interior NCC @ identity (no rotation)")
ax2.set_title("Cross-session registration at identity")
ax2.legend(); ax2.grid(alpha=0.3, axis="x")

fig.suptitle("Arecibo topocentric-observer fix: cross-session registration "
             "(tilt eliminated, NCC@identity up)", fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fix_before_after.png"), dpi=130)
print("wrote fix_before_after.png")
print(f"mean NCC@id geocenter={geo_n.mean():.2f} -> Arecibo={are_n.mean():.2f}")
print(f"max tilt geocenter={geo_t.max():.2f}° -> Arecibo={are_t.max():.2f}°")
