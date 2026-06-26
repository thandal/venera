"""Characterize the 2012 Doppler-centering residual (fo): is it a constant systematic
offset, or correlated with a geometric quantity (which would identify a specific
ephemeris-tuning miscalculation by the data provider in 2012)?

fo = the limb-fit freq_offset (cols), i.e. the residual between the provider's
ephemeris Doppler tuning and the true sub-radar Doppler. We project a subset of looks
for fo, compute candidate geometric Doppler terms from the ephemeris (no .img read),
and test which (if any) explains fo. Candidate miscalculations:
  - constant (clock/processing offset)
  - diurnal / hour-of-day (topocentric vs geocentric tuning rotates with Earth)
  - bistatic-vs-monostatic: f0/c*(v_GBT - v_Arecibo) (tuned to one station, not the
    Arecibo->GBT path)

Usage: .conda/bin/python scripts/characterize_fo.py [n_subset]
"""
import os, sys, glob
import numpy as np
from multiprocessing import Pool
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA=("/home/than/code/venera/arecibo_radar/pds-geosciences.wustl.edu/venus/"
      "arcb_nrao-v-rtls_gbt-3-delaydoppler-v1/vrm_90xx/data/")
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG=os.path.join(ROOT,"results","figures")
F0=2380e6; C=299792458.0
NSUB=int(sys.argv[1]) if len(sys.argv)>1 else 40


def looks():
    from venera.data import parse_lbl
    fs=sorted(glob.glob(DATA+"venus_scp_2012*.img")+glob.glob(DATA+"venus_ocp_2012*.img"))
    return [f for f in fs if parse_lbl(f).get("GEO_POINTING")=="N"]


def _init():
    from venera import spice_setup; spice_setup.furnsh_kernels()


def los_rate(et, station, dt=2.0):
    import cspyce as csp
    from venera.geometry import _topo_offset_j2000, STATIONS
    def rng(t):
        ev=np.asarray(csp.spkpos("VENUS",t,"J2000","CN+S","EARTH")[0],float)
        if station in STATIONS: ev=ev-_topo_offset_j2000(t,station)
        return np.linalg.norm(ev)*1e3
    return (rng(et+dt/2)-rng(et-dt/2))/dt


def fo_of(img):
    from venera.geometry import Spin
    from venera.projection import project_file
    G=np.zeros((4000,8000),np.float32);Gc=np.zeros((4000,8000),np.int32)
    return project_file(img,Spin(),G,Gc)["fit"][0]


def geom(img):
    import cspyce as csp
    from venera.data import parse_lbl
    lbl=parse_lbl(img)
    et=0.5*(csp.str2et(lbl["START_TIME"])+csp.str2et(lbl["STOP_TIME"]))
    base=os.path.basename(img); hh=int(base.split("_")[3][:2])+int(base.split("_")[3][2:4])/60.
    prf=1.0/(lbl["GEO_CODE_LENGTH"]*lbl["GEO_BAUD"]*1e-6); binhz=prf/lbl.get("GEO_TRANSFORM_LENGTH",8192)
    return dict(et=et,hh=hh,binhz=binhz,
                va=los_rate(et,"ARECIBO"),vg=los_rate(et,"GBT"),ve=los_rate(et,"EARTH"))


def main():
    L=looks()
    sub=[L[i] for i in np.linspace(0,len(L)-1,min(NSUB,len(L))).astype(int)]
    with Pool(10,initializer=_init) as p:
        fos=p.map(fo_of,sub)        # projection (reads .img) — the slow part
        gs=p.map(geom,sub)          # ephemeris only — fast
    fo=np.array(fos); et=np.array([g["et"] for g in gs]); hh=np.array([g["hh"] for g in gs])
    va=np.array([g["va"] for g in gs]); vg=np.array([g["vg"] for g in gs]); ve=np.array([g["ve"] for g in gs])
    binhz=np.median([g["binhz"] for g in gs]); t=(et-et.min())/86400.0
    order=np.sort(fo);bc,bn=np.median(fo),-1
    for x in order:
        n=int(np.sum(np.abs(fo-x)<=15))
        if n>bn:bn,bc=n,x
    maj=np.abs(fo-bc)<=15
    print(f"{len(sub)} looks. majority center={bc:+.1f} cols ({maj.sum()}/{len(sub)}); "
          f"bin={binhz*1000:.2f} mHz/col -> {bc*binhz*1000:+.1f} mHz")
    print(f"majority fo: mean={fo[maj].mean():+.2f} std=±{fo[maj].std():.2f} cols; "
          f"slope vs time={np.polyfit(t[maj],fo[maj],1)[0]:+.3f} cols/day")
    foHz=fo*binhz
    cands={"hour-of-day":hh, "time(day)":t,
           "bistatic-mono f0/c*(vg-va)[Hz]":F0/C*(vg-va),
           "topo-geo 2f0/c*(va-ve)[Hz]":2*F0/C*(va-ve)}
    print("\ncorr of majority fo with candidate terms (and term's predicted col-range):")
    for k,v in cands.items():
        cc=np.corrcoef(fo[maj],v[maj])[0,1]
        rng_cols=(np.ptp(v[maj])/binhz) if "Hz" in k else np.nan
        extra=f"  predicts {rng_cols:.0f} cols across session" if "Hz" in k else ""
        print(f"  {k:34s}: corr={cc:+.2f}{extra}")
    print(f"\nmajority fo spread ±{fo[maj].std():.1f} cols = ±{fo[maj].std()*binhz*1000:.2f} mHz")
    print(f"IF provider tuned monostatic-Arecibo, fo would span "
          f"{np.ptp((F0/C*(vg-va))[maj])/binhz:.0f} cols (it spans {np.ptp(fo[maj]):.0f}).")
    fig,axs=plt.subplots(1,3,figsize=(16,5))
    axs[0].scatter(t,fo,c=np.where(maj,'b','r'),s=20);axs[0].axhline(bc,ls='--',c='k',lw=.8)
    axs[0].set_xlabel("day in session");axs[0].set_ylabel("fo (cols)");axs[0].set_title(f"fo vs time (const={bc:+.0f}?)");axs[0].grid(alpha=.3)
    axs[1].scatter(hh[maj],fo[maj],s=20);axs[1].set_xlabel("hour-of-day");axs[1].set_ylabel("fo");axs[1].set_title("fo vs hour (diurnal?)");axs[1].grid(alpha=.3)
    axs[2].scatter((F0/C*(vg-va))[maj],foHz[maj],s=20);axs[2].set_xlabel("bistatic-mono Doppler [Hz]");axs[2].set_ylabel("fo [Hz]");axs[2].set_title("fo vs bistatic differential");axs[2].grid(alpha=.3)
    fig.suptitle("2012 fo residual — constant or geometry-correlated miscalc?",fontsize=12)
    fig.tight_layout();fig.savefig(os.path.join(FIG,"characterize_fo.png"),dpi=120)
    print("\nwrote characterize_fo.png")


if __name__=="__main__":
    main()
