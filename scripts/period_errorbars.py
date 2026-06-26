"""Error bars for the 1988-anchored period estimates from the interior-NCC curve width.

For each 1988<->Y pair: forward-shift Y by the period-induced longitude (no bulk
register — the period IS the alignment), compute the interior overlapping-disk NCC vs
trial period, find the peak, and set the 1-sigma error from how fast the NCC falls off:
  NCC(P) ~= NCC_pk - 0.5*k*(P-Ppk)^2 ;  sigma_NCC = scatter of the curve about a smooth
  fit (measurement noise) ;  sigma_P = sqrt(2*sigma_NCC/k)  (period change that drops
  NCC by one noise sigma below the peak). Also reports the half-width where NCC drops
  by 1% of (peak-min), as a transparent cross-check.

Writes results/figures/period_1988_errorbars.png
"""
import os, sys, glob
import numpy as np
from scipy.ndimage import gaussian_filter, binary_erosion, shift as ndshift
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from venera import spice_setup
from venera.coherence import ncc
import cspyce as csp
spice_setup.furnsh_kernels()
DATA=("/home/than/code/venera/arecibo_radar/pds-geosciences.wustl.edu/venus/"
      "arcb_nrao-v-rtls_gbt-3-delaydoppler-v1/vrm_90xx/data/")
STACKS="results/session_stacks"; FIG="results/figures"; HH,WW=2000,4000
RW0,RW1=int((22+90)/180*HH),int((80+90)/180*HH)   # crop to northern rows (speed)
P_ASSUMED=243.0185; ERODE,LATMIN,LATMAX=20,30,75
PGRID=np.arange(243.012,243.030,0.00025)
def meanday(y):
    ds=[]
    for f in glob.glob(DATA+f"venus_scp_{y}*.lbl")+glob.glob(DATA+f"venus_ocp_{y}*.lbl"):
        ds.append(0.5*(csp.str2et(open(f).read().split("START_TIME")[1].split("=")[1].split()[0])
                       +csp.str2et(open(f).read().split("STOP_TIME")[1].split("=")[1].split()[0])))
    return np.mean(ds)/86400.0
def load(y):
    d=np.load(f"{STACKS}/session_{y}.npz");return d["Gm"][RW0:RW1],d["mask"][RW0:RW1]
def flat(G,m): return (G-gaussian_filter(np.where(m,G,0.),35)/np.maximum(gaussian_filter(m.astype(float),35),1e-6))*m
LB=np.zeros((RW1-RW0,WW),bool); LB[max(int((LATMIN+90)/180*HH)-RW0,0):int((LATMAX+90)/180*HH)-RW0]=True
def curve(GA,MA,GB,MB,dt):
    FA=flat(GA,MA); out=[]
    for P in PGRID:
        wdot=(-360.0/P)-(-360.0/P_ASSUMED); px=(-wdot*dt)/360.0*WW
        FBs=ndshift(flat(GB,MB),(0,px),order=1,mode="constant",cval=0.0)
        MBs=ndshift(MB.astype(float),(0,px),order=1,mode="constant",cval=0)>0.99
        inn=binary_erosion(MA&MBs,iterations=ERODE)&LB
        out.append(ncc(FA,FBs,inn,inn) if inn.sum()>1500 else np.nan)
    return np.array(out)
def main():
    d1988=meanday("1988"); GA,MA=load("1988")
    Ys=["2001","2012","2015","2017","2020"]; res=[]
    fig,ax=plt.subplots(figsize=(10,6))
    for y in Ys:
        GB,MB=load(y); dt=(meanday(y)-d1988)*365.25/365.25*86400/86400  # days
        dt=(meanday(y)-d1988)
        c=curve(GA,MA,GB,MB,dt); g=np.isfinite(c)
        i=int(np.nanargmax(c)); 
        # parabola refine + curvature on +-6 points
        lo,hi=max(i-6,0),min(i+7,len(PGRID)); xx=PGRID[lo:hi]; yy=c[lo:hi]
        co=np.polyfit(xx,yy,2); Ppk=-co[1]/(2*co[0]); k=-2*co[0]  # NCC=co0 P^2... k= -2a (a<0)
        smooth=np.polyval(np.polyfit(PGRID[g],c[g],4),PGRID)
        sigN=np.nanstd((c-smooth)[g])
        sigP=np.sqrt(2*sigN/k) if k>0 else np.nan
        res.append((y,Ppk,sigP,sigN,k,np.nanmax(c)))
        ax.plot(PGRID,c,lw=1,label=f"1988-{y}: {Ppk:.4f}±{sigP:.4f}")
        ax.errorbar([Ppk],[np.nanmax(c)],xerr=[sigP],fmt="o",ms=5,capsize=3)
        print(f"1988-{y}: peak={Ppk:.4f}  sigma_P={sigP:.4f} d  (sigma_NCC={sigN:.4f}, curv k={k:.1f}, peakNCC={np.nanmax(c):.3f})",flush=True)
    for nm,v,col in [("adopted 243.0203",243.0203,"b"),("Campbell 243.0212",243.0212,"g"),("IAU 243.0185",243.0185,"k")]:
        ax.axvline(v,ls="--",lw=.8,color=col,alpha=.7,label=nm)
    ax.set_xlabel("trial period (d)");ax.set_ylabel("interior NCC (1988 vs Y)")
    ax.set_title("1988-anchored period: interior-NCC curves + error bars from peak width")
    ax.legend(fontsize=8);ax.grid(alpha=.3)
    fig.tight_layout();fig.savefig(FIG+"/period_1988_errorbars.png",dpi=130)
    P=np.array([r[1] for r in res]); s=np.array([r[2] for r in res])
    w=1/s**2; pe=np.sum(w*P)/np.sum(w); se=1/np.sqrt(np.sum(w))
    print(f"\ninverse-variance weighted 1988-anchored period = {pe:.4f} ± {se:.4f} d (per-pair sigma {s.round(4)})",flush=True)
    print("wrote period_1988_errorbars.png")
main()
