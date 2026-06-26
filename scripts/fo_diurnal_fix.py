"""Implement + validate the diurnal-fo fix for 2012.

The Doppler-centering residual fo is a smooth function of hour-of-day (a small
residual in the provider's time-varying topocentric tuning) + a constant — NOT a
per-look random quantity. So replace the noisy per-look limb fit with a robust
per-session model fo_model(hour) = a + b*(hour-h0), fit to the majority cluster
(outlier-rejected), and force every look to it.

Validation figures:
  fo_diurnal_model.png        - fo vs hour with the robust fit (model validation)
  maxwell_scatter_ba.png      - per-look Maxwell 2D offset, WIDE fit vs MODEL
  maxwell_stack_ba.png        - Maxwell deep-stack image, WIDE fit vs MODEL

Usage: .conda/bin/python scripts/fo_diurnal_fix.py [n_looks]
"""
import os, sys, glob
import numpy as np
from multiprocessing import Pool
from scipy.ndimage import gaussian_filter, binary_erosion
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA=("/home/than/code/venera/arecibo_radar/pds-geosciences.wustl.edu/venus/"
      "arcb_nrao-v-rtls_gbt-3-delaydoppler-v1/vrm_90xx/data/")
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STACKS=os.path.join(ROOT,"results","session_stacks"); FIG=os.path.join(ROOT,"results","figures")
H,W,DS=4000,8000,2; HH,WW=H//DS,W//DS
# Maxwell crop (template + scatter)
LAT0,LAT1,LON0,LON1=56,74,-12,30
R0=int((LAT0+90)/180*HH);R1=int((LAT1+90)/180*HH);C0=int((LON0+180)/360*WW);C1=int((LON1+180)/360*WW)
# Maxwell display/sharpness box
DLAT0,DLAT1,DLON0,DLON1=50,76,-45,38
DR0=int((DLAT0+90)/180*HH);DR1=int((DLAT1+90)/180*HH);DC0=int((DLON0+180)/360*WW);DC1=int((DLON1+180)/360*WW)
NSUB=int(sys.argv[1]) if len(sys.argv)>1 else 45
def bp(g,m): return (g-gaussian_filter(np.where(m,g,0.),12))*m
d17=np.load(f"{STACKS}/session_2017.npz"); TMPL=bp(d17["Gm"][R0:R1,C0:C1].astype(float),d17["mask"][R0:R1,C0:C1])
def picks(n):
    from venera.data import parse_lbl
    fs=sorted(glob.glob(DATA+"venus_scp_2012*.img")+glob.glob(DATA+"venus_ocp_2012*.img"))
    keep=[f for f in fs if parse_lbl(f).get("GEO_POINTING")=="N"]
    return [keep[i] for i in np.linspace(0,len(keep)-1,min(n,len(keep))).astype(int)]
def _init():
    from venera import spice_setup; spice_setup.furnsh_kernels()
def hourof(img):
    b=os.path.basename(img); return int(b.split("_")[3][:2])+int(b.split("_")[3][2:4])/60.
def proj(args):
    img,foc=args  # foc None=wide fit
    from venera.geometry import Spin
    from venera.projection import project_file
    G=np.zeros((H,W),np.float32);Gc=np.zeros((H,W),np.int32)
    kw=dict() if foc is None else dict(fo_center=float(foc),fo_window=0)
    info=project_file(img,Spin(),G,Gc,**kw)
    Gm=np.divide(G,Gc,out=np.zeros_like(G),where=Gc>0)[::DS,::DS]; mk=(Gc[::DS,::DS]>0)
    return info["fit"][0], Gm.astype(np.float32), mk
def xcorr(a,b):
    a=a-a.mean();b=b-b.mean();cc=np.fft.fftshift(np.fft.irfft2(np.fft.rfft2(a)*np.conj(np.fft.rfft2(b)),s=a.shape))
    pk=np.unravel_index(np.argmax(cc),cc.shape);cy,cx=a.shape[0]//2,a.shape[1]//2
    cc2=cc.copy();cc2[pk[0]-3:pk[0]+4,pk[1]-3:pk[1]+4]=-1e9;sig=cc[pk]/cc2.max() if cc2.max()>0 else 0
    return (pk[0]-cy)/HH*180.,(pk[1]-cx)/WW*360.,float(sig)
def maxoff(Gm,mk):
    g=bp(Gm[R0:R1,C0:C1].astype(float),mk[R0:R1,C0:C1])
    return xcorr(TMPL,g) if mk[R0:R1,C0:C1].mean()>0.4 else None
def stack(parts):
    Gs=np.zeros((HH,WW));Gc=np.zeros((HH,WW),int)
    for Gm,mk in parts: Gs[mk]+=Gm[mk];Gc[mk]+=1
    return np.divide(Gs,Gc,out=np.zeros_like(Gs),where=Gc>0).astype(np.float32),Gc>0
def sharp(G,m):
    box=m[DR0:DR1,DC0:DC1];inn=binary_erosion(box,iterations=10)
    g=G[DR0:DR1,DC0:DC1].astype(float);hp=g-gaussian_filter(np.where(box,g,0.),4)
    return float(np.mean(hp[inn]**2)) if inn.sum()>500 else np.nan
def showmax(ax,G,m,title):
    g=G[DR0:DR1,DC0:DC1].astype(float);cm=m[DR0:DR1,DC0:DC1]
    hp=g-gaussian_filter(np.where(cm,g,0.),30);hp[~cm]=np.nan
    v=hp[np.isfinite(hp)];lo,hi=np.percentile(v,2),np.percentile(v,98)
    ax.imshow(hp,origin="lower",extent=[DLON0,DLON1,DLAT0,DLAT1],cmap="gray",vmin=lo,vmax=hi,aspect="auto")
    ax.set_title(title,fontsize=11);ax.set_xlabel("lon(E)");ax.set_ylabel("lat(N)")

def main():
    looks=picks(NSUB); hrs=np.array([hourof(f) for f in looks])
    with Pool(10,initializer=_init) as p:
        P1=p.map(proj,[(f,None) for f in looks])     # wide fit
    fo0=np.array([r[0] for r in P1])
    # robust diurnal model fo = a + b*(hour-h0)
    h0=hrs.mean(); A=np.c_[np.ones_like(hrs),hrs-h0]; keep=np.ones(len(hrs),bool)
    for _ in range(6):
        coef,*_=np.linalg.lstsq(A[keep],fo0[keep],rcond=None)
        keep=np.abs(fo0-A@coef)<10
    a,b=coef; fo_model=A@coef
    print(f"diurnal model: fo = {a:+.1f} {b:+.1f}*(hour-{h0:.2f})  majority {keep.sum()}/{len(looks)}",flush=True)
    print(f"  residual of majority about model: ±{np.std((fo0-fo_model)[keep]):.1f} cols (was ±{fo0[keep].std():.1f} raw)",flush=True)
    with Pool(10,initializer=_init) as p:
        P2=p.map(proj,list(zip(looks,fo_model)))      # forced to model
    # per-look Maxwell scatter
    ow=[maxoff(r[1],r[2]) for r in P1]; om=[maxoff(r[1],r[2]) for r in P2]
    ow=np.array([o for o in ow if o]); om=np.array([o for o in om if o])
    def rep(a,n):
        ok=a[:,2]>1.05; print(f"{n}: Maxwell scatter dlon±{a[ok,1].std():.3f} dlat±{a[ok,0].std():.3f} (lock {ok.sum()}/{len(a)})",flush=True); return a[ok]
    Ow=rep(ow,"WIDE  fit "); Om=rep(om,"MODEL fo  ")
    Gw,mw=stack([(r[1],r[2]) for r in P1]); Gm,mm=stack([(r[1],r[2]) for r in P2])
    print(f"Maxwell stack sharpness: wide={sharp(Gw,mw):.5f}  model={sharp(Gm,mm):.5f}  (2017~0.0047)",flush=True)
    # FIG 1: model
    fig,ax=plt.subplots(figsize=(8,5))
    ax.scatter(hrs[keep],fo0[keep],s=25,label="majority");ax.scatter(hrs[~keep],fo0[~keep],s=25,c='r',label="outlier(wrong edge)")
    xs=np.linspace(hrs.min(),hrs.max(),50);ax.plot(xs,a+b*(xs-h0),'k-',lw=1.5,label=f"model {a:+.0f}{b:+.0f}*(h-{h0:.1f})")
    ax.set_xlabel("hour of day");ax.set_ylabel("fo (cols)");ax.set_title("2012 fo: diurnal model replaces per-look fit");ax.legend();ax.grid(alpha=.3)
    fig.tight_layout();fig.savefig(FIG+"/fo_diurnal_model.png",dpi=120);plt.close(fig)
    # FIG 2: scatter before/after
    fig,axs=plt.subplots(1,2,figsize=(12,6))
    for ax,a2,t in [(axs[0],Ow,"WIDE per-look fit"),(axs[1],Om,"MODEL diurnal fo")]:
        ax.scatter(a2[:,1],a2[:,0],s=30,alpha=.7);ax.set_xlim(-3,3);ax.set_ylim(-3,3)
        ax.axhline(0,c='k',lw=.5);ax.axvline(0,c='k',lw=.5);ax.grid(alpha=.3)
        ax.set_title(f"{t}\ndlon±{a2[:,1].std():.2f} dlat±{a2[:,0].std():.2f}");ax.set_xlabel("Δlon");ax.set_ylabel("Δlat")
    fig.suptitle("Per-look Maxwell offset: before vs after diurnal-fo (2017 ref ±0.05°)",fontsize=12)
    fig.tight_layout();fig.savefig(FIG+"/maxwell_scatter_ba.png",dpi=120);plt.close(fig)
    # FIG 3: stack before/after
    fig,axs=plt.subplots(1,2,figsize=(13,5))
    showmax(axs[0],Gw,mw,f"2012 Maxwell — WIDE fit  sharp={sharp(Gw,mw):.4f}")
    showmax(axs[1],Gm,mm,f"2012 Maxwell — MODEL fo  sharp={sharp(Gm,mm):.4f}")
    fig.suptitle("2012 Maxwell deep stack: per-look fit vs diurnal-fo model",fontsize=12)
    fig.tight_layout();fig.savefig(FIG+"/maxwell_stack_ba.png",dpi=120);plt.close(fig)
    print("wrote fo_diurnal_model.png, maxwell_scatter_ba.png, maxwell_stack_ba.png",flush=True)

if __name__=="__main__":
    main()
