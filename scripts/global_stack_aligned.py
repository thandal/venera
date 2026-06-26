"""Build the crispest registered global stack: robust (sigma-clipped) combination of
ALL cached looks, with each look longitude-shifted to the rotation period `P_ADOPT`
below so the sessions co-register physically (a period change is an exact per-look
longitude rotation = column shift). Renders the global map + a northern zoom, and an
assumed-period (no-shift) version for before/after comparison.

Usage: .conda/bin/python scripts/global_stack_aligned.py
"""
import os, sys, glob
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE=os.path.join(ROOT,"results","look_cache"); FIG=os.path.join(ROOT,"results","figures")
SST=os.path.join(ROOT,"results","session_stacks")
HH,WW,DS=2000,4000,2; CLIP=3.0
P_ADOPT=243.0216; P_ASSUMED=243.0185   # image build period; §6 science value 243.0206 (sub-px difference)
DWDOT=(-360.0/P_ADOPT)-(-360.0/P_ASSUMED)     # deg/day (adopted - assumed)


def looks():
    return sorted(glob.glob(os.path.join(CACHE,"venus_*_*.npz")))


def shift_px(day, day_ref, align):
    if not align: return 0
    return int(round((-(DWDOT)*(day-day_ref))/360.0*WW))


def place(d, day_ref, align):
    gm,m=d["gm"],d["mask"]; r0,r1,c0,c1=d["bbox"]; hr,hc=r0//DS,c0//DS; h,w=gm.shape
    G=np.zeros((HH,WW),np.float32); M=np.zeros((HH,WW),bool)
    if hr+h>HH or hc+w>WW: h=min(h,HH-hr);w=min(w,WW-hc);gm=gm[:h,:w];m=m[:h,:w]
    G[hr:hr+h,hc:hc+w]=np.where(m,gm,0); M[hr:hr+h,hc:hc+w]=m
    s=shift_px(float(d["day"]),day_ref,align)
    if s: G=np.roll(G,s,axis=1); M=np.roll(M,s,axis=1)
    return G,M


def robust_stack(files, align):
    days=[float(np.load(f,allow_pickle=True)["day"]) for f in files]
    day_ref=float(np.mean(days))
    S=np.zeros((HH,WW)); SS=np.zeros((HH,WW)); N=np.zeros((HH,WW),np.int32)
    for f in files:
        G,M=place(np.load(f,allow_pickle=True),day_ref,align)
        S[M]+=G[M]; SS[M]+=G[M]**2; N[M]+=1
    mu=np.divide(S,N,out=np.zeros_like(S),where=N>0)
    var=np.divide(SS,N,out=np.zeros_like(SS),where=N>0)-mu**2
    sd=np.sqrt(np.clip(var,0,None))
    S2=np.zeros((HH,WW)); N2=np.zeros((HH,WW),np.int32)
    for f in files:
        G,M=place(np.load(f,allow_pickle=True),day_ref,align)
        keep=M&(np.abs(G-mu)<=CLIP*np.maximum(sd,1e-6))
        S2[keep]+=G[keep]; N2[keep]+=1
    return np.divide(S2,N2,out=np.zeros_like(S2),where=N2>0).astype(np.float32), N2>0


def render(G,m,box,title,fn,scale=None):
    LAT0,LAT1,LON0,LON1=box
    r0=int((LAT0+90)/180*HH);r1=int((LAT1+90)/180*HH);c0=int((LON0+180)/360*WW);c1=int((LON1+180)/360*WW)
    g=G[r0:r1,c0:c1].astype(float);cm=m[r0:r1,c0:c1]
    f=g-gaussian_filter(np.where(cm,g,0.),35)/np.maximum(gaussian_filter(cm.astype(float),35),1e-6); f[~cm]=np.nan
    if scale is None:
        v=f[np.isfinite(f)];scale=(np.percentile(v,2),np.percentile(v,99))
    fig,ax=plt.subplots(figsize=(13,7))
    ax.imshow(f,origin="lower",extent=[LON0,LON1,LAT0,LAT1],cmap="gray",vmin=scale[0],vmax=scale[1],aspect="auto",interpolation="bilinear")
    ax.set_title(title,fontsize=12);ax.set_xlabel("longitude (°E)");ax.set_ylabel("latitude (°N)");ax.grid(alpha=.2,color="c",lw=.3)
    fig.tight_layout();fig.savefig(fn,dpi=140);plt.close(fig); return scale


def main():
    fs=looks(); print(f"{len(fs)} looks; adopted P={P_ADOPT}, period shift over 32yr "
                      f"= {abs(DWDOT)*11680:.3f}° ({abs(DWDOT)*11680/360*WW:.1f}px)",flush=True)
    Ga,ma=robust_stack(fs,align=True)
    np.savez_compressed(os.path.join(SST,"global_aligned.npz"),Gm=Ga,mask=ma,period=P_ADOPT)
    NORTH=(20,82,-130,75); ISH=(40,80,-60,45)
    sc=render(Ga,ma,NORTH,f"Global stack — ALL 916 looks, period-aligned (P={P_ADOPT} d)",f"{FIG}/global_aligned_north.png")
    render(Ga,ma,ISH,f"Global stack period-aligned — Ishtar/Maxwell",f"{FIG}/global_aligned_maxwell.png")
    # assumed-period (no shift) for comparison, same scale
    Gn,mn=robust_stack(fs,align=False)
    render(Gn,mn,NORTH,f"Global stack — assumed P={P_ASSUMED} (no period alignment)",f"{FIG}/global_assumed_north.png",scale=sc)
    print("wrote global_aligned_north.png, global_aligned_maxwell.png, global_assumed_north.png",flush=True)
    print(f"global stack saved: results/session_stacks/global_aligned.npz ({int(ma.sum())} px covered)",flush=True)


if __name__=="__main__":
    main()
