"""Big individual flip-figures for visually validating the 2012 fixes.

Renders separate, large, identically-scaled PNGs (flip between them — more sensitive
than red/green overlays):
  flip_2015.png            - 2015 reference (sharp, monostatic)
  flip_2012_before.png     - 2012 diurnal-fo + MONOSTATIC freq_scale (fs-before)
  flip_2012_after.png      - 2012 diurnal-fo + BISTATIC freq_scale (fs-after)
each at the same crop + grayscale, two zooms (Maxwell + wider north).

Also prints the 2012-vs-2015 warp-field decomposition (longitude stretch, latitude
stretch, rotation-about-SRP, shear) before vs after, so we can see what the
freq_scale fix removed and what residual (lat/rotation) remains.

Usage: .conda/bin/python scripts/flip_figures_2012.py
"""
import os, sys, glob
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from venera.registration import tile_displacements, register_maps
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STACKS=os.path.join(ROOT,"results","session_stacks"); CACHE=os.path.join(ROOT,"results","look_cache")
FIG=os.path.join(ROOT,"results","figures"); HH,WW,DS=2000,4000,2
# two zooms
MAXW=(50,78,-40,40); WIDE=(28,80,-110,55)


def stack_from_cache(year):
    Gs=np.zeros((HH,WW),np.float64);Gc=np.zeros((HH,WW),np.int32)
    for f in glob.glob(os.path.join(CACHE,f"venus_*_{year}*.npz")):
        d=np.load(f,allow_pickle=True); gm,m=d["gm"],d["mask"]; r0,r1,c0,c1=d["bbox"]
        hr,hc=r0//DS,c0//DS; h,w=gm.shape
        if hr+h>HH or hc+w>WW: h=min(h,HH-hr);w=min(w,WW-hc);gm=gm[:h,:w];m=m[:h,:w]
        Gs[hr:hr+h,hc:hc+w][m]+=gm[m]; Gc[hr:hr+h,hc:hc+w][m]+=1
    return np.divide(Gs,Gc,out=np.zeros_like(Gs),where=Gc>0).astype(np.float32),Gc>0


def render(G,m,box,title,fn,scale=None):
    LAT0,LAT1,LON0,LON1=box
    r0=int((LAT0+90)/180*HH);r1=int((LAT1+90)/180*HH);c0=int((LON0+180)/360*WW);c1=int((LON1+180)/360*WW)
    g=G[r0:r1,c0:c1].astype(float);cm=m[r0:r1,c0:c1]
    f=g-gaussian_filter(np.where(cm,g,0.),30)/np.maximum(gaussian_filter(cm.astype(float),30),1e-6); f[~cm]=np.nan
    if scale is None:
        v=f[np.isfinite(f)];scale=(np.percentile(v,2),np.percentile(v,99))
    fig,ax=plt.subplots(figsize=(13,8))
    ax.imshow(f,origin="lower",extent=[LON0,LON1,LAT0,LAT1],cmap="gray",vmin=scale[0],vmax=scale[1],aspect="auto",interpolation="bilinear")
    ax.set_title(title,fontsize=13);ax.set_xlabel("longitude (°E)");ax.set_ylabel("latitude (°N)")
    ax.grid(alpha=0.2,color="c",lw=0.4)
    fig.tight_layout();fig.savefig(fn,dpi=150);plt.close(fig); return scale


def warp_decomp(A,mA,Bn,mB,name):
    dr,dc,_=register_maps(A,Bn,valid_a=mA,valid_b=mB,max_shift=70,smooth_px=7,trend_px=55)
    B2=np.roll(np.roll(Bn,int(round(dr)),0),int(round(dc)),1); m2=np.roll(np.roll(mB,int(round(dr)),0),int(round(dc)),1)
    d=tile_displacements(A,B2,mA,m2,tile=256,step=160,min_sig=0.92,max_shift=45)
    if len(d)<6: print(f"  {name}: too few tiles"); return
    d=np.array(d); lat,lon,dlat,dlon=d[:,0],d[:,1],d[:,2],d[:,3]
    De=np.c_[np.ones_like(lon),lon,lat]
    cl=np.linalg.lstsq(De,dlon,rcond=None)[0]; ca=np.linalg.lstsq(De,dlat,rcond=None)[0]
    div_lon=cl[1]; div_lat=ca[2]; curl=ca[1]-cl[2]; shear=cl[1]-ca[2]
    print(f"  {name}: n={len(d)} |disp|med={np.median(np.hypot(dlon,dlat)):.2f}deg | "
          f"lon-stretch={div_lon:+.4f}/deg  lat-stretch={div_lat:+.4f}/deg  "
          f"rotation(curl)={curl:+.4f}  resid_lon={np.std(dlon-De@cl):.2f} resid_lat={np.std(dlat-De@ca):.2f}")


def main():
    G15=np.load(f"{STACKS}/session_2015.npz"); A15,m15=G15["Gm"],G15["mask"]
    db=np.load(f"{STACKS}/session_2012_fsbefore.npz"); Gb,mb=db["Gm"],db["mask"]
    Ga,ma=stack_from_cache("2012")  # current cache = after (bistatic-fs)
    for box,tag in [(MAXW,"maxwell"),(WIDE,"wide")]:
        sc=render(A15,m15,box,"2015 (monostatic reference)",f"{FIG}/flip_{tag}_2015.png")
        render(Gb,mb,box,"2012 BEFORE (monostatic freq_scale)",f"{FIG}/flip_{tag}_2012_before.png",scale=sc)
        render(Ga,ma,box,"2012 AFTER (bistatic freq_scale)",f"{FIG}/flip_{tag}_2012_after.png",scale=sc)
    print("warp decomposition vs 2015 (lon-stretch is what freq_scale targets):")
    warp_decomp(A15,m15,Gb,mb,"2012 BEFORE")
    warp_decomp(A15,m15,Ga,ma,"2012 AFTER ")
    print("wrote flip_{maxwell,wide}_{2015,2012_before,2012_after}.png")


if __name__=="__main__":
    main()
