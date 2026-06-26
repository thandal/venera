"""Production fix for the 2012 (bistatic) Doppler centering: replace the noisy
per-look limb fit with the robust per-session **diurnal** fo model, reproject ALL
2012 looks with it (updating the look cache), and render separate before/after deep-
stack PNGs.

Steps:
  A. stack 2012 from the CURRENT cache  -> session_2012_before.png
  B. fit diurnal model fo(hour) from a sample of 2012 looks (wide fit)
  C. reproject ALL 2012 looks with fo_center = model(hour) -> overwrite cache
  D. stack 2012 from the new cache      -> session_2012_after.png  (matched scale)

Usage: .conda/bin/python scripts/reproject_2012_diurnal.py
"""
import os, sys, glob, time
import numpy as np
from multiprocessing import Pool
from scipy.ndimage import gaussian_filter
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA=("/home/than/code/venera/arecibo_radar/pds-geosciences.wustl.edu/venus/"
      "arcb_nrao-v-rtls_gbt-3-delaydoppler-v1/vrm_90xx/data/")
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE=os.path.join(ROOT,"results","look_cache"); FIG=os.path.join(ROOT,"results","figures")
H,W,DS=4000,8000,2; HH,WW=H//DS,W//DS
LAT0,LAT1,LON0,LON1=25,80,-115,55  # display crop (northern hemisphere)
SAMPLE=55


def hourof(p):
    b=os.path.basename(p); return int(b.split("_")[3][:2])+int(b.split("_")[3][2:4])/60.+int(b.split("_")[3][4:6])/3600.


def looks2012():
    return sorted(glob.glob(DATA+"venus_scp_2012*.img")+glob.glob(DATA+"venus_ocp_2012*.img"))


def _init():
    from venera import spice_setup; spice_setup.furnsh_kernels()


def wide_fo(img):
    from venera.geometry import Spin
    from venera.projection import project_file
    G=np.zeros((H,W),np.float32);Gc=np.zeros((H,W),np.int32)
    return project_file(img,Spin(),G,Gc)["fit"][0]


def reproj_cache(args):
    """Reproject one look with fo_center=foc and write the cache npz (project_all_looks
    format)."""
    img,foc=args
    from venera.geometry import Spin, sub_radar_point, doppler_angle
    from venera.projection import project_file
    base=os.path.basename(img)[:-4]; out=os.path.join(CACHE,base+".npz")
    try:
        G=np.zeros((H,W),np.float32);Gc=np.zeros((H,W),np.int32)
        info=project_file(img,Spin(),G,Gc,fo_center=float(foc),fo_window=0)
        et=info["et_mid"]; lon,lat,_=sub_radar_point(et,Spin())
        rows=np.where(np.any(Gc>0,1))[0]; cols=np.where(np.any(Gc>0,0))[0]
        if rows.size==0: return (base,"empty")
        r0,r1,c0,c1=rows[0],rows[-1],cols[0],cols[-1]
        Gm=np.divide(G,Gc,out=np.zeros_like(G),where=Gc>0)
        np.savez_compressed(out, gm=Gm[r0:r1+1:DS,c0:c1+1:DS].astype(np.float32),
            mask=(Gc[r0:r1+1:DS,c0:c1+1:DS]>0), bbox=np.array([r0,r1,c0,c1]), ds=DS,
            hw=np.array([H,W]), et_mid=et, day=et/86400.0,
            srp_lon=np.degrees(lon), srp_lat=np.degrees(lat),
            doppler_angle=np.degrees(doppler_angle(et,Spin())),
            pol=("scp" if "_scp_" in base else "ocp"),
            pointing=info["pointing"], valid_frac=info["valid_frac"])
        return (base,"ok")
    except Exception as e:
        return (base,f"ERR {e}")


def stack_2012():
    Gs=np.zeros((HH,WW),np.float64);Gc=np.zeros((HH,WW),np.int32)
    for f in glob.glob(os.path.join(CACHE,"venus_*_2012*.npz")):
        d=np.load(f,allow_pickle=True); gm,m=d["gm"],d["mask"]; r0,r1,c0,c1=d["bbox"]
        hr,hc=r0//DS,c0//DS; h,w=gm.shape
        if hr+h>HH or hc+w>WW: h=min(h,HH-hr);w=min(w,WW-hc);gm=gm[:h,:w];m=m[:h,:w]
        Gs[hr:hr+h,hc:hc+w][m]+=gm[m]; Gc[hr:hr+h,hc:hc+w][m]+=1
    return np.divide(Gs,Gc,out=np.zeros_like(Gs),where=Gc>0).astype(np.float32),Gc>0


def render(G,m,title,fn,scale=None):
    r0=int((LAT0+90)/180*HH);r1=int((LAT1+90)/180*HH);c0=int((LON0+180)/360*WW);c1=int((LON1+180)/360*WW)
    crop=G[r0:r1,c0:c1].astype(float);cm=m[r0:r1,c0:c1]
    bg=gaussian_filter(np.where(cm,crop,0.),35)/np.maximum(gaussian_filter(cm.astype(float),35),1e-6)
    f=crop-bg; f[~cm]=np.nan
    if scale is None:
        v=f[np.isfinite(f)];scale=(np.percentile(v,2),np.percentile(v,98))
    fig,ax=plt.subplots(figsize=(10,5))
    ax.imshow(f,origin="lower",extent=[LON0,LON1,LAT0,LAT1],cmap="gray",vmin=scale[0],vmax=scale[1],aspect="auto")
    ax.set_title(title,fontsize=12);ax.set_xlabel("longitude (°E)");ax.set_ylabel("latitude (°N)")
    ax.grid(alpha=0.25,color="c",lw=0.4)
    fig.tight_layout();fig.savefig(fn,dpi=130);plt.close(fig)
    return scale


def main():
    from venera.projection import fit_diurnal_fo
    looks=looks2012(); print(f"2012: {len(looks)} looks",flush=True)
    # A. before stack (current cache)
    Gb,mb=stack_2012(); sc=render(Gb,mb,"2012 deep stack — BEFORE (per-look limb fit)",
                                  os.path.join(FIG,"session_2012_before.png"))
    print("wrote session_2012_before.png",flush=True)
    # B. fit diurnal model from a sample
    samp=[looks[i] for i in np.linspace(0,len(looks)-1,SAMPLE).astype(int)]
    with Pool(10,initializer=_init) as p:
        fos=p.map(wide_fo,samp)
    hrs=np.array([hourof(f) for f in samp])
    model,(a,b,h0)=fit_diurnal_fo(hrs,np.array(fos))
    print(f"diurnal model: fo = {a:+.1f} {b:+.1f}*(hour-{h0:.2f}) cols",flush=True)
    # C. reproject ALL looks with model -> overwrite cache
    t0=time.time(); tasks=[(f,model(hourof(f))) for f in looks]; done=0;err=0
    with Pool(12,initializer=_init) as p:
        for base,st in p.imap_unordered(reproj_cache,tasks):
            done+=1
            if not st.startswith("ok"): err+=1
            if done%40==0: print(f"  reprojected {done}/{len(tasks)} ({(time.time()-t0)/60:.1f}m, err={err})",flush=True)
    print(f"reprojected {done}, {err} errors, {(time.time()-t0)/60:.1f}m",flush=True)
    # D. after stack (matched scale)
    Ga,ma=stack_2012(); render(Ga,ma,"2012 deep stack — AFTER (diurnal-fo centering)",
                               os.path.join(FIG,"session_2012_after.png"),scale=sc)
    print("wrote session_2012_after.png",flush=True)


if __name__=="__main__":
    main()
