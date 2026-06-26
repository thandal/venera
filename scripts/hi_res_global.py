"""High-resolution (4x, 8000x16000 ~2.4 km/px) period-aligned global stack of ALL 916
looks. Projects each look at the data's near-native resolution (strip-processed,
memory-bounded), applies the full corrected pipeline (topocentric observer,
label-delay, bistatic freq_scale, 2012 diurnal-fo), longitude-shifts to the adopted
period, and accumulates a mean. Chunked across processes (each saves a partial sum) to
bound memory and parallelize.

Usage: .conda/bin/python scripts/hi_res_global.py [n_proc]
"""
import os, sys, glob, time
import numpy as np
from multiprocessing import Process
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA=("/home/than/code/venera/arecibo_radar/pds-geosciences.wustl.edu/venus/"
      "arcb_nrao-v-rtls_gbt-3-delaydoppler-v1/vrm_90xx/data/")
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG=os.path.join(ROOT,"results","figures"); SST=os.path.join(ROOT,"results","session_stacks")
TMP="/tmp/claude-1000/-home-than-code-venera/8ba753c0-1b68-49ca-aef5-34440f177b6a/scratchpad"
os.makedirs(TMP,exist_ok=True)
NPROC=int(sys.argv[1]) if len(sys.argv)>1 else 6
H=int(sys.argv[2]) if len(sys.argv)>2 else 8000      # map rows (lat); W=2H (lon)
W=2*H; NLON=int(H*1.3); NLAT=int(H*0.7)               # mesh density ~ map density
CKPT=40                                               # checkpoint every N looks/worker
P_ADOPT=243.0216; P_ASSUMED=243.0185   # image build period; §6 science value 243.0206 (sub-px difference)
DWDOT=(-360.0/P_ADOPT)-(-360.0/P_ASSUMED)
# 2012 diurnal-fo model (from reproject_2012_diurnal): fo = A + B*(hour - H0)
FO_A, FO_B, FO_H0 = 5.8, -8.5, 17.38


def hourof(p):
    b=os.path.basename(p); return int(b.split("_")[3][:2])+int(b.split("_")[3][2:4])/60.+int(b.split("_")[3][4:6])/3600.


def worker(files, day_ref, idx):
    from venera import spice_setup; spice_setup.furnsh_kernels()
    from venera.geometry import Spin
    from venera.projection import project_file
    from venera.data import parse_lbl
    import cspyce as csp
    Ssum=np.zeros((H,W),np.float32); Scnt=np.zeros((H,W),np.int32)
    t0=time.time()
    for i,f in enumerate(files):
        try:
            lbl=parse_lbl(f); bistatic=(lbl.get("RX_STATION")=="GBT")
            G=np.zeros((H,W),np.float32); Gc=np.zeros((H,W),np.int32)
            kw=dict(n_lon=NLON,n_lat=NLAT)
            if bistatic:
                kw.update(fo_center=FO_A+FO_B*(hourof(f)-FO_H0), fo_window=0)
            info=project_file(f,Spin(),G,Gc,**kw)
            day=info["et_mid"]/86400.0
            sft=int(round((-(DWDOT)*(day-day_ref))/360.0*W))
            if sft:
                G=np.roll(G,sft,axis=1); Gc=np.roll(Gc,sft,axis=1)
            mk=Gc>0; Ssum[mk]+=G[mk]; Scnt[mk]+=1
        except Exception as e:
            print(f"[w{idx}] ERR {os.path.basename(f)}: {e}",flush=True)
        if i%20==0: print(f"[w{idx}] {i+1}/{len(files)} ({(time.time()-t0)/60:.1f}m)",flush=True)
        if i and i%CKPT==0:    # checkpoint: overwrite partials so a suspend/crash keeps progress
            np.save(f"{TMP}/hi_sum_{idx}.npy",Ssum); np.save(f"{TMP}/hi_cnt_{idx}.npy",Scnt)
    np.save(f"{TMP}/hi_sum_{idx}.npy",Ssum); np.save(f"{TMP}/hi_cnt_{idx}.npy",Scnt)
    print(f"[w{idx}] done {len(files)} looks {(time.time()-t0)/60:.1f}m",flush=True)


def main():
    nproc=NPROC
    kmpx=(180.0/H)*111.0  # lat km/px
    fs=sorted(glob.glob(DATA+"venus_scp_*.img")+glob.glob(DATA+"venus_ocp_*.img"))
    import cspyce as csp
    from venera import spice_setup; spice_setup.furnsh_kernels()
    # day_ref from label mid-times (cheap)
    from venera.data import parse_lbl
    mids=[]
    for f in fs:
        l=parse_lbl(f); mids.append(0.5*(csp.str2et(l["START_TIME"])+csp.str2et(l["STOP_TIME"]))/86400.0)
    day_ref=float(np.mean(mids))
    print(f"{len(fs)} looks, {nproc} procs, grid {H}x{W} (~{kmpx:.1f}km/px), mesh {NLON}x{NLAT}, "
          f"period {P_ADOPT}, day_ref {day_ref:.0f}",flush=True)
    chunks=[fs[i::nproc] for i in range(nproc)]
    procs=[Process(target=worker,args=(chunks[i],day_ref,i)) for i in range(nproc)]
    for p in procs: p.start()
    for p in procs: p.join()
    Ssum=np.zeros((H,W),np.float64); Scnt=np.zeros((H,W),np.int64)
    for i in range(nproc):
        Ssum+=np.load(f"{TMP}/hi_sum_{i}.npy"); Scnt+=np.load(f"{TMP}/hi_cnt_{i}.npy")
    Gm=np.divide(Ssum,Scnt,out=np.zeros_like(Ssum),where=Scnt>0).astype(np.float32); M=Scnt>0
    np.savez_compressed(f"{SST}/global_fullres.npz",Gm=Gm,mask=M,period=P_ADOPT,res_km=1.2)
    print(f"combined: {int(M.sum())} px covered; saved global_fullres.npz",flush=True)
    render(Gm,M)
    for i in range(nproc):
        os.remove(f"{TMP}/hi_sum_{i}.npy"); os.remove(f"{TMP}/hi_cnt_{i}.npy")


def render(Gm,M):
    from scipy.ndimage import gaussian_filter
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    cols=np.where(M.any(0))[0]; c0,c1=cols[0],cols[-1]
    lon0,lon1=c0/W*360-180,c1/W*360-180; LAT0,LAT1=-85,83
    r0=int((LAT0+90)/180*H);r1=int((LAT1+90)/180*H)
    g=Gm[r0:r1,c0:c1].astype(float);cm=M[r0:r1,c0:c1]
    bgpx=int(H/180*1.5)   # ~1.5deg high-pass background, resolution-aware
    f=g-gaussian_filter(np.where(cm,g,0.),bgpx)/np.maximum(gaussian_filter(cm.astype(float),bgpx),1e-6); f[~cm]=np.nan
    v=f[np.isfinite(f)];lo,hi=np.percentile(v,2),np.percentile(v,99)
    kmpx=(180.0/H)*111.0
    fig,ax=plt.subplots(figsize=(32,18))
    ax.imshow(f,origin="lower",extent=[lon0,lon1,LAT0,LAT1],cmap="gray",vmin=lo,vmax=hi,aspect="auto",interpolation="nearest")
    ax.set_title(f"Venus global radar stack — 916 looks, both hemispheres, period-aligned, ~{kmpx:.1f} km/px",fontsize=14)
    ax.set_xlabel("longitude (°E)");ax.set_ylabel("latitude (°N)")
    fig.tight_layout();fig.savefig(f"{FIG}/global_fullres.png",dpi=300);plt.close(fig)
    print("wrote global_fullres.png",flush=True)


if __name__=="__main__":
    main()
