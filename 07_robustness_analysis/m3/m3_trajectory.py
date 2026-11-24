import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='4'
from pathlib import Path
import time,json,gc
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from m3_extract import O,D,read,zscore

def main():
 assert (O/'cache/EXTRACTION_READY.json').exists();ev=pd.read_parquet(O/'paired/CDH_Original_Clean_Cohort.parquet');m=np.load(O/'cache/Event_Metrics.npy',mmap_mode='r');ok=np.flatnonzero(np.isfinite(m[:,7,0])&np.isfinite(m[:,8,0]));take=np.sort(np.random.default_rng(1701).choice(ok,min(20000,len(ok)),replace=False));sub=ev.iloc[take].copy();sub['Metric_Array_Index']=take;sub.to_parquet(O/'paired/uWUE_Trajectory_Sample.parquet',index=False)
 dates=pd.DatetimeIndex(np.load(O/'cache/dates.npy'));td=np.load(O/'cache/time_days.npy');phase=np.minimum(46,((dates.dayofyear.to_numpy()-1)//8)+1);year=dates.year.to_numpy();tr=np.zeros((2,21,4));start=time.time();ds=Dataset(D/'L4_EcoFunction/Weihe_L4_EcoFunction_RAW_1000m_2000_2024_V3_1.nc');zs=Dataset(D/'L4_EcoFunction/Weihe_L4_EcoFunction_Zscore_1000m_2000_2024_V3_1.nc')
 for y0 in range(0,414,12):
  y1=min(414,y0+12);ix=np.flatnonzero((sub.row>=y0)&(sub.row<y1))
  if not len(ix):continue
  g=read(ds,'GPP',y0,y1);et=read(ds,'ET',y0,y1);u=read(ds,'uWUE',y0,y1);zu=read(zs,'ZuWUE',y0,y1);w=np.where(np.isfinite(u)&np.isfinite(g)&np.isfinite(et)&(g>=.01)&(et>=.05),g/et,np.nan).astype(np.float32);zw=zscore(w,phase,year)
  s=sub.onset_idx.to_numpy(np.int32)[ix];c=(sub.col.to_numpy()[ix]+(sub.row.to_numpy()[ix]-y0)*703).astype(int);base=m[sub.Metric_Array_Index.to_numpy()[ix]][:,[7,8],0]
  for j,off in enumerate(range(-5,16)):
   t=s+off;good=(t>=0)&(t<1142);ids=np.flatnonzero(good);t=t[good];cc=c[good];aa=zu[t,cc]+base[ids,0];bb=zw[t,cc]+base[ids,1];good2=np.isfinite(aa)&np.isfinite(bb);aa=aa[good2];bb=bb[good2]
   for v,q in enumerate([aa,bb]):tr[v,j]+=np.array([len(q),q.sum(),np.dot(q,q),(td[t[good2]]-td[s[ids[good2]]]).sum()])
  print('trajectory sample',y0,y1,'N',len(ix),'sec',round(time.time()-start,1),flush=True);del g,et,u,zu,w,zw;gc.collect()
 np.save(O/'paired/uWUE_Trajectory_Aggregates.npy',tr);(O/'paired/uWUE_Trajectory_Scope.json').write_text(json.dumps(dict(N_Paired_Events=len(take),N_Eligible=len(ok),Selection='simple random without replacement from common finite pre-baseline CDH events',Seed=1701,Common_Finite_Per_Offset=True,Scope='all spatial rows and years sampled; not a spatially restricted extraction-resume subtotal'),indent=2),encoding='utf8');ds.close();zs.close();print('TRAJECTORY READY',flush=True)
if __name__=='__main__':main()
