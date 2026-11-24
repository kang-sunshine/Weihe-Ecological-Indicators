from pathlib import Path
import os
os.environ['OMP_NUM_THREADS']='4';os.environ['OPENBLAS_NUM_THREADS']='1'
import json,time,gc
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from m3_extract import read,zscore,extract,O,D,M,M5

def main():
 ev=pd.read_parquet(O/'paired/CDH_Original_Clean_Cohort.parquet');dates=pd.DatetimeIndex(np.load(O/'cache/dates.npy'));td=np.load(O/'cache/time_days.npy');phase=np.minimum(46,((dates.dayofyear.to_numpy()-1)//8)+1);year=dates.year.to_numpy()
 # Four variants with exactly matched phase support; use small spatial blocks.
 out=np.lib.format.open_memmap(O/'cache/Strict_Sync_Event_Metrics.npy',mode='w+',dtype='float64',shape=(len(ev),4,10));counts=np.zeros(4,np.int64)
 l4=D/'L4_EcoFunction';raw=Dataset(l4/'Weihe_L4_EcoFunction_RAW_1000m_2000_2024_V3_1.nc');state=Dataset(l4/'Weihe_L4_EcoFunction_StateSpace_1000m_2000_2024_V3_1.nc');a=np.load(M/'cache/Acomp_by_row.npy',mmap_mode='r');on=np.load(M5/'cache/Catalogue_Onsets_Union.npy',mmap_mode='r');start=time.time()
 for y0 in range(0,414,6):
  y1=min(414,y0+6);idx=np.flatnonzero((ev.row>=y0)&(ev.row<y1))
  if not len(idx):continue
  zs=read(state,'ZSync',y0,y1);zu=read(state,'ZuWUE',y0,y1);zn=read(state,'ZNIRv',y0,y1);s=read(raw,'Sync',y0,y1);st=np.full_like(s,np.nan);st[3:]=s[:-3];zt=zscore(st,phase,year)
  common=np.isfinite(zs)&np.isfinite(zt);c1=np.where(common,-zs,np.nan);c2=np.where(common,-zt,np.nan)
  aa=np.transpose(a[y0:y1],(1,0,2)).reshape(1142,-1).copy();neg=[np.maximum(0,-np.clip(q,-10,10)) for q in [zt,zu,zn]];bb=np.sqrt((neg[0]**2+neg[1]**2+neg[2]**2)/np.float32(3));joint=np.isfinite(aa)&np.isfinite(bb)&common;aa[~joint]=np.nan;bb[~joint]=np.nan
  cube=np.stack([aa,bb,c1,c2])
  cols=(ev.col.to_numpy()[idx]+(ev.row.to_numpy()[idx]-y0)*703).astype(np.int32);r,_=extract(ev.onset_idx.to_numpy(np.int32)[idx],ev.term_idx.to_numpy(np.int32)[idx],cols,td,cube,np.transpose(on[y0:y1],(1,0,2)).reshape(1142,-1));out[idx]=r[:,:4];out.flush()
  counts+=np.array([np.isfinite(c1).sum(),np.isfinite(c2).sum(),np.isfinite(aa).sum(),np.isfinite(bb).sum()]);print('strict sync',y0,y1,len(idx),'sec',round(time.time()-start,1),flush=True)
  del zs,zu,zn,s,st,zt,common,c1,c2,aa,bb,neg,joint,cube,r;gc.collect()
 assert counts[0]==counts[1] and counts[2]==counts[3]
 (O/'cache/STRICT_SYNC_READY.json').write_text(json.dumps(dict(N_Events=len(ev),Variants=['Acomp_original_common_phase','Acomp_trailing_common_phase','Sync_centered_common_phase','Sync_trailing_common_phase'],Fields=['Baseline','Damage','Peak_Lag_Days','Peak_Value','Legacy120_Observed','Followup_Days','Outcome_Code','Valid_Followup','Peak_Phase','Pre_N_Valid'],Common_Phase_Counts=counts.tolist(),Support='paired indicators use exactly the same finite observation phases before baseline/peak extraction',Seconds=time.time()-start),indent=2),encoding='utf8')
 raw.close();state.close();print('STRICT SYNC READY',flush=True)
if __name__=='__main__':main()
