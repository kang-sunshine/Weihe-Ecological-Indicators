# -*- coding: utf-8 -*-
import json,time,gc,warnings
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from numba import njit
from netCDF4 import Dataset
from core import *

def main():
 dirs()
 if (O/'qc/SYNC_READY.json').exists():print('Sync extraction already complete.');return
 resume=0
 if (O/'qc/cache/CDH_Trailing_Metrics.npy').exists():
  complete=[y for y in range(0,414,6) if all((O/f'outputs/event_details/{h}_SyncLeak_rows_{y:03d}.parquet').exists() for h in HAZ)]
  # A row is fully committed only after metrics flush/checkpoint, not merely sample writes.
  if (O/'qc/SYNC_CHECKPOINT.json').exists():resume=json.loads((O/'qc/SYNC_CHECKPOINT.json').read_text())['Next_Row']
  else:
   # First interrupted run printed156..162 after both metric arrays were flushed.
   resume=162 if complete==list(range(0,162,6)) else 0
   assert resume>0,'Ambiguous initial interrupted checkpoint; inspect before resume'
 warnings.filterwarnings('ignore',category=RuntimeWarning)
 # Execute only extracted pure functions, not the historical modules or producers.
 fn=ast_function(MIN/'code/minor2_core.py','event_metrics');metrics=njit(cache=False)(fn);zscore=ast_function(M3/'code/m3_extract.py','zscore')
 td=np.load(M5/'cache/time_days.npy');dates=np.load(M5/'cache/dates.npy');di=pd.DatetimeIndex(dates);phase=np.minimum(46,(di.dayofyear.to_numpy()-1)//8+1);year=di.year.to_numpy()
 e=events('CDH');mode='r+' if resume else 'w+';tr=np.lib.format.open_memmap(O/'qc/cache/CDH_Trailing_Metrics.npy',mode=mode,dtype='float32',shape=(len(e),18));common=np.lib.format.open_memmap(O/'qc/cache/CDH_CommonPhase_Metrics.npy',mode=mode,dtype='float32',shape=(len(e),2,18))
 catalogs={}
 for h in HAZ:
  d=pd.concat([pd.read_parquet(p,columns=['Event_Num','row','col','onset'],filters=[('row','>=',resume)],use_threads=False) for p in sorted((M2/'catalogues'/h).glob('*.parquet'))],ignore_index=True)
  catalogs[h]=d
 A=np.load(M1/'cache/Acomp_by_row.npy',mmap_mode='r');ons=np.load(M5/'cache/Catalogue_Onsets_Union.npy',mmap_mode='r');act=np.load(M1/'cache/AnyMask_by_row.npy',mmap_mode='r');grid=np.load(M5/'cache/CDH_pixel_blocks_50.npy');assert all(np.array_equal(grid,np.load(M5/'cache'/f'{h}_pixel_blocks_50.npy')) for h in HAZ)
 raw=Dataset(DB/'L4_EcoFunction/Weihe_L4_EcoFunction_RAW_1000m_2000_2024_V3_1.nc');state=Dataset(DB/'L4_EcoFunction/Weihe_L4_EcoFunction_StateSpace_1000m_2000_2024_V3_1.nc')
 for ds,names in [(raw,['Sync','GPP','ET']),(state,['ZSync','ZuWUE','ZNIRv'])]:
  for name in names:ds[name].set_var_chunk_cache(4*1024*1024,1009,.75)
 def read(ds,name,y0,y1):return np.ma.filled(ds[name][:,y0:y1,:],np.nan).astype(np.float32).reshape(1142,-1)
 gates=[];start=time.time();nunknown=0
 oldids=pd.read_parquet(M3/'paired/CDH_Original_Clean_Cohort.parquet',columns=['event_num'],use_threads=False);lookup=np.full(int(oldids.event_num.max())+1,-1,np.int32);lookup[oldids.event_num.to_numpy(int)]=np.arange(len(oldids),dtype=np.int32);del oldids
 for y0 in range(resume,414,6):
  y1=min(414,y0+6);ac=np.transpose(A[y0:y1],(1,0,2)).reshape(1142,-1).copy();zs=read(state,'ZSync',y0,y1);g=read(raw,'GPP',y0,y1);et=read(raw,'ET',y0,y1);pair=np.isfinite(g)&np.isfinite(et);del g,et
  for h,d in catalogs.items():
   ix=np.flatnonzero((d.row.to_numpy()>=y0)&(d.row.to_numpy()<y1));sub=d.iloc[ix];cc=sub.col.to_numpy(np.int32)+(sub.row.to_numpy(np.int32)-y0)*703;v=leakage(sub.onset.to_numpy(np.int32),cc,td,zs,pair,ac)
   out=sub.copy();out['Hazard']=h;out['Pixel_ID']=sub.row.to_numpy(np.int32)*703+sub.col.to_numpy(np.int32);out['Block50']=grid[sub.row,sub.col];nunknown+=int((out.Block50<0).sum())
   for j,name in enumerate(['N_Pre_Sync','N_Structural_Future','N_Used_Future_Pairs','N_Safe_Sync','Original_ZSync_Pre','Safe_ZSync_Pre','N_Pre_Acomp','N_Safe_Acomp','Original_Acomp_Pre','Safe_Acomp_Pre']):out[name]=v[:,j]
   out.to_parquet(O/f'outputs/event_details/{h}_SyncLeak_rows_{y0:03d}.parquet',index=False,compression='zstd')
  del pair
  ix=np.flatnonzero((e.row.to_numpy()>=y0)&(e.row.to_numpy()<y1));cc=e.col.to_numpy(np.int32)[ix]+(e.row.to_numpy(np.int32)[ix]-y0)*703;ss=e.onset_idx.to_numpy(np.int32)[ix];tt=e.term_idx.to_numpy(np.int32)[ix]
  sync=read(raw,'Sync',y0,y1);traw=np.full_like(sync,np.nan);traw[3:]=sync[:-3];zt=zscore(traw,phase,year);zu=read(state,'ZuWUE',y0,y1);zn=read(state,'ZNIRv',y0,y1)
  neg=[np.maximum(0,-np.clip(v,-10,10)) for v in [zt,zu,zn]];trailing=np.sqrt((neg[0]**2+neg[1]**2+neg[2]**2)/np.float32(3));on=np.transpose(ons[y0:y1],(1,0,2)).reshape(1142,-1);active=np.transpose(act[y0:y1],(1,0,2)).reshape(1142,-1)
  tr[ix]=metrics(ss,tt,cc,td,trailing[None,:,:],on,active)[:,0]
  joint=np.isfinite(ac)&np.isfinite(trailing);native=np.where(joint,ac,np.nan);alt=np.where(joint,trailing,np.nan);common[ix]=metrics(ss,tt,cc,td,np.stack([native,alt]),on,active)
  # Match original M3 strict baseline/peak fields on exactly the same phase support.
  oldidx=lookup[e.event_num.to_numpy()[ix]];old=np.load(M3/'cache/Strict_Sync_Event_Metrics.npy',mmap_mode='r')
  for variant in range(2):
   for newfield,oldfield,label in [(0,0,'Baseline'),(1,1,'Damage'),(2,2,'PeakLag')]:
    a=np.asarray(common[ix,variant,newfield]);b=np.asarray(old[oldidx,variant,oldfield]);finite=np.isfinite(a)&np.isfinite(b);mm=int((np.isfinite(a)!=np.isfinite(b)).sum());err=float(np.max(abs(a[finite]-b[finite]))) if finite.any() else 0.;gates.append(dict(RowStart=y0,Variant=variant,Field=label,MissingMismatch=mm,MaxAbs=err,Pass=mm==0 and err<2e-6))
  tr.flush();common.flush();print('SYNC extraction',y0,y1,'sec',round(time.time()-start,1),flush=True)
  (O/'qc/SYNC_CHECKPOINT.json').write_text(json.dumps(dict(Next_Row=y1)),encoding='utf8')
  del ac,zs,sync,traw,zt,zu,zn,neg,trailing,on,active,joint,native,alt,old;gc.collect()
 # Recheck all previously committed blocks too; do not trust only resumed rows.
 raw.close();state.close();gates=[];old=np.load(M3/'cache/Strict_Sync_Event_Metrics.npy',mmap_mode='r')
 for y0 in range(0,414,6):
  ix=np.flatnonzero((e.row.to_numpy()>=y0)&(e.row.to_numpy()<min(414,y0+6)));oldidx=lookup[e.event_num.to_numpy()[ix]]
  for variant in range(2):
   for newfield,oldfield,label in [(0,0,'Baseline'),(1,1,'Damage'),(2,2,'PeakLag')]:
    a=np.asarray(common[ix,variant,newfield]);b=np.asarray(old[oldidx,variant,oldfield]);finite=np.isfinite(a)&np.isfinite(b);mm=int((np.isfinite(a)!=np.isfinite(b)).sum());err=float(np.max(abs(a[finite]-b[finite]))) if finite.any() else 0;gates.append(dict(RowStart=y0,Variant=variant,Field=label,MissingMismatch=mm,MaxAbs=err,Pass=mm==0 and err<2e-6))
 pd.DataFrame(gates).to_csv(O/'qc/Trailing_M3_Baseline_Peak_Bridge.csv',index=False)
 assert all(g['Pass'] for g in gates),'STOP: trailing version conflict'
 counts={h:sum(pq.ParquetFile(p).metadata.num_rows for p in (O/'outputs/event_details').glob(h+'_SyncLeak_rows_*.parquet')) for h in HAZ}
 (O/'qc/SYNC_READY.json').write_text(json.dumps(dict(Completed=True,A1_Full_Catalogue_Counts=counts,A2_CDH_Clean=len(e),A1_Unknown_Block_Records_Resumed_Portion=nunknown,Reference='trailing shifted rawSync then independent2000-2020 phase standardization, same n>=10/ddof1 rules',M3_CommonPhase_Baseline_Peak_Reproduced=True,Recovery='Minor2 exact M5 rule without M3 additional positive-damage gate',Seconds=time.time()-start)),encoding='utf8')
 print('SYNC EXTRACTION READY',time.time()-start,flush=True)
if __name__=='__main__':main()
