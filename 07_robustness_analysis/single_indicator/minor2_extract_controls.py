# -*- coding: utf-8 -*-
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='2'
import json,time,gc,warnings
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from minor2_core import R,O,M1,M5,F,event_metrics,read_block
warnings.filterwarnings('ignore',category=RuntimeWarning)
def z_gpp(a,phase,year):
 z=np.full_like(a,np.nan);means=np.full((46,a.shape[1]),np.nan,np.float32);sds=means.copy();counts=np.zeros((46,a.shape[1]),np.uint8)
 for p in range(1,47):
  ii=np.flatnonzero(phase==p);b=a[(phase==p)&(year<=2020)];n=np.isfinite(b).sum(axis=0);mu=np.nanmean(b,axis=0,dtype=np.float64).astype(np.float32);sd=np.nanstd(b,axis=0,ddof=1,dtype=np.float64).astype(np.float32);ok=(n>=10)&np.isfinite(mu)&np.isfinite(sd)&(sd>1e-6);z[ii]=np.where(ok,(a[ii]-mu)/sd,np.nan).astype(np.float32);means[p-1]=mu;sds[p-1]=sd;counts[p-1]=n
 return z,means,sds,counts
def main():
 assert json.loads((O/'cache/ACOMP_GATE.json').read_text())['Pass'];start=time.time();td=np.load(O/'cache/time_days.npy');phase=np.load(O/'cache/phase.npy');year=np.load(O/'cache/year.npy');ev={};out={};names=['NIRv','GPP','Acomp_CommonPhase','NIRv_CommonPhase','GPP_CommonPhase']
 for h in ['CDH','HEAT','HR']:
  e=pd.read_parquet(O/'paired'/f'{h}_Fixed_Clean_Events.parquet',columns=['event_num','row','col','onset_idx','term_idx']).astype({'event_num':'int32','row':'int16','col':'int16','onset_idx':'int16','term_idx':'int16'});ev[h]=e;out[h]={}
  for n in names:
   path=O/'cache'/h/(n+'.npy');mm=np.lib.format.open_memmap(path,mode='w+',dtype='float32',shape=(len(e),len(F)));mm.flush();mm._mmap.close();out[h][n]=path
 muout=np.lib.format.open_memmap(O/'cache/GPP_Baseline_Mean.npy',mode='w+',dtype='float32',shape=(46,414,703));sdout=np.lib.format.open_memmap(O/'cache/GPP_Baseline_SD.npy',mode='w+',dtype='float32',shape=(46,414,703));nout=np.lib.format.open_memmap(O/'cache/GPP_Baseline_N.npy',mode='w+',dtype='uint8',shape=(46,414,703));A=np.load(M1/'cache/Acomp_by_row.npy',mmap_mode='r');on=np.load(M5/'cache/Catalogue_Onsets_Union.npy',mmap_mode='r');active=np.load(M1/'cache/AnyMask_by_row.npy',mmap_mode='r');phasecounts=[];support=[]
 with Dataset(R/'Weihe database V3/L4_EcoFunction/Weihe_L4_EcoFunction_RAW_1000m_2000_2024_V3_1.nc') as raw,Dataset(R/'Weihe database V3/L4_EcoFunction/Weihe_L4_EcoFunction_StateSpace_1000m_2000_2024_V3_1.nc') as state:
  for name in ['GPP','ET','uWUE']:raw[name].set_var_chunk_cache(4*1024*1024,1009,.75)
  state['ZNIRv'].set_var_chunk_cache(4*1024*1024,1009,.75)
  for y0 in range(0,414,6):
   y1=min(414,y0+6);g=read_block(raw,'GPP',y0,y1);zg,mu,sd,nn=z_gpp(g,phase,year);muout[:,y0:y1]=mu.reshape(46,y1-y0,703);sdout[:,y0:y1]=sd.reshape(46,y1-y0,703);nout[:,y0:y1]=nn.reshape(46,y1-y0,703);zn=read_block(state,'ZNIRv',y0,y1);an=np.maximum(0,-np.clip(zn,-10,10)).astype(np.float32);ag=np.maximum(0,-np.clip(zg,-10,10)).astype(np.float32);ac=np.transpose(A[y0:y1],(1,0,2)).reshape(1142,-1).copy();joint=np.isfinite(ac)&np.isfinite(an)&np.isfinite(ag);et=read_block(raw,'ET',y0,y1);uw=read_block(raw,'uWUE',y0,y1)
   support.append(dict(Y0=y0,Y1=y1,N_RawGPP_Finite=int(np.isfinite(g).sum()),N_RawGPP_Negative=int((np.isfinite(g)&(g<0)).sum()),N_RawGPP_Zero=int((np.isfinite(g)&(g==0)).sum()),N_ZGPP_Finite=int(np.isfinite(zg).sum()),N_ZGPP_Finite_When_ET_Missing=int((np.isfinite(zg)&~np.isfinite(et)).sum()),N_ZGPP_Finite_When_ET_Below0p05=int((np.isfinite(zg)&np.isfinite(et)&(et<.05)).sum()),N_ZGPP_Finite_When_uWUE_Invalid=int((np.isfinite(zg)&~np.isfinite(uw)).sum()),N_Acomp_Finite=int(np.isfinite(ac).sum()),N_NIRv_Finite=int(np.isfinite(an).sum()),N_Common_Phase=int(joint.sum())))
   for p in range(46):
    counts=np.bincount(nn[p],minlength=22)
    for n,v in enumerate(counts):
     if v:phasecounts.append(dict(Phase=p+1,N_Reference_Observations=n,N_Pixels=int(v)))
   cube=np.stack([an,ag,np.where(joint,ac,np.nan),np.where(joint,an,np.nan),np.where(joint,ag,np.nan)]);ons=np.transpose(on[y0:y1],(1,0,2)).reshape(1142,-1);act=np.transpose(active[y0:y1],(1,0,2)).reshape(1142,-1)
   for h,e in ev.items():
    ix=np.flatnonzero((e.row.to_numpy()>=y0)&(e.row.to_numpy()<y1))
    if not len(ix):continue
    ss=e.onset_idx.to_numpy(np.int32)[ix];tt=e.term_idx.to_numpy(np.int32)[ix];cc=e.col.to_numpy(np.int32)[ix]+(e.row.to_numpy(np.int32)[ix]-y0)*703;arr=event_metrics(ss,tt,cc,td,cube,ons,act)
    for k,n in enumerate(names):
     mm=np.load(out[h][n],mmap_mode='r+');mm[ix]=arr[:,k];mm.flush();mm._mmap.close();del mm
   muout.flush();sdout.flush();nout.flush()
   if y0%24==0:print('CONTROL rows',y0,y1,'seconds',round(time.time()-start,1),flush=True)
   del g,zg,mu,sd,nn,zn,an,ag,ac,joint,et,uw,cube,ons,act,arr;gc.collect()
 muout.flush();sdout.flush();nout.flush();pd.DataFrame(support).to_csv(O/'audit/GPP_Independent_Support.csv',index=False);pd.DataFrame(phasecounts).groupby(['Phase','N_Reference_Observations'],as_index=False).N_Pixels.sum().to_csv(O/'supplement/GPP_Reference_Valid_Counts.csv',index=False)
 # Independent accepted M3Acomp observed-legacy check on its positive cohort.
 h='CDH';e=ev[h];m=np.load(O/'cache/CDH/Acomp.npy',mmap_mode='r');oldE=pd.read_parquet(R/'EI_Reviewer1_Major3_Revision/paired/CDH_Original_Clean_Cohort.parquet',columns=['event_num']);oldM=np.load(R/'EI_Reviewer1_Major3_Revision/cache/Event_Metrics.npy',mmap_mode='r');lookup=np.full(4796212,-1,np.int32);lookup[oldE.event_num.to_numpy(int)]=np.arange(len(oldE));ix=lookup[e.event_num.to_numpy(int)];assert (ix>=0).all();choose=(oldM[ix,0,1]>0)&np.isfinite(oldM[ix,0,5]);old=oldM[ix[choose],0,4];new=m[choose,12];good=np.isfinite(old)&np.isfinite(new);mm=int((np.isfinite(old)!=np.isfinite(new)).sum());err=float(np.max(abs(old[good]-new[good])));record=dict(N_Positive_M3_Cohort=int(choose.sum()),N_Finite_Observed_Legacy=int(good.sum()),MissingMismatch=mm,MaxAbs=err,Pass=mm==0 and err<2e-4);(O/'audit/M3_Acomp_Observed_Legacy_Reproduction.json').write_text(json.dumps(record,indent=2),encoding='utf8');assert record['Pass'],record
 (O/'cache/CONTROLS_READY.json').write_text(json.dumps(dict(Indicators=['Acomp','NIRv','GPP'],Extra_Strict_Phase=True,N_Hazards=3,GPP_No_ET_Gate=True,Seconds=time.time()-start),indent=2),encoding='utf8');print('CONTROLS READY',round(time.time()-start,1),flush=True)
if __name__=='__main__':main()
