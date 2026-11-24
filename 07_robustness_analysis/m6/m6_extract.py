# -*- coding: utf-8 -*-
from pathlib import Path
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='2'
import json,time,gc,warnings
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from numba import njit
warnings.filterwarnings('ignore',category=RuntimeWarning)
O=Path(r'.\EI_Reviewer1_Major6_Revision');D=Path(r'.\Weihe database V3');M1=Path(r'.\EI_Reviewer1_Major1_Revision');M5=Path(r'.\EI_Reviewer1_Major5_Revision')
V=['ORIGINAL_NATIVE','ORIGINAL_LOYO_COMMON_PHASE','LOYO','ORIGINAL_EARLY_COMMON_PHASE','EARLY_2000_2010']
FIELDS=['Baseline','Damage','Peak_Lag_Days','Peak_Value','Legacy_Historical','Pre_N_Valid','Peak_Index','M5_Followup_Days','M5_Outcome','M5_Valid','M5_Observed_Legacy','Historical_Observed_Finite_Days']
def read(ds,v,y0,y1):return np.ma.filled(ds[v][:,y0:y1,:],np.nan).astype(np.float32).reshape(1142,-1)

@njit(cache=True)
def events(onset,term,col,td,cube,active,onsets):
 n=len(onset);nv=cube.shape[0];nt=len(td);out=np.full((n,nv,12),np.nan)
 for i in range(n):
  s=onset[i];t=term[i];c=col[i];p0=np.searchsorted(td,td[s]-40);pend=np.searchsorted(td,td[s]+56,side='right');oldend=np.searchsorted(td,td[s]+120,side='right');oldnext=-1;newnext=-1
  for k in range(t+1,oldend):
   if active[k,c]:oldnext=k;break
  oldstop=min(oldend,oldnext) if oldnext>=0 else oldend
  for k in range(t+1,nt):
   if td[k]-td[t]>120:break
   if onsets[k,c]:newnext=k;break
  for v in range(nv):
   bbuf=np.empty(s-p0,np.float32);nn=0
   for k in range(p0,s):
    a=cube[v,k,c]
    if np.isfinite(a):bbuf[nn]=a;nn+=1
   out[i,v,5]=nn
   if nn<3:continue
   z=np.sort(bbuf[:nn]);b=float(z[nn//2]) if nn%2 else float((z[nn//2-1]+z[nn//2])/2.)
   pk=-1;pv=-np.inf
   for k in range(s,pend):
    a=cube[v,k,c]
    if np.isfinite(a) and (pk<0 or a>pv):pk=k;pv=a
   out[i,v,0]=b
   if pk<0:continue
   out[i,v,1]=pv-b;out[i,v,2]=td[pk]-td[s];out[i,v,3]=pv;out[i,v,6]=pk
   # Historical L5.3 legacy: reproduce original active-phase next-event and
   # onset+120 horizon exactly, including its full interval weights.
   consec=0;rec=-1
   for k in range(pk+1,oldstop):
    a=cube[v,k,c]
    if not np.isfinite(a):consec=0;continue
    if a<=b:consec+=1
    else:consec=0
    if consec>=2:rec=k-1;break
   stop=rec if rec>=0 else oldstop;leg=0.;days=0
   for k in range(t+1,stop):
    a=cube[v,k,c]
    if not np.isfinite(a):continue
    dt=td[k+1]-td[k] if k+1<nt else 8
    if dt>0:leg+=max(0.,a-b)*dt;days+=dt
   if days>0:out[i,v,4]=leg;out[i,v,11]=days
   # Independent recovery sensitivity follows M5 PRIMARY, not the old legacy rule.
   cday=min(120,td[-1]-td[t]);reason=2 if cday==120 else 3
   if newnext>=0 and td[newnext]-td[t]<=cday:cday=td[newnext]-td[t];reason=1
   consecutive=0;first=-1;recday=-1;lastvalid=-1;lastscheduled=-1
   for k in range(t,nt):
    day=td[k]-td[t]
    if day>cday or (newnext>=0 and k>=newnext):break
    lastscheduled=day;a=cube[v,k,c]
    if not np.isfinite(a):consecutive=0;first=-1;continue
    lastvalid=day
    if k<=pk:continue
    if a<=b:
     if consecutive==0:first=day
     consecutive+=1
     if consecutive>=2:recday=first;break
    else:consecutive=0;first=-1
   dur=recday if recday>=0 else cday
   if recday>=0:reason=0
   elif lastvalid<lastscheduled:reason=4;dur=max(0,lastvalid)
   out[i,v,7]=dur;out[i,v,8]=reason;out[i,v,9]=int(lastvalid>=0)
   lg=0.;nd=0
   for k in range(t+1,nt):
    day=td[k]-td[t]
    if day>=dur:break
    a=cube[v,k,c]
    if not np.isfinite(a):continue
    dt=min(td[k+1]-td[k] if k+1<nt else 8,dur-day)
    if dt>0:lg+=max(0.,a-b)*dt;nd+=dt
   if nd>0:out[i,v,10]=lg
 return out

def zvariants(a,phase,year,domain,varidx,nh,nl,supp,sdrows,y0,stored_mu,stored_sd):
 zo=np.full_like(a,np.nan);zl=np.full_like(a,np.nan);ze=np.full_like(a,np.nan)
 for p in range(1,47):
  ai=np.flatnonzero(phase==p);bi=np.flatnonzero((phase==p)&(year<=2020));ei=np.flatnonzero((phase==p)&(year<=2010));b=a[bi];n=np.isfinite(b).sum(axis=0);mu32=stored_mu[p-1];sd32=stored_sd[p-1];ok=(n>=10)&np.isfinite(sd32)&(sd32>1e-6)&np.isfinite(mu32)
  orig=np.where(ok,(a[ai]-mu32)/sd32,np.nan).astype(np.float32);zo[ai]=orig;zl[ai]=orig
  be=a[ei];ne=np.isfinite(be).sum(axis=0);me=np.nanmean(be,axis=0,dtype=np.float64).astype(np.float32);se=np.nanstd(be,axis=0,ddof=1,dtype=np.float64).astype(np.float32);oke=(ne>=10)&np.isfinite(se)&(se>1e-6)&np.isfinite(me);ze[ai]=np.where(oke,(a[ai]-me)/se,np.nan).astype(np.float32)
  for j,t in enumerate(bi):
   bref=np.delete(b,j,axis=0);nm=np.isfinite(bref).sum(axis=0);ml=np.nanmean(bref,axis=0,dtype=np.float64).astype(np.float32);sdl=np.nanstd(bref,axis=0,ddof=1,dtype=np.float64).astype(np.float32);valid=(nm>=10)&np.isfinite(ml)&np.isfinite(sdl)&(sdl>1e-6);zl[t]=np.where(valid,(a[t]-ml)/sdl,np.nan).astype(np.float32)
   nl[varidx,year[t]-2000,p-1]+=np.bincount(nm[domain],minlength=22)
  nh[varidx,0,p-1]+=np.bincount(n[domain],minlength=22);nh[varidx,1,p-1]+=np.bincount(ne[domain],minlength=22)
  supp[varidx,0,p-1]+=np.array([domain.sum(),(domain&(n>=10)).sum(),(domain&ok).sum(),(domain&(n>=10)&~ok).sum()]);supp[varidx,1,p-1]+=np.array([domain.sum(),(domain&(ne>=10)).sum(),(domain&oke).sum(),(domain&(ne>=10)&~oke).sum()])
  # Fixed geographic subsampling for SD-ratio diagnostics, unrelated to effects.
  choose=domain&ok&oke&((np.arange(a.shape[1])+y0*703)%37==0)
  if choose.any():sdrows.append(np.column_stack([np.full(choose.sum(),varidx),np.full(choose.sum(),p),se[choose]/sd32[choose]]))
 return zo,zl,ze

def main():
 start=time.time();ev=pd.read_parquet(O/'paired/CDH_Fixed_Original_Clean_Events.parquet');N=len(ev);td=np.load(O/'cache/time_days.npy');phase=np.load(O/'cache/phase.npy');year=np.load(O/'cache/year.npy');raw=Dataset(D/'L4_EcoFunction/Weihe_L4_EcoFunction_RAW_1000m_2000_2024_V3_1.nc');base=Dataset(D/'L4_EcoFunction/Weihe_L4_EcoFunction_PhenologyBaseline_2000_2020_V3_1.nc');state=Dataset(D/'L4_EcoFunction/Weihe_L4_EcoFunction_StateSpace_1000m_2000_2024_V3_1.nc');domain=np.load(Path(r'.\EI_Reviewer1_Major4_Revision\cache\Climate_Valid_Mask.npy'),mmap_mode='r');aorig=np.load(M1/'cache/Acomp_by_row.npy',mmap_mode='r');active=np.load(M1/'cache/AnyMask_by_row.npy',mmap_mode='r');onsets=np.load(M5/'cache/Catalogue_Onsets_Union.npy',mmap_mode='r')
 out=np.lib.format.open_memmap(O/'cache/CDH_Event_Metrics.npy',mode='w+',dtype='float64',shape=(N,5,12));nh=np.zeros((3,2,46,22),np.int64);nl=np.zeros((3,21,46,22),np.int64);supp=np.zeros((3,2,46,4),np.int64);mom=np.zeros((2,25,4,2,6),np.float64);sdrows=[];qc=[]
 for y0 in range(0,414,6):
  y1=min(414,y0+6);dom=domain[y0:y1].reshape(-1);idx=np.flatnonzero((ev.row>=y0)&(ev.row<y1));Z=[];L=[];E=[]
  for k,var in enumerate(['Sync','uWUE','NIRv']):
   a=read(raw,var,y0,y1);mu=np.ma.filled(base[var+'_mean'][:,y0:y1,:],np.nan).astype(np.float32).reshape(46,-1);sd=np.ma.filled(base[var+'_std'][:,y0:y1,:],np.nan).astype(np.float32).reshape(46,-1);z,l,e=zvariants(a,phase,year,dom,k,nh,nl,supp,sdrows,y0,mu,sd);Z.append(z);L.append(l);E.append(e)
   if y0==0 or y0==198:
    original=read(state,['ZSync','ZuWUE','ZNIRv'][k],y0,y1);good=np.isfinite(original)&np.isfinite(z);mm=int((np.isfinite(original)!=np.isfinite(z)).sum());err=float(np.max(np.abs(original[good]-z[good]))) if good.any() else 0.;assert mm==0 and err<1e-5;(qc.append(dict(Check='original_Z_from_raw',Variable=var,Y0=y0,MaxAbs=err,MissingMismatch=mm)))
   assert np.array_equal(np.isfinite(z[year>=2021]),np.isfinite(l[year>=2021]));assert np.array_equal(z[year>=2021][np.isfinite(z[year>=2021])],l[year>=2021][np.isfinite(l[year>=2021])])
   del a
  def acomp(zz):
   neg=[np.maximum(0,-np.clip(q,-10,10)).astype(np.float32) for q in zz];r=np.sqrt((neg[0]**2+neg[1]**2+neg[2]**2)/np.float32(3));r[~(np.isfinite(zz[0])&np.isfinite(zz[1])&np.isfinite(zz[2]))]=np.nan;return r
  A=np.transpose(aorig[y0:y1],(1,0,2)).reshape(1142,-1).copy();Al=acomp(L);Ae=acomp(E);recalc=acomp(Z);assert np.array_equal(np.isfinite(recalc),np.isfinite(A)) and np.nanmax(np.abs(recalc-A))<1e-6
  commonL=np.isfinite(A)&np.isfinite(Al);commonE=np.isfinite(A)&np.isfinite(Ae)
  for pair,zz,aa,mask in [(0,L,Al,commonL),(1,E,Ae,commonE)]:
   for yr in range(2000 if pair==0 else 2011,2025):
    yi=np.flatnonzero(year==yr);valid=mask[yi]&dom[None,:]
    for k,(orig,new) in enumerate(list(zip(Z,zz))+[(A,aa)]):
     for v,src in enumerate([orig,new]):
      t=src[yi][valid].astype(np.float64)
      if len(t):mom[pair,yr-2000,k,v]+=np.array([len(t),t.sum(),np.dot(t,t),(np.abs(t)>10).sum(),(t==0).sum(),(np.abs(t)>4.5).sum()])
  cube=np.stack([A,np.where(commonL,A,np.nan),np.where(commonL,Al,np.nan),np.where(commonE,A,np.nan),np.where(commonE,Ae,np.nan)])
  if len(idx):
   cc=(ev.col.to_numpy()[idx]+(ev.row.to_numpy()[idx]-y0)*703).astype(np.int32);arr=events(ev.onset_idx.to_numpy(np.int32)[idx],ev.term_idx.to_numpy(np.int32)[idx],cc,td,cube,np.transpose(active[y0:y1],(1,0,2)).reshape(1142,-1),np.transpose(onsets[y0:y1],(1,0,2)).reshape(1142,-1));out[idx]=arr
  out.flush();np.savez_compressed(O/'cache/Extraction_Moments_Checkpoint.npz',N_Histogram=nh,N_LOYO_Histogram=nl,Support=supp,Indicator_Moments=mom);np.save(O/'cache/SD_Ratio_Samples.npy',np.vstack(sdrows) if sdrows else np.empty((0,3)));(O/'cache/Checkpoint.json').write_text(json.dumps(dict(Completed_Y1=y1,Seconds=time.time()-start)),encoding='utf8')
  print('M6 block',y0,y1,'events',len(idx),'sec',round(time.time()-start,1),flush=True)
  del Z,L,E,A,Al,Ae,recalc,commonL,commonE,cube,z,l,e,arr;gc.collect()
 source=ev.Acomp_legacy_anomaly_days.to_numpy();ours=out[:,0,4];same=np.isfinite(source)&np.isfinite(ours);miss=int((np.isfinite(source)!=np.isfinite(ours)).sum());err=float(np.max(np.abs(source[same]-ours[same])));dmgerr=float(np.nanmax(np.abs(out[:,0,1]-ev.Acomp_damage.to_numpy())));lagerr=float(np.nanmax(np.abs(out[:,0,2]-ev.Acomp_peak_lag_days.to_numpy())));print('legacy verification',miss,err,'damage',dmgerr,'lag',lagerr,flush=True)
 assert miss==0 and err<1e-3 and dmgerr<1e-6 and lagerr==0,(miss,err,dmgerr,lagerr)
 pd.DataFrame(qc).to_csv(O/'audit/Original_Z_Reproduction.csv',index=False);raw.close();base.close();state.close();(O/'cache/EXTRACTION_READY.json').write_text(json.dumps(dict(N_Events=N,Variants=V,Fields=FIELDS,Historical_Legacy_N=int(np.isfinite(ours).sum()),Historical_Legacy_MissingMismatch=miss,Historical_Legacy_MaxAbs=err,Damage_MaxAbs=dmgerr,Peak_Lag_MaxAbs=lagerr,LOYO_Post2020_Phase_Exact=True,Seconds=time.time()-start),indent=2),encoding='utf8');print('M6 EXTRACTION READY',flush=True)
if __name__=='__main__':main()
