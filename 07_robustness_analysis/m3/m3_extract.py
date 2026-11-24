# Scientific paired sensitivity; writes only M3. No source module is imported.
from pathlib import Path
import os
os.environ['OMP_NUM_THREADS']='4';os.environ['OPENBLAS_NUM_THREADS']='1'
import ast,json,time,gc,warnings
import numpy as np
import pandas as pd
from netCDF4 import Dataset,num2date
from numba import njit
warnings.filterwarnings('ignore',category=RuntimeWarning)
O=Path(r'.\EI_Reviewer1_Major3_Revision');D=Path(r'.\Weihe database V3');M=Path(r'.\EI_Reviewer1_Major1_Revision');M5=Path(r'.\EI_Reviewer1_Major5_Revision')
V=['Acomp_original','Acomp_NIRv_free','NIRv_only','GPP_only','Acomp_trailingSync','Sync_signed','Sync_trailing_signed','uWUE_signed','GPP_ET_signed','Sync_only','uWUE_only']
FIELDS=['Baseline','Damage','Peak_Lag_Days','Peak_Value','Legacy120_Observed','Followup_Days','Outcome_Code','Valid_Followup','Peak_Phase','Pre_N_Valid']

def read(ds,name,y0,y1):return np.ma.filled(ds[name][:,y0:y1,:],np.nan).astype(np.float32).reshape(1142,-1)
def zscore(a,phase,year):
 out=np.full(a.shape,np.nan,np.float32)
 for p in range(1,47):
  idx=np.flatnonzero(phase==p);b=a[(phase==p)&(year<=2020)];n=np.isfinite(b).sum(axis=0)
  mu=np.nanmean(b,axis=0,dtype=np.float64).astype(np.float32);sd=np.nanstd(b,axis=0,ddof=1,dtype=np.float64).astype(np.float32)
  ok=(n>=10)&np.isfinite(mu)&np.isfinite(sd)&(sd>1e-6)
  out[idx]=np.where(ok,(a[idx]-mu)/sd,np.nan).astype(np.float32)
 return out

@njit(cache=True)
def extract(onset,term,cols,td,cube,onsets):
 ne=len(onset);nv=cube.shape[0];nt=len(td);res=np.full((ne,nv,10),np.nan); diag=np.zeros((ne,4),np.int16)
 for e in range(ne):
  s=onset[e];t=term[e];c=cols[e];pre=np.searchsorted(td,td[s]-40);end=np.searchsorted(td,td[s]+56,side='right')
  nextday=-1
  for k in range(t+1,nt):
   dd=td[k]-td[t]
   if dd>120:break
   if onsets[k,c]:nextday=dd;break
  diag[e,0]=nextday
  for v in range(nv):
   buf=np.empty(s-pre,np.float32);n=0
   for k in range(pre,s):
    if np.isfinite(cube[v,k,c]):buf[n]=cube[v,k,c];n+=1
   res[e,v,9]=n
   if n<3:continue
   z=np.sort(buf[:n]);b=float(z[n//2]) if n%2 else float((z[n//2-1]+z[n//2])/2.0)
   res[e,v,0]=b;pk=-1;pv=-np.inf
   for k in range(s,end):
    a=cube[v,k,c]
    if np.isfinite(a) and (pk<0 or a>pv):pv=a;pk=k
   if pk<0:continue
   res[e,v,1]=pv-b;res[e,v,2]=td[pk]-td[s];res[e,v,3]=pv;res[e,v,8]=pk
   # M5 primary: termination origin, post-peak, catalogue onset interruption,
   # inclusive120-day administrative horizon; two consecutive observations.
   if pv<=b:continue
   cday=min(120,td[-1]-td[t]);reason=2 if cday==120 else 3
   if nextday>=0 and nextday<=cday:cday=nextday;reason=1
   rec=-1;confirm=-1;consec=0;first=-1;lastvalid=-1;lastscheduled=-1
   for k in range(t,nt):
    day=td[k]-td[t]
    if day>cday or (nextday>=0 and day>=nextday):break
    lastscheduled=day;value=cube[v,k,c]
    if not np.isfinite(value):consec=0;first=-1;continue
    lastvalid=day
    if k<=pk:continue
    if value<=b:
     if consec==0:first=day
     consec+=1
     if consec>=2:rec=first;confirm=day;break
    else:consec=0;first=-1
   dur=rec if rec>=0 else cday
   if rec>=0:reason=0
   elif lastvalid<lastscheduled:reason=4;dur=max(0,lastvalid)
   res[e,v,5]=dur;res[e,v,6]=reason;res[e,v,7]=int(lastvalid>=0)
   # Common definition for all ablations: observed post-termination positive
   # excess through recovery/interruption/120d. Partial final cells are clipped.
   stop=rec if rec>=0 else dur;legacy=0.;nleg=0
   for k in range(t+1,nt):
    day=td[k]-td[t]
    if day>=stop:break
    a=cube[v,k,c]
    if not np.isfinite(a):continue
    dt=td[k+1]-td[k] if k+1<nt else 8
    dt=min(dt,stop-day)
    if dt>0:legacy+=max(0.,a-b)*dt;nleg+=1
   if nleg>0:res[e,v,4]=legacy
  # Component opposition at original Acomp peak, original baseline differences.
  pk=int(res[e,0,8]) if np.isfinite(res[e,0,8]) else -1
  if pk>=0 and nv>=8:
   nup=0;ndown=0
   for v in [5,7]:
    if np.isfinite(res[e,v,0]):
     dd=cube[v,pk,c]-res[e,v,0]
     if dd>0:nup+=1
     elif dd<0:ndown+=1
   # NIRv-only here uses negative capped Z; test signed NIRv with same cube2.
   if np.isfinite(res[e,2,0]):
    dd=cube[2,pk,c]-res[e,2,0]
    if dd>0:nup+=1
    elif dd<0:ndown+=1
   diag[e,1]=int(nup>0 and ndown>0)
   diag[e,2]=nup;diag[e,3]=ndown
 return res,diag

@njit(cache=True)
def log_terms(onset,cols,td,g,et,vpd,peak):
 out=np.full((len(onset),4),np.nan)
 for e in range(len(onset)):
  s=onset[e];c=cols[e];pk=peak[e]
  if pk<0:continue
  count=0;bg=0.;be=0.;bv=0.
  for k in range(np.searchsorted(td,td[s]-40),s):
   if g[k,c]>0 and et[k,c]>0 and vpd[k,c]>0 and np.isfinite(g[k,c]+et[k,c]+vpd[k,c]):
    bg+=np.log(g[k,c]);be+=np.log(et[k,c]);bv+=np.log(vpd[k,c]);count+=1
  if count<3:continue
  if not (g[pk,c]>0 and et[pk,c]>0 and vpd[pk,c]>0 and np.isfinite(g[pk,c]+et[pk,c]+vpd[pk,c])):continue
  dg=np.log(g[pk,c])-bg/count;de=-(np.log(et[pk,c])-be/count);dv=.5*(np.log(vpd[pk,c])-bv/count)
  out[e,0]=dg;out[e,1]=de;out[e,2]=dv;out[e,3]=dg+de+dv
 return out

def main():
 start=time.time()
 use=['row','col','climate_cumulative_severity','Acomp_pre','Acomp_damage','Acomp_peak_lag_days','Acomp_legacy_anomaly_days','valid_resistance_sample','onset_idx','term_idx','event_num']
 files=sorted((M/'cache/CDH').glob('clean_*.parquet'));assert files
 ev=pd.concat([pd.read_parquet(p,columns=use) for p in files],ignore_index=True).sort_values(['row','col','onset_idx']).reset_index(drop=True)
 assert len(ev)==1755878 and ev.event_num.is_unique
 ev.to_parquet(O/'paired/CDH_Original_Clean_Cohort.parquet',index=False)
 resume_y=int(os.environ.get('M3_RESUME_Y','0'));mode='r+' if resume_y else 'w+'
 N=len(ev);out=np.lib.format.open_memmap(O/'cache/Event_Metrics.npy',mode=mode,dtype='float64',shape=(N,len(V),len(FIELDS)))
 logs=np.lib.format.open_memmap(O/'cache/uWUE_Log_Components.npy',mode=mode,dtype='float64',shape=(N,4));diag=np.lib.format.open_memmap(O/'cache/Event_Diagnostics.npy',mode=mode,dtype='int16',shape=(N,4))
 l4=D/'L4_EcoFunction';raw=Dataset(l4/'Weihe_L4_EcoFunction_RAW_1000m_2000_2024_V3_1.nc');state=Dataset(l4/'Weihe_L4_EcoFunction_StateSpace_1000m_2000_2024_V3_1.nc')
 dates=pd.DatetimeIndex([str(x) for x in num2date(raw['time'][:],raw['time'].units,only_use_cftime_datetimes=False)]);td=(dates-dates[0]).days.to_numpy(np.int32);phase=np.minimum(46,((dates.dayofyear.to_numpy()-1)//8)+1);year=dates.year.to_numpy()
 np.save(O/'cache/time_days.npy',td);np.save(O/'cache/dates.npy',dates.values)
 a=np.load(M/'cache/Acomp_by_row.npy',mmap_mode='r');on=np.load(M5/'cache/Catalogue_Onsets_Union.npy',mmap_mode='r')
 history=[];validcounts=np.zeros((len(V),3),np.int64);trajectory=np.zeros((2,21,4),float);validated=False
 for y0 in range(resume_y,414,6):
  y1=min(414,y0+6);idx=np.flatnonzero((ev.row>=y0)&(ev.row<y1))
  if not len(idx):continue
  cols=((ev.col.to_numpy()[idx])+(ev.row.to_numpy()[idx]-y0)*703).astype(np.int32);ss=ev.onset_idx.to_numpy(np.int32)[idx];tt=ev.term_idx.to_numpy(np.int32)[idx]
  zsync=read(state,'ZSync',y0,y1);zu=read(state,'ZuWUE',y0,y1);zn=read(state,'ZNIRv',y0,y1)
  sync=read(raw,'Sync',y0,y1);g=read(raw,'GPP',y0,y1);et=read(raw,'ET',y0,y1);vpd=read(raw,'VPD',y0,y1);uw=read(raw,'uWUE',y0,y1)
  if not validated and np.isfinite(sync).sum()>1000:
   src=(O/'audit/Step_L4_1_build_EcoFunction_Robust_V3.py').read_text(encoding='utf-8-sig');tree=ast.parse(src)
   names=['cumulative_with_zero','rolling_sum_from_cumulative','calculate_sync_block','centered_window_bounds'];funcs=[q for q in tree.body if isinstance(q,ast.FunctionDef) and q.name in names]
   env={'np':np};exec(compile(ast.Module(body=funcs,type_ignores=[]),'<source functions only>','exec'),env)
   st,en=env['centered_window_bounds'](1142,7);zz=env['calculate_sync_block'](g[:,:,None],et[:,:,None],st,en,5).reshape(sync.shape)
   diff=np.abs(zz-sync);good=np.isfinite(diff);maxdiff=float(diff[good].max());mismatch=int((np.isfinite(zz)!=np.isfinite(sync)).sum());assert maxdiff<1e-5 and mismatch==0,(maxdiff,mismatch)
   # Reuse shift identity, verified against executable helper with trailing bounds.
   trailing=np.full_like(sync,np.nan);trailing[3:]=sync[:-3]
   st=np.maximum(0,np.arange(1142)-6);en=np.arange(1142)+1
   zz=env['calculate_sync_block'](g[:,:,None],et[:,:,None],st,en,5).reshape(sync.shape)
   good=np.isfinite(zz)&np.isfinite(trailing);md=float(np.abs(zz[good]-trailing[good]).max());mm=int((np.isfinite(zz)!=np.isfinite(trailing)).sum());assert md<1e-5 and mm==0,(md,mm)
   history.append(dict(Check='raw_centered_and_shifted_trailing_vs_executable',Y0=y0,Centered_MaxAbs=maxdiff,Centered_MissingMismatch=mismatch,Trailing_MaxAbs=md,Trailing_MissingMismatch=mm));validated=True
  trailing=np.full_like(sync,np.nan);trailing[3:]=sync[:-3];zt=zscore(trailing,phase,year)
  # WUE uses exactly the original accepted uWUE support, including upper QC.
  support=np.isfinite(uw)&np.isfinite(g)&np.isfinite(et)&np.isfinite(vpd)&(g>=.01)&(et>=.05)&(vpd>=.001)
  wue=np.where(support,g/et,np.nan).astype(np.float32);zw=zscore(wue,phase,year);zg=zscore(np.where(support,g,np.nan).astype(np.float32),phase,year)
  negs=[np.maximum(0,-np.clip(z,-10,10)).astype(np.float32) for z in [zsync,zu,zn,zt,zg]]
  joint=np.isfinite(zsync)&np.isfinite(zu)&np.isfinite(zn)
  A=np.transpose(a[y0:y1],(1,0,2)).reshape(1142,-1).copy();A2=np.sqrt((negs[0]*negs[0]+negs[1]*negs[1])/np.float32(2));A2[~joint]=np.nan
  Atr=np.sqrt((negs[3]*negs[3]+negs[1]*negs[1]+negs[2]*negs[2])/np.float32(3));Atr[~(np.isfinite(zt)&np.isfinite(zu)&np.isfinite(zn))]=np.nan
  recalc=np.sqrt((negs[0]*negs[0]+negs[1]*negs[1]+negs[2]*negs[2])/np.float32(3));recalc[~joint]=np.nan
  assert np.array_equal(np.isfinite(A),np.isfinite(recalc)) and np.nanmax(np.abs(A-recalc))<1e-6
  cube=np.stack([A,A2,negs[2],negs[4],Atr,-zsync,-zt,-zu,-zw,negs[0],negs[1]])
  for k in range(len(V)):validcounts[k]+=np.array([np.isfinite(cube[k]).sum(),cube[k].size,joint.sum()])
  ons=np.transpose(on[y0:y1],(1,0,2)).reshape(1142,-1)
  rr,dd=extract(ss,tt,cols,td,cube,ons);out[idx]=rr;diag[idx]=dd
  peaks=np.where(np.isfinite(rr[:,7,8]),rr[:,7,8],-1).astype(np.int32)
  # Same accepted positive raw support for logs; no clipped/Z quantities decomposed.
  lg=log_terms(ss,cols,td,np.where(support,g,np.nan),np.where(support,et,np.nan),np.where(support,vpd,np.nan),peaks);logs[idx]=lg
  # Paired event-centered Z anomaly trajectories, each relative to own pre median.
  pair=np.isfinite(rr[:,7,0])&np.isfinite(rr[:,8,0])
  for j,offset in enumerate(range(-5,16)):
   k=ss+offset;use=pair&(k>=0)&(k<1142);ix=np.flatnonzero(use);k=k[use];cc=cols[use]
   aa=-cube[7,k,cc]+rr[ix,7,0];bb=-cube[8,k,cc]+rr[ix,8,0];ok=np.isfinite(aa)&np.isfinite(bb);aa=aa[ok];bb=bb[ok]
   for v,q in enumerate([aa,bb]):trajectory[v,j]+=np.array([len(q),q.sum(),(q*q).sum(),(td[k[ok]]-td[ss[ix[ok]]]).sum()])
  out.flush();diag.flush();logs.flush()
  print('block',y0,y1,'events',len(idx),'elapsed',round(time.time()-start,1),flush=True)
  del zsync,zu,zn,sync,g,et,vpd,uw,trailing,zt,zw,zg,negs,A,A2,Atr,recalc,cube,ons,rr,dd,lg;gc.collect()
 raw.close();state.close();assert validated
 pd.DataFrame(history).to_csv(O/'audit/Sync_Source_Reproduction.csv',index=False)
 pd.DataFrame(dict(Variant=V,Valid_Phase_Pixels=validcounts[:,0],Total_Grid_Phase_Pixels=validcounts[:,1],Original_Joint_Valid=validcounts[:,2],Scope=f'computed rows{resume_y}:414; operational record, not full-cohort missingness')).to_csv(O/'paired/Phase_Validity.csv',index=False)
 np.save(O/'paired/uWUE_Trajectory_Aggregates.npy',trajectory)
 (O/'cache/EXTRACTION_READY.json').write_text(json.dumps(dict(N_Events=N,Variants=V,Fields=FIELDS,Log_Columns=['Delta_logGPP','Minus_Delta_logET','Half_Delta_logVPD','Delta_loguWUE'],Scope='all existing CDH clean-pre finite severity; original event IDs fixed',Resume_Y=resume_y,Trajectory_Scope=f'rows{resume_y}:414 only; replace with balanced common-event sample for reporting',Legacy_Definition='observed positive excess post-termination, M5 postpeak and catalogue interruption,120d horizon; not historical L5.3 legacy',Seconds=time.time()-start),indent=2),encoding='utf8')
 print('M3 EXTRACTION READY',N,flush=True)
if __name__=='__main__':main()
