# -*- coding: utf-8 -*-
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import json,hashlib,time,gc,sys
import numpy as np
import pandas as pd
from netCDF4 import Dataset,num2date
from numba import njit
from audit_core import R,O,WORK,VARS,event_values,write_case
M5=R/'EI_Reviewer1_Major5_Revision';M1=R/'EI_Reviewer1_Major1_Revision';MIN=R/'EI_Reviewer1_Minor2_SingleIndicator';P=R/'analysis_project';DB=R/'Weihe database V3'
NPIX=414*703;HAZ=['CDH','HEAT','HR'];TOL=2e-6
@njit(cache=True)
def episode_indices(s,t,c,nt,nc):
 size=0
 for i in range(len(s)):size+=t[i]-s[i]+1
 out=np.empty(size,np.int64);owner=np.empty(size,np.int32);k=0
 for i in range(len(s)):
  for a in range(s[i],t[i]+1):out[k]=a*nc+c[i];owner[k]=i;k+=1
 return out,owner
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for q in iter(lambda:f.read(8*1024*1024),b''):h.update(q)
 return h.hexdigest()
def snapshot():
 roots=[R/n for n in ['Weihe database','Weihe database V2','Weihe database V3','analysis_project']]+[p for p in R.glob('EI_*') if p.is_dir()];result={}
 for root in roots:
  for p in root.rglob('*'):
   if p.is_relative_to(O):continue
   if p.is_file():s=p.stat();result[str(p)]=(s.st_size,s.st_mtime_ns)
 return result
def idxdate(series,dates):
 a=series.to_numpy('datetime64[D]');ok=~np.isnat(a);v=np.full(len(a),-1,np.int16);v[ok]=np.searchsorted(dates,a[ok]);assert np.array_equal(dates[v[ok]],a[ok]);return v
def main():
 start=time.time()
 for n in ['scripts','outputs','qc','outputs/event_details']: (O/n).mkdir(parents=True,exist_ok=True)
 if (O/'qc/EXTRACTION_READY.json').exists():print('Existing completed extraction; not repeated.');return
 if WORK.exists() and any(WORK.iterdir()):raise RuntimeError('Incomplete work cache present; inspect rather than append duplicate records.')
 WORK.mkdir(parents=True,exist_ok=True)
 before=snapshot();(O/'qc/Protected_Files_Before.json').write_text(json.dumps(before),encoding='utf8')
 state=DB/'L4_EcoFunction/Weihe_L4_EcoFunction_StateSpace_1000m_2000_2024_V3_1.nc';src=[state,M5/'code/configuration.json',M5/'code/extract_m5.py',M5/'code/survival_m5.py',M5/'M5_Figure6ef_SourceData.csv',MIN/'cache/ACOMP_GATE.json',M5/'cache/CDH_pixel_blocks_50.npy',M5/'cache/dates.npy']+[P/'analysis_code'/s for s in ['Step_L4_1_build_EcoFunction_Robust_V3.py','Step_L4_2_build_Phenology_Zscore_Robust_V1.py','Step_L4_3_build_EcoFunction_StateSpace_Robust_V1.py']]
 # Large NC provenance uses path/stat plus beginning/end hashes; avoid a second full4.5GB read.
 manifest=[]
 for p in src:
  assert p.exists();s=p.stat()
  if s.st_size>500_000_000:
   h=hashlib.sha256()
   with open(p,'rb') as f:h.update(f.read(8*1024*1024));f.seek(-8*1024*1024,2);h.update(f.read())
   method='first+last8MiB SHA256; metadata invariant; not a full-file digest';hs=h.hexdigest()
  else:method='full SHA256';hs=sha(p)
  manifest.append(dict(Path=str(p),Bytes=s.st_size,Mtime_ns=s.st_mtime_ns,SHA256=hs,Hash_Method=method))
 pd.DataFrame(manifest).to_csv(O/'qc/Input_Manifest.csv',index=False,encoding='utf-8-sig')
 dates=np.load(M5/'cache/dates.npy').astype('datetime64[D]');td=np.load(M5/'cache/time_days.npy');grid=np.load(M5/'cache/CDH_pixel_blocks_50.npy');assert grid.max()==75
 W=np.random.default_rng(20261159).multinomial(76,np.full(76,1/76),size=250).astype(np.int32);np.save(O/'qc/Spatial_Bootstrap_250_Weights.npy',W)
 cfg=dict(Date='2026-11-16',Hazards=HAZ,Event_Cohort='original M5 High/Extreme baseline-and-peak cohort only, no newpositiveDamage filter; Low/Moderate not labelled originalM5',Global_Observations='all jointly finite ZSync,ZuWUE,ZNIRv,Acomp phase-pixels2000–2024 within archived basin support',Event_Observations='jointly finite phases within original M5 merged onset..termination interval, gap phases retained; hazards never pooled',Peak_Observations='one observation at original stored Acomp peak per formal M5 event; no alternate peak search',Aneg='max(0,-clip(originalZ,-10,10))',Contribution='float64 numerator/denominator from originalfloat32Z thenfloat32stored shares; exact quantiles of those stored shares',Primary_Component_Recovery='own nativefinite negative-component pre40d median, >=3; both actual composite recovery observations <= ownmedian',Z_Recovery_Sensitivity='own nativefinite rawZ pre40d median, >=3; both actual observations >= ownmedian; independent criterion',Bootstrap_N=250,Seed=20261159,Blocks=76,Block_km=50,Projection='existing M5/M1 AEQD lat35.5 lon107E, metres; grid labels reused, no reconstruction',Record_Spearman_CI='exact pooledmidrank point; fixed-pooled-rank conditional spatial bootstrap for very large GLOBAL/EPISODE series; variance of rank-distribution estimation not included; explicit in rows',Exact_Reranked_Spearman='pixel-mean and event-peak samples rerank within each whole-block replicate',No_New_Recovery_Fit=True,No_Model_Training=True,No_ZIP=True)
 (O/'qc/configuration.json').write_text(json.dumps(cfg,indent=2),encoding='utf8')
 sourceg=pd.read_csv(M5/'M5_Figure6ef_SourceData.csv');events={};cohort=[];rawfiles=[]
 cols=['event_num','row','col','onset_idx','term_idx','Acomp_pre','Ecological_Peak_Date','Recovery_Date','Recovery_Confirmation_Date','Next_Extreme_Onset','Outcome_Code','Valid_Followup','Full_ECDF_Q']
 for h in HAZ:
  parts=[]
  for f in sorted((M5/'event_records'/h).glob('*.parquet')):
   d=pd.read_parquet(f,columns=cols);rawfiles.append(dict(Hazard=h,File=str(f),Rows=len(d),Bytes=f.stat().st_size,Mtime_ns=f.stat().st_mtime_ns))
   e=pd.DataFrame(dict(event_num=d.event_num.astype('int32'),row=d.row.astype('int16'),col=d.col.astype('int16'),onset=d.onset_idx.astype('int16'),term=d.term_idx.astype('int16'),baseline=d.Acomp_pre.astype('float32'),peak=idxdate(d.Ecological_Peak_Date,dates),rec=idxdate(d.Recovery_Date,dates),confirm=idxdate(d.Recovery_Confirmation_Date,dates),next=idxdate(d.Next_Extreme_Onset,dates),outcome=d.Outcome_Code.astype('int8'),valid=d.Valid_Followup.astype('int8'),Q_full=d.Full_ECDF_Q));parts.append(e)
  e=pd.concat(parts,ignore_index=True);assert e.event_num.is_unique;assert (e.Q_full>=75).all();assert (e.peak>=0).all();assert (e['confirm'][e.outcome==0]>=0).all()
  nr=int((e.outcome==0).sum());assert nr==sourceg[sourceg.Event==h].N_recovered.sum()
  cohort.append(dict(Event=h,N_Formal_Events=len(e),N_Formal_Survival_Eligible=int(e.valid.sum()),N_Formal_Confirmed_Recovered=nr,N_50km_Whole_Blocks=76,Scope='original M5 High/Extreme only'))
  events[h]=e
  e.to_parquet(O/f'outputs/event_details/{h}_Formal_M5_Event_Registry.parquet',index=False,compression='zstd')
 pd.DataFrame(rawfiles).to_csv(O/'qc/M5_Event_File_Manifest.csv',index=False);pd.DataFrame(cohort).to_csv(O/'qc/Original_M5_Cohort_Counts.csv',index=False)
 cases=['GLOBAL_COMMON']+[h+'_'+kind for h in HAZ for kind in ['EPISODE','PEAK']]
 registry={k:dict(N=0,stats=np.zeros((76,21)),pixels=np.zeros((NPIX,8)),absdiff=np.zeros((76,3)),zero=np.zeros(76),Cstats=np.zeros((76,4)),Csum_error=0.) for k in cases}
 diag=[];samples=[];globalerr=0.;globalmissing=0;gatebad=0;Bbad=0;Bdone={h:0 for h in HAZ};nt=len(td)
 F=['Acomp_Baseline_Effective','Acomp_First','Acomp_Confirm']+[f'Pre_N_{v}' for v in ['Sync','uWUE','NIRv']]+[f'Pre_A_{v}' for v in ['Sync','uWUE','NIRv']]+[f'Pre_Z_{v}' for v in ['Sync','uWUE','NIRv']]+[f'First_Z_{v}' for v in ['Sync','uWUE','NIRv']]+[f'Confirm_Z_{v}' for v in ['Sync','uWUE','NIRv']]+[f'First_A_{v}' for v in ['Sync','uWUE','NIRv']]+[f'Confirm_A_{v}' for v in ['Sync','uWUE','NIRv']]+['Components_Complete']+[f'Recovered_A_{v}' for v in ['Sync','uWUE','NIRv']]+[f'Recovered_Z_{v}' for v in ['Sync','uWUE','NIRv']]+['Pattern_A','Pattern_Z','RMS_First_Rebuilt','RMS_Confirm_Rebuilt','Pre_N_Acomp','Pre_Acomp_Rebuilt','Recovery_After_Peak_Days','Confirmation_After_Term_Days','Confirmation_Index_Gap']
 with Dataset(state,'r') as ds:
  actual=np.asarray([str(v) for v in num2date(ds['time'][:],ds['time'].units,only_use_cftime_datetimes=False)],dtype='datetime64[D]');assert np.array_equal(actual,dates)
  for name in ['ZSync','ZuWUE','ZNIRv','A_comp']:ds[name].set_var_chunk_cache(4*1024*1024,1009,.75)
  for y0 in range(0,414,12):
   y1=min(414,y0+12);nc=(y1-y0)*703;z=np.empty((3,nt,nc),np.float32)
   for j,name in enumerate(['ZSync','ZuWUE','ZNIRv']):z[j]=np.ma.filled(ds[name][:,y0:y1,:],np.nan).astype(np.float32,copy=False).reshape(nt,nc)
   ac=np.ma.filled(ds['A_comp'][:,y0:y1,:],np.nan).astype(np.float32,copy=False).reshape(nt,nc)
   common=np.isfinite(z).all(axis=0)&np.isfinite(ac);flat=np.flatnonzero(common.ravel());ti=(flat//nc).astype(np.int16);local=(flat%nc).astype(np.int32);pix=local+y0*703;bl=grid.ravel()[pix];assert (bl>=0).all()
   X=np.empty((len(flat),7),np.float32);X[:,:3]=z.reshape(3,-1)[:,flat].T;X[:,3]=ac.ravel()[flat]
   an=np.maximum(0,-np.clip(X[:,:3],-10,10));sq=an*an;rebuilt=np.sqrt(sq.sum(axis=1)/3);err=float(np.max(abs(rebuilt-X[:,3]))) if len(X) else 0.;globalerr=max(globalerr,err);gatebad+=int((abs(rebuilt-X[:,3])>TOL).sum());assert gatebad==0,('STOP:Acomp version mismatch',gatebad,err)
   for j in range(3):X[:,4+j]=np.sqrt((sq[:,(j+1)%3]+sq[:,(j+2)%3])/2)
   # Numerical subtraction can produce tinynegative from rounding; protect only algebraic roundoff, not missing data.
   assert np.isfinite(X).all()
   write_case('GLOBAL_COMMON',X,bl,pix,registry)
   for j in np.linspace(0,len(X)-1,min(32,len(X)),dtype=int):samples.append(dict(row=int(pix[j]//703),col=int(pix[j]%703),Date=str(dates[ti[j]]),Block50=int(bl[j]),ZSync=float(X[j,0]),ZuWUE=float(X[j,1]),ZNIRv=float(X[j,2]),Acomp=float(X[j,3]),Rebuilt=float(rebuilt[j]),Abs_Error=float(abs(rebuilt[j]-X[j,3]))))
   # Within-hazard episodes and originalpeak values reference exact original event IDs.
   for h,e in events.items():
    sel=np.flatnonzero((e.row.to_numpy()>=y0)&(e.row.to_numpy()<y1));sub=e.iloc[sel];cc=(sub.row.to_numpy(np.int32)-y0)*703+sub.col.to_numpy(np.int32);pks=sub.peak.to_numpy(np.int32);loc=pks*nc+cc
    peakX=np.empty((len(sel),7),np.float32);peakX[:,:3]=z.reshape(3,-1)[:,loc].T;peakX[:,3]=ac.ravel()[loc];aa=np.maximum(0,-np.clip(peakX[:,:3],-10,10));qq=aa*aa
    for j in range(3):peakX[:,4+j]=np.sqrt((qq[:,(j+1)%3]+qq[:,(j+2)%3])/2)
    ok=np.isfinite(peakX).all(axis=1);assert ok.all();pp=sub.row.to_numpy(np.int32)*703+sub.col.to_numpy(np.int32);bb=grid.ravel()[pp]
    write_case(h+'_PEAK',peakX,bb,pp,registry)
    peakout=sub[['event_num','row','col','onset','term','peak','Q_full']].copy();peakout['Date']=dates[pks];peakout['Block50']=bb
    for j,n in enumerate(VARS):peakout[n]=peakX[:,j]
    peakout.to_parquet(O/f'outputs/event_details/{h}_Peak_rows_{y0:03d}.parquet',index=False,compression='zstd')
    ii,owner=episode_indices(sub.onset.to_numpy(np.int32),sub.term.to_numpy(np.int32),cc,nt,nc);ep=np.empty((len(ii),7),np.float32);ep[:,:3]=z.reshape(3,-1)[:,ii].T;ep[:,3]=ac.ravel()[ii];vv=np.maximum(0,-np.clip(ep[:,:3],-10,10));qq=vv*vv
    for j in range(3):ep[:,4+j]=np.sqrt((qq[:,(j+1)%3]+qq[:,(j+2)%3])/2)
    ok=np.isfinite(ep).all(axis=1);epix=pp[owner[ok]];eb=bb[owner[ok]];write_case(h+'_EPISODE',ep[ok],eb,epix,registry)
    # B: only real formally confirmed composite recoveries; no censored case is made recovered.
    recmask=sub.outcome.to_numpy()==0;rs=sub.loc[recmask];rr=rs.rec.to_numpy(np.int32);kk=rs['confirm'].to_numpy(np.int32);pk=rs.peak.to_numpy(np.int32);ss=rs.onset.to_numpy(np.int32);tt=rs.term.to_numpy(np.int32);rc=cc[recmask];nn=rs['next'].to_numpy(np.int32)
    assert np.all(rr>pk) and np.all(kk==rr+1) and np.all(td[kk]-td[tt]<=120) and np.all((nn<0)|(kk<nn))
    values=event_values(z,ac,td,ss,tt,rc,pk,rr,kk,rs.baseline.to_numpy(np.float32));complete=values[:,24]==1
    berr=max(float(np.max(abs(values[complete,33]-values[complete,1]))) if complete.any() else 0.,float(np.max(abs(values[complete,34]-values[complete,2]))) if complete.any() else 0.)
    Bbad+=int(((abs(values[complete,33]-values[complete,1])>TOL)|(abs(values[complete,34]-values[complete,2])>TOL)).sum());assert Bbad==0,('STOP:B RMS version mismatch',Bbad,berr)
    assert (values[:,1]<=values[:,0]).all() and (values[:,2]<=values[:,0]).all();assert np.max(abs(values[:,36]-values[:,0]))<1e-6
    result=rs[['event_num','row','col','onset','term','peak','rec','confirm','next','Q_full']].reset_index(drop=True)
    result['Hazard']=h;result['Event_ID']=[h+'_'+str(v).zfill(10) for v in result.event_num];result['Pixel_ID']=result.row.astype(np.int32)*703+result.col.astype(np.int32);result['Block50']=grid[result.row,result.col]
    for j,name in enumerate(F):result[name]=values[:,j]
    result['Recovery_Date']=dates[rr];result['Confirmation_Date']=dates[kk];result['Peak_Date']=dates[pk];result['Onset_Date']=dates[ss];result['Termination_Date']=dates[tt];result['Actual_Confirmation_Gap_Days']=td[kk]-td[rr]
    result.to_parquet(O/f'outputs/event_details/{h}_Recovery_rows_{y0:03d}.parquet',index=False,compression='zstd');Bdone[h]+=len(result)
    diag.append(dict(Hazard=h,Row_Start=y0,N_Formal_Recovered=len(result),N_Complete=int(complete.sum()),N_Excluded=int((~complete).sum()),Max_RMS_Confirmation_Error=berr,N_Actual_5day_Gap=int((result.Actual_Confirmation_Gap_Days==5).sum()),N_Actual_6day_Gap=int((result.Actual_Confirmation_Gap_Days==6).sum()),N_Actual_8day_Gap=int((result.Actual_Confirmation_Gap_Days==8).sum()),N_Date_Order_Failures=0,N_Acomp_Rule_Failures=0))
    del peakX,peakout,ep,ii,owner,vv,values,result;gc.collect()
   del X,z,ac,an,sq,rebuilt,flat,pix,ti,bl,common;gc.collect()
   for case,reg in registry.items():
    np.savez_compressed(WORK/case/'aggregates.npz',stats=reg['stats'],pixels=reg['pixels'],absdiff=reg['absdiff'],zero=reg['zero'],Cstats=reg['Cstats'],N=reg['N'],Csum_error=reg['Csum_error'])
   print('EXTRACT rows',y0,y1,'GLOBAL N',registry['GLOBAL_COMMON']['N'],'B recovered',Bdone,'seconds',round(time.time()-start,1),flush=True)
 for h,c in zip(HAZ,cohort):assert Bdone[h]==c['N_Formal_Confirmed_Recovered']
 pd.DataFrame(diag).to_csv(O/'qc/Recovery_Temporal_And_RMS_Gates.csv',index=False);pd.DataFrame(samples).to_csv(O/'qc/Global_Record_Reconstruction_Audit.csv',index=False)
 counts=[dict(Case=k,N=v['N'],N_Pixels=int((v['pixels'][:,0]>0).sum()),N_Blocks=int((v['stats'][:,0]>0).sum()),N_Acomp_Zero=int(v['zero'].sum()),Contribution_Sum_MaxError=v['Csum_error']) for k,v in registry.items()]
 pd.DataFrame(counts).to_csv(O/'qc/Observation_Cohorts.csv',index=False)
 assert before==snapshot();(O/'qc/EXTRACTION_READY.json').write_text(json.dumps(dict(Global_Acomp_MaxAbs=globalerr,Global_Acomp_Mismatch=gatebad,B_Acomp_Mismatch=Bbad,B_Recovered_Processed=Bdone,Seconds=time.time()-start,Cases=cases),indent=2),encoding='utf8');print('EXTRACTION READY',round(time.time()-start,1),flush=True)
if __name__=='__main__':main()
