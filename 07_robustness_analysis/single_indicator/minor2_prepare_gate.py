# -*- coding: utf-8 -*-
from pathlib import Path
import json,hashlib,shutil,ast,time,gc
import numpy as np
import pandas as pd
from netCDF4 import Dataset,num2date
from minor2_core import R,O,M1,M5,F,event_metrics
def main():
 start=time.time()
 for n in ['audit','cache','paired','supplement','code']:(O/n).mkdir(parents=True,exist_ok=True)
 for base in [R/'Weihe database',R/'Weihe database V3',R/'analysis_project',R]:
  p=base/'AGENTS.md'
  if p.exists():raise RuntimeError('Read applicable instructions first: '+str(p))
 inv=[]
 for base in [R/'Weihe database V3',R/'analysis_project']+[R/f'EI_Reviewer1_Major{k}_Revision' for k in [1,2,3,5,6]]:
  for p in base.rglob('*'):
   if p.is_file():s=p.stat();inv.append(dict(Path=str(p),Bytes=s.st_size,Mtime_ns=s.st_mtime_ns))
 pd.DataFrame(inv).to_csv(O/'audit/Inputs_Before.csv',index=False)
 src=[R/'analysis_project/analysis_code'/n for n in ['Step_L4_1_build_EcoFunction_Robust_V3.py','Step_L4_2_build_Phenology_Zscore_Robust_V1.py','Step_L4_3_build_EcoFunction_StateSpace_Robust_V1.py','Step_L5_3_Build_Event_Severity_Resilience_Metrics_Robust_V2_FAST.py']]+[M5/'code/extract_m5.py',M5/'code/survival_m5.py',M5/'code/analyse_m5.py',R/'EI_Reviewer1_Major3_Revision/code/m3_extract.py',R/'EI_Reviewer1_Major6_Revision/code/m6_extract.py',M1/'audit/Step_L5_4A_Severity_Response_Thresholds_FAST.py']
 sources=[];functions=[]
 for i,p in enumerate(src):
  b=p.read_bytes();dest=O/'audit'/p.name;dest.write_bytes(b);sources.append(dict(Source=str(p),Snapshot=dest.name,SHA256=hashlib.sha256(b).hexdigest()))
  for f in ast.walk(ast.parse(b.decode('utf-8-sig'))):
   if isinstance(f,ast.FunctionDef):functions.append(dict(Source=str(p),Function=f.name,Start=f.lineno,End=f.end_lineno))
 pd.DataFrame(sources).to_csv(O/'audit/Source_Manifest.csv',index=False);pd.DataFrame(functions).to_csv(O/'audit/Source_Function_Locations.csv',index=False)
 raw=R/'Weihe database V3/L4_EcoFunction/Weihe_L4_EcoFunction_RAW_1000m_2000_2024_V3_1.nc';state=R/'Weihe database V3/L4_EcoFunction/Weihe_L4_EcoFunction_StateSpace_1000m_2000_2024_V3_1.nc'
 with Dataset(raw) as a,Dataset(state) as b:
  dates=pd.DatetimeIndex([str(t) for t in num2date(a['time'][:],a['time'].units,only_use_cftime_datetimes=False)]);dd=pd.DatetimeIndex([str(t) for t in num2date(b['time'][:],b['time'].units,only_use_cftime_datetimes=False)]);assert np.array_equal(dates.values,dd.values) and np.array_equal(a['x'][:],b['x'][:]) and np.array_equal(a['y'][:],b['y'][:]);meta={v:{k:a[v].getncattr(k) for k in a[v].ncattrs()} for v in ['GPP','ET','NIRv']}
 (O/'audit/Raw_Input_Attributes.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2,default=lambda v:v.item() if isinstance(v,np.generic) else str(v)),encoding='utf8');td=(dates-dates[0]).days.to_numpy(np.int32);np.save(O/'cache/time_days.npy',td);np.save(O/'cache/dates.npy',dates.values);np.save(O/'cache/phase.npy',np.minimum(46,(dates.dayofyear.to_numpy()-1)//8+1));np.save(O/'cache/year.npy',dates.year.to_numpy());assert np.array_equal(dates.values,np.load(M5/'cache/dates.npy'))
 cfg=dict(Date='2026-11-12',Indicators=['Acomp','NIRv','GPP'],Fields=F,Reference='pixel x calendarphase2000-2020,n>=10,ddof1,SD>1e-6;float64parameter calculation thenfloat32;ZNIRv originalstored',GPP_Input='RAW GPP own finite inheritedQA values only; no ET/VPD/uWUE support or extra GPPthreshold',GPP_M3_Reuse='FORBIDDEN:oldM3GPP imposed uWUEsupport',Aneg='max(0,-clip(Z,-10,10));Acomp unchangedRMS',Event_Cohort='existing hazard-specificclean-pre finiteS catalogue, sameIDs/onset/term/severity',Primary_Pairing='intersection of3 native baseline/peak/validfollowup events, no new positive-damage gate; nativephase masks retained and reported',Strict_Phase_Sensitivity='repeat3 eventmetrics on exact common finite phase support, same unchanged referenceZ; separate from formalnativeAcomp',Recovery='M5PRIMARY termination-origin0..120inclusive;strictpostpeak;2adjacentfinite confirmations before newcatalogueonset;missingreset/trailingcensor',Legacy_Primary='M3/M5-rule actual observed post-termination positive-excess integral, partialfinalcellclipped, stopatownrecovery/interruption/horizon; not censor-adjusted fullburden',Legacy_Secondary='exacthistoricalL5.3onset120/active-next integral, separateversion',Severity='complete original hazard ECDF,float64rightupperrank,tiesunsplit;Low<50,Moderate50<=Q<75,High75<=Q<90,Extreme>=90; preserveM5High/Extreme',Bootstrap_N=250,Minimum_Valid_Bootstrap=200,Seed=20261109,Block_km=50,Shared_Weights='sameM5wholeblockmultinomialweightsseed+50 acrossindicators/groups;directpaireddifferences',Discordance_120='tri-common eligible event subset withknown120status inall3: recovered oradministrative120only;next/missing/recordendcannotbeunrecovered120',Discordance_Observed_Order='single recovered earlier thanAcomp andAcompobservedbeyondthatday, orreverse; unknowncensoredstatusnotreclassified',Legacy_Weak_Diagnostic='numericzero<=1e-12 separatefrommissing; optionalownmatchedpositiveQ25vsAcompownQ75 rankdiagnostic, notcross-scalephysicalburden',Gate='stop allsinglecomparisons unless nativeAcomppre/damage/lag/historicallegacy andallM5records/counts reproduce',No_Source_Writes=True,No_New_HGB_Training=True)
 (O/'code/configuration.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding='utf8')
 use=['event_num','row','col','onset_idx','term_idx','climate_cumulative_severity','Acomp_pre','Acomp_damage','Acomp_peak_lag_days','Acomp_legacy_anomaly_days','valid_resistance_sample'];ev={};out={}
 for h in ['CDH','HEAT','HR']:
  folder=O/'cache'/h;folder.mkdir(exist_ok=True);e=pd.concat([pd.read_parquet(p,columns=use) for p in sorted((M1/'cache'/h).glob('clean_*.parquet'))],ignore_index=True);assert e.event_num.is_unique;e.to_parquet(O/'paired'/f'{h}_Fixed_Clean_Events.parquet',index=False);ev[h]=e;out[h]=np.lib.format.open_memmap(folder/'Acomp.npy',mode='w+',dtype='float32',shape=(len(e),len(F)))
 A=np.load(M1/'cache/Acomp_by_row.npy',mmap_mode='r');on=np.load(M5/'cache/Catalogue_Onsets_Union.npy',mmap_mode='r');active=np.load(M1/'cache/AnyMask_by_row.npy',mmap_mode='r')
 for y0 in range(0,414,6):
  y1=min(414,y0+6);cube=np.transpose(A[y0:y1],(1,0,2)).reshape(1,1142,-1);ons=np.transpose(on[y0:y1],(1,0,2)).reshape(1142,-1);act=np.transpose(active[y0:y1],(1,0,2)).reshape(1142,-1)
  for h,e in ev.items():
   ix=np.flatnonzero((e.row.to_numpy()>=y0)&(e.row.to_numpy()<y1))
   if not len(ix):continue
   ss=e.onset_idx.to_numpy(np.int32)[ix];tt=e.term_idx.to_numpy(np.int32)[ix];cc=e.col.to_numpy(np.int32)[ix]+(e.row.to_numpy(np.int32)[ix]-y0)*703;out[h][ix]=event_metrics(ss,tt,cc,td,cube,ons,act)[:,0]
  if y0%48==0:print('GATE rows',y0,y1,'seconds',round(time.time()-start,1),flush=True)
 for z in out.values():z.flush()
 checks=[];allpass=True
 for h,e in ev.items():
  m=out[h]
  for label,f,col,tol in [('Baseline',0,'Acomp_pre',1e-6),('Damage',1,'Acomp_damage',1e-6),('Peak_Lag',2,'Acomp_peak_lag_days',0),('Historical_Legacy',14,'Acomp_legacy_anomaly_days',2e-4)]:
   a=e[col].to_numpy();b=m[:,f];finite=np.isfinite(a)&np.isfinite(b);mm=int((np.isfinite(a)!=np.isfinite(b)).sum());err=float(np.max(abs(a[finite]-b[finite]))) if finite.any() else 0;ok=mm==0 and err<=tol;allpass&=ok;checks.append(dict(Event=h,Check=label,N_Checked=len(e),N_Finite=int(finite.sum()),Missing_Mismatch=mm,MaxAbs=err,Tolerance=tol,Pass=ok))
  index=np.full(int(e.event_num.max())+1,-1,np.int32);index[e.event_num.to_numpy(int)]=np.arange(len(e));n=0;mis=np.zeros(5,np.int64)
  for p in sorted((M5/'event_records'/h).glob('*.parquet')):
   old=pd.read_parquet(p,columns=['event_num','Followup_Days','Outcome_Code','Valid_Followup','Recovery_Date','Event_Termination','Recovery_Confirmation_Date']);ix=index[old.event_num.to_numpy(int)];assert (ix>=0).all();n+=len(old)
   for j,(f,col) in enumerate([(7,'Followup_Days'),(8,'Outcome_Code'),(9,'Valid_Followup')]):mis[j]+=int((m[ix,f]!=old[col].to_numpy()).sum())
   for j,(f,col) in enumerate([(10,'Recovery_Date'),(11,'Recovery_Confirmation_Date')],3):
    days=(old[col]-old.Event_Termination).dt.days.fillna(-1).to_numpy();mis[j]+=int((m[ix,f]!=days).sum())
  for name,k in [('M5_Duration',0),('M5_Outcome',1),('M5_Valid',2),('M5_Recovery_Day',3),('M5_Confirmation_Day',4)]:checks.append(dict(Event=h,Check=name,N_Checked=n,Missing_Mismatch=int(mis[k]),MaxAbs=int(mis[k]),Tolerance=0,Pass=mis[k]==0));allpass&=mis[k]==0
  full=np.load(M1/'cache'/h/'all_severity_sorted.npy',mmap_mode='r');q=100*np.searchsorted(full,e.climate_cumulative_severity.to_numpy(),side='right')/len(full);np.save(O/'cache'/h/'Full_ECDF_Q.npy',q);blockmap=np.load(M5/'cache'/f'{h}_pixel_blocks_50.npy',mmap_mode='r');blocks=blockmap[e.row.to_numpy(int),e.col.to_numpy(int)];assert (blocks>=0).all();np.save(O/'cache'/h/'Block50.npy',blocks);B=int(blockmap.max()+1);ix=(q>=75)&(m[:,9]==1);g=(q[ix]>=90).astype(int);dt=m[ix,7].astype(int);reason=m[ix,8].astype(int);C=np.bincount((((blocks[ix]*2+g)*5+reason)*121+dt),minlength=B*2*5*121).reshape(B,2,5,121);old=np.load(M5/'cache'/f'{h}_Aggregates.npz')['counts_50_PRIMARY_POSTPEAK_CATALOG_INCLUSIVE'];ok=np.array_equal(C,old);allpass&=ok;checks.append(dict(Event=h,Check='M5_Spatial_Counts_Exact',N_Checked=int(C.sum()),MaxAbs=int(np.max(abs(C-old))),Tolerance=0,Pass=ok))
 pd.DataFrame(checks).to_csv(O/'audit/Acomp_Gate_Verification.csv',index=False);(O/'cache/ACOMP_GATE.json').write_text(json.dumps(dict(Pass=bool(allpass),N_Checks=len(checks),N_Indicators_Compared=0,Seconds=time.time()-start),indent=2),encoding='utf8');print(pd.DataFrame(checks).to_string(index=False),flush=True)
 if not allpass:raise RuntimeError('FAIL: originalAcompnotreproduced; singleindicatorsNOTSTARTED')
 print('ACOMP GATE PASS',round(time.time()-start,1),flush=True)
if __name__=='__main__':main()
